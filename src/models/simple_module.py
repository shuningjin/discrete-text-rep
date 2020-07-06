import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import math
import logging as log
from src.task import pad_idx, eos_idx
from src.utils.util import input_from_batch
from src import device, cuda_device


def one_hot_argmax(y_soft, dim=-1):
    """
    Example:
    y_soft = [0.5, 0.2, 0.3] # logits vector (normalized or unnormalized)
    y_hard = [1., 0, 0]      # one-hot vector for argmax
    """
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    return y_hard


# Auxillary for transformer
def generate_square_subsequent_mask(sz, cuda_device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    mask = mask.to(device)
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PackedSequneceUtil(object):
    def __init__(self):
        self.is_packed = False
        self.pack_shape = None

    def preprocess(self, input):
        self.is_packed = isinstance(input, PackedSequence)
        if self.is_packed:
            input, *self.pack_shape = input
        return input

    def postprocess(self, output, pad):
        assert self.is_packed
        packed_ouput = PackedSequence(output, *self.pack_shape)
        padded_output = pad_packed_sequence(
            packed_ouput, batch_first=True, padding_value=pad
        )[0]
        return padded_output


class Classifier(nn.Module):
    def __init__(self, d_inp, n_classes, cls_type="log_reg", dropout=0.2, d_hid=512):
        super().__init__()

        # logistic regression
        if cls_type == "log_reg":
            classifier = nn.Linear(d_inp, n_classes)
        # mlp
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, n_classes),
            )
        # InferSent
        elif cls_type == "fancy_mlp":
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, n_classes),
            )
        self.classifier = classifier

    def forward(self, seq_emb):
        logits = self.classifier(seq_emb)
        return logits


class Pooler(nn.Module):
    """ Do pooling, possibly with a projection beforehand """

    def __init__(self, project=True, d_inp=512, d_proj=512, pool_type="max"):
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

        if self.pool_type == "attn":
            d_in = d_proj if project else d_inp
            self.attn = nn.Linear(d_in, 1, bias=False)

    def forward(self, sequence, mask):
        """
        sequence: (bsz, T, d_inp)
        mask: nopad_mask (bsz, T) or (bsz, T, 1)
        """
        # no pad in sequence
        if mask is None:
            mask = torch.ones(sequence.shape[:2], device=device)

        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)  # (bsz, T, 1)
        pad_mask = mask == 0
        proj_seq = self.project(sequence)  # (bsz, T, d_proj) or (bsz, T, d_inp)

        if self.pool_type == "max":
            proj_seq = proj_seq.masked_fill(pad_mask, -float("inf"))
            seq_emb = proj_seq.max(dim=1)[0]

        elif self.pool_type == "mean":
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1).float()

        elif self.pool_type == "final":
            idxs = mask.expand_as(proj_seq).sum(dim=1, keepdim=True).long() - 1
            seq_emb = proj_seq.gather(dim=1, index=idxs).squeeze(dim=1)

        elif self.pool_type == "first":
            seq_emb = proj_seq[:, 0]

        elif self.pool_type == "none":
            seq_emb = proj_seq

        return seq_emb

    def forward_dict(self, output_dict):
        """
        Arg - output_dict with keys:
            'output': sequence of vectors
            'nopad_mask': sequence mask, with 1 for non pad positions and 0 elsewhere
            'final_state' (optional): final hidden state of lstm
        Return - an aggregated vector
        """
        sequence = output_dict["sequence"]
        mask = output_dict["nopad_mask"]

        if self.pool_type == "final_state":
            assert "final_state" in output_dict
            out = output_dict["final_state"]
        else:
            out = self.forward(sequence, mask)
        return out


""" Baseline Encoder Classifiers """


class EncoderClassifier(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.encoder = encoder

        if config.classifier.freeze_encoder == 1:
            log.info("Freeze encoder parameters.")
            for param in self.encoder.parameters():
                param.detach_()
                param.requires_grad = False
        else:
            log.info("Training encoder parameters.")

        enc_dim = self.encoder.get_output_dim()
        assert config.classifier.aggregate in ["final_state", "max", "mean"]
        self.pooler = Pooler(
            project=False, d_inp=enc_dim, pool_type=config.classifier.aggregate
        )
        self.classifier = Classifier(enc_dim, config.cls_class, config.classifier.type)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.metric = BooleanAccuracy()

    def forward(self, batch):
        sent = input_from_batch(batch)["enc_in"]
        output_dict = self.encoder(sent)
        pooled_out = self.pooler.forward_dict(output_dict)
        logits = self.classifier(pooled_out)

        loss = self.criterion(logits, batch["labels"])
        pred = torch.argmax(logits, dim=-1)
        self.metric(pred, batch["labels"])

        out = {"pred": pred, "loss": loss}
        return out


class CBOWEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.cbow.d_model, padding_idx=0
        )
        self.output_dim = config.cbow.d_model

    def forward(self, sent):
        emb = self.embedding(sent)
        nopad_mask = sent != pad_idx

        return {"sequence": emb, "nopad_mask": nopad_mask}

    def get_output_dim(self):
        return self.output_dim


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size, config.lstm.d_model, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=config.lstm.d_model,
            hidden_size=config.lstm.h_dim,
            num_layers=config.lstm.enc_nlayers,
            bidirectional=(config.lstm.enc_ndirections == 2),
            bias=True,
            dropout=0,
            batch_first=True,
        )
        self.output_dim = config.lstm.h_dim * config.lstm.enc_ndirections

    def forward(self, enc_in):
        nopad_mask = enc_in != pad_idx
        nopad_lengths = nopad_mask.sum(dim=-1).long()
        batch_size = enc_in.shape[0]

        inp_embs = self.embedding(enc_in)
        # lstm
        packed_embs = pack_padded_sequence(
            inp_embs, lengths=nopad_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        packed_output, (hT, _) = self.lstm(packed_embs)

        output = pad_packed_sequence(packed_output, batch_first=True)[0]
        final = (
            hT.view(self.lstm.num_layers, self.lstm.bidirectional + 1, batch_size, -1)[
                -1
            ]
            .transpose(0, 1)
            .contiguous()
            .view(batch_size, -1)
        )

        return {"sequence": output, "nopad_mask": nopad_mask, "final_state": final}

    def get_output_dim(self):
        return self.output_dim


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.transformer.d_model, padding_idx=0
        )
        self.pos_encoder = PositionalEncoding(
            d_model=config.transformer.d_model, dropout=config.transformer.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            dim_feedforward=config.transformer.d_ffn,
            dropout=config.transformer.dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer.enc_nlayer
        )
        self.output_dim = config.transformer.d_model

    def forward(self, src):
        src_pad_mask = src == 0
        src_nopad_mask = src != 0
        nopad_lengths = src_nopad_mask.sum(dim=-1).long()

        src_emb = self.embedding(src).transpose(0, 1)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask).transpose(
            0, 1
        )

        return {"sequence": memory, "nopad_mask": src_nopad_mask}

    def get_output_dim(self):
        return self.output_dim
