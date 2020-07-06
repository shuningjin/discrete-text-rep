import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import logging as log

from src.models.simple_module import (
    PositionalEncoding,
    generate_square_subsequent_mask,
    Classifier,
    Pooler,
)
from src.models import (
    ConcreteQuantizer,
    HardEMQuantizer,
    DVQ,
    VectorQuantizer,
)
from src.task import pad_idx, eos_idx
from src.utils.util import input_from_batch
from src import device, cuda_device


class QuantizerforClassification(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.encoder = encoder

        # freeze
        self.freeze = config.classifier.freeze_encoder
        self._init_freeze()

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        enc_out_dim = self.encoder.get_output_dim()
        D = self.encoder.quantizer.D

        # reembedding
        self.reembedding = config.classifier.reembedding
        if self.reembedding == 1:
            if config.quantizer.level == "word":
                # add padding idx
                self.embedding_layers = nn.ModuleList(
                    [
                        nn.Embedding(config.quantizer.K + 1, D, padding_idx=0)
                        for _ in range(config.quantizer.M)
                    ]
                )
            elif config.quantizer.level == "sentence":
                # no padding idx
                self.embedding_layers = nn.ModuleList(
                    [
                        nn.Embedding(config.quantizer.K, D, padding_idx=None)
                        for _ in range(config.quantizer.M)
                    ]
                )

        # merge word
        self.quantizer_level = config.quantizer.level
        self.merge_word_split = config.classifier.merge_word
        if config.quantizer.level == "word":
            if self.merge_word_split == "concat":
                enc_out_dim = D * config.quantizer.M
            elif self.merge_word_split in ["sum", "sum_tanh", "sum_relu"]:
                enc_out_dim = D
                if self.merge_word_split in ["sum_tanh", "sum_relu"]:
                    self.bias = nn.Parameter(torch.randn(D))
        elif config.quantizer.level == "sentence":
            enc_out_dim = D

        # add layers on top
        self.add_layer = config.classifier.add_layer
        enc_out_dim = self._init_add_layer(config, enc_out_dim)

        self.pooler = Pooler(
            project=False, d_inp=enc_out_dim, pool_type=config.classifier.aggregate
        )

        self.classifier = Classifier(
            enc_out_dim, n_classes=config.cls_class, cls_type=config.classifier.type
        )

    def forward(self, batch):

        input = input_from_batch(batch)
        enc_in = input["enc_in"]
        batch_size = enc_in.shape[0]

        # encode
        enc_outdict = self.encoder(enc_in)

        # reemb
        if self.reembedding == 0:
            emb_seq = enc_outdict["quantizer_out"]["quantized_stack"]
        elif self.reembedding == 1:
            idx_seq = enc_outdict["quantizer_out"]["encoding_indices"]
            emb_seq = self._reemb(idx_seq)
        # merge word split
        emb_seq = self._merge_word_split(emb_seq)
        enc_outdict["sequence"] = emb_seq

        # add layer
        enc_outdict = self._add_layer(enc_outdict)

        pooled_out = self.pooler.forward_dict(enc_outdict)
        logits = self.classifier(pooled_out)

        loss = self.criterion(logits, batch["labels"])
        pred = torch.argmax(logits, dim=-1)

        out = {
            "pred": pred,
            "loss": loss,
        }

        return out

    # TODO
    def _init_freeze(self):
        if self.freeze == 1:
            log.info("Freeze all parameters in pretrained model.")
            for param in self.encoder.parameters():
                param.detach_()
                param.requires_grad = False
            # if isinstance(self.encoder.quantizer, VectorQuantizer):
            #    self.encoder.quantizer.EMA = 0
            if isinstance(self.encoder.quantizer, DVQ):
                for i in self.encoder.quantizer.vq_layers:
                    i.ema = 0
            elif isinstance(self.encoder.quantizer, ConcreteQuantizer) or isinstance(
                self.encoder.quantizer, HardEMQuantizer
            ):
                self.encoder.quantizer.force_eval = True
        elif self.freeze == 0:
            log.info("Careful! Train all parameters, no additional loss.")

    def _init_add_layer(self, config, enc_out_dim):
        if self.add_layer == "transformer":
            log.info("add: transformer encoder layer.")
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=enc_out_dim, nhead=4, dim_feedforward=256, dropout=0.1
            )
        elif self.add_layer == "lstm":
            log.info("add: bilstm layer.")
            self.lstm = nn.LSTM(
                input_size=enc_out_dim,
                hidden_size=50,
                num_layers=1,
                bidirectional=True,
                bias=True,
                dropout=0,
                batch_first=True,
            )
            enc_out_dim = 50 * 2
        elif self.add_layer == "ffn":
            log.info("add: feedforward layer.")
            assert config.quantizer.level == "sentence"
            assert config.classifier.aggregate == "none"

            self.ffn = nn.Sequential(
                nn.Linear(enc_out_dim * config.quantizer.M, 256),
                nn.Tanh(),
                nn.Dropout(0.1),
            )
            enc_out_dim = 256
        elif self.add_layer == "none":
            log.info("add: no layer.")
        return enc_out_dim

    def _reemb(self, idx_seq):
        # idx_seq: bsz × T(optional) × M
        indice_list = idx_seq.split(1, dim=-1)
        emb_list = []
        for idx, emb in zip(indice_list, self.embedding_layers):
            # idx: bsz × T(optional）
            idx = idx.squeeze(-1)
            # word: shift by 1 so embeddings are zero indexed, original padding is -1
            # setence: no need to shift, as no padding
            if self.quantizer_level == "word":
                idx = idx + 1
            # e: bsz × T(optional）× D
            e = emb(idx)
            emb_list.append(e)
            # emb_seq: bsz × T(optional）× M × D
            emb_seq = torch.stack(emb_list, dim=-2)
        return emb_seq

    def _merge_word_split(self, emb_seq):
        # word: emb_seq: bsz × T × M × D
        if self.quantizer_level == "word":
            if self.merge_word_split == "concat":
                # bsz × T × (M * D)
                emb_seq = emb_seq.view([*emb_seq.shape[:-2]] + [-1])
            elif self.merge_word_split == "sum":
                # bsz × T × D
                emb_seq = torch.sum(emb_seq, dim=-2)
            elif self.merge_word_split == "sum_tanh":
                # bsz × T × D
                emb_seq = torch.tanh(torch.sum(emb_seq, dim=-2) + self.bias)
            elif self.merge_word_split == "sum_relu":
                # bsz × T × D
                emb_seq = torch.relu(torch.sum(emb_seq, dim=-2) + self.bias)
        return emb_seq

    def _add_layer(self, enc_outdict):
        if self.add_layer == "transformer":
            enc_outdict["sequence"] = self.encoder_layer(
                enc_outdict["sequence"].transpose(0, 1),
                src_key_padding_mask=(enc_outdict["nopad_mask"] == 0),
            ).transpose(0, 1)
        elif self.add_layer == "lstm":
            seq = enc_outdict["sequence"]
            batch_size = seq.shape[0]
            nopad_mask = enc_outdict["nopad_mask"]
            nopad_lengths = nopad_mask.sum(dim=-1).long()
            packed_embs = pack_padded_sequence(
                seq, lengths=nopad_lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hT, _) = self.lstm(packed_embs)
            output = pad_packed_sequence(packed_output, batch_first=True)[0]
            final = (
                hT.view(
                    self.lstm.num_layers, self.lstm.bidirectional + 1, batch_size, -1
                )[-1]
                .transpose(0, 1)
                .contiguous()
                .view(batch_size, -1)
            )
            enc_outdict["sequence"] = output
            enc_outdict["final_state"] = final
        elif self.add_layer == "ffn":
            # bsz × M × D
            seq = enc_outdict["sequence"]
            bsz = seq.shape[0]
            # bsz × (M * D), concat M split
            seq = seq.reshape(bsz, -1)
            enc_outdict["sequence"] = self.ffn(seq)
        return enc_outdict


class TransformerQuantizerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_embeddings = config.vocab_size

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

        # sentence or word
        self.quantizer_level = config.quantizer.level
        # decompose
        split = config.quantizer.M
        if config.quantizer.level == "word":
            assert (
                config.transformer.d_model % split == 0
            ), "transformer.d_model must be divisible by quantizer_split"
            D = config.transformer.d_model // split
        elif config.quantizer.level == "sentence":
            D = config.transformer.d_model

        # specific to quantizer
        # word
        if config.quantizer.type == "vq":
            self.quantizer = DVQ(
                config,
                num_embeddings=config.quantizer.K,
                embedding_dim=D,
                split=split,
                decompose_option="slice",
            )
            self.project_before_quantizer = lambda x: x
        elif config.quantizer.type == "concrete":
            self.quantizer = ConcreteQuantizer(
                config, num_embeddings=config.quantizer.K, embedding_dim=D, split=split
            )
            self.project_before_quantizer = nn.Linear(
                config.transformer.d_model, config.quantizer.M * config.quantizer.K
            )
        elif config.quantizer.type == "em":
            self.quantizer = HardEMQuantizer(
                config, num_embeddings=config.quantizer.K, embedding_dim=D, split=split
            )
            self.project_before_quantizer = nn.Linear(
                config.transformer.d_model, config.quantizer.M * config.quantizer.K
            )

        # sentence
        if self.quantizer_level == "sentence":
            if config.quantizer.type == "vq":
                d_proj = config.quantizer.M * config.transformer.d_model
            elif config.quantizer.type == "concrete":
                d_proj = config.quantizer.M * config.quantizer.K
            elif config.quantizer.type == "em":
                d_proj = config.quantizer.M * config.quantizer.K
            self.pooler = Pooler(
                project=True,
                pool_type="mean",
                d_inp=config.transformer.d_model,
                d_proj=d_proj,
            )

        self.output_dim = config.transformer.d_model

    def forward(self, src):
        src_pad_mask = src == 0
        src_nopad_mask = src != 0
        nopad_lengths = src_nopad_mask.sum(dim=-1).long()

        src_emb = self.embedding(src).transpose(0, 1)
        src_emb = self.pos_encoder(src_emb)
        src_mask = None

        memory = self.encoder(
            src_emb, src_key_padding_mask=src_pad_mask, mask=src_mask
        ).transpose(0, 1)

        if self.quantizer_level == "word":
            # bsz × T × (M * D) or bsz × T × (M * K)
            memory = self.project_before_quantizer(memory)
            packed_memory = pack_padded_sequence(
                memory, lengths=nopad_lengths, batch_first=True, enforce_sorted=False
            )
            quantizer_out = self.quantizer(packed_memory)
            # bsz × T × (M * D)
            enc_out = quantizer_out["quantized"]

        elif self.quantizer_level == "sentence":
            # seq2vec: mean pooling to get sentence representation
            # bsz × (M * D) or bsz × (M * K)
            pooled_memory = self.pooler(memory, src_nopad_mask)
            quantizer_out = self.quantizer(pooled_memory)
            # mask is filled with 1, bsz × M
            # src_nopad_mask = torch.ones_like(quantizer_out['encoding_indices'])
            src_nopad_mask = quantizer_out["encoding_indices"] != -1
            # bsz × M × D
            enc_out = quantizer_out["quantized_stack"]

        return {
            "quantizer_out": quantizer_out,
            "nopad_mask": src_nopad_mask,
            "sequence": enc_out,
        }

    def get_output_dim(self):
        return self.output_dim


class DecodingUtil(object):
    def __init__(self, vsize):
        self.criterion = nn.NLLLoss(reduction="none")
        self.vsize = vsize

    def forward(self, logprobs, dec_out_gold):

        # reconstruction loss
        loss_reconstruct = self.criterion(
            logprobs.contiguous().view(-1, self.vsize), dec_out_gold.view(-1)
        )
        # mask out padding
        nopad_mask = (dec_out_gold != pad_idx).view(-1).float()
        nll = (loss_reconstruct * nopad_mask).view(logprobs.shape[:-1]).detach()
        loss_reconstruct = (loss_reconstruct * nopad_mask).sum()

        # post-processing
        nopad_mask2 = dec_out_gold != pad_idx
        pred_idx = torch.argmax(logprobs, dim=2)
        pred_idx = pred_idx * nopad_mask2.long()

        ntokens = nopad_mask.sum().item()

        return {
            "loss": loss_reconstruct,
            "pred_idx": pred_idx,
            "nopad_mask": nopad_mask2,
            "ntokens": ntokens,
            "nll": nll,
        }


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()

        self.encoder = encoder

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            dim_feedforward=config.transformer.d_ffn,
            dropout=config.transformer.dropout,
        )

        self.num_embeddings = config.vocab_size
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer.d_model, self.num_embeddings),
            nn.LogSoftmax(dim=-1),
        )

        self.decoding_util = DecodingUtil(config.vocab_size)
        self.init_weights()

        self.kl_fbp = config.concrete.kl.fbp_threshold
        self.kl_beta = config.concrete.kl.beta

    def forward(self, batch):

        input = input_from_batch(batch)
        src = input["enc_in"]
        bsz = src.shape[0]

        # encoder
        enc_outdict = self.encoder(src)
        memory = enc_outdict["sequence"].transpose(0, 1)
        return self.decode(input, memory, enc_outdict)

    def decode(self, input, memory, enc_outdict):

        # teacher forcing
        dec_out_gold = input["dec_out_gold"]
        tgt = input["dec_in"]
        tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
        tgt_emb = self.encoder.pos_encoder(tgt_emb)

        bsz = input["enc_in"].shape[0]

        tgt_pad_mask = tgt == 0

        # causal masking
        tgt_mask = generate_square_subsequent_mask(len(tgt_emb), cuda_device)

        src_nopad_mask = enc_outdict["nopad_mask"]
        src_pad_mask = src_nopad_mask == 0
        output = self.decoder_layer(
            tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )

        output = output.transpose(0, 1)

        # classifier
        logprobs = self.classifier(output)

        dec_outdict = self.decoding_util.forward(logprobs, dec_out_gold)
        loss_reconstruct = dec_outdict["loss"]
        pred_idx = dec_outdict["pred_idx"]
        ntokens = dec_outdict["ntokens"]

        # total loss
        quantizer_out = enc_outdict["quantizer_out"]

        if type(self.encoder.quantizer) in [DVQ, VectorQuantizer]:
            loss = loss_reconstruct + quantizer_out["loss"]
            result = {
                "loss_commit": quantizer_out["loss_commit"],
                "min_distances": quantizer_out["min_distances"],
            }
        elif type(self.encoder.quantizer) == ConcreteQuantizer:
            actual_kl = quantizer_out["kl"]
            if self.training:
                # apply thershold to kl (batch mean), actual_kl is sum
                # fbp_kl = torch.clamp(actual_kl, min=self.kl_fbp * bsz)
                # loss = loss_reconstruct + fbp_kl
                if actual_kl < (self.kl_fbp * bsz):
                    loss = loss_reconstruct
                else:
                    loss = loss_reconstruct + self.kl_beta * actual_kl
            else:
                loss = loss_reconstruct

            result = {
                "kl": actual_kl,
            }
        elif type(self.encoder.quantizer) == HardEMQuantizer:
            loss = loss_reconstruct
            result = {}

        result.update(
            {
                # 'z_q': quantizer_out['quantized'].detach(),
                "indices": quantizer_out["encoding_indices"].detach(),
                "loss_reconstruct": loss_reconstruct.detach(),
                "loss": loss,
                "pred_idx": pred_idx.detach(),
                "ntokens": ntokens,
                "nll": dec_outdict["nll"],
            }
        )
        return result

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
