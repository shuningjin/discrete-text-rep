import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.simple_module import one_hot_argmax, PackedSequneceUtil


class HardEMQuantizer(nn.Module):
    def __init__(self, config, num_embeddings, embedding_dim, split):
        super().__init__()

        self.K = num_embeddings
        self.D = embedding_dim
        self.M = split
        self.relax = config.em.relax

        self.embeddings = nn.Parameter(torch.randn(self.M, self.K, self.D))
        self.force_eval = False

    def forward(self, logits):
        """
        Case 1: sequence of vector
        Arg: logits - bsz x T x (M * K), tensor
                    - or N x (M * K), packed sequence
        Return: quantized - bsz x T x (M * D)

        Case 2: single vector
        Arg: logits - bsz x (M * K), tensor
        Return: quantized - bsz x (M * D)
        """

        # support PackedSequence
        packed_seq_util = PackedSequneceUtil()
        logits = packed_seq_util.preprocess(logits)

        # reshape logits: B_flatten x M x K, with B_flatten = bsz, bsz * T, N
        bsz = logits.shape[0]
        assert logits.shape[-1] == self.M * self.K
        logits = logits.view(-1, self.M, self.K)

        # z: B_flatten x M x K
        # force_eval: target training (freeze param), M step
        if self.training and not self.force_eval:
            if self.relax:
                z = F.softmax(logits, dim=2)
            else:
                logits = F.softmax(logits, dim=2)
                z_hard = one_hot_argmax(logits)
                z = (z_hard - logits).detach() + logits
        else:
            z = one_hot_argmax(logits)

        quantized_stack = (
            z.transpose(0, 1).bmm(self.embeddings).transpose(0, 1)
        )  # B_flatten x M x D
        # if prob is not one-hot, this is not exact index
        encoding_indices = torch.argmax(z, dim=-1)

        if packed_seq_util.is_packed:
            quantized_stack = packed_seq_util.postprocess(quantized_stack, pad=0.0)
            quantized = quantized_stack.view(
                [*quantized_stack.shape[:-2]] + [self.M * self.D]
            )
            encoding_indices = packed_seq_util.postprocess(encoding_indices, pad=-1)
            z = packed_seq_util.postprocess(z, pad=0.0)
        else:
            quantized_stack = quantized_stack.view(bsz, -1, self.M, self.D).squeeze(1)
            quantized = quantized_stack.reshape(bsz, -1, self.M * self.D).squeeze(1)
            encoding_indices = encoding_indices.view(bsz, -1, self.M).squeeze(1)
            z = z.view(bsz, -1, self.M, self.K).squeeze(1)

        # debug
        # print(z)

        return {
            # B x T (optional) x (M * D)
            "quantized": quantized,
            # B x T (optional) x M x D
            "quantized_stack": quantized_stack,
            # B x T (optional) x M
            "encoding_indices": encoding_indices,
        }
