""" import standard library """
import itertools
import os
import logging as log
import argparse
import time
import _pickle as pkl
from typing import Any, Dict, Iterable, List, Sequence, Type, Union
import pyhocon
import types
import sys
from tqdm import tqdm
import copy
from collections import defaultdict, OrderedDict
import re
import random
import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.nn import (
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from torch.nn import LayerNorm, MultiheadAttention, Linear, Dropout

from allennlp.nn.util import move_to_device, device_mapping
from allennlp.data import Instance, Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import (
    TextField,
    LabelField,
    ListField,
    MetadataField,
    MultiLabelField,
    SpanField,
)
from allennlp.training.metrics import (
    Average,
    BooleanAccuracy,
    CategoricalAccuracy,
    F1Measure,
)
from src.task import pad_idx, eos_idx

""" utility function """


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda_device = 0
    else:
        device = torch.device("cpu")
        cuda_device = -1
    # print("device: " + str(device))
    return device, cuda_device


def set_log(log_path=None, mode="a"):
    log.basicConfig(
        format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
    )
    log.getLogger().setLevel(log.INFO)
    log.getLogger().handlers.clear()
    log.getLogger().addHandler(log.StreamHandler())
    if log_path is not None:
        log.getLogger().addHandler(log.FileHandler(log_path, mode=mode))


def get_encoder_state_dict(model_state_dict, encoder_name="encoder"):
    encoder_dict = OrderedDict({})
    old_encoder_keys = [
        k for k in model_state_dict.keys() if k.startswith(encoder_name)
    ]
    for k in old_encoder_keys:
        new_k = k[len(encoder_name) + 1 :]
        encoder_dict[new_k] = model_state_dict[k]
    return encoder_dict


def load_encoder(encoder, ckpt_path):
    from src import device, cuda_device

    checkpoint = torch.load(ckpt_path, map_location=device_mapping(cuda_device))
    model_state_dict = checkpoint["model"]
    encoder_state_dict = get_encoder_state_dict(model_state_dict, "encoder")
    encoder.load_state_dict(encoder_state_dict, strict=False)

    print(encoder_state_dict.keys())
    log.info(f"Load pretrained model: {ckpt_path}")
    perplexity = re.findall(".*best(.*).pt", ckpt_path)[0]
    log.info("pretrain perplexity: %.2f", float(perplexity))


def log_assert(condition, message=""):
    try:
        assert condition, message
    except AssertionError as e:
        log.info(e)
        raise e


def current_run_ckpt(config):
    assert config.ckpt_path == "current"
    run_dir_list = os.listdir(config.run_dir)
    regex = "^" + config.task + ".*pt$"
    ckpt_count = 0
    for filename in run_dir_list:
        if re.match(regex, filename):
            ckpt_count += 1
            ckptname = filename
            ckpt_path = os.path.join(config.run_dir, ckptname)
    log_assert(ckpt_count > 0, "no checkpoint found.")
    log_assert(ckpt_count == 1, f"multiple checkpoints found in {config.run_dir}")
    return ckpt_path


def calculate_kl_bound(config):
    # average length for each task
    T_map = {"ag": 34, "db": 49, "yelp-full": 140}

    ratio = config.concrete.kl.fbp_ratio
    if ratio != "none":
        log_assert(0 <= ratio)  # if ratio > 1, equals to unpenalized
        log_assert(
            config.concrete.kl.type == "categorical",
            "bound only applies to categorical kl",
        )
        log_assert(
            config.concrete.kl.prior_logits == "uniform",
            "bound only applies to uniform prior",
        )
        M = config.quantizer.M
        K = config.quantizer.K
        T = T_map[config.task]
        if config.quantizer.level == "sentence":
            log.info(
                f"Apply FBP as a ratio of uniform prior bound: M * log(K) = {M:d} * log({K:d})",
            )
            bound = M * math.log(K)
        elif config.quantizer.level == "word":
            log.info(
                f"Apply FBP as a ratio of uniform prior bound: T * M * log(K) = {T:d} * {M:d} * log({K:d})"
            )
            bound = T * M * math.log(K)
        config.concrete.kl.fbp_threshold = math.ceil(bound * ratio)
        log.info(
            f"Bound: {bound:.4f}, Ratio: {ratio:.4f}, FBP: {config.concrete.kl.fbp_threshold:d}"
        )


def input_from_batch(batch):
    sent = batch["input1"]["words"]  # shape (batch_size, seq_len)
    batch_size, seq_len = sent.size()

    # no <SOS> and <EOS>
    enc_in = sent[:, 1:-1].clone()
    enc_in[enc_in == eos_idx] = pad_idx

    # no <SOS>
    dec_out_gold = sent[:, 1:].contiguous()

    # no <EOS>
    dec_in = sent[:, :-1].clone()
    dec_in[dec_in == eos_idx] = pad_idx

    out = {
        "batch_size": batch_size,
        "dec_in": dec_in,
        "dec_out_gold": dec_out_gold,
        "enc_in": enc_in,
        "sent": sent,
    }
    return out
