import os
import logging as log
import itertools
import _pickle as pkl
from collections import defaultdict
import csv
import re
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Sequence, Type, Union
from overrides import overrides

import torch
import pandas as pd
import numpy as np

from allennlp.data import Instance, Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import (
    TextField,
    LabelField,
    MetadataField,
)


# NOTE: these are not the same as AllenNLP SOS, EOS tokens
# NOTE: pad and unk tokens are created by AllenNLP vocabs by default
# AllenNLP unk token: @@UNKNOWN@@, AllenNLP pad token: @@PADDING@@
SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"
SPECIALS = [SOS_TOK, EOS_TOK]
pad_idx = 0
eos_idx = 3
# sos_idx = 2
# PAD_TOK = "@@PADDING@@"


def _get_words(task, words_by_freq_path: str):
    """
    Get all words for all sentences
    Return mapping words to frequencies.
    """
    word2freq = defaultdict(int)

    def update_vocab_freqs(sentence):
        """Update counts for words in the sentence"""
        for word in sentence:
            word2freq[word] += 1
        return

    for sentence in task.sentences:
        update_vocab_freqs(sentence)

    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)

    # for debugging
    with open(words_by_freq_path, "w") as f:
        f.write("total: " + "\t" + str(len(words_by_freq)) + "\n")
        for (word, freq) in words_by_freq:
            f.write(word + "\t" + str(freq) + "\n")

    return words_by_freq


def _get_vocab(words_by_freq, max_v_sizes, word_freq_thresh):
    """Build vocabulary by selecting the most frequent tokens"""
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes)

    words_by_freq = dict(words_by_freq)
    try:
        words_by_freq.pop("<unk>")  # remove special token, TODO
    except KeyError:
        pass

    for special in SPECIALS:
        vocab.add_token_to_namespace(special, "tokens")

    for word, freq in list(words_by_freq.items())[: max_v_sizes["word"]]:
        if freq >= word_freq_thresh:
            vocab.add_token_to_namespace(word, "tokens")

    return vocab


# based on task.sentences
def build_vocab(config, task, vocab_dir: str, prefix=""):
    """ Build vocabulary from scratch, reading data from tasks. """
    log.info("Building vocab from scratch.")
    max_v_sizes = {"word": config.max_word_v_size}
    words_by_freq_path = os.path.join(config.exp_dir, prefix + "words_by_freq.txt")
    word2freq = _get_words(task, words_by_freq_path)
    vocab = _get_vocab(word2freq, max_v_sizes, config.word_freq_thresh)
    vocab.save_to_files(vocab_dir)
    log.info("Saved vocab to " + vocab_dir)
    return vocab


def build_embeddings(config, vocab, emb_pkl_path: str):
    """ Build word embeddings from scratch,
    using precomputed fastText / GloVe embeddings. """

    # for all word in vocab of data: pad -> 0, pretrained_exist -> load, other -> random
    log.info("Building pretrained embedding weight.")
    word_v_size, unk_idx, pad_idx = (
        vocab.get_vocab_size("tokens"),
        vocab.get_token_index(vocab._oov_token),
        vocab.get_token_index(vocab._padding_token),
    )

    embeddings = np.random.randn(word_v_size, config.d_model)
    embeddings[pad_idx] = 0.0

    with open(
        config.word_embs_path, "r", encoding="utf-8", newline="\n", errors="ignore"
    ) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(" ", 1)  # only split first space
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                embeddings[idx] = np.array(list(map(float, vec.split())))

    embeddings = torch.FloatTensor(embeddings)
    log.info("Finished loading pretrained embedding weight.")

    # Save/cache the word embeddings
    pkl.dump(embeddings, open(emb_pkl_path, "wb"))
    log.info("Saved embeddings pkl to " + emb_pkl_path)
    return embeddings


def sentence_to_text_field(sent: Sequence[str], indexers: Any):
    """ Helper function to map a sequence of tokens into a sequence of
    AllenNLP Tokens, then wrap in a TextField with the given indexers """
    return TextField(list(map(Token, sent)), token_indexers=indexers)


""" example use of indexer
d = {}
d['sent1'] = TextField(list(map(Token, ['<SOS>', 'hello' , 'of', 'the', '<EOS>'])), token_indexers=indexers)
c = Instance(d)
c.index_fields(vocab)
c.as_tensor_dict() # output {'sent1': {'words': tensor([2, 1, 7, 4, 3])}}
"""


def build_indexers():
    indexers = {}
    indexers["words"] = SingleIdTokenIndexer()
    return indexers


def clean_string(string):
    """
    Tokenization/string cleaning for yelp data set
    Based on https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\"\"", ' " ', string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()


def clean_tokenize_truncate(x, max_length):
    x = clean_string(x)
    x = x.split(" ")
    x = x[:max_length]
    return x


class Task(object):
    def __init__(self):
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.sentences = None
        self.n_train_examples = self.n_val_examples = self.n_test_examples = 0

        # self.name = name
        self.class_number = None
        self.max_length = None

        self.train_path = self.val_path = self.test_path = None

    @staticmethod
    def process_split(split, indexers, vocab) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(input1, label):
            """ from multiple types in one column create multiple fields """
            d = {}
            d["sent_str1"] = MetadataField(" ".join(input1))
            input1 = ["<SOS>"] + input1 + ["<EOS>"]
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["labels"] = LabelField(
                label, label_namespace="labels", skip_indexing=True
            )
            d = Instance(d)
            d.index_fields(vocab)
            return d

        instances = map(lambda x: _make_instance(x[0], x[1]), tqdm(list(zip(*split))))
        return list(instances)

    def load_data(self):
        d = {}
        for split in ["train", "val", "test"]:
            split_path = self.__dict__[f"{split}_path"]
            split_text = self.read_data(split_path, max_length=self.max_length)
            d[f"{split}_data_text"] = split_text
            d[f"n_{split}_examples"] = len(split_text[0])
        self.__dict__.update(d)
        self.sentences = self.train_data_text[0]

    @staticmethod
    def read_data(path):
        raise NotImplementedError()


class AGNews(Task):
    def __init__(self):
        # name = "ag_news"
        super().__init__()
        self.class_number = 4
        self.file_by_split = dict(
            train="ag_news_csv/train.train.csv",
            val="ag_news_csv/train.dev.csv",
            test="ag_news_csv/test.csv",
        )
        self.max_length = 400

    @staticmethod
    def read_data(path, max_length):
        def label_fn(x):
            return x - 1

        rows = pd.read_csv(
            path,
            sep=",",
            error_bad_lines=False,
            header=None,
            skiprows=None,
            quoting=0,
            keep_default_na=False,
            encoding="utf-8",
        )

        label_fn = label_fn if label_fn is not None else (lambda x: x)
        labels = rows[0].apply(lambda x: label_fn(x))
        sentences = rows[2]
        sentences = sentences.apply(lambda x: clean_tokenize_truncate(x, max_length))
        return sentences.tolist(), labels.tolist()


class YelpFull(Task):
    def __init__(self):
        # name = "ag_news"
        super().__init__()
        self.class_number = 5
        self.file_by_split = dict(
            train="yelp_review_full_csv/train.train.csv",
            val="yelp_review_full_csv/train.dev.csv",
            test="yelp_review_full_csv/test.csv",
        )
        self.max_length = 400

    @staticmethod
    def read_data(path, max_length):
        def label_fn(x):
            return x - 1

        rows = pd.read_csv(
            path,
            sep=",",
            error_bad_lines=False,
            header=None,
            skiprows=None,
            quoting=0,
            keep_default_na=False,
            encoding="utf-8",
        )

        label_fn = label_fn if label_fn is not None else (lambda x: x)
        labels = rows[0].apply(lambda x: label_fn(x))
        sentences = rows[1]
        sentences = sentences.apply(lambda x: clean_tokenize_truncate(x, max_length))
        return sentences.tolist(), labels.tolist()


class DBPedia(Task):
    def __init__(self):
        # name = "ag_news"
        super().__init__()
        self.class_number = 14
        self.file_by_split = dict(
            train="dbpedia_csv/train.train.csv",
            val="dbpedia_csv/train.dev.csv",
            test="dbpedia_csv/test.csv",
        )
        self.max_length = 400

    @staticmethod
    def read_data(path, max_length):
        def label_fn(x):
            return x - 1

        rows = pd.read_csv(
            path,
            sep=",",
            error_bad_lines=False,
            header=None,
            skiprows=None,
            quoting=0,
            keep_default_na=False,
            encoding="utf-8",
        )

        label_fn = label_fn if label_fn is not None else (lambda x: x)
        labels = rows[0].apply(lambda x: label_fn(x))
        sentences = rows[2]
        sentences = sentences.apply(lambda x: clean_tokenize_truncate(x, max_length))
        return sentences.tolist(), labels.tolist()


def get_task(config):

    task_class_mapping = {
        "ag": AGNews,
        "db": DBPedia,
        "yelp-full": YelpFull,
    }

    task_class = task_class_mapping[config.task]
    task = task_class()
    config.cls_class = task.class_number

    task_pkl_path = os.path.join(config.exp_dir, config.task + ".pkl")
    vocab_dir = os.path.join(config.exp_dir, config.task + "_vocab")

    if not os.path.exists(task_pkl_path):
        assert (config.task) in task_class_mapping.keys(), "invalid task name"
        task.train_path = os.path.join(config.data_dir, task.file_by_split["train"])
        task.val_path = os.path.join(config.data_dir, task.file_by_split["val"])
        task.test_path = os.path.join(config.data_dir, task.file_by_split["test"])

        task.load_data()
        indexers = build_indexers()
        vocab = build_vocab(config, task, vocab_dir, config.task + "_")
        task.train_data = task_class.process_split(
            task.train_data_text, indexers, vocab
        )
        task.val_data = task_class.process_split(task.val_data_text, indexers, vocab)
        task.test_data = task_class.process_split(task.test_data_text, indexers, vocab)

        # remove attributes to save memory
        del task.train_data_text
        del task.val_data_text
        del task.test_data_text
        del task.sentences

        # save task pickle
        pkl.dump(task, open(task_pkl_path, "wb"))

    else:
        # print(task_pkl_path)
        task = pkl.load(open(task_pkl_path, "rb"))
        log.info("Loaded vocab from " + vocab_dir)
        vocab = Vocabulary.from_files(vocab_dir)

    for namespace, mapping in vocab._index_to_token.items():
        log.info("Vocab namespace " + namespace + ": size " + str(len(mapping)))

    return task, vocab
