import logging as log
import time
import math
import os
import itertools
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import random
import copy
from typing import Any, Dict, Iterable, List, Sequence, Type, Union
from overrides import overrides
import torch
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from allennlp.nn.util import move_to_device, device_mapping
from allennlp.data.iterators import BucketIterator, BasicIterator

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from src.utils.util import current_run_ckpt, set_device
from src.models import VectorQuantizer, DVQ, ConcreteQuantizer
from src.trainer.trainer_util import *
from src import device, cuda_device

np.set_printoptions(precision=3)


class Trainer(object):
    def __init__(self, config, model, vocab=None):

        self.model = model.to(device)
        self.vocab = vocab

        if config.pretrain.tensorboard:
            self.get_tensorboard(config)


    def get_optimizer(self, config):
        model = self.model
        if config.pretrain.use_noam == 0:
            optimizer = O.Adam(model.parameters(), lr=config.pretrain.lr)
        elif config.pretrain.use_noam == 1:
            optimizer = NoamOpt.get_opt(model.parameters(), config)
        self.optimizer = optimizer

    def train_step(self, config, batch):
        batch_size = batch["input1"]["words"].size(0)
        model = self.model
        optimizer = self.optimizer

        model.train()
        optimizer.zero_grad()
        output_dict = model(batch)
        loss = output_dict["loss"]
        loss.div(batch_size).backward()
        if config.pretrain.grad_norm:
            clip_grad_norm_(model.parameters(), config.pretrain.grad_norm)
        optimizer.step()
        # training progress statistics
        self.scorer.update(output_dict, batch_size)

    def get_tensorboard(self, config):
        self.tb_train_writer = SummaryWriter(
            os.path.join(config.run_dir, "tensorboard_train")
        )
        self.tb_validation_writer = SummaryWriter(
            os.path.join(config.run_dir, "tensorboard_val")
        )

    @staticmethod
    def get_train_generator(task, config):
        # data iterator
        sorting_keys = [("input1", "num_tokens")]
        iterator = BucketIterator(
            sorting_keys=sorting_keys,
            max_instances_in_memory=None,
            batch_size=config.pretrain.batch_size,
            biggest_batch_first=True,
        )
        tr_generator = iterator(task.train_data, num_epochs=None, shuffle=True)
        tr_generator = move_to_device(tr_generator, cuda_device)
        n_tr_batches = math.ceil(task.n_train_examples / config.pretrain.batch_size)
        return tr_generator, n_tr_batches

    def load_train_checkpoint(self, config, tr_generator, n_tr_batches):
        # reload checkpoint
        prev_ckpt = "none"
        if config.ckpt_path != "none":
            ckpt_path = config.ckpt_path
            if config.ckpt_path == "current":
                ckpt_path = current_run_ckpt(config)
            self.global_step = load_checkpoint(
                self.model,
                tr_generator,
                n_tr_batches,
                ckpt_path,
                self.metric,
                self.optimizer,
            )
            prev_ckpt = ckpt_path
        return prev_ckpt

    def train(self, config, task, cuda_device=cuda_device):

        self.get_optimizer(config)
        self.global_step = 0
        self.metric = MetricForEarlyStop(
            config.pretrain.patience, should_decrease=True
        )  # perplexity
        tr_generator, n_tr_batches = self.get_train_generator(task, config)
        self.scorer = MetricForPretrain(config.quantizer.type)
        prev_ckpt = self.load_train_checkpoint(config, tr_generator, n_tr_batches)
        start_time = time.time()

        for batch in tr_generator:
            batch = move_to_device(batch, cuda_device)
            self.global_step += 1
            epoch_idx = int((self.global_step - 1) / n_tr_batches) + 1

            # max epoch
            if epoch_idx > config.pretrain.max_epochs:
                log.info(f"\nBest metric: {self.metric.best:.2f}")
                log.info(
                    f"Finish {config.pretrain.max_epochs} epochs in {(time.time() - start_time) /60:.2f} minutes"
                )
                return

            self.train_step(config, batch)

            # log train
            if self.global_step % config.pretrain.log_every == 0:
                log.info(
                    f"\nTrain: Epoch {epoch_idx}, Batch {(self.global_step - 1) % n_tr_batches + 1}/{n_tr_batches}"
                )
                logs = self.scorer.calculate(reset=False)
                log_string = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                log.info(log_string)

                if config.pretrain.tensorboard:
                    logs["epoch"] = epoch_idx
                    for key, value in logs.items():
                        self.tb_train_writer.add_scalar(key, value, self.global_step)

            # validation
            if self.global_step % config.pretrain.val_interval == 0:
                self.scorer.reset()

                log.info("\nValidating...")
                cur_metric, val_logs = self.validate(task, config, quiet=False)
                val_logs["epoch"] = epoch_idx
                if config.pretrain.tensorboard:
                    for key, value in val_logs.items():
                        self.tb_validation_writer.add_scalar(
                            key, value, self.global_step
                        )

                # early stop, save best checkpoint
                is_best_so_far, should_stop = self.metric.check_history(cur_metric)
                if is_best_so_far:
                    log.info(f"\nUpdating best metric: {self.metric.best:.2f}")
                    try:
                        os.remove(prev_ckpt)
                    except OSError:
                        pass
                    prev_ckpt = save_checkpoint(
                        self.model,
                        self.global_step,
                        config,
                        self.metric,
                        self.optimizer,
                    )
                if should_stop:
                    log.info(f"\nBest metric: {self.metric.best:.2f}")
                    log.info(
                        f"Out of patience. Finish {epoch_idx} epochs in {(time.time() - start_time) / 60:.2f} minutes",
                    )
                    return


    def validate(self, task, config, quiet=False):
        model = self.model
        max_data_points = min(task.n_val_examples, config.pretrain.val_data_limit)

        val_generator = BasicIterator(
            config.pretrain.batch_size, instances_per_epoch=max_data_points
        )(task.val_data, num_epochs=1, shuffle=False)

        val_generator = move_to_device(val_generator, cuda_device)
        n_val_batches = math.ceil(max_data_points / config.pretrain.batch_size)

        scorer = MetricForPretrain(config.quantizer.type)

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_generator, 1):
                batch = move_to_device(batch, cuda_device)
                batch_size = batch["input1"]["words"].size(0)

                output_dict = model(batch)
                scorer.update(output_dict, batch_size)

            self.check_representation(batch, output_dict, self.vocab)

            assert batch_idx == n_val_batches and scorer.nexamples == max_data_points

            # log summary
            val_logs = scorer.calculate(reset=False)
            # concrete quantizer does not sample when evaluation
            # it takes argmax, so no kl, loss = loss_reconstruct
            if config.quantizer.type == "concrete":
                del val_logs["kl"]
                del val_logs["loss_reconstruct"]
            log_string = " | ".join([f"{k}: {v:.4f}" for k, v in val_logs.items()])
            log.info(f"\nValidation: " + log_string)
            return val_logs["perplexity"], val_logs

    @staticmethod
    def check_representation(batch, output_dict, vocab):
        head = "--- Last Batch in Validation Set ---"
        log.info("\n" + head)
        indices = output_dict["indices"].squeeze().cpu().numpy().tolist()
        sentences = batch["sent_str1"]
        pred_idx = output_dict["pred_idx"]
        pred_token = np.array(
            [
                vocab.get_token_from_index(i, namespace="tokens")
                for i in pred_idx.view(-1).cpu().numpy()
            ]
        ).reshape(pred_idx.shape)
        pred_sent = [
            " ".join(filter(lambda x: x != "@@PADDING@@", i)) for i in pred_token
        ]
        for i in range(len(sentences)):
            clean_indices = filter_indices(indices[i])

            log.info(f"{clean_indices} {sentences[i]}\n[Reconstruct]: {pred_sent[i]}",)
        log.info("-" * len(head))


class EMTrainer(Trainer):
    def __init__(self, *argv):
        super().__init__(*argv)

    @overrides
    def get_optimizer(self, config):
        model = self.model
        enc_param = [
            param
            for name, param in model.encoder.named_parameters()
            if "quantizer.embeddings" not in name
        ]
        dec_param = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("encoder") or name == "encoder.quantizer.embeddings"
        ]

        if config.pretrain.use_noam == 0:
            enc_optimizer = O.Adam(enc_param, lr=config.pretrain.lr)
            dec_optimizer = O.Adam(dec_param, lr=config.pretrain.lr)
        elif config.pretrain.use_noam == 1:
            enc_optimizer = NoamOpt.get_opt(enc_param, config)
            dec_optimizer = NoamOpt.get_opt(dec_param, config)
        self.optimizer = [enc_optimizer, dec_optimizer]

    @overrides
    def train_step(self, config, batch):
        batch_size = batch["input1"]["words"].size(0)
        model = self.model

        enc_optimizer, dec_optimizer = self.optimizer
        enc_param = enc_optimizer.param_groups[0]["params"]
        dec_param = dec_optimizer.param_groups[0]["params"]

        model.train()

        # E step: maximize joint likelihood wrt z for e_iters steps
        for _ in range(config.pretrain.em_iter):
            # print("E")
            enc_optimizer.zero_grad()
            model.encoder.quantizer.force_eval = False
            output_dict = model(batch)
            loss = output_dict["loss"]  # assumes no KL term
            loss.div(batch_size).backward()
            if config.pretrain.grad_norm:
                clip_grad_norm_(enc_param, config.pretrain.grad_norm)
            enc_optimizer.step()
        # TODO: maybe store the best output instead of taking the last one...

        # M step: take best/last result
        # print("M")
        dec_optimizer.zero_grad()
        if config.quantizer.type == "em":
            model.encoder.quantizer.force_eval = True
        output_dict = model(batch)
        loss = output_dict["loss"]  # assumes no KL term
        loss.div(batch_size).backward()
        if config.pretrain.grad_norm:
            clip_grad_norm_(dec_param, config.pretrain.grad_norm)
        dec_optimizer.step()

        self.scorer.update(output_dict, batch_size)


class CLSTrainer(object):
    def __init__(self, model):
        self.model = model.to(device)

    def _get_optimizer(self, config):
        model = self.model
        if config.target.use_noam == 0:
            optimizer = O.Adam(model.parameters(), lr=config.target.lr)
        elif config.target.use_noam == 1:
            optimizer = NoamOpt.get_opt(model, config)
        self.optimizer = optimizer

    @staticmethod
    def _sample(config, task):
        def subsample_random(train_data, sample_size, seed):
            total_size = len(train_data)
            index = list(range(total_size))
            random.seed(seed)
            random.shuffle(index)
            sample_index = index[:sample_size]
            sample_data = [train_data[i] for i in sample_index]
            return sample_data

        train_ratio = config.target.train_ratio
        train_num = config.target.train_num

        # use a small portion for training
        full_n_train_examples = task.n_train_examples
        assert not (
            train_ratio != "none" and train_num != "none"
        ), "specify either train ratio or number"
        if train_ratio != "none":
            assert 0 < train_ratio <= 1
            task.n_train_examples = math.ceil(full_n_train_examples * train_ratio)
        elif train_num != "none":
            assert 0 < train_num <= full_n_train_examples
            train_ratio = train_num / full_n_train_examples
            task.n_train_examples = train_num
        else:  # train_ratio == train_num == 'none'
            train_ratio = 1
        if train_ratio < 1:
            if config.target.sample_first:
                task.train_data = task.train_data[: task.n_train_examples]
            else:
                task.train_data = subsample_random(
                    task.train_data, task.n_train_examples, config.seed
                )
            [print(i) for i in task.train_data[:5]]
        log.info(
            f"taregt train - ratio: {train_ratio:.4f}, instance: {task.n_train_examples} out of {full_n_train_examples}"
        )
        return task

    @staticmethod
    def _get_train_generator(task, config):
        # data iterator
        sorting_keys = [("input1", "num_tokens")]
        iterator = BucketIterator(
            sorting_keys=sorting_keys,
            max_instances_in_memory=None,
            batch_size=config.target.batch_size,
            biggest_batch_first=True,
        )
        tr_generator = iterator(task.train_data, num_epochs=None, shuffle=True)
        tr_generator = move_to_device(tr_generator, cuda_device)
        n_tr_batches = math.ceil(task.n_train_examples / config.target.batch_size)
        return tr_generator, n_tr_batches

    def train(self, config, task):

        task = self._sample(config, task,)
        tr_generator, n_tr_batches = self._get_train_generator(task, config)
        self._get_optimizer(config)
        model = self.model
        optimizer = self.optimizer

        step = 0
        metric = MetricForEarlyStop(config.target.patience, should_decrease=False)
        scorer = MetricForClassification()

        prev_ckpt = "none"

        start_time = time.time()
        for batch in tr_generator:
            step += 1
            epoch_idx = int((step - 1) / n_tr_batches) + 1

            if epoch_idx > config.target.max_epochs:
                log.info(f"\nBest metric: {metric.best:.4f}")
                log.info(
                    f"Finish {config.target.max_epochs} epochs in {(time.time() - start_time) / 60:.2f} minutes"
                )
                break

            batch = move_to_device(batch, cuda_device)
            batch_size = batch["input1"]["words"].size(0)
            model.train()
            optimizer.zero_grad()
            out = model(batch)
            loss = out["loss"]
            loss.div(batch_size).backward()
            if config.target.grad_norm:
                clip_grad_norm_(model.parameters(), config.target.grad_norm)
            optimizer.step()
            scorer.update(out, batch)

            # train log
            if step % config.target.log_every == 0:
                log.info(
                    f"\nTrain: Epoch {epoch_idx}, Batch {(step - 1) % n_tr_batches + 1}/{n_tr_batches}"
                )
                logs = scorer.calculate(reset=False)
                log_string = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                log.info(log_string)

            # validation
            if step % config.target.val_interval == 0:
                cur_metric = self.validate(config, self.model, task, split="val")
                is_best_so_far, should_stop = metric.check_history(cur_metric)

                if is_best_so_far:
                    log.info(f"\nUpdating best metric: {metric.best:.4f}")
                    if config.target.test == 1:
                        # best_model_state = model.state_dict()
                        best_model = copy.deepcopy(model)

                    # try: os.remove(prev_ckpt)
                    # except OSError: pass
                    # prev_ckpt = save_checkpoint2(model, step, config.sts_run_dir, metric.best)

                if should_stop:
                    log.info(f"\nBest metric: {metric.best:.4f}")
                    log.info(
                        f"Out of patience. Finish {epoch_idx} epochs in {(time.time() - start_time) / 60:.2f} minutes"
                    )
                    # log.info (model.encoder.embedding.weight)
                    break

                scorer.reset()

            # time estimation
            if step % 500 == 0:
                elapsed_time = (time.time() - start_time) / 60
                est_epoch_time = elapsed_time * (n_tr_batches / step)
                log.info(
                    f"Elapsed time for {step} step: {elapsed_time:.2f} minutes. Estimated time for one epoch: {est_epoch_time:.2f} mintues. \n"
                )

        if config.target.test == 1:
            log.info("\n---- Test Result -----")
            test_metric = self.validate(config, best_model, task, split="test")
            log.info(f"\nTest metric: {test_metric:.4f}")

    @staticmethod
    def validate(config, model, task, split="val"):
        # model = self.model
        pred = []
        scorer = MetricForClassification()

        if split == "val":
            data = task.val_data
            n_examples = min(task.n_val_examples, config.target.val_data_limit)
        elif split == "test":
            data = task.test_data
            n_examples = task.n_test_examples

        val_iter = BasicIterator(
            config.target.batch_size, instances_per_epoch=n_examples
        )(data, num_epochs=1, shuffle=False)
        n_val_batches = math.ceil(n_examples / config.target.batch_size)

        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                batch = move_to_device(batch, cuda_device)
                out = model(batch)
                scorer.update(out, batch)

        # log
        log.info("\nValidation Summary:")
        logs = scorer.calculate(reset=False)
        log_string = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        log.info(log_string)

        metric = logs["accuracy"]
        return metric
