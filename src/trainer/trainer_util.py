from typing import Any, Dict, Iterable, List, Sequence, Type, Union
import torch, math, os
import logging as log

from allennlp.nn.util import move_to_device, device_mapping
from src import device, cuda_device
import itertools


def filter_indices(li):
    if isinstance(li[0], list):
        clean_indices = list(filter(lambda x: x[0] >= 0, li))
    elif isinstance(li[0], int):
        clean_indices = list(filter(lambda x: x >= 0, li))
    return clean_indices


class NoamOpt(object):
    """
    Modified from this: https://www.aclweb.org/anthology/W18-2509.pdf
    Optim wrapper that implements rate, used in the Transformer paper
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    @staticmethod
    def get_std_opt(model_param):
        return NoamOpt(
            512,
            2,
            4000,
            torch.optim.Adam(model_param, lr=0, betas=(0.9, 0.98), eps=1e-9),
        )

    @staticmethod
    def get_opt(model_param, config):
        return NoamOpt(
            config.transformer.d_model,
            config.noam.factor,
            config.noam.warmup,
            torch.optim.Adam(
                model_param,
                lr=config.noam.lr,
                betas=(config.noam.beta1, config.noam.beta2),
                eps=config.noam.eps,
                weight_decay=config.noam.weight_decay,
            ),
        )

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "_step": self._step,
            "_rate": self._rate,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._step = state_dict["_step"]
        self._rate = state_dict["_rate"]

    def __getattr__(self, k):
        """
        extended dot access to optimizer attributes, e.g. param_groups
        """
        if k in self.__dict__:
            return getattr(self, k)
        elif k in self.optimizer.__dict__:
            return getattr(self.optimizer, k)
        else:
            name = self.__class__.__name__
            optimizer_name = self.optimizer.__class__.__name__
            raise AttributeError(
                f"'{name}' or '{optimizer_name}' has no attribute '{k}'"
            )


def save_checkpoint(model, step, config, metric, optimizers):
    checkpoint_state = {"model": model.state_dict(), "step": step}
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
    for i, opt in enumerate(optimizers):
        checkpoint_state["optimizer" + str(i)] = opt.state_dict()
    if metric is not None:
        if type(metric) is MetricForEarlyStop:
            checkpoint_state["metric"] = metric.state_dict()
            best_metric = metric.best
        else:
            assert type(metric) is float
            best_metric = metric
        checkpoint_path = os.path.join(
            config.run_dir, f"{config.task}_ckpt{step}_best{best_metric:.2f}.pt"
        )
    else:
        checkpoint_path = os.path.join(config.run_dir, f"{config.task}_ckpt{step}.pt")
    torch.save(checkpoint_state, checkpoint_path)
    log.info(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    model, generator, n_tr_batches, checkpoint_path, metric, optimizers
):
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
    checkpoint = torch.load(checkpoint_path, map_location=device_mapping(cuda_device))
    model.load_state_dict(checkpoint["model"], strict=False)
    step = checkpoint["step"]
    if metric is not None:
        metric.load_state_dict(checkpoint["metric"])

    for i, opt in enumerate(optimizers):
        opt.load_state_dict(checkpoint["optimizer" + str(i)])
    for _ in itertools.islice(generator, step % n_tr_batches):
        pass
    log.info(f"Load checkpoint: {checkpoint_path}")
    # return model, generator, step, metric, optimizers
    return step


# lightweight: target traininng
def save_checkpoint2(model, step, run_dir, score):
    checkpoint_state = {"model": model.state_dict(), "step": step}
    checkpoint_path = os.path.join(run_dir, "ckpt{}_best{:.2f}.pt".format(step, score))
    # torch.save(checkpoint_state, checkpoint_path)
    log.info(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


# check dev performance for early stopping
class MetricForEarlyStop(object):
    def __init__(self, patience, should_decrease=False):

        self.patience = patience
        self.best_fn = min if should_decrease else max
        self.metric_history = []
        self.best = None

    def check_history(self, cur_score):
        """
        Given a the history of the performance on a metric
        and the current score, check if current score is
        best so far and if out of patience.
        """
        self.metric_history.append(cur_score)
        best_score = self.best_fn(self.metric_history)
        self.best = best_score

        best_index = self.metric_history.index(best_score)
        cur_index = len(self.metric_history) - 1

        is_best_so_far = cur_index == best_index
        out_of_patience = cur_index - best_index >= self.patience
        # if is_best_so_far: metric_history = metric_history[-1:]

        return is_best_so_far, out_of_patience

    def state_dict(self) -> Dict:
        return {"_metric_best": self.best, "_metric_history": self.metric_history}

    def load_state_dict(self, state_dict):
        self.best = state_dict["_metric_best"]
        self.metric_history = state_dict["_metric_history"]


class MetricForPretrain(object):
    def __init__(self, quantizer_name):
        self.quantizer = quantizer_name
        self.reset()

    def reset(self):
        self.ntokens = 0
        self.nexamples = 0

        if self.quantizer == "vq":
            self.losses = dict(loss=0.0, loss_reconstruct=0.0, loss_commit=0.0)
        elif self.quantizer == "concrete":
            self.losses = dict(loss=0.0, loss_reconstruct=0.0, kl=0.0)
        elif self.quantizer == "em":
            self.losses = dict(loss=0.0, loss_reconstruct=0.0,)
            #self.losses = dict(loss=0.0)

    def update(self, output_dict, bsz):
        for k in self.losses:
            loss = output_dict[k]
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.losses[k] += loss
        self.ntokens += output_dict["ntokens"]
        self.nexamples += bsz

    def calculate(self, reset=True):
        perplexity = math.exp(self.losses["loss_reconstruct"] / self.ntokens)
        losses = {k: v / self.nexamples for k, v in self.losses.items()}
        logs = {"perplexity": perplexity}
        logs.update(losses)
        if reset:
            self.reset()
        return logs


class MetricForClassification(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.nexamples = 0
        self.correct = 0
        self.loss = 0.0

    def update(self, output_dict, batch):
        loss = output_dict["loss"].item()
        bsz = batch["labels"].shape[0]
        correct = (output_dict["pred"] == batch["labels"]).float().sum().item()
        self.loss += loss
        self.nexamples += bsz
        self.correct += correct

    def calculate(self, reset=True):
        accuracy = self.correct / self.nexamples
        loss = self.loss / self.nexamples
        logs = dict(loss=loss, accuracy=accuracy)
        if reset:
            self.reset()
        return logs
