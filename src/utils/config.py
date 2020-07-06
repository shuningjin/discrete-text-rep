"""
modify from https://github.com/nyu-mll/jiant/blob/d6faf55d79d38feba6666922ee62c082dc49ce2f/jiant/utils/config.py
"""

import argparse
import json
import logging as log
import os
import random
import sys
import time
import types
import re
from typing import Iterable, Sequence, Type, Union

import pyhocon

def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")
    # Configuration files
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        nargs="+",
        default="/home-nfs/sjin/code3/default.conf",
        help="Config file(s) (.conf) for model parameters.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        type=str,
        default=None,
        help="Parameter overrides, as valid HOCON string.",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="If true, only use a small portion of training data."
    )
    parser.add_argument(
        "--pass_subrun", "-p", action="store_true", help="If true, pass existing sub run."
    )
    return parser.parse_args(cl_arguments)


class Params(object):
    """Params handler object.

    This functions as a nested dict, but allows for seamless dot-style access, similar to
    tf.HParams but without the type validation. For example:

    p = Params(name="model", num_layers=4)
    p.name  # "model"
    p['data'] = Params(path="file.txt")
    p.data.path  # "file.txt"
    """

    @staticmethod
    def clone(source, strict=True):
        if isinstance(source, pyhocon.ConfigTree):
            return Params(**source.as_plain_ordered_dict())
        elif isinstance(source, Params):
            return Params(**source.as_dict())
        elif isinstance(source, dict):
            return Params(**source)
        elif strict:
            raise ValueError("Cannot clone from type: " + str(type(source)))
        else:
            return None

    def __getitem__(self, k):
        return getattr(self, k)

    def __contains__(self, k):
        return k in self._known_keys

    def __setitem__(self, k, v):
        assert isinstance(k, str)
        if isinstance(self.get(k, None), types.FunctionType):
            raise ValueError("Invalid parameter name (overrides reserved name '%s')." % k)

        converted_val = Params.clone(v, strict=False)
        if converted_val is not None:
            setattr(self, k, converted_val)
        else:  # plain old data
            setattr(self, k, v)
        self._known_keys.add(k)

    def __delitem__(self, k):
        if k not in self:
            raise ValueError("Parameter %s not found.", k)
        delattr(self, k)
        self._known_keys.remove(k)

    def __init__(self, **kw):
        """Create from a list of key-value pairs."""
        self._known_keys = set()
        for k, v in kw.items():
            self[k] = v

    def get(self, k, default=None):
        if "." in k:
            keys = k.split(".")
            d = self.as_dict()
            for key in keys:
                d = d[key]
            return d
        else:
            return getattr(self, k, default)

    def keys(self):
        return sorted(self._known_keys)

    def as_dict(self):
        """Recursively convert to a plain dict."""

        def convert(v):
            return v.as_dict() if isinstance(v, Params) else v

        return {k: convert(self[k]) for k in self.keys()}

    def __repr__(self):
        return self.as_dict().__repr__()

    def __str__(self):
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)




def params_from_file(config_files: Union[str, Iterable[str]], overrides: str = None):
    # Argument handling is as follows:
    # 1) read config file into pyhocon.ConfigTree
    # 2) merge overrides into the ConfigTree
    # 3) validate specific parameters with custom logic
    config_string = ""
    if isinstance(config_files, str):
        config_files = [config_files]
    for config_file in config_files:
        with open(config_file) as fd:
            log.info("Loading config from %s", config_file)
            config_string += fd.read()
            config_string += "\n"
    if overrides:
        log.info("Config overrides: %s", overrides)
        # Append overrides to file to allow for references and injection.
        config_string += "\n"
        config_string += overrides
    basedir = os.path.dirname(config_file)  # directory context for includes
    config = pyhocon.ConfigFactory.parse_string(config_string, basedir=basedir)
    config = Params.clone(config)

    # convert scientific notation string to float, recursively
    # eg. 1e-3 -> 0.001, 1e3 or 1e+3 -> 1000
    def scientific2float(d):
        for k in d.keys():
            v = d[k]
            if isinstance(v, str) and re.match('^\d*e[-+]?\d*$', v):
                try: d[k] = float(v)
                except ValueError: pass
            if isinstance(v, Params):
                d[k] = scientific2float(v)
        return d

    config = scientific2float(config)

    return config
