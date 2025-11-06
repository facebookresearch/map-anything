# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Miscellaneous utility functions.
"""

import logging
import os
import random

import numpy as np
import torch


class StreamToLogger:
    """
    A class that redirects stream writes to a logger.

    This class can be used to redirect stdout or stderr to a logger
    by implementing a file-like interface with write and flush methods.

    Parameters:
    - logger: A logger instance that will receive the log messages
    - log_level: The logging level to use (default: logging.INFO)
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        """
        Write the buffer content to the logger.

        Parameters:
        - buf: The string buffer to write
        """
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        """
        Flush method to comply with file-like object interface.
        This method is required but does nothing in this implementation.
        """
        pass


def seed_everything(seed: int = 42):
    """
    Set the `seed` value for torch and numpy seeds. Also turns on
    deterministic execution for cudnn.

    Parameters:
    - seed: A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")


def invalid_to_nans(arr, valid_mask, ndim=999):
    """
    Replace invalid values in an array with NaN values based on a validity mask.

    Parameters:
    - arr: Input array (typically a PyTorch tensor)
    - valid_mask: Boolean mask indicating valid elements (True) and invalid elements (False)
    - ndim: Maximum number of dimensions to keep; flattens dimensions if arr.ndim > ndim

    Returns:
    - Modified array with invalid values replaced by NaN
    """
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float("nan")
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    """
    Replace invalid values in an array with zeros based on a validity mask.

    Parameters:
    - arr: Input array (typically a PyTorch tensor)
    - valid_mask: Boolean mask indicating valid elements (True) and invalid elements (False)
    - ndim: Maximum number of dimensions to keep; flattens dimensions if arr.ndim > ndim

    Returns:
    - Tuple containing:
      - Modified array with invalid values replaced by zeros
      - nnz: Number of non-zero (valid) elements per sample in the batch
    """
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = (
            arr[..., 0].numel() // len(arr) if len(arr) else 0
        )  # Number of pixels per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz
