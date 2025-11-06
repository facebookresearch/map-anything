# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for managing computation device
"""

import numpy as np
import torch


def to_device(batch, device, callback=None, non_blocking=False):
    """
    Transfer data to another device (i.e. GPU, CPU:torch, CPU:numpy).

    This function recursively processes nested data structures (lists, tuples, dicts)
    and transfers each tensor to the specified device.

    Args:
        batch: Data to transfer (list, tuple, dict of tensors or other objects)
        device: Target device - pytorch device (e.g., 'cuda', 'cpu') or 'numpy'
        callback: Optional function that would be called on every element before processing
        non_blocking: If True, allows asynchronous copy to GPU (may be faster)

    Returns:
        Data with the same structure as input but with tensors transferred to target device
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {
            k: to_device(v, device, non_blocking=non_blocking) for k, v in batch.items()
        }

    if isinstance(batch, (tuple, list)):
        return type(batch)(
            to_device(x, device, non_blocking=non_blocking) for x in batch
        )

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


def to_numpy(x):
    """Convert data to numpy arrays.

    Args:
        x: Input data (can be tensor, array, or nested structure)

    Returns:
        Data with the same structure but with tensors converted to numpy arrays
    """
    return to_device(x, "numpy")


def to_cpu(x):
    """Transfer data to CPU.

    Args:
        x: Input data (can be tensor, array, or nested structure)

    Returns:
        Data with the same structure but with tensors moved to CPU
    """
    return to_device(x, "cpu")


def to_cuda(x):
    """Transfer data to CUDA device (GPU).

    Args:
        x: Input data (can be tensor, array, or nested structure)

    Returns:
        Data with the same structure but with tensors moved to GPU
    """
    return to_device(x, "cuda")
