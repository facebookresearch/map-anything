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


def get_device(preferred=None):
    """Auto-detect best available computation device.

    Args:
        preferred: Optional preferred device type ('cuda', 'mps', 'cpu').
                  If None, auto-detects in order: CUDA > MPS > CPU.

    Returns:
        torch.device: The best available device
    """
    if preferred is not None:
        if isinstance(preferred, str):
            return torch.device(preferred)
        return preferred

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_capabilities(device):
    """Query device-specific capabilities.

    Args:
        device: torch.device to query

    Returns:
        dict: Device capabilities including:
            - 'bf16_supported': bool - whether bfloat16 is supported
            - 'device_type': str - device type for autocast ('cuda', 'mps', 'cpu')
    """
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]

    capabilities = {
        "device_type": device_type,
    }

    if device_type == "cuda":
        capabilities["bf16_supported"] = torch.cuda.is_bf16_supported()
    elif device_type == "mps":
        capabilities["bf16_supported"] = False
    else:
        capabilities["bf16_supported"] = False

    return capabilities


def is_memory_query_supported(device):
    """Check if memory query operations are supported for the device.

    Args:
        device: torch.device to check

    Returns:
        bool: True if memory operations like mem_get_info() are supported
    """
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    return device_type == "cuda"


def empty_cache(device=None):
    """Clear GPU cache if backend supports it.

    Args:
        device: torch.device to clear cache for. If None, uses auto-detected device.
    """
    if device is None:
        device = get_device()

    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_amp_dtype(device, requested_dtype="bf16"):
    """Determine the best available dtype for mixed precision.

    Args:
        device: torch.device to query
        requested_dtype: str or torch.dtype - preferred dtype ('bf16', 'fp16', 'fp32')

    Returns:
        tuple: (dtype_used: torch.dtype, dtype_name: str)
                dtype_used is the actual torch.dtype that will be used
                dtype_name is the string representation for logging
    """
    if isinstance(requested_dtype, str):
        requested_dtype = requested_dtype.lower()

    if requested_dtype in ["fp32", "float32"]:
        return torch.float32, "fp32"

    capabilities = get_device_capabilities(device)
    if requested_dtype in ["bf16", "bfloat16"]:
        if capabilities["bf16_supported"]:
            return torch.bfloat16, "bf16"
        else:
            return torch.float16, "fp16"

    if requested_dtype in ["fp16", "float16"]:
        return torch.float16, "fp16"

    return torch.float32, "fp32"


def get_autocast_device_type(device):
    """Get the device type string for torch.autocast.

    Args:
        device: torch.device or string

    Returns:
        str: Device type for autocast ('cuda', 'mps', 'cpu')
    """
    if hasattr(device, "type"):
        return device.type
    return str(device).split(":")[0]
