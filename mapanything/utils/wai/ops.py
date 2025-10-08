# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
This utils script contains PORTAGE of wai-core ops methods for MapAnything.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_numpy(
    data: torch.Tensor | np.ndarray | int | float,
    dtype: np.dtype | str | type = np.float32,
) -> np.ndarray:
    """
    Convert data to a NumPy array with the specified dtype (default: float32).

    This function handles conversion from NumPy arrays and PyTorch tensors to a NumPy array.

    Args:
        data: Input data (torch.Tensor, np.ndarray, or scalar)
        dtype: Target data type (NumPy dtype, str, or type). Default: np.float32.

    Returns:
        Converted data as NumPy array with specified dtype.
    """
    # Set default dtype if not defined
    assert dtype is not None, "dtype cannot be None"
    dtype = np.dtype(dtype)

    # Handle torch.Tensor
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(dtype)

    # Handle numpy.ndarray
    if isinstance(data, np.ndarray):
        return data.astype(dtype)

    # Handle scalar values
    if isinstance(data, (int, float)):
        return np.array(data, dtype=dtype)

    raise NotImplementedError(f"Unsupported data type: {type(data)}")


def get_dtype_device(
    data: torch.Tensor | np.ndarray | dict | list,
) -> tuple[torch.dtype | np.dtype | None, torch.device | str | type | None]:
    """
    Determine the data type and device of the input data.

    This function recursively inspects the input data and determines its data type
    and device. It handles PyTorch tensors, NumPy arrays, dictionaries, and lists.

    Args:
        data: Input data (torch.Tensor, np.ndarray, dict, list, or other)

    Returns:
        tuple: (dtype, device) where:
            - dtype: The data type (torch.dtype or np.dtype)
            - device: The device (torch.device, 'cpu', 'cuda:X', or np.ndarray)

    Raises:
        ValueError: If tensors in a dictionary are on different CUDA devices
    """
    if isinstance(data, torch.Tensor):
        return data.dtype, data.device

    if isinstance(data, np.ndarray):
        return data.dtype, np.ndarray

    if isinstance(data, dict):
        dtypes = {get_dtype_device(v)[0] for v in data.values()}
        devices = {get_dtype_device(v)[1] for v in data.values()}
        cuda_devices = {device for device in devices if str(device).startswith("cuda")}
        cpu_devices = {device for device in devices if str(device).startswith("cpu")}
        if (len(cuda_devices) > 0) or (len(cpu_devices) > 0):
            # torch.tensor
            dtype = torch.float
            if all(dtype == torch.half for dtype in dtypes):
                dtype = torch.half
            device = None
            if len(cuda_devices) > 1:
                raise ValueError("All tensors must be on the same device")
            if len(cuda_devices) == 1:
                device = list(cuda_devices)[0]
            if (device is None) and (len(cpu_devices) == 1):
                device = list(cpu_devices)[0]
        else:
            dtype = np.float32
            # Fix typo in numpy float16 check
            if all(dtype == np.float16 for dtype in dtypes):
                dtype = np.float16
            device = np.ndarray

    elif isinstance(data, list):
        if not data:  # Handle empty list case
            return None, None
        dtype, device = get_dtype_device(data[0])

    else:
        return np.float32, np.ndarray

    return dtype, device


def to_dtype_device(
    data: torch.Tensor | np.ndarray | dict | list | int | float,
    dtype: torch.dtype | np.dtype | str | None = None,
    device: torch.device | str | type | None = None,
    convert_scalar: bool = False,
) -> torch.Tensor | np.ndarray | dict | list | int | float:
    """
    Convert data to specified dtype and device.

    This function handles conversion between numpy arrays and PyTorch tensors,
    as well as recursive conversion for nested data structures like dictionaries and lists.

    Args:
        data: Input data (torch.Tensor, np.ndarray, dict, list, or scalar)
        dtype: Target data type (torch dtype, numpy dtype, or None)
        device: Target device (torch device, 'cuda', 'cpu', np.ndarray, torch.Tensor, or None)
        convert_scalar: Whether to convert scalar values (int, float) to tensors/arrays

    Returns:
        Converted data with specified dtype and on specified device
    """
    # Handle case where device is passed in dtype parameter
    if device is None:
        if dtype is None:
            raise ValueError("Either `dtype` or `device` must be provided.")

        if str(dtype).startswith("cuda") or str(dtype).startswith("cpu"):
            device = dtype
            dtype = None
        else:
            raise NotImplementedError()

    # Set default dtype based on device
    if dtype is None:
        if device is not None and (
            str(device).startswith("cuda") or str(device).startswith("cpu")
        ):
            dtype = torch.float
        else:
            dtype = np.float32

    # Handle torch.Tensor
    if isinstance(data, torch.Tensor):
        if device == np.ndarray:
            return data.detach().cpu().numpy().astype(dtype)
        return data.to(device=device, dtype=dtype)

    # Handle numpy.ndarray
    elif isinstance(data, np.ndarray):
        if device == torch.Tensor:
            return torch.from_numpy(data).to(dtype=dtype, device=device)
        return data.astype(dtype)

    # Handle dictionary (recursively)
    elif isinstance(data, dict):
        return {
            k: to_dtype_device(v, dtype, device, convert_scalar=convert_scalar)
            for k, v in data.items()
        }

    # Handle list (recursively)
    elif isinstance(data, list):
        return [
            to_dtype_device(x, dtype, device, convert_scalar=convert_scalar)
            for x in data
        ]

    # Handle scalar values
    else:
        if convert_scalar and isinstance(data, (int, float)):
            if device == np.ndarray:
                # Fix: scalars don't have astype method
                return np.array(data, dtype=dtype)
            else:
                return torch.tensor(data, dtype=dtype, device=device)

    # Return original data if no conversion was applied
    return data


def crop(
    data: np.ndarray | torch.Tensor | Image.Image,
    bbox: tuple[int, int, int, int] | tuple[int, int],
) -> np.ndarray | torch.Tensor | Image.Image:
    """
    Crop data of different formats (numpy arrays, PyTorch tensors, PIL Images) to a target size.

    Args:
        data: Input data to resize (numpy.ndarray, torch.Tensor, or PIL.Image.Image)
        size: Target size as tuple (offset_height, offset_width, height, width) or tuple (height, width)

    Returns:
        Cropped data in the same format as the input
    """
    if len(bbox) == 4:
        offset_height, offset_width, target_height, target_width = bbox
    elif len(bbox) == 2:
        target_height, target_width = bbox
        offset_height, offset_width = 0, 0
    else:
        raise ValueError(f"Unsupported size length {len(bbox)}.")

    end_height = offset_height + target_height
    end_width = offset_width + target_width

    if any([sz < 0 for sz in bbox]):
        raise ValueError("Bounding box can't have negative values.")

    if isinstance(data, np.ndarray):
        if (
            max(offset_height, end_height) > data.shape[0]
            or max(offset_width, end_width) > data.shape[1]
        ):
            raise ValueError("Invalid bounding box.")
        cropped_data = data[offset_height:end_height, offset_width:end_width, ...]
        return cropped_data

    # Handle PIL images
    elif isinstance(data, Image.Image):
        if (
            max(offset_height, end_height) > data.size[1]
            or max(offset_width, end_width) > data.size[0]
        ):
            raise ValueError("Invalid bounding box.")
        return data.crop((offset_width, offset_height, end_width, end_height))

    # Handle PyTorch tensors
    elif isinstance(data, torch.Tensor):
        if data.is_nested:
            # special handling for nested tensors
            return torch.stack([crop(nested_tensor, bbox) for nested_tensor in data])
        if (
            max(offset_height, end_height) > data.shape[-2]
            or max(offset_width, end_width) > data.shape[-1]
        ):
            raise ValueError("Invalid bounding box.")
        cropped_data = data[..., offset_height:end_height, offset_width:end_width]
        return cropped_data
    else:
        raise TypeError(f"Unsupported data type '{type(data)}'.")


def to_torch_device_contiguous(
    data_dict: dict[str, dict | np.ndarray | torch.Tensor],
    device: torch.device | str,
    contiguous: bool = False,
) -> dict[str, dict | torch.Tensor]:
    """
    This function handles conversion between a dict of heterogeneous numpy arrays and torch tensors,
    supporting recursion and creation of torch contiguous tensors.

    Args:
        data: Input data (torch.Tensor, np.ndarray, dict, list, or scalar)
        device: Target device (torch device, 'cuda', 'cpu')

    Returns:
        A dict of torch tensors, optionally contiguous in memory and loaded on the specified device.
    """

    result_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            result_dict[k] = to_torch_device_contiguous(v, device, contiguous)
        elif isinstance(v, np.ndarray):
            result_dict[k] = torch.from_numpy(v).to(device)
            if contiguous:
                result_dict[k] = result_dict[k].contiguous()
        elif isinstance(v, torch.Tensor):
            result_dict[k] = v.to(device).contiguous()
        else:
            raise ValueError(f"Found an unsupported value type {type(v)=} for key {k}.")
    return result_dict


def stack(
    data: list[
        dict[str, torch.Tensor | np.ndarray]
        | list[torch.Tensor | np.ndarray]
        | tuple[torch.Tensor | np.ndarray]
    ],
) -> dict[str, torch.Tensor | np.ndarray] | list[torch.Tensor | np.ndarray]:
    """
    Stack a list of dictionaries into a single dictionary with stacked values.
    Or when given a list of sublists, stack the sublists using torch or numpy stack
    if the items are of equal size, or nested tensors if the items are PyTorch tensors
    of different size.

    This utility function is similar to PyTorch's collate function, but specifically
    designed for stacking dictionaries of numpy arrays or PyTorch tensors.

    Args:
        data (list): A list of dictionaries with the same keys, where values are
                    either numpy arrays or PyTorch tensors.
                    OR
                    A list of sublist, where the values of sublists are PyTorch tensors
                    or np arrays.

    Returns:
        dict: A dictionary with the same keys as input dictionaries, but with values
              stacked along a new first dimension.
        OR
        list: If the input was a list with sublists, it returns a list with a stacked
            output for each original input sublist.

    Raises:
        ValueError: If dictionaries in the list have inconsistent keys.
        NotImplementedError: If input is not a list or contains non-dictionary elements.
    """
    if not isinstance(data, list):
        raise NotImplementedError(f"Stack: Data type not supported: {data}")

    if len(data) == 0:
        return data

    if all(isinstance(entry, dict) for entry in data):
        stacked_data = {}
        keys = list(data[0].keys())
        if any(set(entry.keys()) != set(keys) for entry in data):
            raise ValueError("Data not consistent for stacking")

        for key in keys:
            stacked_data[key] = []
            for entry in data:
                stacked_data[key].append(entry[key])

            # stack it according to data format
            if all(isinstance(v, np.ndarray) for v in stacked_data[key]):
                stacked_data[key] = np.stack(stacked_data[key])
            elif all(isinstance(v, torch.Tensor) for v in stacked_data[key]):
                # Check if all tensors have the same shape
                first_shape = stacked_data[key][0].shape
                if all(tensor.shape == first_shape for tensor in stacked_data[key]):
                    stacked_data[key] = torch.stack(stacked_data[key])
                else:
                    # Use nested tensors if shapes are not consistent
                    stacked_data[key] = torch.nested.nested_tensor(stacked_data[key])
        return stacked_data

    if all(isinstance(entry, list) for entry in data):
        # new stacked data will be a list with all of the sublist
        stacked_data = []
        for sublist in data:
            # stack it according to data format
            if all(isinstance(v, np.ndarray) for v in sublist):
                stacked_data.append(np.stack(sublist))
            elif all(isinstance(v, torch.Tensor) for v in sublist):
                # Check if all tensors have the same shape
                first_shape = sublist[0].shape
                if all(tensor.shape == first_shape for tensor in sublist):
                    stacked_data.append(torch.stack(sublist))
                else:
                    # Use nested tensors if shapes are not consistent
                    stacked_data.append(torch.nested.nested_tensor(sublist))
        return stacked_data

    raise NotImplementedError(f"Stack: Data type not supported: {data}")


def resize(
    data: np.ndarray | torch.Tensor | Image.Image,
    size: tuple[int, int] | int | None = None,
    scale: float | None = None,
    modality_format: str | None = None,
) -> np.ndarray | torch.Tensor | Image.Image:
    """
    Resize data of different formats (numpy arrays, PyTorch tensors, PIL Images) to a target size.

    Args:
        data: Input data to resize (numpy.ndarray, torch.Tensor, or PIL.Image.Image)
        size: Target size as tuple (height, width) or single int for long-side scaling
        scale: Scale factor to apply to the original dimensions
        modality_format: Type of data being resized ('depth', 'normals', or None)
                         Affects interpolation method used

    Returns:
        Resized data in the same format as the input

    Raises:
        ValueError: If neither size nor scale is provided, or if both are provided
        TypeError: If data is not a supported type
    """
    # Validate input parameters
    if size is not None and scale is not None:
        raise ValueError("Only one of size or scale should be provided.")

    # Calculate size from scale if needed
    if size is None:
        if scale is None:
            raise ValueError("Either size or scale must be provided.")

        size = (1, 1)
        if isinstance(data, (np.ndarray, torch.Tensor)):
            size = (int(data.shape[-2] * scale), int(data.shape[-1] * scale))
        elif isinstance(data, Image.Image):
            size = (int(data.size[1] * scale), int(data.size[0] * scale))
        else:
            raise TypeError(f"Unsupported data type '{type(data)}'.")

    # Handle long-side scaling when size is a single integer
    elif isinstance(size, int):
        long_side = size
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if isinstance(data, torch.Tensor) and data.is_nested:
                raise ValueError(
                    "Long-side scaling not support for nested tensors, use fixed size instead."
                )
            h, w = data.shape[-2], data.shape[-1]
        elif isinstance(data, Image.Image):
            w, h = data.size
        else:
            raise TypeError(f"Unsupported data type '{type(data)}'.")
        if h > w:
            size = (long_side, int(w * long_side / h))
        else:
            size = (int(h * long_side / w), long_side)

    target_height, target_width = size

    # Set interpolation method based on modality
    if modality_format in ["depth", "normals"]:
        interpolation = Image.Resampling.NEAREST
        torch_interpolation = "nearest"
    else:
        interpolation = Image.Resampling.LANCZOS
        torch_interpolation = "bilinear"

    # Handle numpy arrays
    if isinstance(data, np.ndarray):
        pil_image = Image.fromarray(data)
        resized_image = pil_image.resize((target_width, target_height), interpolation)
        return np.array(resized_image)

    # Handle PIL images
    elif isinstance(data, Image.Image):
        return data.resize((target_width, target_height), interpolation)

    # Handle PyTorch tensors
    elif isinstance(data, torch.Tensor):
        if data.is_nested:
            # special handling for nested tensors
            return torch.stack(
                [
                    resize(nested_tensor, size, scale, modality_format)
                    for nested_tensor in data
                ]
            )
        original_dim = data.ndim
        if original_dim == 2:  # (H, W)
            data = data.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions
        elif original_dim == 3:  # (C/B, H W)
            if modality_format == "depth":
                data = data.unsqueeze(1)  # channel batch dimension
            else:
                data = data.unsqueeze(0)  # Add batch dimension
        resized_tensor = F.interpolate(
            data,
            size=(target_height, target_width),
            mode=torch_interpolation,
            align_corners=False if torch_interpolation != "nearest" else None,
        )
        if original_dim == 2:
            return resized_tensor.squeeze(0).squeeze(
                0
            )  # Remove batch and channel dimensions
        elif original_dim == 3:
            if modality_format == "depth":
                return resized_tensor.squeeze(1)  # Remove channel dimension

            return resized_tensor.squeeze(0)  # Remove batch dimension
        else:
            return resized_tensor

    else:
        raise TypeError(f"Unsupported data type '{type(data)}'.")
