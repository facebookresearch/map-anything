import inspect
from functools import partial, wraps
from numbers import Number
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint

    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__

        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    assert torch.__version__ >= "2.0", "SDPA requires PyTorch 2.0 or later"

    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)  # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    module.__class__ = _AttentionWrapper
    return module


def sync_ddp_hook(
    state, bucket: torch.distributed.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = torch.distributed.group.WORLD
    world_size = group_to_use.size()
    grad = bucket.buffer()
    grad.div_(world_size)
    torch.distributed.all_reduce(grad, group=group_to_use)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5

    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u, v], dim=-1)
    return uv


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    return optim_shift


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor = None,
    focal: torch.Tensor = None,
    downsample_size: Tuple[int, int] = (64, 64),
):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(
        width, height, dtype=points.dtype, device=points.device
    )  # (H, W, 2)

    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_lr = (
        F.interpolate(
            uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest"
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = (
        None
        if mask is None
        else F.interpolate(
            mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest"
        ).squeeze(1)
        > 0
    )

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = (
            points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        )
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(
                uv_lr_i_np, points_lr_i_np
            )
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(
        optim_shift, device=points.device, dtype=points.dtype
    ).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(
            optim_focal, device=points.device, dtype=points.dtype
        ).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise

    return wrapper


def get_device(args, kwargs):
    device = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            if device is None:
                device = arg.device
            elif device != arg.device:
                raise ValueError("All tensors must be on the same device.")
    return device


def get_args_order(func, args, kwargs):
    """
    Get the order of the arguments of a function.
    """
    names = inspect.getfullargspec(func).args
    names_idx = {name: i for i, name in enumerate(names)}
    args_order = []
    kwargs_order = {}
    for name, arg in kwargs.items():
        if name in names:
            kwargs_order[name] = names_idx[name]
            names.remove(name)
    for i, arg in enumerate(args):
        if i < len(names):
            args_order.append(names_idx[names[i]])
    return args_order, kwargs_order


def broadcast_args(args, kwargs, args_dim, kwargs_dim):
    spatial = []
    for arg, arg_dim in zip(
        args + list(kwargs.values()), args_dim + list(kwargs_dim.values())
    ):
        if isinstance(arg, torch.Tensor) and arg_dim is not None:
            arg_spatial = arg.shape[: arg.ndim - arg_dim]
            if len(arg_spatial) > len(spatial):
                spatial = [1] * (len(arg_spatial) - len(spatial)) + spatial
            for j in range(len(arg_spatial)):
                if spatial[-j] < arg_spatial[-j]:
                    if spatial[-j] == 1:
                        spatial[-j] = arg_spatial[-j]
                    else:
                        raise ValueError("Cannot broadcast arguments.")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and args_dim[i] is not None:
            args[i] = torch.broadcast_to(
                arg, [*spatial, *arg.shape[arg.ndim - args_dim[i] :]]
            )
    for key, arg in kwargs.items():
        if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
            kwargs[key] = torch.broadcast_to(
                arg, [*spatial, *arg.shape[arg.ndim - kwargs_dim[key] :]]
            )
    return args, kwargs, spatial


@suppress_traceback
def batched(*dims):
    """
    Decorator that allows a function to be called with batched arguments.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, device=torch.device("cpu"), **kwargs):
            args = list(args)
            # get arguments dimensions
            args_order, kwargs_order = get_args_order(func, args, kwargs)
            args_dim = [dims[i] for i in args_order]
            kwargs_dim = {key: dims[i] for key, i in kwargs_order.items()}
            # convert to torch tensor
            device = get_device(args, kwargs) or device
            for i, arg in enumerate(args):
                if isinstance(arg, (Number, list, tuple)) and args_dim[i] is not None:
                    args[i] = torch.tensor(arg, device=device)
            for key, arg in kwargs.items():
                if (
                    isinstance(arg, (Number, list, tuple))
                    and kwargs_dim[key] is not None
                ):
                    kwargs[key] = torch.tensor(arg, device=device)
            # broadcast arguments
            args, kwargs, spatial = broadcast_args(args, kwargs, args_dim, kwargs_dim)
            for i, (arg, arg_dim) in enumerate(zip(args, args_dim)):
                if isinstance(arg, torch.Tensor) and arg_dim is not None:
                    args[i] = arg.reshape([-1, *arg.shape[arg.ndim - arg_dim :]])
            for key, arg in kwargs.items():
                if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
                    kwargs[key] = arg.reshape(
                        [-1, *arg.shape[arg.ndim - kwargs_dim[key] :]]
                    )
            # call function
            results = func(*args, **kwargs)
            type_results = type(results)
            results = list(results) if isinstance(results, (tuple, list)) else [results]
            # restore spatial dimensions
            for i, result in enumerate(results):
                results[i] = result.reshape([*spatial, *result.shape[1:]])
            if type_results is tuple:
                results = tuple(results)
            elif type_results is list:
                results = list(results)
            else:
                results = results[0]
            return results

        return wrapper

    return decorator


def image_uv(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1].

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None:
        left = 0
    if top is None:
        top = 0
    if right is None:
        right = width
    if bottom is None:
        bottom = height
    u = torch.linspace(
        (left + 0.5) / width,
        (right - 0.5) / width,
        right - left,
        device=device,
        dtype=dtype,
    )
    v = torch.linspace(
        (top + 0.5) / height,
        (bottom - 0.5) / height,
        bottom - top,
        device=device,
        dtype=dtype,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u, v], dim=-1)

    return uv


@batched(2, 1, 2, 2)
def unproject_cv(
    uv_coord: torch.Tensor,
    depth: torch.Tensor = None,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None,
) -> torch.Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (torch.Tensor): [..., N] depth value
        extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix
        intrinsics (torch.Tensor): [..., 3, 3] intrinsics matrix

    Returns:
        points (torch.Tensor): [..., N, 3] 3d points
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = torch.cat([uv_coord, torch.ones_like(uv_coord[..., :1])], dim=-1)
    points = points @ torch.inverse(intrinsics).transpose(-2, -1)
    if depth is not None:
        points = points * depth[..., None]
    if extrinsics is not None:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        points = (points @ torch.inverse(extrinsics).transpose(-2, -1))[..., :3]
    return points


def depth_to_points(
    depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor = None
):
    height, width = depth.shape[-2:]
    uv = image_uv(width=width, height=height, dtype=depth.dtype, device=depth.device)
    pts = unproject_cv(
        uv,
        depth,
        intrinsics=intrinsics[..., None, :, :],
        extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None,
    )

    return pts


@batched(0, 0, 0, 0, 0, 0)
def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    Args:
        focal_x (float | torch.Tensor): focal length in x axis
        focal_y (float | torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    N = fx.shape[0]
    ret = torch.zeros((N, 3, 3), dtype=fx.dtype, device=fx.device)
    zeros, ones = (
        torch.zeros(N, dtype=fx.dtype, device=fx.device),
        torch.ones(N, dtype=fx.dtype, device=fx.device),
    )
    ret = torch.stack(
        [fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1
    ).unflatten(-1, (3, 3))
    return ret
