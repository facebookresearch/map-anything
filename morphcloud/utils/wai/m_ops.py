# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def m_dot(
    transform: torch.Tensor,
    points: torch.Tensor | list,
    maintain_shape: bool = False,
) -> torch.Tensor | list:
    """
    Apply batch matrix multiplication between transform matrices and points.

    Args:
        transform: Batch of transformation matrices [..., 3/4, 3/4]
        points: Batch of points [..., N, 3] or a list of points
        maintain_shape: If True, preserves the original shape of points

    Returns:
        Transformed points with shape [..., N, 3] or a list of transformed points
    """
    if isinstance(points, list):
        return [m_dot(t, p, maintain_shape) for t, p in zip(transform, points)]

    # Store original shape and flatten batch dimensions
    orig_shape = points.shape
    batch_dims = points.shape[:-3]

    # Reshape to standard batch format
    transform_flat = transform.reshape(-1, transform.shape[-2], transform.shape[-1])
    points_flat = points.reshape(transform_flat.shape[0], -1, points.shape[-1])

    # Apply transformation
    pts = torch.bmm(
        transform_flat[:, :3, :3],
        points_flat[..., :3].permute(0, 2, 1).to(transform_flat.dtype),
    ).permute(0, 2, 1)

    if transform.shape[-1] == 4:
        pts = pts + transform_flat[:, :3, 3].unsqueeze(1)

    # Restore original shape
    if maintain_shape:
        return pts.reshape(orig_shape)
    else:
        return pts.reshape(*batch_dims, -1, 3)


def m_unproject(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    cam2world: torch.Tensor = None,
    img_grid: torch.Tensor = None,
    valid: torch.Tensor = None,
    H: int | None = None,
    W: int | None = None,
    img_feats: torch.Tensor = None,
    maintain_shape: bool = False,
) -> torch.Tensor:
    """
    Unproject 2D image points with depth values to 3D points in camera or world space.

    Args:
        depth: Depth values, either a tensor of shape ...xHxW or a float value
        intrinsic: Camera intrinsic matrix of shape ...x3x3
        cam2world: Optional camera-to-world transformation matrix of shape ...x4x4
        img_grid: Optional pre-computed image grid. If None, will be created
        valid: Optional mask for valid depth values or minimum depth threshold
        H: Image height (required if depth is a scalar)
        W: Image width (required if depth is a scalar)
        img_feats: Optional image features to append to 3D points
        maintain_shape: If True, preserves the original shape of points

    Returns:
        3D points in camera or world space, with optional features appended
    """
    # Get device and shape information from intrinsic matrix
    device = intrinsic.device
    pre_shape = intrinsic.shape[:-2]  # Batch dimensions

    # Validate inputs
    if isinstance(depth, (int, float)) and H is None:
        raise ValueError("H must be provided if depth is a scalar")

    # Determine image dimensions from depth if not provided
    if isinstance(depth, torch.Tensor) and H is None:
        H, W = depth.shape[-2:]

    # Create image grid if not provided
    if img_grid is None:
        # Create coordinate grid with shape HxWx3 (last dimension is homogeneous)
        img_grid = _create_image_grid(H, W, device)
        # Add homogeneous coordinate
        img_grid = torch.cat([img_grid, torch.ones_like(img_grid[..., :1])], -1)

    # Expand img_grid to match batch dimensions of intrinsic
    if img_grid.dim() <= intrinsic.dim():
        img_grid = img_grid.unsqueeze(0)
        img_grid = img_grid.expand(*pre_shape, *img_grid.shape[-3:])

    # Handle valid mask or minimum depth threshold
    depth_mask = None
    if valid is not None:
        if isinstance(valid, float):
            # Create mask for minimum depth value
            depth_mask = depth > valid
        elif isinstance(valid, torch.Tensor):
            depth_mask = valid

        # Apply mask to image grid and other inputs
        img_grid = masking(img_grid, depth_mask, dim=intrinsic.dim())
        if not isinstance(depth, (int, float)):
            depth = masking(depth, depth_mask, dim=intrinsic.dim() - 1)
        if img_feats is not None:
            img_feats = masking(img_feats, depth_mask, dim=intrinsic.dim() - 1)

    # Unproject 2D points to 3D camera space
    cam_pts: torch.Tensor = m_dot(
        m_inverse_intrinsics(intrinsic),
        img_grid[..., [1, 0, 2]],
        maintain_shape=True,
    )
    # Scale by depth values
    cam_pts = mult(cam_pts, depth.unsqueeze(-1))

    # Transform to world space if cam2world is provided
    if cam2world is not None:
        cam_pts = m_dot(cam2world, cam_pts, maintain_shape=True)

    # Append image features if provided
    if img_feats is not None:
        if isinstance(cam_pts, list):
            if isinstance(cam_pts[0], list):
                # Handle nested list case
                result = []
                for batch_idx, batch in enumerate(cam_pts):
                    batch_result = []
                    for view_idx, view in enumerate(batch):
                        batch_result.append(
                            torch.cat([view, img_feats[batch_idx][view_idx]], -1)
                        )
                    result.append(batch_result)
                cam_pts = result
            else:
                # Handle single list case
                cam_pts = [
                    torch.cat([pts, feats], -1)
                    for pts, feats in zip(cam_pts, img_feats)
                ]
        else:
            # Handle tensor case
            cam_pts = torch.cat([cam_pts, img_feats], -1)

    if maintain_shape:
        return cam_pts

    # Flatten last dimension
    return cam_pts.reshape(*pre_shape, -1, 3)


def m_project(
    world_pts: torch.Tensor,
    intrinsic: torch.Tensor,
    world2cam: torch.Tensor | None = None,
    maintain_shape: bool = False,
) -> torch.Tensor:
    """
    Project 3D world points to 2D image coordinates.

    Args:
        world_pts: 3D points in world coordinates
        intrinsic: Camera intrinsic matrix
        world2cam: Optional transformation from world to camera coordinates
        maintain_shape: If True, preserves the original shape of points

    Returns:
        Image points with coordinates in img_y,img_x,z order
    """
    # Transform points from world to camera space if world2cam is provided
    cam_pts: torch.Tensor = world_pts
    if world2cam is not None:
        cam_pts = m_dot(world2cam, world_pts, maintain_shape=maintain_shape)

    # Get shapes to properly expand intrinsics
    shared_dims = intrinsic.shape[:-2]
    extra_dims = cam_pts.shape[len(shared_dims) : -1]

    # Expand intrinsics to match cam_pts shape
    expanded_intrinsic = intrinsic.view(*shared_dims, *([1] * len(extra_dims)), 3, 3)
    expanded_intrinsic = expanded_intrinsic.expand(*shared_dims, *extra_dims, 3, 3)

    # Project points from camera space to image space
    depth_abs = cam_pts[..., 2].abs().clamp(min=1e-5)
    return torch.stack(
        [
            expanded_intrinsic[..., 1, 1] * cam_pts[..., 1] / depth_abs
            + expanded_intrinsic[..., 1, 2],
            expanded_intrinsic[..., 0, 0] * cam_pts[..., 0] / depth_abs
            + expanded_intrinsic[..., 0, 2],
            cam_pts[..., 2],
        ],
        -1,
    )


def in_image(
    image_pts: torch.Tensor | list,
    H: int,
    W: int,
    min_depth: float = 0.0,
) -> torch.Tensor | list:
    """
    Check if image points are within the image boundaries.

    Args:
        image_pts: Image points in pixel coordinates
        H: Image height
        W: Image width
        min_depth: Minimum valid depth

    Returns:
        Boolean mask indicating which points are within the image
    """
    is_list = isinstance(image_pts, list)
    if is_list:
        return [in_image(pts, H, W, min_depth=min_depth) for pts in image_pts]

    in_image_mask = (
        torch.all(image_pts >= 0, -1)
        & (image_pts[..., 0] < H)
        & (image_pts[..., 1] < W)
    )
    if (min_depth is not None) and image_pts.shape[-1] == 3:
        in_image_mask &= image_pts[..., 2] > min_depth
    return in_image_mask


def _create_image_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Create a coordinate grid for image pixels.

    Args:
        H: Image height
        W: Image width
        device: Computation device

    Returns:
        Image grid with shape HxWx3 (last dimension is homogeneous)
    """
    y_coords = torch.arange(H, device=device)
    x_coords = torch.arange(W, device=device)

    # Use meshgrid with indexing="ij" for correct orientation
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Stack coordinates and add homogeneous coordinate
    img_grid = torch.stack([y_grid, x_grid, torch.ones_like(y_grid)], dim=-1)

    return img_grid


def masking(
    X: torch.Tensor | list,
    mask: torch.Tensor | list,
    dim: int = 3,
) -> torch.Tensor | list:
    """
    Apply a Boolean mask to tensor or list elements.
    Handles nested structures by recursively applying the mask.

    Args:
        X: Input tensor or list to be masked
        mask: Boolean mask to apply
        dim: Dimension threshold for recursive processing

    Returns:
        Masked tensor or list with the same structure as input
    """
    if isinstance(X, list) or (isinstance(X, torch.Tensor) and X.dim() >= dim):
        return [masking(x, m, dim) for x, m in zip(X, mask)]
    return X[mask]


def m_inverse_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of camera intrinsics matrices analytically.
    This is much faster than using torch.inverse() for intrinsics matrices.

    The intrinsics matrix has the form:
    K = [fx  s  cx]
        [0   fy cy]
        [0   0   1]

    And its inverse is:
    K^-1 = [1/fx  -s/(fx*fy)  (s*cy-cx*fy)/(fx*fy)]
           [0     1/fy        -cy/fy            ]
           [0     0           1                 ]

    Args:
        intrinsics: Camera intrinsics matrices of shape [..., 3, 3]

    Returns:
        Inverse intrinsics matrices of shape [..., 3, 3]
    """
    # Extract the components of the intrinsics matrix
    fx = intrinsics[..., 0, 0]
    s = intrinsics[..., 0, 1]  # skew, usually 0
    cx = intrinsics[..., 0, 2]
    fy = intrinsics[..., 1, 1]
    cy = intrinsics[..., 1, 2]

    # Create output tensor with same shape and device
    inv_intrinsics = torch.zeros_like(intrinsics)

    # Compute the inverse analytically
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 0, 1] = -s / (fx * fy)
    inv_intrinsics[..., 0, 2] = (s * cy - cx * fy) / (fx * fy)
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 1, 2] = -cy / fy
    inv_intrinsics[..., 2, 2] = 1.0

    return inv_intrinsics


def mult(
    A: torch.Tensor | np.ndarray | list | float | int,
    B: torch.Tensor | np.ndarray | list | float | int,
) -> torch.Tensor | np.ndarray | list | float | int:
    """
    Multiply two objects with support for lists, tensors, arrays, and scalars.
    Handles nested structures by recursively applying multiplication.

    Args:
        A: First operand (tensor, array, list, or scalar)
        B: Second operand (tensor, array, list, or scalar)

    Returns:
        Result of multiplication with the same structure as inputs
    """
    if isinstance(A, list) and isinstance(B, (int, float)):
        return [mult(a, B) for a in A]
    if isinstance(B, list) and isinstance(A, (int, float)):
        return [mult(A, b) for b in B]
    if isinstance(A, list) and isinstance(B, list):
        return [mult(a, b) for a, b in zip(A, B)]
    return A * B
