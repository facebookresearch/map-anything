# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utilities for geometry operations.

References: DUSt3R, MoGe
"""

from numbers import Number
from typing import Tuple, Union

import einops as ein
import numpy as np
import torch
import torch.nn.functional as F

from mapanything.utils.misc import invalid_to_zeros
from mapanything.utils.warnings import no_warnings


def depthmap_to_camera_frame(depthmap, intrinsics):
    """
    Convert depth image to a pointcloud in camera frame.

    Args:
        - depthmap: HxW or BxHxW torch tensor
        - intrinsics: 3x3 or Bx3x3 torch tensor

    Returns:
        pointmap in camera frame (HxWx3 or BxHxWx3 tensor), and a mask specifying valid pixels.
    """
    # Add batch dimension if not present
    if depthmap.dim() == 2:
        depthmap = depthmap.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width = depthmap.shape
    device = depthmap.device

    # Compute 3D point in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    depth_z = depthmap
    xx = (x_grid - cx) * depth_z / fx
    yy = (y_grid - cy) * depth_z / fy
    pts3d_cam = torch.stack((xx, yy, depth_z), dim=-1)

    # Compute mask of valid non-zero depth pixels
    valid_mask = depthmap > 0.0

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        pts3d_cam = pts3d_cam.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

    return pts3d_cam, valid_mask


def depthmap_to_world_frame(depthmap, intrinsics, camera_pose=None):
    """
    Convert depth image to a pointcloud in world frame.

    Args:
        - depthmap: HxW or BxHxW torch tensor
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - camera_pose: 4x4 or Bx4x4 torch tensor

    Returns:
        pointmap in world frame (HxWx3 or BxHxWx3 tensor), and a mask specifying valid pixels.
    """
    pts3d_cam, valid_mask = depthmap_to_camera_frame(depthmap, intrinsics)

    if camera_pose is not None:
        # Add batch dimension if not present
        if camera_pose.dim() == 2:
            camera_pose = camera_pose.unsqueeze(0)
            pts3d_cam = pts3d_cam.unsqueeze(0)
            squeeze_batch_dim = True
        else:
            squeeze_batch_dim = False

        # Convert points from camera frame to world frame
        pts3d_cam_homo = torch.cat(
            [pts3d_cam, torch.ones_like(pts3d_cam[..., :1])], dim=-1
        )
        pts3d_world = ein.einsum(
            camera_pose, pts3d_cam_homo, "b i k, b h w k -> b h w i"
        )
        pts3d_world = pts3d_world[..., :3]

        # Remove batch dimension if it was added
        if squeeze_batch_dim:
            pts3d_world = pts3d_world.squeeze(0)
    else:
        pts3d_world = pts3d_cam

    return pts3d_world, valid_mask


def transform_pts3d(pts3d, transformation):
    """
    Transform 3D points using a 4x4 transformation matrix.

    Args:
        - pts3d: HxWx3 or BxHxWx3 torch tensor
        - transformation: 4x4 or Bx4x4 torch tensor

    Returns:
        transformed points (HxWx3 or BxHxWx3 tensor)
    """
    # Add batch dimension if not present
    if pts3d.dim() == 3:
        pts3d = pts3d.unsqueeze(0)
        transformation = transformation.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Convert points to homogeneous coordinates
    pts3d_homo = torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], dim=-1)

    # Transform points
    transformed_pts3d = ein.einsum(
        transformation, pts3d_homo, "b i k, b h w k -> b h w i"
    )
    transformed_pts3d = transformed_pts3d[..., :3]

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        transformed_pts3d = transformed_pts3d.squeeze(0)

    return transformed_pts3d


def project_pts3d_to_image(pts3d, intrinsics, return_z_dim):
    """
    Project 3D points to image plane (assumes pinhole camera model with no distortion).

    Args:
        - pts3d: HxWx3 or BxHxWx3 torch tensor
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - return_z_dim: bool, whether to return the third dimension of the projected points

    Returns:
        projected points (HxWx2)
    """
    if pts3d.dim() == 3:
        pts3d = pts3d.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Project points to image plane
    projected_pts2d = ein.einsum(intrinsics, pts3d, "b i k, b h w k -> b h w i")
    projected_pts2d[..., :2] /= projected_pts2d[..., 2].unsqueeze(-1).clamp(min=1e-6)

    # Remove the z dimension if not required
    if not return_z_dim:
        projected_pts2d = projected_pts2d[..., :2]

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        projected_pts2d = projected_pts2d.squeeze(0)

    return projected_pts2d


def get_rays_in_camera_frame(intrinsics, height, width, normalize_to_unit_sphere):
    """
    Convert camera intrinsics to a raymap (ray origins + directions) in camera frame.
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool

    Returns:
        - ray_origins: (HxWx3 or BxHxWx3) tensor
        - ray_directions: (HxWx3 or BxHxWx3) tensor
    """
    # Add batch dimension if not present
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size = intrinsics.shape[0]
    device = intrinsics.device

    # Compute rays in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    ray_origins = torch.zeros((batch_size, height, width, 3), device=device)
    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    ray_directions = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)

    # Normalize ray directions to unit sphere if required (else rays will lie on unit plane)
    if normalize_to_unit_sphere:
        ray_directions = ray_directions / torch.norm(
            ray_directions, dim=-1, keepdim=True
        )

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        ray_origins = ray_origins.squeeze(0)
        ray_directions = ray_directions.squeeze(0)

    return ray_origins, ray_directions


def get_rays_in_world_frame(
    intrinsics, height, width, normalize_to_unit_sphere, camera_pose=None
):
    """
    Convert camera intrinsics & camera_pose (if provided) to a raymap (ray origins + directions) in camera or world frame (if camera_pose is provided).
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool
        - camera_pose: 4x4 or Bx4x4 torch tensor

    Returns:
        - ray_origins: (HxWx3 or BxHxWx3) tensor
        - ray_directions: (HxWx3 or BxHxWx3) tensor
    """
    # Get rays in camera frame
    ray_origins, ray_directions = get_rays_in_camera_frame(
        intrinsics, height, width, normalize_to_unit_sphere
    )

    if camera_pose is not None:
        # Add batch dimension if not present
        if camera_pose.dim() == 2:
            camera_pose = camera_pose.unsqueeze(0)
            ray_origins = ray_origins.unsqueeze(0)
            ray_directions = ray_directions.unsqueeze(0)
            squeeze_batch_dim = True
        else:
            squeeze_batch_dim = False

        # Convert rays from camera frame to world frame
        ray_origins_homo = torch.cat(
            [ray_origins, torch.ones_like(ray_origins[..., :1])], dim=-1
        )
        ray_directions_homo = torch.cat(
            [ray_directions, torch.zeros_like(ray_directions[..., :1])], dim=-1
        )
        ray_origins_world = ein.einsum(
            camera_pose, ray_origins_homo, "b i k, b h w k -> b h w i"
        )
        ray_directions_world = ein.einsum(
            camera_pose, ray_directions_homo, "b i k, b h w k -> b h w i"
        )
        ray_origins_world = ray_origins_world[..., :3]
        ray_directions_world = ray_directions_world[..., :3]

        # Remove batch dimension if it was added
        if squeeze_batch_dim:
            ray_origins_world = ray_origins_world.squeeze(0)
            ray_directions_world = ray_directions_world.squeeze(0)
    else:
        ray_origins_world = ray_origins
        ray_directions_world = ray_directions

    return ray_origins_world, ray_directions_world


def recover_pinhole_intrinsics_from_ray_directions(
    ray_directions, use_geometric_calculation=False
):
    """
    Recover pinhole camera intrinsics from ray directions, supporting both batched and non-batched inputs.

    Args:
        ray_directions: Tensor of shape [H, W, 3] or [B, H, W, 3] containing unit normalized ray directions

    Returns:
        Dictionary containing camera intrinsics (fx, fy, cx, cy) as tensors
    """
    # Add batch dimension if not present
    if ray_directions.dim() == 3:  # [H, W, 3]
        ray_directions = ray_directions.unsqueeze(0)  # [1, H, W, 3]
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width, _ = ray_directions.shape
    device = ray_directions.device

    # Create pixel coordinate grid
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )

    # Expand grid for all batches
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]

    # Determine if high resolution or not
    is_high_res = height * width > 1000000

    if is_high_res or use_geometric_calculation:
        # For high-resolution cases, use direct geometric calculation
        # Define key points
        center_h, center_w = height // 2, width // 2
        quarter_w, three_quarter_w = width // 4, 3 * width // 4
        quarter_h, three_quarter_h = height // 4, 3 * height // 4

        # Get rays at key points
        center_rays = ray_directions[:, center_h, center_w, :].clone()  # [B, 3]
        left_rays = ray_directions[:, center_h, quarter_w, :].clone()  # [B, 3]
        right_rays = ray_directions[:, center_h, three_quarter_w, :].clone()  # [B, 3]
        top_rays = ray_directions[:, quarter_h, center_w, :].clone()  # [B, 3]
        bottom_rays = ray_directions[:, three_quarter_h, center_w, :].clone()  # [B, 3]

        # Normalize rays to have dz = 1
        center_rays = center_rays / center_rays[:, 2].unsqueeze(1)  # [B, 3]
        left_rays = left_rays / left_rays[:, 2].unsqueeze(1)  # [B, 3]
        right_rays = right_rays / right_rays[:, 2].unsqueeze(1)  # [B, 3]
        top_rays = top_rays / top_rays[:, 2].unsqueeze(1)  # [B, 3]
        bottom_rays = bottom_rays / bottom_rays[:, 2].unsqueeze(1)  # [B, 3]

        # Calculate fx directly (vectorized across batch)
        fx_left = (quarter_w - center_w) / (left_rays[:, 0] - center_rays[:, 0])
        fx_right = (three_quarter_w - center_w) / (right_rays[:, 0] - center_rays[:, 0])
        fx = (fx_left + fx_right) / 2  # Average for robustness

        # Calculate cx
        cx = center_w - fx * center_rays[:, 0]

        # Calculate fy and cy
        fy_top = (quarter_h - center_h) / (top_rays[:, 1] - center_rays[:, 1])
        fy_bottom = (three_quarter_h - center_h) / (
            bottom_rays[:, 1] - center_rays[:, 1]
        )
        fy = (fy_top + fy_bottom) / 2

        cy = center_h - fy * center_rays[:, 1]
    else:
        # For standard resolution, use regression with sampling for efficiency
        # Sample a grid of points (but more dense than for high-res)
        step_h = max(1, height // 50)
        step_w = max(1, width // 50)

        h_indices = torch.arange(0, height, step_h, device=device)
        w_indices = torch.arange(0, width, step_w, device=device)

        # Extract subset of coordinates
        x_sampled = x_grid[:, h_indices[:, None], w_indices[None, :]]  # [B, H', W']
        y_sampled = y_grid[:, h_indices[:, None], w_indices[None, :]]  # [B, H', W']
        rays_sampled = ray_directions[
            :, h_indices[:, None], w_indices[None, :], :
        ]  # [B, H', W', 3]

        # Reshape for linear regression
        x_flat = x_sampled.reshape(batch_size, -1)  # [B, N]
        y_flat = y_sampled.reshape(batch_size, -1)  # [B, N]

        # Extract ray direction components
        dx = rays_sampled[..., 0].reshape(batch_size, -1)  # [B, N]
        dy = rays_sampled[..., 1].reshape(batch_size, -1)  # [B, N]
        dz = rays_sampled[..., 2].reshape(batch_size, -1)  # [B, N]

        # Compute ratios for linear regression
        ratio_x = dx / dz  # [B, N]
        ratio_y = dy / dz  # [B, N]

        # Since torch.linalg.lstsq doesn't support batched input, we'll use a different approach
        # For x-direction: x = cx + fx * (dx/dz)
        # We can solve this using normal equations: A^T A x = A^T b
        # Create design matrices
        ones = torch.ones_like(x_flat)  # [B, N]
        A_x = torch.stack([ones, ratio_x], dim=2)  # [B, N, 2]
        b_x = x_flat.unsqueeze(2)  # [B, N, 1]

        # Compute A^T A and A^T b for each batch
        ATA_x = torch.bmm(A_x.transpose(1, 2), A_x)  # [B, 2, 2]
        ATb_x = torch.bmm(A_x.transpose(1, 2), b_x)  # [B, 2, 1]

        # Solve the system for each batch
        solution_x = torch.linalg.solve(ATA_x, ATb_x).squeeze(2)  # [B, 2]
        cx, fx = solution_x[:, 0], solution_x[:, 1]

        # Repeat for y-direction
        A_y = torch.stack([ones, ratio_y], dim=2)  # [B, N, 2]
        b_y = y_flat.unsqueeze(2)  # [B, N, 1]

        ATA_y = torch.bmm(A_y.transpose(1, 2), A_y)  # [B, 2, 2]
        ATb_y = torch.bmm(A_y.transpose(1, 2), b_y)  # [B, 2, 1]

        solution_y = torch.linalg.solve(ATA_y, ATb_y).squeeze(2)  # [B, 2]
        cy, fy = solution_y[:, 0], solution_y[:, 1]

    # Create intrinsics matrices
    batch_size = fx.shape[0]
    intrinsics = torch.zeros(batch_size, 3, 3, device=ray_directions.device)

    # Fill in the intrinsics matrices
    intrinsics[:, 0, 0] = fx  # focal length x
    intrinsics[:, 1, 1] = fy  # focal length y
    intrinsics[:, 0, 2] = cx  # principal point x
    intrinsics[:, 1, 2] = cy  # principal point y
    intrinsics[:, 2, 2] = 1.0  # bottom-right element is always 1

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        intrinsics = intrinsics.squeeze(0)

    return intrinsics


def transform_rays(ray_origins, ray_directions, transformation):
    """
    Transform 6D rays (ray origins and ray directions) using a 4x4 transformation matrix.

    Args:
        - ray_origins: HxWx3 or BxHxWx3 torch tensor
        - ray_directions: HxWx3 or BxHxWx3 torch tensor
        - transformation: 4x4 or Bx4x4 torch tensor
        - normalize_to_unit_sphere: bool, whether to normalize the transformed ray directions to unit length

    Returns:
        transformed ray_origins (HxWx3 or BxHxWx3 tensor) and ray_directions (HxWx3 or BxHxWx3 tensor)
    """
    # Add batch dimension if not present
    if ray_origins.dim() == 3:
        ray_origins = ray_origins.unsqueeze(0)
        ray_directions = ray_directions.unsqueeze(0)
        transformation = transformation.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Transform ray origins and directions
    ray_origins_homo = torch.cat(
        [ray_origins, torch.ones_like(ray_origins[..., :1])], dim=-1
    )
    ray_directions_homo = torch.cat(
        [ray_directions, torch.zeros_like(ray_directions[..., :1])], dim=-1
    )
    transformed_ray_origins = ein.einsum(
        transformation, ray_origins_homo, "b i k, b h w k -> b h w i"
    )
    transformed_ray_directions = ein.einsum(
        transformation, ray_directions_homo, "b i k, b h w k -> b h w i"
    )
    transformed_ray_origins = transformed_ray_origins[..., :3]
    transformed_ray_directions = transformed_ray_directions[..., :3]

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        transformed_ray_origins = transformed_ray_origins.squeeze(0)
        transformed_ray_directions = transformed_ray_directions.squeeze(0)

    return transformed_ray_origins, transformed_ray_directions


def convert_z_depth_to_depth_along_ray(z_depth, intrinsics):
    """
    Convert z-depth image to depth along camera rays.

    Args:
        - z_depth: HxW or BxHxW torch tensor
        - intrinsics: 3x3 or Bx3x3 torch tensor

    Returns:
        - depth_along_ray: HxW or BxHxW torch tensor
    """
    # Add batch dimension if not present
    if z_depth.dim() == 2:
        z_depth = z_depth.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Get rays in camera frame
    batch_size, height, width = z_depth.shape
    _, ray_directions = get_rays_in_camera_frame(
        intrinsics, height, width, normalize_to_unit_sphere=False
    )

    # Compute depth along ray
    pts3d_cam = z_depth[..., None] * ray_directions
    depth_along_ray = torch.norm(pts3d_cam, dim=-1)

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        depth_along_ray = depth_along_ray.squeeze(0)

    return depth_along_ray


def convert_raymap_z_depth_quats_to_pointmap(ray_origins, ray_directions, depth, quats):
    """
    Convert raymap (ray origins + directions on unit plane), z-depth and
    unit quaternions (representing rotation) to a pointmap in world frame.

    Args:
        - ray_origins: (HxWx3 or BxHxWx3) torch tensor
        - ray_directions: (HxWx3 or BxHxWx3) torch tensor
        - depth: (HxWx1 or BxHxWx1) torch tensor
        - quats: (HxWx4 or BxHxWx4) torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - pointmap: (HxWx3 or BxHxWx3) torch tensor
    """
    # Add batch dimension if not present
    if ray_origins.dim() == 3:
        ray_origins = ray_origins.unsqueeze(0)
        ray_directions = ray_directions.unsqueeze(0)
        depth = depth.unsqueeze(0)
        quats = quats.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width, _ = depth.shape
    device = depth.device

    # Normalize the quaternions to ensure they are unit quaternions
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    # Convert quaternions to pixel-wise rotation matrices
    qx, qy, qz, qw = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    rot_mat = (
        torch.stack(
            [
                qw**2 + qx**2 - qy**2 - qz**2,
                2 * (qx * qy - qw * qz),
                2 * (qw * qy + qx * qz),
                2 * (qw * qz + qx * qy),
                qw**2 - qx**2 + qy**2 - qz**2,
                2 * (qy * qz - qw * qx),
                2 * (qx * qz - qw * qy),
                2 * (qw * qx + qy * qz),
                qw**2 - qx**2 - qy**2 + qz**2,
            ],
            dim=-1,
        )
        .reshape(batch_size, height, width, 3, 3)
        .to(device)
    )

    # Compute 3D points in local camera frame
    pts3d_local = depth * ray_directions

    # Rotate the local points using the quaternions
    rotated_pts3d_local = ein.einsum(
        rot_mat, pts3d_local, "b h w i k, b h w k -> b h w i"
    )

    # Compute 3D point in world frame associated with each pixel
    pts3d = ray_origins + rotated_pts3d_local

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        pts3d = pts3d.squeeze(0)

    return pts3d


def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    Args:
        - quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - rot_matrix: 3x3 or Bx3x3 torch tensor
    """
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Ensure the quaternion is normalized
    quat = quat / quat.norm(dim=1, keepdim=True)
    x, y, z, w = quat.unbind(dim=1)

    # Compute the rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Construct the rotation matrix
    rot_matrix = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=1,
    ).view(-1, 3, 3)

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        rot_matrix = rot_matrix.squeeze(0)

    return rot_matrix


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def quaternion_inverse(quat):
    """
    Compute the inverse of a quaternion.

    Args:
        - quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - inv_quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
    """
    # Unsqueeze batch dimension if not present
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Compute the inverse
    quat_conj = quat.clone()
    quat_conj[:, :3] = -quat_conj[:, :3]
    quat_norm = torch.sum(quat * quat, dim=1, keepdim=True)
    inv_quat = quat_conj / quat_norm

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        inv_quat = inv_quat.squeeze(0)

    return inv_quat


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    Args:
        - q1: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - q2: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - qm: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
    """
    # Unsqueeze batch dimension if not present
    if q1.dim() == 1:
        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Unbind the quaternions
    x1, y1, z1, w1 = q1.unbind(dim=1)
    x2, y2, z2, w2 = q2.unbind(dim=1)

    # Compute the product
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    # Stack the components
    qm = torch.stack([x, y, z, w], dim=1)

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        qm = qm.squeeze(0)

    return qm


def transform_pose_using_quats_and_trans_2_to_1(quats1, trans1, quats2, trans2):
    """
    Transform quats and translation of pose2 from absolute frame (pose2 to world) to relative frame (pose2 to pose1).

    Args:
        - quats1: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans1: 3 or Bx3 torch tensor
        - quats2: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans2: 3 or Bx3 torch tensor

    Returns:
        - quats: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans: 3 or Bx3 torch tensor
    """
    # Unsqueeze batch dimension if not present
    if quats1.dim() == 1:
        quats1 = quats1.unsqueeze(0)
        trans1 = trans1.unsqueeze(0)
        quats2 = quats2.unsqueeze(0)
        trans2 = trans2.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Compute the inverse of view1's pose
    inv_quats1 = quaternion_inverse(quats1)
    R1_inv = quaternion_to_rotation_matrix(inv_quats1)
    t1_inv = -1 * ein.einsum(R1_inv, trans1, "b i j, b j -> b i")

    # Transform view2's pose to view1's frame
    quats = quaternion_multiply(inv_quats1, quats2)
    trans = ein.einsum(R1_inv, trans2, "b i j, b j -> b i") + t1_inv

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        quats = quats.squeeze(0)
        trans = trans.squeeze(0)

    return quats, trans


def convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
    ray_directions, depth_along_ray, pose_trans, pose_quats
):
    """
    Convert ray directions, depth along ray, pose translation, and
    unit quaternions (representing pose rotation) to a pointmap in world frame.

    Args:
        - ray_directions: (HxWx3 or BxHxWx3) torch tensor
        - depth_along_ray: (HxWx1 or BxHxWx1) torch tensor
        - pose_trans: (3 or Bx3) torch tensor
        - pose_quats: (4 or Bx4) torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - pointmap: (HxWx3 or BxHxWx3) torch tensor
    """
    # Add batch dimension if not present
    if ray_directions.dim() == 3:
        ray_directions = ray_directions.unsqueeze(0)
        depth_along_ray = depth_along_ray.unsqueeze(0)
        pose_trans = pose_trans.unsqueeze(0)
        pose_quats = pose_quats.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width, _ = depth_along_ray.shape
    device = depth_along_ray.device

    # Normalize the quaternions to ensure they are unit quaternions
    pose_quats = pose_quats / torch.norm(pose_quats, dim=-1, keepdim=True)

    # Convert quaternions to rotation matrices (B x 3 x 3)
    rot_mat = quaternion_to_rotation_matrix(pose_quats)

    # Get pose matrix (B x 4 x 4)
    pose_mat = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    pose_mat[:, :3, :3] = rot_mat
    pose_mat[:, :3, 3] = pose_trans

    # Compute 3D points in local camera frame
    pts3d_local = depth_along_ray * ray_directions

    # Compute 3D points in world frame
    pts3d_homo = torch.cat([pts3d_local, torch.ones_like(pts3d_local[..., :1])], dim=-1)
    pts3d_world = ein.einsum(pose_mat, pts3d_homo, "b i k, b h w k -> b h w i")
    pts3d_world = pts3d_world[..., :3]

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        pts3d_world = pts3d_world.squeeze(0)

    return pts3d_world


def xy_grid(
    W,
    H,
    device=None,
    origin=(0, 0),
    unsqueeze=None,
    cat_dim=-1,
    homogeneous=False,
    **arange_kw,
):
    """
    Generate a coordinate grid of shape (H,W,2) or (H,W,3) if homogeneous=True.

    Args:
        W (int): Width of the grid
        H (int): Height of the grid
        device (torch.device, optional): Device to place the grid on. If None, uses numpy arrays
        origin (tuple, optional): Origin coordinates (x,y) for the grid. Default is (0,0)
        unsqueeze (int, optional): Dimension to unsqueeze in the output tensors
        cat_dim (int, optional): Dimension to concatenate the x,y coordinates. If None, returns tuple
        homogeneous (bool, optional): If True, adds a third dimension of ones to make homogeneous coordinates
        **arange_kw: Additional keyword arguments passed to np.arange or torch.arange

    Returns:
        numpy.ndarray or torch.Tensor: Coordinate grid where:
            - output[j,i,0] = i + origin[0] (x-coordinate)
            - output[j,i,1] = j + origin[1] (y-coordinate)
            - output[j,i,2] = 1 (if homogeneous=True)
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        def arange(*a, **kw):
            return torch.arange(*a, device=device, **kw)

        meshgrid, stack = torch.meshgrid, torch.stack

        def ones(*a):
            return torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)

    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """
    Apply a geometric transformation to a set of 3-D points.

    Args:
        Trf: 3x3 or 4x4 projection matrix (typically a Homography) or batch of matrices
            with shape (B, 3, 3) or (B, 4, 4)
        pts: numpy/torch/tuple of coordinates with shape (..., 2) or (..., 3)
        ncol: int, number of columns of the result (2 or 3)
        norm: float, if not 0, the result is projected on the z=norm plane
            (homogeneous normalization)

    Returns:
        Array or tensor of projected points with the same type as input and shape (..., ncol)
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # Adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # Optimized code
    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /=, it will lead to a bug
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)

    return res


def inv(mat):
    """
    Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def closed_form_pose_inverse(
    pose_matrices, rotation_matrices=None, translation_vectors=None
):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 pose matrices in a batch.

    If `rotation_matrices` and `translation_vectors` are provided, they must correspond to the rotation and translation
    components of `pose_matrices`. Otherwise, they will be extracted from `pose_matrices`.

    Args:
        pose_matrices: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        rotation_matrices (optional): Nx3x3 array or tensor of rotation matrices.
        translation_vectors (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as input `pose_matrices`.

    Shapes:
        pose_matrices: (N, 4, 4)
        rotation_matrices: (N, 3, 3)
        translation_vectors: (N, 3, 1)
    """
    # Check if pose_matrices is a numpy array or a torch tensor
    is_numpy = isinstance(pose_matrices, np.ndarray)

    # Validate shapes
    if pose_matrices.shape[-2:] != (4, 4) and pose_matrices.shape[-2:] != (3, 4):
        raise ValueError(
            f"pose_matrices must be of shape (N,4,4), got {pose_matrices.shape}."
        )

    # Extract rotation_matrices and translation_vectors if not provided
    if rotation_matrices is None:
        rotation_matrices = pose_matrices[:, :3, :3]
    if translation_vectors is None:
        translation_vectors = pose_matrices[:, :3, 3:]

    # Compute the inverse of input SE3 matrices
    if is_numpy:
        rotation_transposed = np.transpose(rotation_matrices, (0, 2, 1))
        new_translation = -np.matmul(rotation_transposed, translation_vectors)
        inverted_matrix = np.tile(np.eye(4), (len(rotation_matrices), 1, 1))
    else:
        rotation_transposed = rotation_matrices.transpose(1, 2)
        new_translation = -torch.bmm(rotation_transposed, translation_vectors)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(rotation_matrices), 1, 1)
        inverted_matrix = inverted_matrix.to(rotation_matrices.dtype).to(
            rotation_matrices.device
        )
    inverted_matrix[:, :3, :3] = rotation_transposed
    inverted_matrix[:, :3, 3:] = new_translation

    return inverted_matrix


def relative_pose_transformation(trans_01, trans_02):
    r"""
    Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = relative_pose_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError(
            "Input trans_01 type is not a torch.Tensor. Got {}".format(type(trans_01))
        )
    if not torch.is_tensor(trans_02):
        raise TypeError(
            "Input trans_02 type is not a torch.Tensor. Got {}".format(type(trans_02))
        )
    if trans_01.dim() not in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4. Got {}".format(trans_01.shape)
        )
    if trans_02.dim() not in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4. Got {}".format(trans_02.shape)
        )
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(
            "Input number of dims must match. Got {} and {}".format(
                trans_01.dim(), trans_02.dim()
            )
        )

    # Convert to Nx4x4 if inputs are 4x4
    squeeze_batch_dim = False
    if trans_01.dim() == 2:
        trans_01 = trans_01.unsqueeze(0)
        trans_02 = trans_02.unsqueeze(0)
        squeeze_batch_dim = True

    # Compute inverse of trans_01 using closed form
    trans_10 = closed_form_pose_inverse(trans_01)

    # Compose transformations using matrix multiplication
    trans_12 = torch.matmul(trans_10, trans_02)

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        trans_12 = trans_12.squeeze(0)

    return trans_12


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0

    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(
    depthmap, camera_intrinsics, camera_pose, **kw
):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = (
            np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        )

    return X_world, valid_mask


def get_absolute_pointmaps_and_rays_info(
    depthmap, camera_intrinsics, camera_pose, **kw
):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array),
        a mask specifying valid pixels,
        ray origins of absolute coordinates (HxWx3 array),
        ray directions of absolute coordinates (HxWx3 array),
        depth along ray (HxWx1 array),
        ray directions of camera/local coordinates (HxWx3 array),
        pointmap of camera/local coordinates (HxWx3 array).
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: pinhole & there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    fu = camera_intrinsics[0, 0]
    fv = camera_intrinsics[1, 1]
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    # Get the rays on the unit plane
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_cam = (u - cu) / fu
    y_cam = (v - cv) / fv
    z_cam = np.ones_like(x_cam)
    ray_dirs_cam_on_unit_plane = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(
        np.float32
    )

    # Compute the 3d points in the local camera coordinate system
    pts_cam = depthmap[..., None] * ray_dirs_cam_on_unit_plane

    # Get the depth along the ray and compute the ray directions on the unit sphere
    depth_along_ray = np.linalg.norm(pts_cam, axis=-1, keepdims=True)
    ray_directions_cam = ray_dirs_cam_on_unit_plane / np.linalg.norm(
        ray_dirs_cam_on_unit_plane, axis=-1, keepdims=True
    )

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0

    # Get the ray origins in absolute coordinates and the ray directions in absolute coordinates
    ray_origins_world = np.zeros_like(ray_directions_cam)
    ray_directions_world = ray_directions_cam
    pts_world = pts_cam
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates
        ray_origins_world = ray_origins_world + t_cam2world[None, None, :]
        ray_directions_world = np.einsum(
            "ik, vuk -> vui", R_cam2world, ray_directions_cam
        )
        pts_world = ray_origins_world + ray_directions_world * depth_along_ray

    return (
        pts_world,
        valid_mask,
        ray_origins_world,
        ray_directions_world,
        depth_along_ray,
        ray_directions_cam,
        pts_cam,
    )


def adjust_camera_params_for_rotation(camera_params, original_size, k):
    """
    Adjust camera parameters for rotation.

    Args:
        camera_params: Camera parameters [fx, fy, cx, cy, ...]
        original_size: Original image size as (width, height)
        k: Number of 90-degree rotations counter-clockwise (k=3 means 90 degrees clockwise)

    Returns:
        Adjusted camera parameters
    """
    fx, fy, cx, cy = camera_params[:4]
    width, height = original_size

    if k % 4 == 1:  # 90 degrees counter-clockwise
        new_fx, new_fy = fy, fx
        new_cx, new_cy = height - cy, cx
    elif k % 4 == 2:  # 180 degrees
        new_fx, new_fy = fx, fy
        new_cx, new_cy = width - cx, height - cy
    elif k % 4 == 3:  # 90 degrees clockwise (270 counter-clockwise)
        new_fx, new_fy = fy, fx
        new_cx, new_cy = cy, width - cx
    else:  # No rotation
        return camera_params

    adjusted_params = [new_fx, new_fy, new_cx, new_cy]
    if len(camera_params) > 4:
        adjusted_params.extend(camera_params[4:])

    return adjusted_params


def adjust_pose_for_rotation(pose, k):
    """
    Adjust camera pose for rotation.

    Args:
        pose: 4x4 camera pose matrix (camera-to-world, OpenCV convention - X right, Y down, Z forward)
        k: Number of 90-degree rotations counter-clockwise (k=3 means 90 degrees clockwise)

    Returns:
        Adjusted 4x4 camera pose matrix
    """
    # Create rotation matrices for different rotations
    if k % 4 == 1:  # 90 degrees counter-clockwise
        rot_transform = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    elif k % 4 == 2:  # 180 degrees
        rot_transform = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif k % 4 == 3:  # 90 degrees clockwise (270 counter-clockwise)
        rot_transform = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    else:  # No rotation
        return pose

    # Apply the transformation to the pose
    adjusted_pose = pose
    adjusted_pose[:3, :3] = adjusted_pose[:3, :3] @ rot_transform.T

    return adjusted_pose


def crop_to_aspect_ratio(image, depth, camera_params, target_ratio=1.5):
    """
    Crop image and depth to the largest possible target aspect ratio while
    keeping the left side if aspect ratio is wider and the bottom of image if the aspect ratio is taller.

    Args:
        image: PIL image
        depth: Depth map as numpy array
        camera_params: Camera parameters [fx, fy, cx, cy, ...]
        target_ratio: Target width/height ratio

    Returns:
        Cropped image, cropped depth, adjusted camera parameters
    """
    width, height = image.size
    fx, fy, cx, cy = camera_params[:4]
    current_ratio = width / height

    if abs(current_ratio - target_ratio) < 1e-6:
        # Already at target ratio
        return image, depth, camera_params

    if current_ratio > target_ratio:
        # Image is wider than target ratio, crop width
        new_width = int(height * target_ratio)
        left = 0
        right = new_width

        # Crop image
        cropped_image = image.crop((left, 0, right, height))

        # Crop depth
        if len(depth.shape) == 3:
            cropped_depth = depth[:, left:right, :]
        else:
            cropped_depth = depth[:, left:right]

        # Adjust camera parameters
        new_cx = cx - left
        adjusted_params = [fx, fy, new_cx, cy] + list(camera_params[4:])

    else:
        # Image is taller than target ratio, crop height
        new_height = int(width / target_ratio)
        top = max(0, height - new_height)
        bottom = height

        # Crop image
        cropped_image = image.crop((0, top, width, bottom))

        # Crop depth
        if len(depth.shape) == 3:
            cropped_depth = depth[top:bottom, :, :]
        else:
            cropped_depth = depth[top:bottom, :]

        # Adjust camera parameters
        new_cy = cy - top
        adjusted_params = [fx, fy, cx, new_cy] + list(camera_params[4:])

    return cropped_image, cropped_depth, adjusted_params


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5

    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5

    return K


def normalize_depth_using_non_zero_pixels(depth, return_norm_factor=False):
    """
    Normalize the depth by the average depth of non-zero depth pixels.

    Args:
        depth (torch.Tensor): Depth tensor of size [B, H, W, 1].
    Returns:
        normalized_depth (torch.Tensor): Normalized depth tensor.
        norm_factor (torch.Tensor): Norm factor tensor of size B.
    """
    assert depth.ndim == 4 and depth.shape[3] == 1
    # Calculate the sum and count of non-zero depth pixels for each batch
    valid_depth_mask = depth > 0
    valid_sum = torch.sum(depth * valid_depth_mask, dim=(1, 2, 3))
    valid_count = torch.sum(valid_depth_mask, dim=(1, 2, 3))

    # Calculate the norm factor
    norm_factor = valid_sum / (valid_count + 1e-8)
    while norm_factor.ndim < depth.ndim:
        norm_factor.unsqueeze_(-1)

    # Normalize the depth by the norm factor
    norm_factor = norm_factor.clip(min=1e-8)
    normalized_depth = depth / norm_factor

    # Create the output tuple
    output = (
        (normalized_depth, norm_factor.squeeze(-1).squeeze(-1).squeeze(-1))
        if return_norm_factor
        else normalized_depth
    )

    return output


def normalize_pose_translations(pose_translations, return_norm_factor=False):
    """
    Normalize the pose translations by the average norm of the non-zero pose translations.

    Args:
        pose_translations (torch.Tensor): Pose translations tensor of size [B, V, 3]. B is the batch size, V is the number of views.
    Returns:
        normalized_pose_translations (torch.Tensor): Normalized pose translations tensor of size [B, V, 3].
        norm_factor (torch.Tensor): Norm factor tensor of size B.
    """
    assert pose_translations.ndim == 3 and pose_translations.shape[2] == 3
    # Compute distance of all pose translations to origin
    pose_translations_dis = pose_translations.norm(dim=-1)  # [B, V]
    non_zero_pose_translations_dis = pose_translations_dis > 0  # [B, V]

    # Calculate the average norm of the translations across all views (considering only views with non-zero translations)
    sum_of_all_views_pose_translations = pose_translations_dis.sum(dim=1)  # [B]
    count_of_all_views_with_non_zero_pose_translations = (
        non_zero_pose_translations_dis.sum(dim=1)
    )  # [B]
    norm_factor = sum_of_all_views_pose_translations / (
        count_of_all_views_with_non_zero_pose_translations + 1e-8
    )  # [B]

    # Normalize the pose translations by the norm factor
    norm_factor = norm_factor.clip(min=1e-8)
    normalized_pose_translations = pose_translations / norm_factor.unsqueeze(
        -1
    ).unsqueeze(-1)

    # Create the output tuple
    output = (
        (normalized_pose_translations, norm_factor)
        if return_norm_factor
        else normalized_pose_translations
    )

    return output


def normalize_multiple_pointclouds(
    pts_list, valid_masks=None, norm_mode="avg_dis", ret_factor=False
):
    """
    Normalize multiple point clouds using a joint normalization strategy.

    Args:
        pts_list: List of point clouds, each with shape (..., H, W, 3) or (B, H, W, 3)
        valid_masks: Optional list of masks indicating valid points in each point cloud
        norm_mode: String in format "{norm}_{dis}" where:
            - norm: Normalization strategy (currently only "avg" is supported)
            - dis: Distance transformation ("dis" for raw distance, "log1p" for log(1+distance),
                  "warp-log1p" to warp points using log distance)
        ret_factor: If True, return the normalization factor as the last element in the result list

    Returns:
        List of normalized point clouds with the same shapes as inputs.
        If ret_factor is True, the last element is the normalization factor.
    """
    assert all(pts.ndim >= 3 and pts.shape[-1] == 3 for pts in pts_list)
    if valid_masks is not None:
        assert len(pts_list) == len(valid_masks)

    norm_mode, dis_mode = norm_mode.split("_")

    # Gather all points together (joint normalization)
    nan_pts_list = [
        invalid_to_zeros(pts, valid_masks[i], ndim=3)
        if valid_masks
        else invalid_to_zeros(pts, None, ndim=3)
        for i, pts in enumerate(pts_list)
    ]
    all_pts = torch.cat([nan_pts for nan_pts, _ in nan_pts_list], dim=1)
    nnz_list = [nnz for _, nnz in nan_pts_list]

    # Compute distance to origin
    all_dis = all_pts.norm(dim=-1)
    if dis_mode == "dis":
        pass  # do nothing
    elif dis_mode == "log1p":
        all_dis = torch.log1p(all_dis)
    elif dis_mode == "warp-log1p":
        # Warp input points before normalizing them
        log_dis = torch.log1p(all_dis)
        warp_factor = log_dis / all_dis.clip(min=1e-8)
        for i, pts in enumerate(pts_list):
            H, W = pts.shape[1:-1]
            pts_list[i] = pts * warp_factor[:, i * (H * W) : (i + 1) * (H * W)].view(
                -1, H, W, 1
            )
        all_dis = log_dis
    else:
        raise ValueError(f"bad {dis_mode=}")

    # Compute normalization factor
    norm_factor = all_dis.sum(dim=1) / (sum(nnz_list) + 1e-8)
    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts_list[0].ndim:
        norm_factor.unsqueeze_(-1)

    # Normalize points
    res = [pts / norm_factor for pts in pts_list]
    if ret_factor:
        res.append(norm_factor)

    return res


def apply_log_to_norm(input_data):
    """
    Normalize the input data and apply a logarithmic transformation based on the normalization factor.

    Args:
        input_data (torch.Tensor): The input tensor to be normalized and transformed.

    Returns:
        torch.Tensor: The transformed tensor after normalization and logarithmic scaling.
    """
    org_d = input_data.norm(dim=-1, keepdim=True)
    input_data = input_data / org_d.clip(min=1e-8)
    input_data = input_data * torch.log1p(org_d)
    return input_data


def angle_diff_vec3(v1, v2, eps=1e-12):
    """
    Compute angle difference between 3D vectors.

    Args:
        v1: torch.Tensor of shape (..., 3)
        v2: torch.Tensor of shape (..., 3)
        eps: Small epsilon value for numerical stability

    Returns:
        torch.Tensor: Angle differences in radians
    """
    cross_norm = torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps
    dot_prod = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross_norm, dot_prod)


def angle_diff_vec3_numpy(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12):
    """
    Compute angle difference between 3D vectors using NumPy.

    Args:
        v1 (np.ndarray): First vector of shape (..., 3)
        v2 (np.ndarray): Second vector of shape (..., 3)
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-12.

    Returns:
        np.ndarray: Angle differences in radians
    """
    return np.arctan2(
        np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1) + eps, (v1 * v2).sum(axis=-1)
    )


@no_warnings(category=RuntimeWarning)
def points_to_normals(
    point: np.ndarray, mask: np.ndarray = None, edge_threshold: float = None
) -> np.ndarray:
    """
    Calculate normal map from point map. Value range is [-1, 1].

    Args:
        point (np.ndarray): shape (height, width, 3), point map
        mask (optional, np.ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
        edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

    Returns:
        normal (np.ndarray): shape (height, width, 3), normal map.
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ]
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    valid = (
        np.stack(
            [
                mask[:-2, 1:-1] & mask[1:-1, :-2],
                mask[1:-1, :-2] & mask[2:, 1:-1],
                mask[2:, 1:-1] & mask[1:-1, 2:],
                mask[1:-1, 2:] & mask[:-2, 1:-1],
            ]
        )
        & mask[None, 1:-1, 1:-1]
    )
    if edge_threshold is not None:
        view_angle = angle_diff_vec3_numpy(pts[None, 1:-1, 1:-1, :], normal)
        view_angle = np.minimum(view_angle, np.pi - view_angle)
        valid = valid & (view_angle < np.deg2rad(edge_threshold))

    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal


def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Create a sliding window view of the input array along a specified axis.

    This function creates a memory-efficient view of the input array with sliding windows
    of the specified size and stride. The window dimension is appended to the end of the
    output array's shape. This is useful for operations like convolution, pooling, or
    any analysis that requires examining local neighborhoods in the data.

    Args:
        x (np.ndarray): Input array with shape (..., axis_size, ...)
        window_size (int): Size of the sliding window
        stride (int): Stride of the sliding window (step size between consecutive windows)
        axis (int, optional): Axis to perform sliding window over. Defaults to -1 (last axis)

    Returns:
        np.ndarray: View of the input array with shape (..., n_windows, ..., window_size),
                   where n_windows = (axis_size - window_size + 1) // stride

    Raises:
        AssertionError: If window_size is larger than the size of the specified axis

    Example:
        >>> x = np.array([1, 2, 3, 4, 5, 6])
        >>> sliding_window_1d(x, window_size=3, stride=2)
        array([[1, 2, 3],
               [3, 4, 5]])
    """
    assert x.shape[axis] >= window_size, (
        f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    )
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1) // stride,
        *x.shape[axis + 1 :],
        window_size,
    )
    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1 :],
        x.strides[axis],
    )
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding


def sliding_window_nd(
    x: np.ndarray,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Create sliding windows along multiple dimensions of the input array.

    This function applies sliding_window_1d sequentially along multiple axes to create
    N-dimensional sliding windows. This is useful for operations that need to examine
    local neighborhoods in multiple dimensions simultaneously.

    Args:
        x (np.ndarray): Input array
        window_size (Tuple[int, ...]): Size of the sliding window for each axis
        stride (Tuple[int, ...]): Stride of the sliding window for each axis
        axis (Tuple[int, ...]): Axes to perform sliding window over

    Returns:
        np.ndarray: Array with sliding windows along the specified dimensions.
                   The window dimensions are appended to the end of the shape.

    Note:
        The length of window_size, stride, and axis tuples must be equal.

    Example:
        >>> x = np.random.rand(10, 10)
        >>> windows = sliding_window_nd(x, window_size=(3, 3), stride=(2, 2), axis=(-2, -1))
        >>> # Creates 3x3 sliding windows with stride 2 in both dimensions
    """
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(
    x: np.ndarray,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
) -> np.ndarray:
    """
    Create 2D sliding windows over the input array.

    Convenience function for creating 2D sliding windows, commonly used for image
    processing operations like convolution, pooling, or patch extraction.

    Args:
        x (np.ndarray): Input array
        window_size (Union[int, Tuple[int, int]]): Size of the 2D sliding window.
                                                  If int, same size is used for both dimensions.
        stride (Union[int, Tuple[int, int]]): Stride of the 2D sliding window.
                                             If int, same stride is used for both dimensions.
        axis (Tuple[int, int], optional): Two axes to perform sliding window over.
                                         Defaults to (-2, -1) (last two dimensions).

    Returns:
        np.ndarray: Array with 2D sliding windows. The window dimensions (height, width)
                   are appended to the end of the shape.

    Example:
        >>> image = np.random.rand(100, 100)
        >>> patches = sliding_window_2d(image, window_size=8, stride=4)
        >>> # Creates 8x8 patches with stride 4 from the image
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(
    x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1
):
    """
    Perform 1D max pooling on the input array.

    Max pooling reduces the dimensionality of the input by taking the maximum value
    within each sliding window. This is commonly used in neural networks and signal
    processing for downsampling and feature extraction.

    Args:
        x (np.ndarray): Input array
        kernel_size (int): Size of the pooling kernel
        stride (int): Stride of the pooling operation
        padding (int, optional): Amount of padding to add on both sides. Defaults to 0.
        axis (int, optional): Axis to perform max pooling over. Defaults to -1.

    Returns:
        np.ndarray: Max pooled array with reduced size along the specified axis

    Note:
        - For floating point arrays, padding is done with np.nan values
        - For integer arrays, padding is done with the minimum value of the dtype
        - np.nanmax is used to handle NaN values in the computation

    Example:
        >>> x = np.array([1, 3, 2, 4, 5, 1, 2])
        >>> max_pool_1d(x, kernel_size=3, stride=2)
        array([3, 5, 2])
    """
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        padding_arr = np.full(
            (*x.shape[:axis], padding, *x.shape[axis + 1 :]),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(
    x: np.ndarray,
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Perform N-dimensional max pooling on the input array.

    This function applies max_pool_1d sequentially along multiple axes to perform
    multi-dimensional max pooling. This is useful for downsampling multi-dimensional
    data while preserving the most important features.

    Args:
        x (np.ndarray): Input array
        kernel_size (Tuple[int, ...]): Size of the pooling kernel for each axis
        stride (Tuple[int, ...]): Stride of the pooling operation for each axis
        padding (Tuple[int, ...]): Amount of padding for each axis
        axis (Tuple[int, ...]): Axes to perform max pooling over

    Returns:
        np.ndarray: Max pooled array with reduced size along the specified axes

    Note:
        The length of kernel_size, stride, padding, and axis tuples must be equal.
        Max pooling is applied sequentially along each axis in the order specified.

    Example:
        >>> x = np.random.rand(10, 10, 10)
        >>> pooled = max_pool_nd(x, kernel_size=(2, 2, 2), stride=(2, 2, 2),
        ...                      padding=(0, 0, 0), axis=(-3, -2, -1))
        >>> # Reduces each dimension by half with 2x2x2 max pooling
    """
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
):
    """
    Perform 2D max pooling on the input array.

    Convenience function for 2D max pooling, commonly used in computer vision
    and image processing for downsampling images while preserving important features.

    Args:
        x (np.ndarray): Input array
        kernel_size (Union[int, Tuple[int, int]]): Size of the 2D pooling kernel.
                                                  If int, same size is used for both dimensions.
        stride (Union[int, Tuple[int, int]]): Stride of the 2D pooling operation.
                                             If int, same stride is used for both dimensions.
        padding (Union[int, Tuple[int, int]]): Amount of padding for both dimensions.
                                              If int, same padding is used for both dimensions.
        axis (Tuple[int, int], optional): Two axes to perform max pooling over.
                                         Defaults to (-2, -1) (last two dimensions).

    Returns:
        np.ndarray: 2D max pooled array with reduced size along the specified axes

    Example:
        >>> image = np.random.rand(64, 64)
        >>> pooled = max_pool_2d(image, kernel_size=2, stride=2, padding=0)
        >>> # Reduces image size from 64x64 to 32x32 with 2x2 max pooling
    """
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)


@no_warnings(category=RuntimeWarning)
def depth_edge(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.

    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff = max_pool_2d(
            depth, kernel_size, stride=1, padding=kernel_size // 2
        ) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = max_pool_2d(
            np.where(mask, depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + max_pool_2d(
            np.where(mask, -depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol

    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def depth_aliasing(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the map that indicates the aliasing of x depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff_max = (
            max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        )
        diff_min = (
            max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
        )
    else:
        diff_max = (
            max_pool_2d(
                np.where(mask, depth, -np.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            - depth
        )
        diff_min = (
            max_pool_2d(
                np.where(mask, -depth, -np.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            + depth
        )
    diff = np.minimum(diff_max, diff_min)

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


@no_warnings(category=RuntimeWarning)
def normals_edge(
    normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None
) -> np.ndarray:
    """
    Compute the edge mask from normal map.

    Args:
        normal (np.ndarray): shape (..., height, width, 3), normal map
        tol (float): tolerance in degrees

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3, (
        "normal should be of shape (..., height, width, 3)"
    )
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    padding = kernel_size // 2
    normals_window = sliding_window_2d(
        np.pad(
            normals,
            (
                *([(0, 0)] * (normals.ndim - 3)),
                (padding, padding),
                (padding, padding),
                (0, 0),
            ),
            mode="edge",
        ),
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2),
    )
    if mask is None:
        angle_diff = np.arccos(
            (normals[..., None, None] * normals_window).sum(axis=-3)
        ).max(axis=(-2, -1))
    else:
        mask_window = sliding_window_2d(
            np.pad(
                mask,
                (*([(0, 0)] * (mask.ndim - 3)), (padding, padding), (padding, padding)),
                mode="edge",
            ),
            window_size=kernel_size,
            stride=1,
            axis=(-3, -2),
        )
        angle_diff = np.where(
            mask_window,
            np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)),
            0,
        ).max(axis=(-2, -1))

    angle_diff = max_pool_2d(
        angle_diff, kernel_size, stride=1, padding=kernel_size // 2
    )
    edge = angle_diff > np.deg2rad(tol)
    return edge
