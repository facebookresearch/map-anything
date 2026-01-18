# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Multi-view depth consistency confidence computation.

This module provides utilities for computing per-pixel confidence scores based on
multi-view depth projection consistency, as an alternative to learning-based confidence.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import einsum

from mapanything.utils.geometry import (
    closed_form_pose_inverse,
    depthmap_to_camera_frame,
)
from mapanything.utils.wai.intersection_check import (
    create_frustum_from_intrinsics,
    frustum_intersection_check,
)


def _in_image(
    pts: torch.Tensor, height: int, width: int, min_depth: float = 0.0
) -> torch.Tensor:
    """
    Check if projected points are within image boundaries.

    Args:
        pts: Tensor of shape (..., 3) containing (x, y, depth) coordinates
        height: Image height
        width: Image width
        min_depth: Minimum valid depth value

    Returns:
        Boolean tensor indicating valid points
    """
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z > min_depth)
    return valid


def _project_pts3d_to_image_with_depth(
    pts3d: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Project 3D points to image plane, returning (x, y, depth).

    Args:
        pts3d: BxHxWx3 or BxNx3 tensor of 3D points in camera frame
        intrinsics: Bx3x3 camera intrinsics

    Returns:
        Tensor of same shape as pts3d with (x, y, depth) values
    """
    # Project points: K @ pts3d
    projected = einsum(intrinsics, pts3d, "b i k, b ... k -> b ... i")
    # Normalize by depth (z)
    depth = projected[..., 2:3].clamp(min=1e-6)
    xy = projected[..., :2] / depth
    return torch.cat([xy, projected[..., 2:3]], dim=-1)


def _compute_frustum_intersection_matrix(
    intrinsics: List[torch.Tensor],
    camera_poses: List[torch.Tensor],
    depth_z: List[torch.Tensor],
    frustum_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute frustum intersection matrix for all view pairs.

    Args:
        intrinsics: List of (B, 3, 3) intrinsics per view
        camera_poses: List of (B, 4, 4) cam2world poses per view
        depth_z: List of (B, H, W, 1) depth maps per view
        frustum_chunk_size: Chunk size for frustum intersection computation
        device: Device for computation

    Returns:
        (num_views, num_views) boolean tensor indicating overlapping views
    """
    # Compute near/far values from depth maps
    near_vals = []
    far_vals = []
    for depth in depth_z:
        valid_depth = depth[depth > 0]
        if valid_depth.numel() > 0:
            near_vals.append(valid_depth.min().item())
            far_vals.append(valid_depth.max().item())
        else:
            near_vals.append(0.1)
            far_vals.append(100.0)

    near_vals = torch.tensor(near_vals, device=device)
    far_vals = torch.tensor(far_vals, device=device)

    # Stack intrinsics for all views (take first batch element for frustum computation)
    stacked_intrinsics = torch.stack([intr[0] for intr in intrinsics], dim=0)
    stacked_poses = torch.stack([pose[0] for pose in camera_poses], dim=0)

    # Create frustums in camera frame
    frustums = create_frustum_from_intrinsics(stacked_intrinsics, near_vals, far_vals)

    # Transform frustums to world frame
    frustums_homog = torch.cat([frustums, torch.ones_like(frustums[:, :, :1])], dim=-1)
    frustums_world = einsum(stacked_poses, frustums_homog, "b i k, b v k -> b v i")
    frustums_world = frustums_world[:, :, :3]

    # Compute frustum intersection
    frustum_intersection = frustum_intersection_check(
        frustums_world, chunk_size=frustum_chunk_size, device=device
    )

    return frustum_intersection


def compute_multiview_depth_confidence(
    depth_z: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    camera_poses: List[torch.Tensor],
    depth_assoc_abs_thresh: float = 0.02,
    depth_assoc_rel_thresh: float = 0.02,
    frustum_chunk_size: int = 50,
    depth_masks: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """
    Compute multi-view depth consistency confidence for each view.

    This function computes per-pixel confidence scores based on how consistently
    each pixel's depth is observed across multiple views. A pixel is considered
    confident if its depth, when projected to other views, matches the depth
    observed in those views.

    Args:
        depth_z: List of (B, H, W, 1) Z-depth maps per view
        intrinsics: List of (B, 3, 3) camera intrinsic matrices per view
        camera_poses: List of (B, 4, 4) cam2world transformation matrices per view
        depth_assoc_abs_thresh: Absolute depth threshold for inlier matching (default: 0.02)
        depth_assoc_rel_thresh: Relative depth threshold for inlier matching (default: 0.02)
        frustum_chunk_size: Chunk size for frustum intersection computation (default: 50)
        depth_masks: Optional list of (B, H, W) boolean masks per view to filter valid depth regions.
            If provided, only pixels where the mask is True will be considered for confidence computation.

    Returns:
        List of (B, H, W) confidence tensors, one per view, with values in [0, 1]
    """
    num_views = len(depth_z)
    if num_views < 2:
        # Single view: return all ones
        return [torch.ones_like(depth_z[0][..., 0])]

    device = depth_z[0].device
    batch_size, height, width, _ = depth_z[0].shape

    # Compute frustum intersection matrix for efficiency
    frustum_intersection = _compute_frustum_intersection_matrix(
        intrinsics, camera_poses, depth_z, frustum_chunk_size, device
    )

    # Compute pts3d in camera frame, world frame, and world2cam transforms for each view
    pts3d_world = []
    world2cam = []
    for i in range(num_views):
        # depthmap_to_camera_frame expects (B, H, W) depth and (B, 3, 3) intrinsics
        depth_squeezed = depth_z[i][..., 0]  # (B, H, W)
        pts_cam, _ = depthmap_to_camera_frame(depth_squeezed, intrinsics[i])

        # Transform to world frame
        pts_homog = torch.cat([pts_cam, torch.ones_like(pts_cam[..., :1])], dim=-1)
        pts_world = einsum(camera_poses[i], pts_homog, "b i k, b h w k -> b h w i")
        pts3d_world.append(pts_world[..., :3])

        # Compute world2cam transformation
        world2cam.append(closed_form_pose_inverse(camera_poses[i]))

    # Compute confidence for each view
    confidence_list = []
    for src_idx in range(num_views):
        # Get overlapping view indices based on frustum intersection
        ov_mask = frustum_intersection[src_idx]
        ov_inds = ov_mask.nonzero(as_tuple=True)[0]

        # Exclude self
        ov_inds = ov_inds[ov_inds != src_idx]

        if len(ov_inds) == 0:
            # No overlapping views, return ones
            confidence_list.append(torch.ones(batch_size, height, width, device=device))
            continue

        # Valid depth mask for source view
        valid_src_depth = depth_z[src_idx][..., 0] > 0  # (B, H, W)
        if depth_masks is not None:
            valid_src_depth = valid_src_depth & depth_masks[src_idx]

        # Accumulators for inliers and outliers
        inlier_count = torch.zeros(batch_size, height, width, device=device)
        outlier_count = torch.zeros(batch_size, height, width, device=device)

        # Project source points to each overlapping view
        for tgt_idx in ov_inds:
            tgt_idx = tgt_idx.item()

            # Transform source world points to target camera frame
            src_pts_world = pts3d_world[src_idx]  # (B, H, W, 3)
            src_pts_homog = torch.cat(
                [src_pts_world, torch.ones_like(src_pts_world[..., :1])], dim=-1
            )
            src_pts_tgt_cam = einsum(
                world2cam[tgt_idx], src_pts_homog, "b i k, b h w k -> b h w i"
            )
            src_pts_tgt_cam = src_pts_tgt_cam[..., :3]

            # Project to target image plane
            projected = _project_pts3d_to_image_with_depth(
                src_pts_tgt_cam, intrinsics[tgt_idx]
            )  # (B, H, W, 3) with (x, y, expected_depth)

            # Check which points are valid (in image and positive depth)
            valid_projection = (
                _in_image(projected, height, width, min_depth=0.04) & valid_src_depth
            )

            # Sample depth at projected locations
            # Normalize coordinates for grid_sample: [-1, 1]
            normalized_pts = (
                2.0
                * projected[..., :2]
                / torch.tensor(
                    [width - 1, height - 1], device=device, dtype=projected.dtype
                )
                - 1.0
            )
            normalized_pts = normalized_pts.clamp(-1.0, 1.0)

            # grid_sample expects (N, C, H, W) input and (N, H, W, 2) grid
            tgt_depth_map = depth_z[tgt_idx].permute(0, 3, 1, 2)  # (B, 1, H, W)
            sampled_depth = F.grid_sample(
                tgt_depth_map,
                normalized_pts,
                mode="nearest",
                align_corners=True,
                padding_mode="zeros",
            )  # (B, 1, H, W)
            sampled_depth = sampled_depth[:, 0]  # (B, H, W)

            # Expected depth is the z-coordinate of projected points
            expected_depth = projected[..., 2]

            # Compute depth association threshold
            depth_thresh = (
                depth_assoc_abs_thresh + depth_assoc_rel_thresh * expected_depth
            )

            # Compute reprojection error
            reproj_error = torch.abs(expected_depth - sampled_depth)

            # Count inliers and outliers
            is_inlier = (reproj_error < depth_thresh) & valid_projection
            is_outlier = (reproj_error >= depth_thresh) & valid_projection

            inlier_count += is_inlier.float()
            outlier_count += is_outlier.float()

        # Compute confidence as ratio of inliers
        epsilon = 1e-10
        confidence = inlier_count / (inlier_count + outlier_count + epsilon)
        confidence_list.append(confidence)

    return confidence_list
