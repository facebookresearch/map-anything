# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import einsum, repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapanything.utils.wai.basic_dataset import BasicSceneframeDataset
from mapanything.utils.wai.intersection_check import (
    create_frustum_from_intrinsics,
    frustum_intersection_check,
)
from mapanything.utils.wai.m_ops import in_image, m_dot, m_project, m_unproject
from mapanything.utils.wai.ops import resize

logger = logging.getLogger("covis_utils")


def resize_sparse_depth(
    depth: torch.Tensor,
    scale: Optional[float] = None,
    size: Optional[Any] = None,
) -> torch.Tensor:
    """Resize sparse depth maps (e.g. projected LiDAR) with a masked min-pool.

    Each output pixel gets the minimum of the valid (> 0) input depths that fall
    into it, or 0 if there are none. Nearest resizing would drop most of the
    sparse points and leave almost no valid pixels at low resolution.

    Args:
        depth: Depth maps (B, H, W), dense or nested tensor
        scale: Scale factor to apply to the original dimensions
        size: Target size as (height, width) or int for long-side scaling

    Returns:
        Resized depth maps (B, out_h, out_w)
    """
    resized = []
    for d in depth:
        h, w = d.shape
        if scale is not None:
            out_h, out_w = int(h * scale), int(w * scale)
        elif isinstance(size, int):
            if h > w:
                out_h, out_w = size, int(w * size / h)
            else:
                out_h, out_w = int(h * size / w), size
        else:
            out_h, out_w = size
        ys, xs = torch.nonzero(d > 0, as_tuple=True)
        out_inds = (ys * out_h // h) * out_w + (xs * out_w // w)
        out = torch.full((out_h * out_w,), float("inf"), device=d.device)
        out.scatter_reduce_(0, out_inds, d[ys, xs], reduce="amin")
        out[torch.isinf(out)] = 0
        resized.append(out.view(out_h, out_w))
    return torch.stack(resized)


def load_scene_data(cfg: Any, scene_name: str, device: str) -> Dict[str, Any]:
    """Load and preprocess scene data.

    Args:
        cfg: Configuration object containing parameters
        scene_name: Name of the scene to process
        device: Device to use for computation ('cuda' or 'cpu')

    Returns:
        Dict containing preprocessed scene data including depths, intrinsics, etc.
    """
    # Create a dataloader that only parses a single scene.
    # This ensures that every loaded frame belongs to this scene.
    cfg.scene_filters = [scene_name]
    single_scene_dataset = BasicSceneframeDataset(cfg)
    dataloader = DataLoader(
        single_scene_dataset,
        cfg.batch_size,
        collate_fn=single_scene_dataset.collate_fn,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    depths = []
    depth_h = 0
    depth_w = 0
    valid_depth_masks = []
    intrinsics = []
    cam2worlds = []
    world_pts3d = []
    frame_names = []

    # Load cameras and depth maps for all frames.
    for batch in tqdm(dataloader, f"Loading depth maps ({scene_name})"):
        if "frame_name" in batch:
            frame_names.extend(batch["frame_name"])

        depth = batch["depth"].to(device)

        # Check for depth confidence values if available
        depth_confidences = batch.get("depth_confidence")
        if depth_confidences is not None:
            depth_confidences = depth_confidences.to(device)

        # Validate configuration parameters
        if (
            cfg.get("downscale_factor") is not None
            and cfg.get("target_size") is not None
        ):
            raise ValueError("Either set downscale_factor or target_size, not both.")

        # Determine scaling parameters
        scale, size = None, None
        if cfg.get("downscale_factor") is not None and cfg.downscale_factor > 1:
            scale = 1 / cfg.downscale_factor
        elif cfg.get("target_size"):
            size = cfg.target_size  # Rescale to a fixed resolution

        # Rescale depth maps if needed
        if scale or size:
            if cfg.get("sparse_depth", False):
                depth = resize_sparse_depth(depth, scale=scale, size=size)
            else:
                depth = resize(depth, scale=scale, size=size, modality_format="depth")
            if depth_confidences is not None:
                depth_confidences = resize(
                    depth_confidences,
                    scale=scale,
                    size=size,
                    modality_format="depth",
                )

        # Ensure consistent depth map dimensions
        if depth_h == 0 and depth_w == 0:
            depth_h, depth_w = depth.shape[-2:]
        elif depth_h != depth.shape[-2] or depth_w != depth.shape[-1]:
            raise ValueError("Depth resolutions vary, set target_size in the config")

        # Create mask for valid depth values
        valid_depth_mask = depth > 0
        if depth_confidences is not None:
            valid_depth_mask &= depth_confidences > cfg.depth_confidence_thresh

        # Calculate scaling factors for intrinsics
        scale_h = depth_h / torch.tensor(batch["h"])
        scale_w = depth_w / torch.tensor(batch["w"])

        # Adjust intrinsics for the scaled depth maps
        depth_intrinsics = torch.clone(batch["intrinsics"])
        depth_intrinsics[:, :1] *= scale_w[:, None, None]
        depth_intrinsics[:, 1:2] *= scale_h[:, None, None]

        # Unproject the depth to 3D points
        depth_intrinsics = depth_intrinsics.to(device)
        curr_batch_cam2worlds = batch["extrinsics"].to(device)
        curr_batch_world_pts3d = m_unproject(
            depth,
            depth_intrinsics,
            curr_batch_cam2worlds,
        )

        # Store processed data
        depths.append(depth)
        valid_depth_masks.append(valid_depth_mask)
        intrinsics.append(depth_intrinsics)
        cam2worlds.append(curr_batch_cam2worlds)
        list_curr_batch_world_pts3d = list(
            torch.unbind(curr_batch_world_pts3d.cpu(), dim=0)
        )
        world_pts3d.extend(
            list_curr_batch_world_pts3d
        )  # Keep on CPU to save GPU memory

    depths = torch.cat(depths)
    intrinsics = torch.cat(intrinsics)
    cam2worlds = torch.cat(cam2worlds)
    valid_depth_masks = torch.cat(valid_depth_masks)

    # sanity checks
    num_frames = depths.shape[0]
    assert intrinsics.shape[0] == num_frames, (
        f"First dim of concatentated intrinsics {intrinsics.shape[0]} doesn't match with expected num of frames: {num_frames}"
    )
    assert cam2worlds.shape[0] == num_frames, (
        f"First dim of concatentated extrinsics {cam2worlds.shape[0]} doesn't match with expected num of frames: {num_frames}"
    )
    assert valid_depth_masks.shape[0] == num_frames, (
        f"First dim of concatentated valid depth masks doesn't match with expected num of frames: {num_frames}"
    )
    assert len(world_pts3d) == num_frames, (
        f"Length of list containing 3d points in world frame {len(world_pts3d)} doesn't match with expected num of frames: {num_frames}"
    )

    result = {
        "depths": depths,
        "depth_dims": (depth_h, depth_w),
        "valid_depth_masks": valid_depth_masks,
        "intrinsics": intrinsics,
        "cam2worlds": cam2worlds,
        "world_pts3d": world_pts3d,
    }

    # Add optional data if available
    if frame_names:
        result["frame_names"] = frame_names

    return result


def compute_frustum_intersection(
    cfg: Any,
    depths: torch.Tensor,
    valid_depth_masks: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2worlds: torch.Tensor,
    device: str,
) -> Optional[torch.Tensor]:
    """Compute frustum intersection between views if enabled.

    Args:
        cfg: Configuration object containing parameters
        depths: Tensor of depth maps for all frames
        valid_depth_masks: Boolean masks indicating valid depth values
        intrinsics: Camera intrinsic matrices for all frames
        cam2worlds: Camera extrinsic matrices (cam2world) for all frames
        device: Device to use for computation

    Returns:
        Tensor containing frustum intersection results or None if disabled
    """
    if not cfg.get("perform_frustum_check", True):
        return None

    near_vals = torch.tensor(
        [
            depth[valid_mask].min() if valid_mask.any() else torch.tensor(0)
            for depth, valid_mask in zip(depths, valid_depth_masks)
        ]
    ).to(device)

    far_vals = torch.tensor(
        [
            depth[valid_mask].max() if valid_mask.any() else torch.tensor(0)
            for depth, valid_mask in zip(depths, valid_depth_masks)
        ]
    ).to(device)

    frustums = create_frustum_from_intrinsics(intrinsics, near_vals, far_vals)
    frustums_homog = torch.cat([frustums, torch.ones_like(frustums[:, :, :1])], dim=-1)
    frustums_world = einsum(cam2worlds, frustums_homog, "b i k, b v k -> b v i")
    frustums_world = frustums_world[:, :, :3]

    # Compute batched frustum intersection check
    frustum_intersection = frustum_intersection_check(
        frustums_world, chunk_size=500, device=device
    )

    # Free up memory by removing unneeded tensors
    del frustums, frustums_homog, frustums_world, near_vals, far_vals
    torch.cuda.empty_cache()

    return frustum_intersection


def project_points_to_views(
    idx: int,
    ov_inds: torch.Tensor,
    depth_h: int,
    depth_w: int,
    valid_depth_masks: torch.Tensor,
    cam2worlds: torch.Tensor,
    world_pts3d: List[torch.Tensor],
    intrinsics: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project points from one view to other views.

    Args:
        idx: Index of the source view
        ov_inds: Indices of overlapping views
        depth_h: Height of depth maps
        depth_w: Width of depth maps
        valid_depth_masks: Boolean masks indicating valid depth values
        cam2worlds: Camera extrinsic matrices (cam2world) for all frames
        world_pts3d: List of 3D points in world coordinates for all frames
        intrinsics: Camera intrinsic matrices for all frames
        device: Device to use for computation

    Returns:
        Tuple containing:
        - reprojected_pts: Tensor of reprojected points
        - valid_mask: Boolean mask indicating valid reprojections
        - view_cam_pts3d: Tensor of 3D points in camera coordinates
    """
    v_rem = len(ov_inds)

    # Project the depth map v into all the overlapping views in this chunk
    view_cam_pts3d = m_dot(
        torch.inverse(cam2worlds[ov_inds]),
        repeat(world_pts3d[idx].to(device), "... -> V ...", V=v_rem),
        maintain_shape=True,
    )

    reprojected_pts = m_project(view_cam_pts3d, intrinsics[ov_inds]).reshape(
        v_rem, depth_h, depth_w, 3
    )

    # Filter out points which are outside the image boundaries
    valid_mask = (
        in_image(reprojected_pts, depth_h, depth_w, min_depth=0.04)
        & valid_depth_masks[idx]
    )

    return reprojected_pts, valid_mask, view_cam_pts3d


def sample_depths_at_reprojections(
    reprojected_pts: torch.Tensor,
    depths: torch.Tensor,
    ov_inds: torch.Tensor,
    depth_h: int,
    depth_w: int,
    device: str,
    window: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample depths at reprojected points.

    Args:
        reprojected_pts: Tensor of reprojected points
        depths: Tensor of depth maps for all frames
        ov_inds: Indices of overlapping views
        depth_h: Height of depth maps
        depth_w: Width of depth maps
        device: Device to use for computation
        window: Size of the lookup window. If > 1, take the valid depth in the
            window x window neighborhood that is closest to the expected depth
            (for sparse depth, where most reprojections land on empty pixels)

    Returns:
        Tuple containing:
        - depth_lu: Sampled depth values
        - expected_depth: Expected depth values at reprojected points
    """
    expected_depth = reprojected_pts[..., 2]
    target_depths = depths[ov_inds].unsqueeze(1)
    pts_xy = reprojected_pts[..., [1, 0]]
    image_size = torch.tensor([depth_w - 1, depth_h - 1], device=device)

    # Sample depths at reprojected points (and their neighbors if window > 1)
    depth_lu = None
    r = window // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            offset = torch.tensor([dx, dy], device=device)
            normalized_pts = 2 * (pts_xy + offset) / image_size - 1
            normalized_pts = torch.clamp(normalized_pts, min=-1.0, max=1.0)
            lu = F.grid_sample(
                target_depths,
                normalized_pts,
                mode="nearest",
                align_corners=True,
            )[:, 0]
            if depth_lu is None:
                depth_lu = lu
            else:
                closer = (lu > 0) & (
                    (depth_lu <= 0)
                    | ((lu - expected_depth).abs() < (depth_lu - expected_depth).abs())
                )
                depth_lu = torch.where(closer, lu, depth_lu)

    return depth_lu, expected_depth
