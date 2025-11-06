# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import shutil
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
from argconf import argconf_parse
from tqdm import tqdm
from wai_processing.utils.covis_utils import (
    compute_frustum_intersection,
    load_scene_data,
    project_points_to_views,
    sample_depths_at_reprojections,
)
from wai_processing.utils.state import (
    set_processing_state,
)

from morphcloud.utils.wai.core import (
    get_frame,
    load_data,
    nest_modality,
    set_frame,
    store_data,
)
from morphcloud.utils.wai.scene_frame import get_scene_names

logger = logging.getLogger("covisibility-confidence")


def _process_frame(
    cfg: Any,
    idx: int,
    depths: torch.Tensor,
    depth_h: int,
    depth_w: int,
    valid_depth_masks: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2worlds: torch.Tensor,
    world_pts3d: list,
    frustum_intersection: Optional[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Process a single frame to compute covisibility and covisibility confidence.

    Args:
        cfg: Configuration object containing parameters
        idx: Index of the frame to process
        depths: Tensor of depth maps for all frames
        depth_h: Height of depth maps
        depth_w: Width of depth maps
        valid_depth_masks: Boolean masks indicating valid depth values
        intrinsics: Camera intrinsic matrices for all frames
        cam2worlds: Camera extrinsic matrices (cam2world) for all frames
        world_pts3d: List of 3D points in world coordinates for all frames
        frustum_intersection: Tensor containing frustum intersection results or None
        device: Device to use for computation

    Returns:
        torch.Tensor: Computed covisibility confidence map with shape (depth_h, depth_w)
    """
    num_frames = depths.shape[0]

    # Get the remaining views which pass the frustum intersection check
    if cfg.get("perform_frustum_check", True) and frustum_intersection is not None:
        ov_inds = frustum_intersection[idx].argwhere()[:, 0].to(device)
    else:
        ov_inds = torch.arange(num_frames).to(device)

    if len(ov_inds) == 0:
        return torch.zeros((depth_h, depth_w), device=device)

    # Process overlapping views in sub-chunks if needed
    ov_chunk_size = 4000

    valid_depth_projection_covis_sum = torch.zeros(
        (depths.shape[-2], depths.shape[-1]), device=device
    )
    invalid_depth_projection_covis_sum = torch.zeros(
        (depths.shape[-2], depths.shape[-1]), device=device
    )

    # Process in chunks to avoid OOM
    for ov_start in range(0, len(ov_inds), ov_chunk_size):
        ov_end = min(ov_start + ov_chunk_size, len(ov_inds))
        ov_inds_chunk = ov_inds[ov_start:ov_end]
        v_rem = len(ov_inds_chunk)
        if v_rem == 0:
            continue

        # Project the depth map using utility function
        reprojected_pts, valid_mask, _ = project_points_to_views(
            idx,
            ov_inds_chunk,
            depth_h,
            depth_w,
            valid_depth_masks,
            cam2worlds,
            world_pts3d,
            intrinsics,
            device,
        )

        # If any points are valid, compute the covisibility
        if valid_mask.any():
            # Sample depths at reprojected points
            depth_lu, expected_depth = sample_depths_at_reprojections(
                reprojected_pts, depths, ov_inds_chunk, depth_h, depth_w, device
            )
            # Calculate reprojection error and depth association threshold
            reprojection_error = torch.abs(expected_depth - depth_lu)

            # Calculate depth association threshold for covisibility map
            depth_assoc_thres_map = (
                cfg.depth_assoc_error_thres_covis
                + cfg.depth_assoc_rel_error_thres_covis * expected_depth
            )

            # calculate inliers of the reprojections
            valid_depth_projection_covis = (
                reprojection_error < depth_assoc_thres_map
            ) & valid_mask
            valid_depth_projection_covis_sum_chunked = valid_depth_projection_covis.sum(
                dim=0
            ).int()

            # calculate outliers of the reprojections
            invalid_depth_projection_covis = (
                reprojection_error > depth_assoc_thres_map
            ) & valid_mask
            invalid_depth_projection_covis_sum_chunked = (
                invalid_depth_projection_covis.sum(dim=0).int()
            )

            # sum it up in case its chunked
            invalid_depth_projection_covis_sum += (
                invalid_depth_projection_covis_sum_chunked
            )
            valid_depth_projection_covis_sum += valid_depth_projection_covis_sum_chunked

    # Calculate confidence map as a relative number of inliers to outliers
    # will result in a confidence score between 0..1
    epsilon = 1e-10  # Small epsilon to prevent division by zero
    covis_depth_confidence = valid_depth_projection_covis_sum / (
        valid_depth_projection_covis_sum + invalid_depth_projection_covis_sum + epsilon
    )

    # Free memory
    torch.cuda.empty_cache()

    return covis_depth_confidence


def compute_covisibility_map(
    cfg: Any, scene_name: str, overwrite: bool = False
) -> None:
    """Compute covisibility maps for a scene.

    Args:
        cfg: Configuration object containing parameters
        scene_name: Name of the scene to process
        overwrite: Whether to overwrite existing output files

    Returns:
        None. Results are saved to disk.
    """
    device = cfg.get("device", "cuda")

    # Setup scene data
    scene_root = Path(cfg.root) / scene_name
    scene_meta = load_data(Path(scene_root, "scene_meta.json"), "scene_meta")

    # Create output directory
    out_path = scene_root / cfg.out_path
    out_path_confidence = out_path / "depth_confidence"
    if out_path_confidence.exists() and overwrite:
        shutil.rmtree(out_path_confidence)

    # Load scene data using the utility function
    frame_data = load_scene_data(cfg, scene_name, device)
    depths = frame_data["depths"]
    depth_h, depth_w = frame_data["depth_dims"]
    valid_depth_masks = frame_data["valid_depth_masks"]
    intrinsics = frame_data["intrinsics"]
    cam2worlds = frame_data["cam2worlds"]
    world_pts3d = frame_data["world_pts3d"]
    frame_names = frame_data["frame_names"]

    # Perform frustum intersection check if enabled
    frustum_intersection = compute_frustum_intersection(
        cfg, depths, valid_depth_masks, intrinsics, cam2worlds, device
    )

    # Compute pairwise covisibility for each view
    num_frames = depths.shape[0]
    logger.info("Computing pairwise overlap for each view ...")

    for idx in tqdm(
        range(num_frames),
        f"Computing exhaustive pairwise covisibility for each view ({scene_name})",
    ):
        # Process current frame
        covis_depth_confidence = _process_frame(
            cfg,
            idx,
            depths,
            depth_h,
            depth_w,
            valid_depth_masks,
            intrinsics,
            cam2worlds,
            world_pts3d,
            frustum_intersection,
            device,
        )

        # Save depth confidence map
        frame_name = frame_names[idx]
        rel_covis_maps_path = f"depth_confidence/{frame_name}.exr"
        store_data(
            out_path / rel_covis_maps_path,
            covis_depth_confidence,
            "scalar",
        )

        # Update frame metadata
        frame = get_frame(scene_meta, frame_name)
        frame[f"{cfg.method_name}_depth_confidence"] = (
            f"{cfg.out_path}/{rel_covis_maps_path}"
        )
        set_frame(scene_meta, frame_name, frame, sort=True)

        # Update frame modalities if needed
        frame_modalities = scene_meta["frame_modalities"]
        frame_modalities_depth_conf = nest_modality(
            frame_modalities, "depth_confidence"
        )
        frame_modalities_depth_conf[cfg.method_name] = {
            "frame_key": f"{cfg.method_name}_depth_confidence",
            "format": "scalar",
        }
        frame_modalities["depth_confidence"] = frame_modalities_depth_conf
        scene_meta["frame_modalities"] = frame_modalities

    store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    import sys

    logger.debug("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        logger.debug(f"  [{i}]: {arg}")

    cfg = argconf_parse()

    if cfg.get("root") is None:
        raise ValueError(
            "Specify the root via: 'python scripts/depth_consistency_confidence.py root=<root_path>'"
        )
    if cfg.get("frame_modalities") is None:
        raise ValueError("Specify the modality to use for depth in frame_modalities'")

    logger.info("Running depth_consistency_confidence using config:")
    for key, value in dict(cfg).items():
        logger.info(f"  {key}: {value}")

    overwrite = cfg.get("overwrite", False)
    if overwrite:
        logger.warning("Careful: Overwrite enabled!")

    device = cfg.get("device", "cuda")
    scene_names = get_scene_names(cfg)
    logger.info(f"processing: {len(scene_names)} scenes")
    logger.debug(f"scene_names = {scene_names}")

    for scene_name in tqdm(scene_names, "Processing scenes"):
        logger.info(f"Processing: {scene_name}")
        scene_root = Path(cfg.root) / scene_name
        set_processing_state(scene_root, "depth_consistency_confidence", "running")
        try:
            compute_covisibility_map(cfg, scene_name, overwrite=overwrite)
            set_processing_state(scene_root, "depth_consistency_confidence", "finished")
        except Exception:
            logging.error(
                f"Computing depth_consistency_confidence failed on scene '{scene_name}'"
            )
            trace_message = traceback.format_exc()
            logger.error(trace_message)
            set_processing_state(
                scene_root,
                "depth_consistency_confidence",
                "failed",
                message=trace_message,
            )
            continue
