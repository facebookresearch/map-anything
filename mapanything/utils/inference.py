# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference utilities.
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import torch

from mapanything.utils.geometry import (
    depth_edge,
    get_rays_in_camera_frame,
    normals_edge,
    points_to_normals,
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
    rotation_matrix_to_quaternion,
)
from mapanything.utils.image import rgb

# Hard constraints - exactly what users can provide
ALLOWED_VIEW_KEYS = {
    "img",  # Required - input images
    "data_norm_type",  # Required - normalization type of the input images
    "depth_z",  # Optional - Z depth maps
    "ray_directions",  # Optional - ray directions in camera frame
    "intrinsics",  # Optional - pinhole camera intrinsics (conflicts with ray_directions)
    "camera_poses",  # Optional - camera poses
    "is_metric_scale",  # Optional - whether inputs are metric scale
    "true_shape",  # Optional - original image shape
    "idx",  # Optional - index of the view
    "instance",  # Optional - instance info of the view
}

REQUIRED_KEYS = {"img", "data_norm_type"}

# Define conflicting keys that cannot be used together
CONFLICTING_KEYS = [
    ("intrinsics", "ray_directions")  # Both represent camera projection
]


def loss_of_one_batch_multi_view(
    batch,
    model,
    criterion,
    device,
    use_amp=False,
    amp_dtype="bf16",
    ret=None,
    ignore_keys=None,
):
    """
    Calculate loss for a batch with multiple views.

    Args:
        batch (list): List of view dictionaries containing input data.
        model (torch.nn.Module): Model to run inference with.
        criterion (callable, optional): Loss function to compute the loss.
        device (torch.device): Device to run the computation on.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
        amp_dtype (str, optional): Floating point type to use for automatic mixed precision. Options: ["fp32", "fp16", "bf16"]. Defaults to "bf16".
        ret (str, optional): If provided, return only the specified key from the result dictionary.
        ignore_keys (set, optional): Set of keys to ignore when moving tensors to device.
                                   Defaults to {"dataset", "label", "instance",
                                   "idx", "true_shape", "rng", "data_norm_type"}.

    Returns:
        dict or Any: If ret is None, returns a dictionary containing views, predictions, and loss.
                     Otherwise, returns the value associated with the ret key.
    """
    # Move necessary tensors to device
    if ignore_keys is None:
        ignore_keys = set(
            [
                "depthmap",
                "dataset",
                "label",
                "instance",
                "idx",
                "true_shape",
                "rng",
                "data_norm_type",
            ]
        )
    for view in batch:
        for name in view.keys():
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    # Determine the mixed precision floating point type
    if use_amp:
        if amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Run model and compute loss
    with torch.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):
        preds = model(batch)
        with torch.autocast("cuda", enabled=False):
            loss = criterion(batch, preds) if criterion is not None else None

    result = {f"view{i + 1}": view for i, view in enumerate(batch)}
    result.update({f"pred{i + 1}": pred for i, pred in enumerate(preds)})
    result["loss"] = loss

    return result[ret] if ret else result


def validate_input_views_for_inference(
    views: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Strict validation and preprocessing of input views.

    Args:
        views: List of view dictionaries

    Returns:
        Validated and preprocessed views

    Raises:
        ValueError: For invalid keys, missing required keys, conflicting inputs, or invalid camera pose constraints
    """
    # Ensure input is not empty
    if not views:
        raise ValueError("At least one view must be provided")

    # Track which views have camera poses
    views_with_poses = []

    # Validate each view
    for view_idx, view in enumerate(views):
        # Check for invalid keys
        provided_keys = set(view.keys())
        invalid_keys = provided_keys - ALLOWED_VIEW_KEYS
        if invalid_keys:
            raise ValueError(
                f"View {view_idx} contains invalid keys: {invalid_keys}. "
                f"Allowed keys are: {sorted(ALLOWED_VIEW_KEYS)}"
            )

        # Check for missing required keys
        missing_keys = REQUIRED_KEYS - provided_keys
        if missing_keys:
            raise ValueError(f"View {view_idx} missing required keys: {missing_keys}")

        # Check for conflicting keys
        for conflict_set in CONFLICTING_KEYS:
            present_conflicts = [key for key in conflict_set if key in provided_keys]
            if len(present_conflicts) > 1:
                raise ValueError(
                    f"View {view_idx} contains conflicting keys: {present_conflicts}. "
                    f"Only one of {conflict_set} can be provided at a time."
                )

        # Check depth constraint: If depth is provided, intrinsics or ray_directions must also be provided
        if "depth_z" in provided_keys:
            if (
                "intrinsics" not in provided_keys
                and "ray_directions" not in provided_keys
            ):
                raise ValueError(
                    f"View {view_idx} depth constraint violation: If 'depth_z' is provided, "
                    f"then 'intrinsics' or 'ray_directions' must also be provided. "
                    f"Z Depth values require camera calibration information to be meaningful for an image."
                )

        # Track views with camera poses
        if "camera_poses" in provided_keys:
            views_with_poses.append(view_idx)

    # Cross-view constraint: If any view has camera_poses, view 0 must have them too
    if views_with_poses and 0 not in views_with_poses:
        raise ValueError(
            f"Camera pose constraint violation: Views {views_with_poses} have camera_poses, "
            f"but view 0 (reference view) does not. When using camera_poses, the first view "
            f"must also provide camera_poses to serve as the reference frame."
        )

    return views


def preprocess_input_views_for_inference(
    views: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Pre-process input views to match the expected internal input format.

    The following steps are performed:
    1. Convert intrinsics to ray directions when required. If ray directions are already provided, unit normalize them.
    2. Convert depth_z to depth_along_ray
    3. Convert camera_poses to the expected input keys (camera_pose_quats and camera_pose_trans)
    4. Default is_metric_scale to True when not provided

    Args:
        views: List of view dictionaries

    Returns:
        Preprocessed views with consistent internal format
    """
    processed_views = []

    for view_idx, view in enumerate(views):
        # Copy the view dictionary to avoid modifying the original input
        processed_view = dict(view)

        # Step 1: Convert intrinsics to ray_directions when required. If ray_directions are provided, unit normalize them.
        if "intrinsics" in view:
            images = view["img"]
            height, width = images.shape[-2:]
            intrinsics = view["intrinsics"]
            _, ray_directions = get_rays_in_camera_frame(
                intrinsics=intrinsics,
                height=height,
                width=width,
                normalize_to_unit_sphere=True,
            )
            processed_view["ray_directions"] = ray_directions
            del processed_view["intrinsics"]
        elif "ray_directions" in view:
            ray_directions = view["ray_directions"]
            ray_norm = torch.norm(ray_directions, dim=-1, keepdim=True)
            processed_view["ray_directions"] = ray_directions / (ray_norm + 1e-8)

        # Step 2: Convert depth_z to depth_along_ray
        if "depth_z" in view:
            depth_z = view["depth_z"]
            ray_directions = processed_view["ray_directions"]
            ray_directions_unit_plane = ray_directions / ray_directions[..., 2:3]
            pts3d_cam = depth_z[..., None] * ray_directions_unit_plane
            depth_along_ray = torch.norm(pts3d_cam, dim=-1, keepdim=True)
            processed_view["depth_along_ray"] = depth_along_ray
            del processed_view["depth_z"]

        # Step 3: Convert camera_poses to expected input keys
        if "camera_poses" in view:
            camera_poses = view["camera_poses"]
            if isinstance(camera_poses, tuple) and len(camera_poses) == 2:
                quats, trans = camera_poses
                processed_view["camera_pose_quats"] = quats
                processed_view["camera_pose_trans"] = trans
            elif torch.is_tensor(camera_poses) and camera_poses.shape[-2:] == (4, 4):
                rotation_matrices = camera_poses[:, :3, :3]
                translation_vectors = camera_poses[:, :3, 3]
                quats = rotation_matrix_to_quaternion(rotation_matrices)
                processed_view["camera_pose_quats"] = quats
                processed_view["camera_pose_trans"] = translation_vectors
            else:
                raise ValueError(
                    f"View {view_idx}: camera_poses must be either a tuple of (quats, trans) "
                    f"or a tensor of (B, 4, 4) transformation matrices."
                )
            del processed_view["camera_poses"]

        # Step 4: Default is_metric_scale to True when not provided
        if "is_metric_scale" not in processed_view:
            # Get batch size from the image tensor
            batch_size = view["img"].shape[0]
            # Default to True for all samples in the batch
            processed_view["is_metric_scale"] = torch.ones(
                batch_size, dtype=torch.bool, device=view["img"].device
            )

        # Rename keys to match expected model input format
        if "ray_directions" in processed_view:
            processed_view["ray_directions_cam"] = processed_view["ray_directions"]
            del processed_view["ray_directions"]

        # Append the processed view to the list
        processed_views.append(processed_view)

    return processed_views


def postprocess_model_outputs_for_inference(
    raw_outputs: List[Dict[str, torch.Tensor]],
    input_views: List[Dict[str, Any]],
    apply_mask: bool = True,
    mask_edges: bool = True,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10,
) -> List[Dict[str, torch.Tensor]]:
    """
    Post-process raw model outputs by copying raw outputs and adding essential derived fields.

    This function simplifies the raw model outputs by:
    1. Copying all raw outputs as-is
    2. Adding denormalized images (img_no_norm)
    3. Adding Z depth (depth_z) from camera frame points
    4. Recovering pinhole camera intrinsics from ray directions
    5. Adding camera pose matrices (camera_poses) if pose data is available
    6. Applying mask to dense geometry outputs if requested (supports edge masking and confidence masking)

    Args:
        raw_outputs: List of raw model output dictionaries, one per view
        input_views: List of original input view dictionaries, one per view
        apply_mask: Whether to apply non-ambiguous mask to dense outputs. Defaults to True.
        mask_edges: Whether to compute an edge mask based on normals and depth and apply it to the output. Defaults to True.
        apply_confidence_mask: Whether to apply the confidence mask to the output. Defaults to False.
        confidence_percentile: The percentile to use for the confidence threshold. Defaults to 10.

    Returns:
        List of processed output dictionaries containing:
            - All original raw outputs (after masking dense geometry outputs if requested)
            - 'img_no_norm': Denormalized RGB images (B, H, W, 3)
            - 'depth_z': Z depth from camera frame (B, H, W, 1) if points in camera frame available
            - 'intrinsics': Recovered pinhole camera intrinsics (B, 3, 3) if ray directions available
            - 'camera_poses': 4x4 pose matrices (B, 4, 4) if pose data available
            - 'mask': comprehensive mask for dense geometry outputs (B, H, W, 1) if requested

    """
    processed_outputs = []

    for view_idx, (raw_output, original_view) in enumerate(
        zip(raw_outputs, input_views)
    ):
        # Start by copying all raw outputs
        processed_output = dict(raw_output)

        # 1. Add denormalized images
        img = original_view["img"]  # Shape: (B, 3, H, W)
        data_norm_type = original_view["data_norm_type"][0]
        img_hwc = rgb(img, data_norm_type)

        # Convert numpy back to torch if needed (rgb returns numpy)
        if isinstance(img_hwc, np.ndarray):
            img_hwc = torch.from_numpy(img_hwc).to(img.device)

        processed_output["img_no_norm"] = img_hwc

        # 2. Add Z depth if we have camera frame points
        if "pts3d_cam" in processed_output:
            processed_output["depth_z"] = processed_output["pts3d_cam"][..., 2:3]

        # 3. Recover pinhole camera intrinsics from ray directions if available
        if "ray_directions" in processed_output:
            intrinsics = recover_pinhole_intrinsics_from_ray_directions(
                processed_output["ray_directions"]
            )
            processed_output["intrinsics"] = intrinsics

        # 4. Add camera pose matrices if both translation and quaternions are available
        if "cam_trans" in processed_output and "cam_quats" in processed_output:
            cam_trans = processed_output["cam_trans"]  # (B, 3)
            cam_quats = processed_output["cam_quats"]  # (B, 4)
            batch_size = cam_trans.shape[0]

            # Convert quaternions to rotation matrices
            rotation_matrices = quaternion_to_rotation_matrix(cam_quats)  # (B, 3, 3)

            # Create 4x4 pose matrices
            pose_matrices = (
                torch.eye(4, device=img.device).unsqueeze(0).repeat(batch_size, 1, 1)
            )
            pose_matrices[:, :3, :3] = rotation_matrices
            pose_matrices[:, :3, 3] = cam_trans

            processed_output["camera_poses"] = pose_matrices  # (B, 4, 4)

        # 5. Apply comprehensive mask to dense geometry outputs if requested
        if apply_mask:
            final_mask = None

            # Start with non-ambiguous mask if available
            if "non_ambiguous_mask" in processed_output:
                non_ambiguous_mask = (
                    processed_output["non_ambiguous_mask"].cpu().numpy()
                )  # (B, H, W)
                final_mask = non_ambiguous_mask

            # Apply confidence mask if requested and available
            if apply_confidence_mask and "conf" in processed_output:
                confidences = processed_output["conf"].cpu()  # (B, H, W)
                # Compute percentile threshold for each batch element
                batch_size = confidences.shape[0]
                conf_mask = torch.zeros_like(confidences, dtype=torch.bool)
                percentile_threshold = (
                    torch.quantile(
                        confidences.reshape(batch_size, -1),
                        confidence_percentile / 100.0,
                        dim=1,
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )  # Shape: (B, 1, 1)

                # Compute mask for each batch element
                conf_mask = confidences > percentile_threshold
                conf_mask = conf_mask.numpy()

                if final_mask is not None:
                    final_mask = final_mask & conf_mask
                else:
                    final_mask = conf_mask

            # Apply edge mask if requested and we have the required data
            if mask_edges and final_mask is not None and "pts3d" in processed_output:
                # Get 3D points for edge computation
                pred_pts3d = processed_output["pts3d"].cpu().numpy()  # (B, H, W, 3)
                batch_size, height, width = final_mask.shape

                edge_masks = []
                for b in range(batch_size):
                    batch_final_mask = final_mask[b]  # (H, W)
                    batch_pts3d = pred_pts3d[b]  # (H, W, 3)

                    if batch_final_mask.any():  # Only compute if we have valid points
                        # Compute normals and normal-based edge mask
                        normals, normals_mask = points_to_normals(
                            batch_pts3d, mask=batch_final_mask
                        )
                        normal_edges = normals_edge(
                            normals, tol=edge_normal_threshold, mask=normals_mask
                        )

                        # Compute depth-based edge mask
                        depth_z = (
                            processed_output["depth_z"][b].squeeze(-1).cpu().numpy()
                        )
                        depth_edges = depth_edge(
                            depth_z, rtol=edge_depth_threshold, mask=batch_final_mask
                        )

                        # Combine both edge types
                        edge_mask = ~(depth_edges & normal_edges)
                        edge_masks.append(edge_mask)
                    else:
                        # No valid points, keep all as invalid
                        edge_masks.append(np.zeros_like(batch_final_mask, dtype=bool))

                # Stack batch edge masks and combine with final mask
                edge_mask = np.stack(edge_masks, axis=0)  # (B, H, W)
                final_mask = final_mask & edge_mask

            # Apply final mask to dense geometry outputs if we have a mask
            if final_mask is not None:
                # Convert mask to torch tensor
                final_mask_torch = torch.from_numpy(final_mask).to(
                    processed_output["pts3d"].device
                )
                final_mask_torch = final_mask_torch.unsqueeze(-1)  # (B, H, W, 1)

                # Apply mask to dense geometry outputs (zero out invalid regions)
                dense_geometry_keys = [
                    "pts3d",
                    "pts3d_cam",
                    "depth_along_ray",
                    "depth_z",
                ]
                for key in dense_geometry_keys:
                    if key in processed_output:
                        processed_output[key] = processed_output[key] * final_mask_torch

                # Add mask to processed output
                processed_output["mask"] = final_mask_torch

        processed_outputs.append(processed_output)

    return processed_outputs
