# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Offline Inference with Local Weights

Usage:
    python demo_local_weight.py --help
"""

import argparse
import json
import os
from time import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local
from mapanything.utils.image import load_images
from mapanything.utils.viz import predictions_to_glb

LOCAL_CONFIG = {
    "path": "configs/train.yaml",
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "checkpoint_path": "ckpt/model.safetensors",
    "config_json_path": "config.json",
    "trained_with_amp": True,
    "trained_with_amp_dtype": "bf16",
    "data_norm_type": "dinov2",
    "patch_size": 14,
    "resolution": 518,
    "strict": False,
}


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser for offline inference script."""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Run inference from local weights only"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images for reconstruction",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=True,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mapanything.glb",
        help="Output path for GLB file",
    )
    parser.add_argument(
        "--local_config",
        type=json.loads,
        default=LOCAL_CONFIG,
        help='Local config for loading a MapAnything model. To set this argument pass a string in this format: \'{"key": "value", ...}\')',
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(
        f"Initializing MapAnything model from local weights with config: {args.local_config}"
    )
    model = initialize_mapanything_local(args.local_config, device)
    print("Successfully loaded pretrained weights")

    print(f"Loading images from: {args.image_folder}")
    views = load_images(args.image_folder)
    if len(views) == 0:
        raise ValueError(f"No images found in {args.image_folder}")
    print(f"Loaded {len(views)} views")

    print("Running inference...")
    start_time = time()
    outputs = model.infer(
        views,
        memory_efficient_inference=args.memory_efficient_inference,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=50,
    )
    duration = time() - start_time
    print(f"Inference finished in {duration:.3f} s")

    world_points_list = []
    images_list = []
    masks_list = []

    for pred in outputs:
        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]

        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask &= valid_mask.cpu().numpy()
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        if args.save_glb:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

    if args.save_glb:
        print(f"Saving GLB file to: {args.output_path}")
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        scene_3d = predictions_to_glb(predictions, as_mesh=False)
        scene_3d.export(args.output_path)
        print(f"Successfully saved GLB file: {args.output_path}")
    else:
        print("Skipping GLB export (--save_glb not specified)")


if __name__ == "__main__":
    main()
