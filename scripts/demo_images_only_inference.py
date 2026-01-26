# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Images-Only Inference with Visualization

Usage:
    python demo_images_only_inference.py --help
"""

import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import rerun as rr
import torch

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import load_images
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)


def log_data_to_rerun(
    image,
    depthmap,
    pose,
    intrinsics,
    pts3d,
    mask,
    base_name,
    pts_name,
    viz_mask=None,
    log_only_imgs_for_rerun_cams=False,
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    if not log_only_imgs_for_rerun_cams:
        rr.log(
            f"{base_name}/pinhole/depth",
            rr.DepthImage(depthmap),
        )
        if viz_mask is not None:
            rr.log(
                f"{base_name}/pinhole/mask",
                rr.SegmentationImage(viz_mask.astype(int)),
            )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )


def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Visualize metric 3D reconstruction from images"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images for reconstruction",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable visualization with Rerun",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.glb",
        help="Output path for GLB file (default: output.glb)",
    )
    parser.add_argument(
        "--video_viz_for_rerun",
        action="store_true",
        default=False,
        help="Video visualization for Rerun - logs views with time index",
    )
    parser.add_argument(
        "--log_only_imgs_for_rerun_cams",
        action="store_true",
        default=False,
        help="Log only images for Rerun camera - no depth, mask, etc.",
    )

    return parser


def main():
    # Parser for arguments and Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model from HuggingFace
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)

    # Load images
    print(f"Loading images from: {args.image_folder}")
    views = load_images(args.image_folder)
    print(f"Loaded {len(views)} views")

    # Run model inference with memory-efficient defaults
    print("Running inference...")
    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("Inference complete!")

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        if args.video_viz_for_rerun:
            viz_string = "MapAnything_Video_Visualization"
        else:
            viz_string = "MapAnything_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
        # Extract data from predictions
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

        # Compute new pts3d using depth, intrinsics, and camera pose
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy arrays
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Store data for GLB export if needed
        if args.save_glb:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            if args.video_viz_for_rerun:
                rr.set_time("stable_time", sequence=view_idx)
                view_base_name = "mapanything/view"
            else:
                view_base_name = f"mapanything/view_{view_idx}"
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),
                intrinsics=intrinsics_torch.cpu().numpy(),
                pts3d=pts3d_np,
                mask=mask,
                base_name=view_base_name,
                pts_name=f"mapanything/pointcloud_view_{view_idx}",
                viz_mask=mask,
                log_only_imgs_for_rerun_cams=args.log_only_imgs_for_rerun_cams,
            )

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")

    # Export GLB if requested
    if args.save_glb:
        print(f"Saving GLB file to: {args.output_path}")

        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Create predictions dict for GLB export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=True)

        # Save GLB file
        scene_3d.export(args.output_path)
        print(f"Successfully saved GLB file: {args.output_path}")
    else:
        print("Skipping GLB export (--save_glb not specified)")


if __name__ == "__main__":
    main()
