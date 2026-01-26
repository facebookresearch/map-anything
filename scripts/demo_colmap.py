# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script to export MapAnything outputs in COLMAP format.

This script runs MapAnything inference on images and exports the results
to COLMAP format with scene-adaptive voxel downsampling.

Usage:
    python demo_colmap.py --images_dir=/path/to/images/ --output_dir=/path/to/output/

Output structure:
    output_dir/
        images/           # Processed images (model input resolution)
            img1.jpg
            img2.jpg
            ...
        sparse/
            cameras.bin
            images.bin
            points3D.bin
            points.ply
"""

import argparse
import glob
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.colmap_export import export_predictions_to_colmap
from mapanything.utils.image import load_images
from mapanything.utils.misc import seed_everything
from mapanything.utils.viz import predictions_to_glb

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="MapAnything COLMAP Export Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save COLMAP output (will contain images/ and sparse/ subfolders)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--voxel_fraction",
        type=float,
        default=0.01,
        help="Fraction of IQR-based scene extent for voxel size (0.01 = 1%%)",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Explicit voxel size in meters (overrides --voxel_fraction)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Also save dense reconstruction as GLB file",
    )
    parser.add_argument(
        "--skip_point2d",
        action="store_true",
        default=False,
        help="Skip Point2D backprojection for faster export (some tools may require Point2D)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("MapAnything COLMAP Export")
    print("=" * 60)
    print(f"Input images: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.voxel_size is not None:
        print(f"Voxel size: {args.voxel_size}m (explicit)")
    else:
        print(f"Voxel fraction: {args.voxel_fraction} (adaptive)")
    print(f"Random seed: {args.seed}")

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")

    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()
    print("Model loaded successfully!")

    # Get image paths
    if not os.path.isdir(args.images_dir):
        raise ValueError(f"Images directory not found: {args.images_dir}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(os.path.join(args.images_dir, ext)))
    image_path_list = sorted(image_path_list)

    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {args.images_dir}")

    print(f"Found {len(image_path_list)} images")

    # Get image names for COLMAP output
    image_names = [os.path.basename(path) for path in image_path_list]

    # Load and preprocess images
    print("Loading images...")
    views = load_images(image_path_list)
    print(f"Loaded {len(views)} views")

    # Run inference with memory-efficient defaults
    print("Running MapAnything inference...")
    with torch.no_grad():
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Export to COLMAP format (includes saving processed images)
    print("Exporting to COLMAP format...")
    _ = export_predictions_to_colmap(
        outputs=outputs,
        processed_views=views,
        image_names=image_names,
        output_dir=args.output_dir,
        voxel_fraction=args.voxel_fraction,
        voxel_size=args.voxel_size,
        data_norm_type=model.encoder.data_norm_type,
        save_ply=True,
        save_images=True,
        skip_point2d=args.skip_point2d,
    )

    print(f"COLMAP reconstruction saved to: {args.output_dir}")

    # Export GLB if requested
    if args.save_glb:
        glb_output_path = os.path.join(args.output_dir, "dense_mesh.glb")
        print(f"Saving GLB file to: {glb_output_path}")

        # Collect data for GLB export
        world_points_list = []
        images_list = []
        masks_list = []

        for pred in outputs:
            pts3d = pred["pts3d"][0].cpu().numpy()
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
            valid_depth = pts3d[..., 2] > 0
            combined_mask = mask & valid_depth

            world_points_list.append(pts3d)
            images_list.append(pred["img_no_norm"][0].cpu().numpy())
            masks_list.append(combined_mask)

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
        scene_3d.export(glb_output_path)
        print(f"Successfully saved GLB file: {glb_output_path}")

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
