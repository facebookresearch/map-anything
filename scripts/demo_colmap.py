# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Demo script to get MapAnything outputs in COLMAP format.

Reference: VGGT (https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py)
"""

import argparse
import copy
import glob
import os
from PIL.ImageOps import exif_transpose
import PIL
import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torchvision import transforms as tvf

from mapanything.models import MapAnything
from mapanything.third_party.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.utils.image import rgb, load_images, find_closest_aspect_ratio
from mapanything.utils.cropping import rescale_image_and_other_optional_info, crop_image_and_other_optional_info
from mapanything.utils.misc import seed_everything
from mapanything.utils.viz import predictions_to_glb
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args():
    parser = argparse.ArgumentParser(description="MapAnything COLMAP Demo")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save COLMAP outputs (defaults to images_dir parent)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=10,
        help="The percentile to use for the confidence threshold for depth filtering. Defaults to 10.",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    return parser.parse_args()


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def demo_fn(model, device, dtype, args, images_dir, output_dir):
    # Print configuration
    print(f"\nProcessing images from: {images_dir}")

    sparse_reconstruction_dir = os.path.join(output_dir, "sparse")

    if os.path.exists(sparse_reconstruction_dir):
        print(f"Reconstruction already exists at {sparse_reconstruction_dir}, skipping...")
        return True

    # Get image paths and preprocess them
    image_path_list = glob.glob(os.path.join(images_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {images_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    images = load_images(image_path_list)

    outputs = model.infer(
        images, memory_efficient_inference=args.memory_efficient_inference, confidence_percentile=args.conf_percentile
    )

    intrinsic = np.stack([outputs[i]["intrinsics"][0].cpu().numpy() for i in range(len(outputs))])
    extrinsic = np.stack([closed_form_pose_inverse(outputs[i]["camera_poses"])[0].cpu().numpy() for i in range(len(outputs))])
    points_3d = np.stack([outputs[i]["pts3d"][0].cpu().numpy() for i in range(len(outputs))])
    images = np.stack([images[i]["img"][0].cpu().numpy() for i in range(len(images))])

    shared_camera = (
        False  # in the feedforward manner, we do not support shared camera
    )
    camera_type = (
        "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera
    )

    num_frames, height, width, _ = points_3d.shape
    
    image_size = np.array([width, height])

    # Denormalize images before computing RGB values
    points_rgb_images = F.interpolate(
        torch.from_numpy(images).to(torch.float32),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    # Convert normalized images back to RGB [0,1] range using the rgb function
    points_rgb_list = []
    for i in range(points_rgb_images.shape[0]):
        # rgb function expects single image tensor and returns numpy array in [0,1] range
        rgb_img = rgb(points_rgb_images[i], model.encoder.data_norm_type)
        points_rgb_list.append(rgb_img)

    # Stack and convert to uint8
    points_rgb = np.stack(points_rgb_list)  # Shape: (N, H, W, 3)
    points_rgb = (points_rgb * 255).astype(np.uint8)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # Filter points based on zero depth
    valid_mask = points_3d[..., 2] > 0

    points_3d = points_3d[valid_mask]
    points_xyf = points_xyf[valid_mask]
    points_rgb = points_rgb[valid_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction = rename_colmap_recons(
        reconstruction,
        base_image_path_list,
    )

    print(f"Saving reconstruction to {output_dir}/sparse")

    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(
        os.path.join(output_dir, "sparse/points.ply")
    )
    return True


def rename_colmap_recons(
    reconstruction,
    image_paths,
):    
    """Rename COLMAP reconstruction images to original names."""
    for pyimageid in reconstruction.images:
        # Reshaped the padded & resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pyimage.name = image_paths[pyimageid - 1]

    return reconstruction


if __name__ == "__main__":
    args = parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Set device and dtype
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Init model
    print("Loading MapAnything model from huggingface ...")
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:        
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # Validate images directory
    if not os.path.isdir(args.images_dir):
        raise ValueError(f"Images directory not found: {args.images_dir}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.images_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    try:
        with torch.no_grad():
            success = demo_fn(model, device, dtype, args, args.images_dir, args.output_dir)
            if success:
                print(f"\n✅ Successfully processed images from {args.images_dir}")
            else:
                print(f"\n❌ Processing failed")
    except Exception as e:
        print(f"\n❌ Error processing images: {e}")
        import traceback
        traceback.print_exc()