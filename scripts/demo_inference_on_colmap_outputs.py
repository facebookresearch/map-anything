# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything demo using COLMAP reconstructions as input

This script demonstrates MapAnything inference on COLMAP format data.
By default MapAnything uses the calibration and poses from COLMAP as input.

The data is expected to be organized in a folder with subfolders:
- images/: containing image files (.jpg, .jpeg, .png)
- sparse/: containing COLMAP reconstruction files (.bin or .txt format)
  - cameras.bin/txt
  - images.bin/txt
  - points3D.bin/txt

Usage:
    python demo_inference_on_colmap_outputs.py --help
"""

import argparse
import glob
import os
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pycolmap
import rerun as rr
import torch
import trimesh
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.colmap import get_camera_matrix, qvec2rotmat, read_model
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.utils.image import preprocess_inputs, rgb
from mapanything.utils.viz import predictions_to_glb, script_add_rerun_args
from mapanything.third_party.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


def load_colmap_data(images_path, sparse_path, stride=1, verbose=False, ext=".bin"):
    """
    Load COLMAP format data for MapAnything inference.

    Expected folder structure:
    colmap_path/
      images/
        img1.jpg
        img2.jpg
        ...
      sparse/
        cameras.bin/txt
        images.bin/txt
        points3D.bin/txt

    Args:
        images_path (str): Path to the images folder
        sparse_path (str): Path to the sparse COLMAP folder
        stride (int): Load every nth image (default: 1)
        verbose (bool): Print progress messages
        ext (str): COLMAP file extension (".bin" or ".txt")

    Returns:
        list: List of view dictionaries for MapAnything inference
    """
    # Check that required folders exist
    if not os.path.exists(images_path):
        raise ValueError(f"Images folder not found at: {images_path}")
    if not os.path.exists(sparse_path):
        raise ValueError(f"Sparse folder not found at: {sparse_path}")

    if verbose:
        print(f"Images folder: {images_path}")
        print(f"Sparse folder: {sparse_path}")
        print(f"Using COLMAP file extension: {ext}")

    # Read COLMAP model
    try:
        cameras, images_colmap, points3D = read_model(sparse_path, ext=ext)
    except Exception as e:
        raise ValueError(f"Failed to read COLMAP model from {sparse_path}: {e}")

    if verbose:
        print(
            f"Loaded COLMAP model with {len(cameras)} cameras, {len(images_colmap)} images, {len(points3D)} 3D points"
        )

    # Get list of available image files
    available_images = set()
    for f in os.listdir(images_path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            available_images.add(f)

    if not available_images:
        raise ValueError(f"No image files found in {images_path}")

    views_example = []
    image_names_in_order = []  # Track image names in the order they're added to views
    processed_count = 0

    # Get a list of all colmap image names
    colmap_image_names = set(img_info.name for img_info in images_colmap.values())
    # Find the unposed images (in images/ but not in colmap)
    unposed_images = available_images - colmap_image_names
    unposed_images = sorted(list(unposed_images))

    if verbose:
        print(f"Found {len(unposed_images)} images without COLMAP poses")

    # Process images in COLMAP order
    for img_id, img_info in sorted(images_colmap.items()):
        # Apply stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        img_name = img_info.name

        # Check if image file exists
        image_path = os.path.join(images_path, img_name)
        if not os.path.exists(image_path):
            if verbose:
                print(f"Warning: Image file not found for {img_name}, skipping")
            processed_count += 1
            continue

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)  # (H, W, 3) - [0, 255]

            # Get camera info
            cam_info = cameras[img_info.camera_id]
            cam_params = cam_info.params

            # Get intrinsic matrix
            K, _ = get_camera_matrix(
                camera_params=cam_params, camera_model=cam_info.model
            )

            # Get pose (COLMAP provides world2cam, we need cam2world)
            # COLMAP: world2cam rotation and translation
            C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

            # Create 4x4 world2cam pose matrix
            world2cam_matrix = np.eye(4)
            world2cam_matrix[:3, :3] = C_R_G
            world2cam_matrix[:3, 3] = C_t_G

            # Convert to cam2world using closed form pose inverse
            pose_matrix = closed_form_pose_inverse(world2cam_matrix[None, :, :])[0]

            # Convert to tensors
            image_tensor = torch.from_numpy(image_array)  # (H, W, 3)
            intrinsics_tensor = torch.from_numpy(K.astype(np.float32))  # (3, 3)
            pose_tensor = torch.from_numpy(pose_matrix.astype(np.float32))  # (4, 4)

            # Create view dictionary for MapAnything inference
            view = {
                "img": image_tensor,  # (H, W, 3) - [0, 255]
                "intrinsics": intrinsics_tensor,  # (3, 3)
                "camera_poses": pose_tensor,  # (4, 4) in OpenCV cam2world convention
                "is_metric_scale": torch.tensor([False]),  # COLMAP data is non-metric
            }

            views_example.append(view)
            image_names_in_order.append(img_name)
            processed_count += 1

            if verbose:
                print(
                    f"Loaded view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})"
                )
                print(f"  - Camera ID: {img_info.camera_id}")
                print(f"  - Camera Model: {cam_info.model}")
                print(f"  - Image ID: {img_id}")

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load data for {img_name}: {e}")
            processed_count += 1
            continue

    # process unposed images (without COLMAP poses)
    for img_name in unposed_images:
        # Apply stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        image_path = os.path.join(images_path, img_name)

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)  # (H, W, 3) - [0, 255]

            # Convert to tensor
            image_tensor = torch.from_numpy(image_array)  # (H, W, 3)

            view = {
                "img": image_tensor,  # (H, W, 3) - [0, 255]
                # No intrinsics or pose available
            }

            views_example.append(view)
            image_names_in_order.append(img_name)
            processed_count += 1

            if verbose:
                print(
                    f"Loaded unposed view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})"
                )

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load data for {img_name}: {e}")
            processed_count += 1
            continue

    if not views_example:
        raise ValueError("No valid images found")

    if verbose:
        print(f"Successfully loaded {len(views_example)} views with stride={stride}")

    return views_example, image_names_in_order


def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
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
            image_plane_distance=0.1,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
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
        description="MapAnything demo using COLMAP reconstructions as input"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to directory containing input images",
    )
    parser.add_argument(
        "--sparse_dir",
        type=str,
        required=True,
        help="Path to COLMAP sparse reconstruction directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where outputs will be saved",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Load every nth image (default: 1)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        choices=[".bin", ".txt"],
        help="COLMAP file extension (default: .bin)",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
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
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose printouts for loading",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--save_input_images",
        action="store_true",
        default=False,
        help="Save input images alongside GLB output (requires --save_glb)",
    )
    parser.add_argument(
        "--save_colmap",
        action="store_true",
        default=True,
        help="Save reconstruction in COLMAP format",
    )
    parser.add_argument(
        "--ignore_calibration_inputs",
        action="store_true",
        default=False,
        help="Ignore COLMAP calibration inputs (use only images and poses)",
    )
    parser.add_argument(
        "--ignore_pose_inputs",
        action="store_true",
        default=False,
        help="Ignore COLMAP pose inputs (use only images and calibration)",
    )

    return parser

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


def process_scene(model, images_path, sparse_path, output_dir, args, scene_name=None):
    """Process a single scene and save outputs."""
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")

    try:
        views_example, image_names_in_order = load_colmap_data(
            images_path,
            sparse_path,
            stride=args.stride,
            verbose=args.verbose,
            ext=args.ext,
        )
        print(f"Loaded {len(views_example)} views")
    except Exception as e:
        print(f"ERROR - Failed to load COLMAP data: {e}")
        return False

    # Preprocess inputs
    print("Preprocessing COLMAP inputs...")
    processed_views = preprocess_inputs(views_example, verbose=False)

    # Run model inference
    print("Running MapAnything inference...")
    outputs = model.infer(
        processed_views,
        memory_efficient_inference=args.memory_efficient_inference,
        ignore_calibration_inputs=args.ignore_calibration_inputs,
        ignore_depth_inputs=True,
        ignore_pose_inputs=args.ignore_pose_inputs,
        ignore_depth_scale_inputs=True,
        ignore_pose_scale_inputs=True,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("Inference complete!")

    if args.save_colmap:
        intrinsic = np.stack([outputs[i]["intrinsics"][0].cpu().numpy() for i in range(len(outputs))])
        extrinsic = np.stack([closed_form_pose_inverse(outputs[i]["camera_poses"])[0].cpu().numpy() for i in range(len(outputs))])
        points_3d = np.stack([outputs[i]["pts3d"][0].cpu().numpy() for i in range(len(outputs))])
        depth_conf = np.stack([outputs[i]["conf"][0].cpu().numpy() for i in range(len(outputs))])
        images = np.stack([processed_views[i]["img"][0].cpu().numpy() for i in range(len(processed_views))])

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

        # Filter points based on depth validity and confidence threshold
        # Use confidence threshold to filter out low-quality predictions
        # conf_threshold = 1.5  # Adjust this value based on quality needs (typical range: 1.0-3.0)
        # valid_mask = (points_3d[..., 2] > 0) & (depth_conf >= conf_threshold)
        valid_mask = (points_3d[..., 2] > 0)

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
            image_names_in_order,
        )

        print(f"Saving reconstruction to {output_dir}/sparse")
        sparse_reconstruction_dir = os.path.join(output_dir, "sparse")
        os.makedirs(sparse_reconstruction_dir, exist_ok=True)
        reconstruction.write(sparse_reconstruction_dir)

        # Save point cloud for fast visualization
        trimesh.PointCloud(points_3d, colors=points_rgb).export(
            os.path.join(output_dir, "sparse/points.ply")
        )
    return True


def main():
    # Parser for arguments
    parser = get_parser()
    args = parser.parse_args()
    script_add_rerun_args(parser)

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
    model.eval()
    print("✅ Successfully loaded model")

    # Validate input directories
    if not os.path.isdir(args.images_dir):
        raise ValueError(f"Images directory not found: {args.images_dir}")
    if not os.path.isdir(args.sparse_dir):
        raise ValueError(f"Sparse directory not found: {args.sparse_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the scene
    try:
        with torch.no_grad():
            success = process_scene(
                model, 
                args.images_dir, 
                args.sparse_dir, 
                args.output_dir, 
                args, 
                scene_name=os.path.basename(args.images_dir)
            )
            if success:
                print(f"\n✅ Successfully processed scene")
            else:
                print(f"\n❌ Processing failed")
    except Exception as e:
        print(f"\n❌ Error processing scene: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
