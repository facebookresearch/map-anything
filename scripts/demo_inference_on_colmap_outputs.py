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
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import rerun as rr
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.colmap import get_camera_matrix, qvec2rotmat, read_model
from mapanything.utils.geometry import closed_form_pose_inverse, depthmap_to_world_frame
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb, script_add_rerun_args


def load_colmap_data(colmap_path, stride=1, verbose=False, ext=".bin"):
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
        colmap_path (str): Path to the main folder containing images/ and sparse/ subfolders
        stride (int): Load every nth image (default: 50)
        verbose (bool): Print progress messages
        ext (str): COLMAP file extension (".bin" or ".txt")

    Returns:
        list: List of view dictionaries for MapAnything inference
    """
    # Define paths
    images_folder = os.path.join(colmap_path, "images")
    sparse_folder = os.path.join(colmap_path, "sparse")

    # Check that required folders exist
    if not os.path.exists(images_folder):
        raise ValueError(f"Required folder 'images' not found at: {images_folder}")
    if not os.path.exists(sparse_folder):
        raise ValueError(f"Required folder 'sparse' not found at: {sparse_folder}")

    if verbose:
        print(f"Loading COLMAP data from: {colmap_path}")
        print(f"Images folder: {images_folder}")
        print(f"Sparse folder: {sparse_folder}")
        print(f"Using COLMAP file extension: {ext}")

    # Read COLMAP model
    try:
        cameras, images_colmap, points3D = read_model(sparse_folder, ext=ext)
    except Exception as e:
        raise ValueError(f"Failed to read COLMAP model from {sparse_folder}: {e}")

    if verbose:
        print(
            f"Loaded COLMAP model with {len(cameras)} cameras, {len(images_colmap)} images, {len(points3D)} 3D points"
        )

    # Get list of available image files
    available_images = set()
    for f in os.listdir(images_folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            available_images.add(f)

    if not available_images:
        raise ValueError(f"No image files found in {images_folder}")

    views_example = []
    processed_count = 0

    # Get a list of all colmap image names
    colmap_image_names = set(img_info.name for img_info in images_colmap.values())
    # Find the unposed images (in images/ but not in colmap)
    unposed_images = available_images - colmap_image_names

    if verbose:
        print(f"Found {len(unposed_images)} images without COLMAP poses")

    # Process images in COLMAP order
    for img_id, img_info in images_colmap.items():
        # Apply stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        img_name = img_info.name

        # Check if image file exists
        image_path = os.path.join(images_folder, img_name)
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

        image_path = os.path.join(images_folder, img_name)

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

    return views_example


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
        "--colmap_path",
        type=str,
        required=True,
        help="Path to folder containing images/ and sparse/ subfolders",
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
        "--output_directory",
        type=str,
        default="colmap_mapanything_output",
        help="Output directory for GLB file and input images",
    )
    parser.add_argument(
        "--save_input_images",
        action="store_true",
        default=False,
        help="Save input images alongside GLB output (requires --save_glb)",
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

    # Load COLMAP data
    print(f"Loading COLMAP data from: {args.colmap_path}")
    views_example = load_colmap_data(
        args.colmap_path,
        stride=args.stride,
        verbose=args.verbose,
        ext=args.ext,
    )
    print(f"Loaded {len(views_example)} views")

    # Preprocess inputs to the expected format
    print("Preprocessing COLMAP inputs...")
    processed_views = preprocess_inputs(views_example, verbose=False)

    # Run model inference
    print("Running MapAnything inference on COLMAP data...")
    outputs = model.infer(
        processed_views,
        memory_efficient_inference=args.memory_efficient_inference,
        # Control which COLMAP inputs to use/ignore
        ignore_calibration_inputs=args.ignore_calibration_inputs,  # Whether to use COLMAP calibration or not
        ignore_depth_inputs=True,  # COLMAP doesn't provide depth (can recover from sparse points but convoluted)
        ignore_pose_inputs=args.ignore_pose_inputs,  # Whether to use COLMAP poses or not
        ignore_depth_scale_inputs=True,  # No depth data
        ignore_pose_scale_inputs=True,  # COLMAP poses are non-metric
        # Use amp for better performance
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("COLMAP inference complete!")

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_COLMAP_Inference_Visualization"
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
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),
                intrinsics=intrinsics_torch.cpu().numpy(),
                pts3d=pts3d_np,
                mask=mask,
                base_name=f"mapanything/view_{view_idx}",
                pts_name=f"mapanything/pointcloud_view_{view_idx}",
                viz_mask=mask,
            )

    # Convey that the visualization is complete
    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")

    # Export GLB if requested
    if args.save_glb:
        # Create output directory structure
        scene_output_dir = args.output_directory
        os.makedirs(scene_output_dir, exist_ok=True)
        scene_prefix = os.path.basename(scene_output_dir)

        glb_output_path = os.path.join(
            scene_output_dir, f"{scene_prefix}_mapanything_colmap_output.glb"
        )
        print(f"Saving GLB file to: {glb_output_path}")

        # Save processed input images if requested
        if args.save_input_images:
            # Create processed images directory
            processed_images_dir = os.path.join(
                scene_output_dir, f"{scene_prefix}_input_images"
            )
            os.makedirs(processed_images_dir, exist_ok=True)
            print(f"Saving processed input images to: {processed_images_dir}")

            # Save each processed input image from outputs
            for view_idx, pred in enumerate(outputs):
                # Get processed image (RGB, 0-255)
                processed_image = (
                    pred["img_no_norm"][0].cpu().numpy() * 255
                )  # (H, W, 3)

                # Convert to PIL Image and save as PNG
                img_pil = Image.fromarray(processed_image.astype(np.uint8))
                img_path = os.path.join(processed_images_dir, f"view_{view_idx}.png")
                img_pil.save(img_path)

            print(
                f"Saved {len(outputs)} processed input images to: {processed_images_dir}"
            )

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
        scene_3d.export(glb_output_path)
        print(f"Successfully saved GLB file: {glb_output_path}")
        print(f"All outputs saved to: {scene_output_dir}")
    else:
        print("Skipping GLB export (--save_glb not specified)")
        if args.save_input_images:
            print("Warning: --save_input_images has no effect without --save_glb")


if __name__ == "__main__":
    main()
