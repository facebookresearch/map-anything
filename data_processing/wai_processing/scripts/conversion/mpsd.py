# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import logging
import os
from pathlib import Path
from shutil import rmtree

import cv2
import numpy as np
import torch
from argconf import argconf_parse
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
)

from morphcloud.utils.wai.core import store_data
from morphcloud.utils.wai.scene_frame import _filter_scenes

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logger = logging.getLogger(__name__)


def convert_scene(cfg, scene_name, scene_to_recon_split_map):
    """
    Process a MPSD scene into the WAI format.
    Resize the original RGB image to the size of the depth map.
    Convert the original depth (in centimeters) to metric depth.
    Convert the normalized focal values to intrinsics and shot information to opencv cam2world extrinsics.
    All the processed information (images, metric depth, intrinsics, extrinsics) is saved in the WAI format.

    Args:
        cfg: Configuration dictionary.
        scene_name: Name of the scene in WAI format (e.g., "geoeven_4_2019-03-17T16_16_24.807705").
        reconstruction_split: The reconstruction split name (e.g., "geoeven_4").

    Expected root directory structure for the raw MPSD dataset (containing ~76K scenes):
    .
    └── mpsd/
        ├── reconstruction_data/
        │   ├── geoeven_4/
        │   │   ├── 2019-03-17T16_16_24.807705/
        │   │   |   ├── image_list.txt
        │   │   |   ├── reconstruction.json
        │   │   |   ├── tracks.csv
        │   │   ├── 2019-03-17T20_16_47.489407/
        │   │   ├── ...
        │   ├── geoeven_4_extension/
        │   │   ├── 2019-11-14T11_36_12.984067/
        │   │   ├── 2019-11-14T22_35_36.125519/
        │   │   ├── ...
        │   ├── geoeven_4_extension_2/
        │   │   ├── 2019-11-22T19_22_12.424412/
        │   │   ├── 2019-11-22T19_22_24.724955/
        │   │   ├── ...
        │   ├── all_camera_params.json
        │   ├── image_key_mapper.json
        │   ├── shot_to_camera.json
        ├── train/
        │   ├── __epk-nlU2UVQ7bBPQKbiA.jpg
        │   ├── __epk-nlU2UVQ7bBPQKbiA.png
        │   ├── ...
        ├── val/
        │   ├── __eA1Z6W5hlzKYvgheyoXg.jpg
        │   ├── __eA1Z6W5hlzKYvgheyoXg.png
        │   ├── ...
        ├── LICENSE.txt
        ├── readme_mpsd.md
        ├── train_recon_folder_names.npy
        ├── train.json
        ├── val_recon_folder_names.npy
        ├── val.json
    """

    # Extract reconstruction_split from scene_name
    reconstruction_split = scene_to_recon_split_map[scene_name]

    # Extract folder_name from scene_name (remove reconstruction_split_ prefix)
    folder_name = scene_name.replace(f"{reconstruction_split}_", "")

    # Setup paths
    mpsd_root = Path(cfg.original_root)
    reconstruction_folder = (
        mpsd_root / "reconstruction_data" / reconstruction_split / folder_name
    )
    target_scene_root = Path(cfg.root) / scene_name

    # Create output directories
    images_dir = target_scene_root / "images"
    depth_dir = target_scene_root / "depth"

    images_dir.mkdir(parents=True, exist_ok=False)
    depth_dir.mkdir(parents=True, exist_ok=False)

    # Load the MPSD train and val metadata files (contains focal values)
    train_metadata_path = mpsd_root / "train.json"
    val_metadata_path = mpsd_root / "val.json"

    with open(train_metadata_path, "r") as f:
        train_metadata = json.load(f)
    with open(val_metadata_path, "r") as f:
        val_metadata = json.load(f)

    # Combine the metadata while keeping track of the split information
    combined_metadata = {}
    for image_name, metadata in train_metadata.items():
        metadata_copy = metadata.copy()
        metadata_copy["split"] = "train"
        combined_metadata[image_name] = metadata_copy
    for image_name, metadata in val_metadata.items():
        metadata_copy = metadata.copy()
        metadata_copy["split"] = "val"
        combined_metadata[image_name] = metadata_copy

    # Load reconstruction data
    image_list_path = reconstruction_folder / "image_list.txt"
    reconstruction_json_path = reconstruction_folder / "reconstruction.json"
    with open(image_list_path, "r") as f:
        image_list = f.read().splitlines()
    image_list = [
        image_name.split("/")[-1] for image_name in image_list
    ]  # Remove path prefix
    with open(reconstruction_json_path, "r") as f:
        reconstruction_data = json.load(f)

    # Get camera pose metadata (shots)
    pose_metadata = reconstruction_data[0]["shots"]

    # Initialize WAI frames list
    wai_frames = []

    # Process each frame
    num_valid_frames = 0
    for image_name in tqdm(natsorted(image_list)):
        # Skip if image is not in metadata or pose data
        if (image_name not in combined_metadata) or (image_name not in pose_metadata):
            continue

        num_valid_frames += 1

        # Get the split, focal value & pose of the image
        img_metadata = combined_metadata[image_name]
        split = img_metadata["split"]
        focal = img_metadata["focal"]

        # Get camera pose (extrinsics)
        axis_angle = pose_metadata[image_name]["rotation"]
        rotation_matrix, _ = cv2.Rodrigues(np.array(axis_angle))
        translation = pose_metadata[image_name]["translation"]

        # Create world-to-camera and camera-to-world matrices
        w2c_pose = np.eye(4)
        w2c_pose[:3, :3] = rotation_matrix
        w2c_pose[:3, 3] = translation
        c2w_pose = np.linalg.inv(w2c_pose)

        # Get the raw image and depth paths
        raw_image_path = mpsd_root / split / f"{image_name}.jpg"
        raw_depth_path = mpsd_root / split / f"{image_name}.png"

        # Load the image and depth
        image = Image.open(raw_image_path)
        depth = np.array(Image.open(raw_depth_path))
        depth = depth / 100.0  # Convert from centimeters to meters

        # Get image dimensions
        depth_height, depth_width = depth.shape

        # Resize the image to match the depth map dimensions
        image = image.resize((depth_width, depth_height))

        # Save the image
        image_filename = f"{image_name}.jpg"
        image_path = images_dir / image_filename
        store_data(image_path, image, "image")
        rel_image_path = Path("images") / image_filename

        # Save the depth
        depth_filename = f"{image_name}.exr"
        depth_path = depth_dir / depth_filename
        store_data(depth_path, torch.tensor(depth), "depth")
        rel_depth_path = Path("depth") / depth_filename

        # Compute intrinsics
        fx = focal * max(depth_width, depth_height)
        fy = focal * max(depth_width, depth_height)
        cx = depth_width / 2
        cy = depth_height / 2

        # Create WAI frame entry
        frame = {
            "frame_name": image_name,
            "file_path": str(rel_image_path),
            "image": str(rel_image_path),
            "depth": str(rel_depth_path),
            "h": depth_height,
            "w": depth_width,
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "transform_matrix": c2w_pose.tolist(),
        }

        wai_frames.append(frame)

    # Create base scene metadata with common fields
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "scene_modalities": {},
    }

    # Handle scenes with too few valid frames
    if num_valid_frames < 2:
        logger.warning(
            f"Scene {scene_name} has fewer than 2 valid frames, creating scene meta with empty frames list"
        )

        # Delete created subfolders as they won't be needed
        if images_dir.exists():
            rmtree(images_dir)
        if depth_dir.exists():
            rmtree(depth_dir)

        # Add skipped-specific fields
        scene_meta.update(
            {
                "frames": [],  # Empty frames list
                "frame_modalities": {},  # Empty frame modalities
                "skipped_reason": f"Scene has only {num_valid_frames} valid frames (minimum required: 2)",
            }
        )
    else:
        # Add valid scene-specific fields
        scene_meta.update(
            {
                "frames": wai_frames,
                "frame_modalities": {
                    "image": {"frame_key": "image", "format": "image"},
                    "depth": {"frame_key": "depth", "format": "depth"},
                },
            }
        )

    # Save scene metadata
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


def get_original_scene_names(cfg):
    # Get the root directory of the MPSD dataset
    mpsd_root = Path(cfg.original_root)
    reconstruction_data_dir = mpsd_root / "reconstruction_data"

    # Create a list of all scene names and a mapping to their reconstruction splits
    all_scene_names = []

    # Process all reconstruction splits (geoeven_4, geoeven_4_extension, etc.)
    for recon_split_dir in reconstruction_data_dir.glob("*"):
        if not recon_split_dir.is_dir():
            continue

        reconstruction_split = recon_split_dir.name
        logger.info(
            f"Getting all scenes under reconstruction split: {reconstruction_split}"
        )

        # Each timestamp directory in the reconstruction split is a scene
        for folder_name in os.listdir(recon_split_dir):
            # Format scene name as "{reconstruction_split}_{folder_name}"
            scene_name = f"{reconstruction_split}_{folder_name}"
            all_scene_names.append(scene_name)

    # scene filter for batch processing
    all_scene_names = _filter_scenes(
        cfg.original_root, all_scene_names, cfg.get("scene_filters")
    )
    return all_scene_names


def get_recon_split_map(cfg):
    # Get the root directory of the MPSD dataset
    mpsd_root = Path(cfg.original_root)
    reconstruction_data_dir = mpsd_root / "reconstruction_data"

    # Create a dict for the reconstruction splits
    scene_to_recon_split_map = {}

    # Process all reconstruction splits (geoeven_4, geoeven_4_extension, etc.)
    for recon_split_dir in reconstruction_data_dir.glob("*"):
        if not recon_split_dir.is_dir():
            continue

        reconstruction_split = recon_split_dir.name
        logger.info(
            f"Getting all scenes under reconstruction split: {reconstruction_split}"
        )

        # Each timestamp directory in the reconstruction split is a scene
        for folder_name in os.listdir(recon_split_dir):
            # Format scene name as "{reconstruction_split}_{folder_name}"
            scene_name = f"{reconstruction_split}_{folder_name}"
            scene_to_recon_split_map[scene_name] = reconstruction_split

    return scene_to_recon_split_map


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/mpsd.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)

    scene_to_recon_split_map = get_recon_split_map(cfg)

    convert_scenes_wrapper(
        converter_func=convert_scene,
        cfg=cfg,
        get_original_scene_names_func=get_original_scene_names,
        scene_to_recon_split_map=scene_to_recon_split_map,
    )
