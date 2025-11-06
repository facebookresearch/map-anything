# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from argconf import argconf_parse
from natsort import natsorted
from tqdm import tqdm
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

from morphcloud.utils.wai.core import store_data
from morphcloud.utils.wai.scene_frame import _filter_scenes

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logger = logging.getLogger(__name__)


def load_spring_extrinsics(file_path):
    "Load Spring camera extrinsics"
    data = np.loadtxt(file_path)
    extrinsics_matrices = data.reshape(-1, 4, 4)
    return extrinsics_matrices


def load_spring_intrinsics(file_path):
    "Load Spring camera intrinsics"
    data = np.loadtxt(file_path)
    intrinsic_matrices = []
    for fx, fy, cx, cy in data:
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsic_matrices.append(intrinsic_matrix)
    return np.array(intrinsic_matrices)


def readDsp5Disp(filename):
    "Read dsp5 disparity file"
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(
                f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?"
            )
        return f["disparity"][()]


def load_spring_depth(path, intrinsics, baseline=0.065):
    """
    Load Spring metric depth from disparity using the baseline between cameras.

    Args:
        path: Path to the disparity file
        intrinsics: Camera intrinsics matrix
        baseline: Distance between left and right cameras in meters (default: 0.065m)

    Returns:
        Metric depth map
    """
    disparity = readDsp5Disp(path)
    disparity = disparity[::2, ::2]
    disparity_validity_mask = disparity > 0
    depth = intrinsics[0, 0] * baseline / disparity
    depth = np.where(disparity_validity_mask, depth, 0)
    depth = depth.copy()
    return depth


def process_spring_scene(cfg, scene_name, split_map):
    """
    Process a Spring scene into the WAI format.
    Convert the disparity to metric depth and save it in the default WAI format (.exr).
    The left & right RGB images along with the metric depth, intrinsics, extrinsics and skymaps (binary masks)
    are processed to WAI format.

    Expected root directory structure for the raw Spring dataset:
    .
    └── spring/
        ├── train/
        │   ├── 0001/
        │   │   ├── cam_data/
        │   │   ├── disp1_left/
        │   │   ├── disp1_right/
        │   │   ├── disp2_BW_left/
        │   │   ├── disp2_BW_right/
        │   │   ├── disp2_FW_left/
        │   │   ├── disp2_FW_right/
        │   │   ├── flow_BW_left/
        │   │   ├── flow_BW_right/
        │   │   ├── flow_FW_left/
        │   │   ├── flow_FW_right/
        │   │   ├── frame_left/
        │   │   ├── frame_right/
        │   │   ├── maps/
        │   ├── 0002/
        │   ├── ...
        ├── test/
        │   ├── 0003/
        │   │   ├── cam_data/
        │   │   ├── frame_left/
        │   │   ├── frame_right/
        │   ├── 0019/
        │   ├── ...
    """
    # Setup paths
    split = split_map[scene_name]
    spring_root = Path(cfg.original_root)
    scene_dir = spring_root / split / scene_name
    target_scene_root = Path(cfg.root) / scene_name

    # Create output directories
    images_dir = target_scene_root / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    if split == "train":
        depth_dir = target_scene_root / "depth"
        depth_dir.mkdir(parents=True, exist_ok=False)
        skymask_dir = target_scene_root / "skymasks"
        skymask_dir.mkdir(parents=True, exist_ok=False)

    # Load camera parameters
    intrinsics = load_spring_intrinsics(scene_dir / "cam_data" / "intrinsics.txt")

    # Extrinsics are only available for the train set
    has_extrinsics = False
    if split == "train":
        extrinsics_path = scene_dir / "cam_data" / "extrinsics.txt"
        left_frame_extrinsics = load_spring_extrinsics(extrinsics_path)
        has_extrinsics = True

    # Get all image files
    left_image_dir = scene_dir / "frame_left"
    right_image_dir = scene_dir / "frame_right"
    left_image_files = natsorted(os.listdir(left_image_dir))

    # Initialize WAI frames list
    wai_frames = []

    # Process each frame
    for idx, left_image_name in enumerate(tqdm(left_image_files)):
        # Get right file name and frame number
        right_image_name = left_image_name.replace("frame_left", "frame_right")
        frame_num = left_image_name.split(".")[0].replace("frame_left_", "")

        # Symlink original images
        left_image_path = left_image_dir / left_image_name
        right_image_path = right_image_dir / right_image_name
        rel_left_image_path = Path("images") / left_image_name
        rel_right_image_path = Path("images") / right_image_name
        os.symlink(left_image_path, target_scene_root / rel_left_image_path)
        os.symlink(right_image_path, target_scene_root / rel_right_image_path)

        # Get image dimensions
        img = cv2.imread(str(left_image_path))
        h, w = img.shape[:2]

        # Process data available only for training split
        if split == "train":
            # Get disparity file paths
            left_disp_path = scene_dir / "disp1_left" / f"disp1_left_{frame_num}.dsp5"
            right_disp_path = (
                scene_dir / "disp1_right" / f"disp1_right_{frame_num}.dsp5"
            )

            # Convert disparity to depth and save
            # Left depth
            left_depth = load_spring_depth(left_disp_path, intrinsics[idx])
            rel_left_depth_path = Path("depth") / (
                left_image_name.replace(".png", ".exr")
            )
            store_data(
                target_scene_root / rel_left_depth_path,
                torch.tensor(left_depth),
                "depth",
            )
            # Right depth
            right_depth = load_spring_depth(right_disp_path, intrinsics[idx])
            rel_right_depth_path = Path("depth") / (
                right_image_name.replace(".png", ".exr")
            )
            store_data(
                target_scene_root / rel_right_depth_path,
                torch.tensor(right_depth),
                "depth",
            )

            # Get skymask file names
            left_skymask_path = (
                scene_dir / "maps" / "skymap_left" / f"skymap_left_{frame_num}.png"
            )
            right_skymask_path = (
                scene_dir / "maps" / "skymap_right" / f"skymap_right_{frame_num}.png"
            )

            # Load skymask and save
            # Left skymask
            left_skymask = cv2.imread(str(left_skymask_path), cv2.IMREAD_UNCHANGED)
            left_skymask = cv2.resize(
                left_skymask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            rel_left_skymask_path = Path("skymasks") / left_image_name
            store_data(
                target_scene_root / rel_left_skymask_path,
                torch.tensor(left_skymask),
                "binary",
            )
            # Right skymask
            right_skymask = cv2.imread(str(right_skymask_path), cv2.IMREAD_UNCHANGED)
            right_skymask = cv2.resize(
                right_skymask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            rel_right_skymask_path = Path("skymasks") / right_image_name
            store_data(
                target_scene_root / rel_right_skymask_path,
                torch.tensor(right_skymask),
                "binary",
            )
        else:
            rel_left_depth_path = None
            rel_right_depth_path = None
            rel_left_skymask_path = None
            rel_right_skymask_path = None

        # Create WAI frame entries for left and right images
        left_frame = {
            "frame_name": left_image_name.split(".")[0],
            "file_path": str(rel_left_image_path),
            "image": str(rel_left_image_path),
            "h": h,
            "w": w,
            "fl_x": float(intrinsics[idx][0, 0]),
            "fl_y": float(intrinsics[idx][1, 1]),
            "cx": float(intrinsics[idx][0, 2]),
            "cy": float(intrinsics[idx][1, 2]),
        }

        right_frame = {
            "frame_name": right_image_name.split(".")[0],
            "file_path": str(rel_right_image_path),
            "image": str(rel_right_image_path),
            "h": h,
            "w": w,
            "fl_x": float(intrinsics[idx][0, 0]),
            "fl_y": float(intrinsics[idx][1, 1]),
            "cx": float(intrinsics[idx][0, 2]),
            "cy": float(intrinsics[idx][1, 2]),
        }

        # Add transform matrices if extrinsics are available
        if has_extrinsics:
            # Get the camera to world poses
            left_w2c_pose = left_frame_extrinsics[idx]
            right_w2c_pose = left_frame_extrinsics[idx].copy()
            baseline = 0.065
            right_w2c_pose[0, 3] -= baseline
            left_c2w_pose = np.linalg.inv(left_w2c_pose)
            right_c2w_pose = np.linalg.inv(right_w2c_pose)

            # Add transform matrices to frames
            left_frame["transform_matrix"] = left_c2w_pose.tolist()
            right_frame["transform_matrix"] = right_c2w_pose.tolist()

        # Add depth paths if available
        if rel_left_depth_path:
            left_frame["depth"] = str(rel_left_depth_path)
        if rel_right_depth_path:
            right_frame["depth"] = str(rel_right_depth_path)

        # Add skymask paths if available
        if rel_left_skymask_path:
            left_frame["skymask"] = str(rel_left_skymask_path)
        if rel_right_skymask_path:
            right_frame["skymask"] = str(rel_right_skymask_path)

        # Add frames to WAI frames list
        wai_frames.append(left_frame)
        wai_frames.append(right_frame)

    # Construct scene metadata
    frame_modalities = {
        "image": {"frame_key": "image", "format": "image"},
    }

    # Add depth and skymask modalities if available
    if any("depth" in frame for frame in wai_frames):
        frame_modalities["depth"] = {"frame_key": "depth", "format": "depth"}

    if any("skymask" in frame for frame in wai_frames):
        frame_modalities["skymask"] = {"frame_key": "skymask", "format": "binary"}

    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "scene_modalities": {},
        "frames": wai_frames,
        "frame_modalities": frame_modalities,
    }

    # Save scene metadata
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


def get_original_scene_names(cfg):
    # Get the root directory of the Spring dataset
    spring_root = Path(cfg.original_root)

    # Create a list of all scene names and a mapping to their splits
    all_scene_names = []
    # Process both train and test splits
    for split in ["train", "test"]:
        split_dir = spring_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist, skipping")
            continue

        # Each numbered directory in the split is a scene
        for scene_dir in split_dir.glob("*"):
            if scene_dir.is_dir() and scene_dir.name.isdigit():
                # Use just the scene number as the scene name (e.g., "0001")
                scene_name = scene_dir.name
                all_scene_names.append(scene_name)
    # scene filter for batch processing
    all_scene_names = _filter_scenes(
        cfg.original_root, all_scene_names, cfg.get("scene_filters")
    )
    return all_scene_names


def get_split_map(cfg):
    # Get the root directory of the Spring dataset
    spring_root = Path(cfg.original_root)

    scene_to_split_map = {}

    # Process both train and test splits
    for split in ["train", "test"]:
        split_dir = spring_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist, skipping")
            continue

        # Each numbered directory in the split is a scene
        for scene_dir in split_dir.glob("*"):
            if scene_dir.is_dir() and scene_dir.name.isdigit():
                # Use just the scene number as the scene name (e.g., "0001")
                scene_name = scene_dir.name
                scene_to_split_map[scene_name] = split
    return scene_to_split_map


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/spring.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)

    split_map = get_split_map(cfg)
    convert_scenes_wrapper(
        process_spring_scene,
        cfg,
        get_original_scene_names_func=get_original_scene_names,
        split_map=split_map,
    )
