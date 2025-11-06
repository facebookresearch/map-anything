# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from argconf import argconf_parse
from scipy.spatial.transform import Rotation
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,  # noqa: F401, Needed for launch_slurm.py
)

from morphcloud.utils.wai.core import store_data

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def process_paralleldomain4d_scene(cfg, scene_name):
    """
    Process a Parallel Domain 4D scene into the WAI format.
    The original RGBA images are symlinked.
    Depths, camera params and poses are processed to WAI format.

    Expected root directory structure for the raw Parallel Domain 4D dataset:
    .
    └── paralleldomain4d/
        ├── scene_000000/
        │   ├── depth/
        │   │   ├── camera0/
        │   │   |   ├── 000000000000000005.npz
        │   │   |   ├── 000000000000000015.npz
        │   │   |   ├── ...
        │   │   ├── ...
        │   │   ├── camera15/
        │   │   ├── yaw-0/
        │   │   ├── yaw-60/
        │   │   ├── yaw-neg-60/
        │   ├── rgb/
        │   │   ├── camera0/
        │   │   |   ├── 000000000000000005.npz
        │   │   |   ├── 000000000000000015.npz
        │   │   |   ├── ...
        │   │   ├── ...
        │   │   ├── camera15/
        │   │   ├── yaw-0/
        │   │   ├── yaw-60/
        │   │   ├── yaw-neg-60/
        │   ├── other modalities ...
        │   ├── localization_1675829175834000_1675829180734000.json
        │   ├── scene_6e420602a6de7a4fb1ed4671775efa83424c85c5.json
        ├── ...
        ├── scene_000020
        ├── ...
    """
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)
    wai_frames = []

    # Load the scene metadata
    scene_meta = glob.glob(str(scene_root / "scene_*.json"))
    scene_meta = scene_meta[0]
    with open(scene_meta, "r") as f:
        scene_meta = json.load(f)

    # Load the scene calibration metadata
    scene_calib_file = os.listdir(scene_root / "calibration")[0]
    with open(scene_root / "calibration" / scene_calib_file, "r") as f:
        scene_calib = json.load(f)

    # Create mapping from camera name to intrinsics
    modality_intrinsics_list = scene_calib["intrinsics"]
    modality_list = scene_calib["names"]
    modality_to_intrinsics = {}
    for modality, modality_intrinsics in zip(modality_list, modality_intrinsics_list):
        modality_to_intrinsics[modality] = modality_intrinsics

    # Loop over the data and only process the camera based data
    for data_entry in scene_meta["data"]:
        if "image" in data_entry["datum"]:
            # Get the rgb image path
            rgb_path = data_entry["datum"]["image"]["filename"]

            # Get the depth image path
            depth_path = data_entry["datum"]["image"]["annotations"]["6"]

            # Ensure that the rgb and depth paths exist before processing
            if not (
                (scene_root / rgb_path).exists() and (scene_root / depth_path).exists()
            ):
                continue

            # Get the camera name and file name from the rgb path
            _, camera_name, file_name = rgb_path.split("/")
            file_name = os.path.splitext(file_name)[0]

            # Symlink the rgb image to the target scene root
            rgb_target_path = image_dir / f"{file_name}_{camera_name}.png"
            os.symlink(scene_root / rgb_path, rgb_target_path)

            # Load the depth image
            depth = np.load(scene_root / depth_path)
            depth = depth["data"]

            # Mask out invalid depth
            depth_validity_mask = depth < 500
            depth = np.where(depth_validity_mask, depth, 0)
            depth = depth.copy()

            # Convert the depth image to a torch tensor
            depth_tensor = torch.from_numpy(depth)

            # Save the depth image to the target scene root
            depth_target_path = depth_dir / f"{file_name}_{camera_name}.exr"
            store_data(depth_target_path, depth_tensor, "depth")

            # Get the camera intrinsics (dict with keys cx, cy, fx, fy)
            intrinsics = modality_to_intrinsics[camera_name]

            # Get the pose for the current frame (in LFU convention)
            translation_lfu = data_entry["datum"]["image"]["pose"]["translation"]
            rotation_lfu = data_entry["datum"]["image"]["pose"]["rotation"]

            # Convert the pose to a 4x4 matrix
            pose_lfu = np.eye(4, dtype=np.float32)
            pose_lfu[:3, :3] = Rotation.from_quat(
                [
                    rotation_lfu["qx"],
                    rotation_lfu["qy"],
                    rotation_lfu["qz"],
                    rotation_lfu["qw"],
                ],
                scalar_first=False,
            ).as_matrix()
            pose_lfu[:3, 3] = np.array(
                [translation_lfu["x"], translation_lfu["y"], translation_lfu["z"]]
            )

            # Convert the pose to OpenCV convention
            lfu_to_rdf = np.array(
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            pose_rdf = lfu_to_rdf @ pose_lfu

            # Store wai frame metadata
            wai_frame = {
                "frame_name": f"{file_name}_{camera_name}",
                "image": str(rgb_target_path),
                "file_path": str(rgb_target_path),
                "depth": str(depth_target_path),
                "transform_matrix": pose_rdf.tolist(),
                "h": depth.shape[0],
                "w": depth.shape[1],
                "fl_x": intrinsics["fx"],
                "fl_y": intrinsics["fy"],
                "cx": intrinsics["cx"],
                "cy": intrinsics["cy"],
            }
            wai_frames.append(wai_frame)

    # Construct overall scene metadata
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
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {
                "frame_key": "depth",
                "format": "depth",
            },
        },
    }
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/paralleldomain4d.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_paralleldomain4d_scene, cfg)
