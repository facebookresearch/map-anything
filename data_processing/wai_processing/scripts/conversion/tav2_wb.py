# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from natsort import natsorted
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,  # noqa: F401, Needed for launch_slurm.py
)

from morphcloud.utils.wai.core import load_data, store_data


def process_tav2_wb_scene(cfg, scene_name):
    """
    Process a TAv2-WB scene into the WAI format.
    The RGB images and depth maps are symlinked to the WAI format.
    The intrinsics and poses are loaded from the numpy files and stored in the WAI format.
    The poses are already in opencv cam2world convention.

    Expected root directory structure for the raw TAv2-WB dataset:
    .
    └── tav2_wb/
        ├── AbandonedCable/
        │   ├── camera_params/
        │   │   ├── 00000021_0.npy
        │   │   ├── ...
        │   ├── depth/
        │   │   ├── 00000021_0.exr
        │   │   ├── ...
        │   ├── images/
        │   │   ├── 00000021_0.png
        │   │   ├── ...
        │   ├── poses/
        │   │   ├── 00000021_0.npy
        │   │   ├── ...
        ├── ...
        ├── DesertGasStation
        ├── ...
        ├── PolarSciFi
        └── ...
    """
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)
    wai_frames = []

    # Get all file names
    images_path = scene_root / "images"
    depths_path = scene_root / "depth"
    camera_params_path = scene_root / "camera_params"
    poses_path = scene_root / "poses"

    # Get all image files
    image_files = [f for f in os.listdir(images_path) if f.endswith(".png")]
    image_files = natsorted(image_files)

    # Loop over all files and process them
    for image_file in image_files:
        # Extract frame name without extension
        frame_name = image_file.split(".")[0]

        # Symlink original images to WAI path
        image_path = images_path / image_file
        rel_target_image_path = Path("images") / image_file
        os.symlink(image_path, target_scene_root / rel_target_image_path)

        # Symlink depth maps to WAI path
        depth_file = f"{frame_name}.exr"
        depth_path = depths_path / depth_file
        rel_depth_out_path = Path("depth") / depth_file
        os.symlink(depth_path, target_scene_root / rel_depth_out_path)

        # Load intrinsics from camera_params
        camera_param_file = f"{frame_name}.npy"
        camera_param_path = camera_params_path / camera_param_file
        intrinsics = np.load(camera_param_path)

        # Load camera pose
        pose_file = f"{frame_name}.npy"
        pose_path = poses_path / pose_file
        Rt_cam2world_opencv = np.load(pose_path)

        # Load image and get the size
        image = load_data(image_path, "image")
        h, w = image.shape[1:]  # (C, H, W)

        # Store WAI frame metadata
        wai_frame = {
            "frame_name": frame_name,
            "image": str(rel_target_image_path),
            "file_path": str(rel_target_image_path),
            "depth": str(rel_depth_out_path),
            "transform_matrix": Rt_cam2world_opencv.tolist(),
            "h": h,
            "w": w,
            "fl_x": float(intrinsics[0, 0]),
            "fl_y": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
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
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/tav2_wb.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_tav2_wb_scene, cfg)
