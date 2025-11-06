# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path

import numpy as np
import torch
from argconf import argconf_parse
from natsort import natsorted
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,  # noqa: F401, Needed for launch_slurm.py
)

from morphcloud.utils.wai.core import load_data, store_data

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def process_mvs_synth_scene(cfg, scene_name):
    """
    Process a MVS-Synth scene into the WAI format.
    Symlink the original rgb image.
    Load the depth (already in .exr format), filter sky depth and save to WAI format.
    Convert the pose and intrinsics information to WAI metadata format.

    Expected root directory structure for the raw MVS-Synth dataset:
    .
    └── mvs_synth/
        ├── 0000/
        │   ├── depths/
        │   │   ├── 0000.exr
        │   │   ├── ...
        │   ├── images/
        │   │   ├── 0000.png
        │   │   ├── ...
        │   ├── poses/
        │   │   ├── 0000.json
        ├── ...
        ├── 0027
        ├── ...
        ├── 0062
        └── ...
    """
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    wai_frames = []

    # Get all image file names from the images directory
    images_dir = scene_root / "images"
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    image_files = natsorted(image_files)

    # Loop over all files and process them
    for image_file in image_files:
        file_name = image_file.replace(".png", "")

        # Symlink original images to WAI path
        image_path = scene_root / "images" / image_file
        rel_target_image_path = Path("images") / image_file
        target_image_path = target_scene_root / rel_target_image_path
        if target_image_path.exists():
            target_image_path.unlink()
        os.symlink(image_path, target_image_path)

        # Load depth map using WAI load_data function
        depth_path = scene_root / "depths" / f"{file_name}.exr"
        depthmap = load_data(depth_path, "depth")
        if isinstance(depthmap, torch.Tensor):
            depthmap = depthmap.numpy()

        # Apply sky mask (sky is set to np.inf, filter to 0)
        depthmap = np.where(depthmap == np.inf, 0, depthmap)
        depthmap = depthmap.copy()

        # Convert depth to metric
        # Determined by checking length of cars (Range Rover) and standing people
        depthmap = depthmap / 10.0

        # Save depth map to EXR file using WAI
        rel_depth_out_path = Path("depth") / f"{file_name}.exr"
        store_data(
            target_scene_root / rel_depth_out_path,
            torch.tensor(depthmap),
            "depth",
        )

        # Load pose information from JSON file
        pose_path = scene_root / "poses" / f"{file_name}.json"
        with open(pose_path, "r") as f:
            cam_info = json.load(f)

        # Extract pose (world-to-camera -> camera-to-world)
        H_w2c = np.array(cam_info["extrinsic"], dtype=np.float32)
        H_c2w = np.linalg.inv(H_w2c)

        # Convert left-handed system to right-handed one
        # RUF -> RDF
        flip_y = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        H_c2w = flip_y @ H_c2w

        # Convert pose translation to metric
        # Determined by checking length of cars (Range Rover) and standing people
        H_c2w[:3, 3] = H_c2w[:3, 3] / 10.0

        # Store WAI frame metadata
        wai_frame = {
            "frame_name": file_name,
            "image": str(rel_target_image_path),
            "file_path": str(rel_target_image_path),
            "depth": str(rel_depth_out_path),
            "transform_matrix": H_c2w.tolist(),
            "h": depthmap.shape[0],
            "w": depthmap.shape[1],
            "fl_x": float(cam_info["f_x"]),
            "fl_y": float(cam_info["f_y"]),
            "cx": float(cam_info["c_x"]),
            "cy": float(cam_info["c_y"]),
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
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/mvs_synth.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_mvs_synth_scene, cfg)
