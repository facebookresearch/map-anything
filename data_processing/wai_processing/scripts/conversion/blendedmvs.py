# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import re
from pathlib import Path

import numpy as np
import torch
from argconf import argconf_parse
from natsort import natsorted

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,  # noqa: F401, Needed for launch_slurm.py
)

from morphcloud.utils.wai.core import store_data


def load_blendedmvs_pfm_file(file_path):
    "Function to load PFM format depth map"
    with open(file_path, "rb") as file:
        header = file.readline().decode("UTF-8").strip()

        if header == "PF":
            is_color = True
        elif header == "Pf":
            is_color = False
        else:
            raise ValueError("The provided file is not a valid PFM file.")

        dimensions = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
        if dimensions:
            img_width, img_height = map(int, dimensions.groups())
        else:
            raise ValueError("Invalid PFM header format.")

        endian_scale = float(file.readline().decode("UTF-8").strip())
        if endian_scale < 0:
            dtype = "<f"  # little-endian
        else:
            dtype = ">f"  # big-endian

        data_buffer = file.read()
        img_data = np.frombuffer(data_buffer, dtype=dtype)

        if is_color:
            img_data = np.reshape(img_data, (img_height, img_width, 3))
        else:
            img_data = np.reshape(img_data, (img_height, img_width))

        img_data = cv2.flip(img_data, 0)

    return img_data


def load_blendedmvs_pose(path, ret_44=False):
    "Load camera pose for BlendedMVS"
    f = open(path)
    RT = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)
    assert RT.shape == (4, 4)
    RT = np.linalg.inv(RT)  # world2cam to cam2world

    K = np.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
    assert K.shape == (3, 3)

    if ret_44:
        return K, RT

    return K, RT[:3, :3], RT[:3, 3]


def process_blendedmvs_scene(cfg, scene_name):
    """
    Process a BlendedMVS scene into the WAI format.
    The PFM format depth maps are converted to default WAI depth format (exr).

    Expected root directory structure for the raw BlendedMVS dataset:
    .
    └── blendedmvs/
        ├── 000000000000000000000000/
        │   ├── blended_images/
        │   │   ├── 00000000_masked.jpg
        │   │   ├── 00000000.jpg
        │   │   ├── ...
        │   ├── cams/
        │   │   ├── 00000000_cam.txt
        │   │   ├── ...
        │   ├── rendered_depth_maps/
        │   │   ├── 00000000.pfm
        ├── ...
        ├── 5a2a95f032a1c655cfe3de62
        ├── ...
        ├── 5858dbcab338a62ad5001081
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
    root_cam_dir = scene_root / "cams"
    file_names = [f[:-8] for f in os.listdir(root_cam_dir) if not f.startswith("pair")]
    file_names = natsorted(file_names)

    # Loop over all files and process them
    for file_name in file_names:
        # Symlink original images to WAI path
        image_path = scene_root / "blended_images" / f"{file_name}.jpg"
        rel_target_image_path = Path("images") / f"{file_name}{image_path.suffix}"
        os.symlink(image_path, target_scene_root / rel_target_image_path)

        # Load depth map from PFM file
        depthmap = load_blendedmvs_pfm_file(
            scene_root / "rendered_depth_maps" / f"{file_name}.pfm"
        )
        depthmap = depthmap.copy()

        # Save depth map to EXR file using WAI
        rel_depth_out_path = Path("depth") / (file_name + ".exr")
        store_data(
            target_scene_root / rel_depth_out_path,
            torch.tensor(depthmap),
            "depth",
        )

        # Load intrinsics and camera pose
        intrinsics, Rt_cam2world_opencv = load_blendedmvs_pose(
            scene_root / "cams" / f"{file_name}_cam.txt",
            ret_44=True,
        )

        # Store WAI frame metadata
        wai_frame = {
            "frame_name": file_name,
            "image": str(rel_target_image_path),
            "file_path": str(rel_target_image_path),
            "depth": str(rel_depth_out_path),
            "transform_matrix": Rt_cam2world_opencv.tolist(),
            "h": depthmap.shape[0],
            "w": depthmap.shape[1],
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
        "scale_type": "colmap",
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
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/blendedmvs.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_blendedmvs_scene, cfg)
