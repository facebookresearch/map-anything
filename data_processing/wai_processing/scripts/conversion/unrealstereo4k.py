# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

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

from morphcloud.utils.wai.core import store_data

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def process_unrealstereo4k_scene(cfg, scene_name):
    """
    Process a UnrealStereo4K scene into the WAI format.
    The original RGBA images are symlinked.
    The disparity maps (.npy) are converted to depth maps and saved in WAI format (.exr).
    Both camera 0 and camera 1 images are processed as separate frames.

    Expected root directory structure for the raw UnrealStereo4K dataset:
    .
    └── unrealstereo4k/
        ├── 00000/
        │   ├── Disp0/
        │   │   ├── 00000.npy
        │   │   ├── ...
        │   ├── Disp1/
        │   │   ├── 00000.npy
        │   │   ├── ...
        │   ├── Extrinsics0/
        │   │   ├── 00000.txt
        │   │   ├── ...
        │   ├── Extrinsics1/
        │   │   ├── 00000.txt
        │   │   ├── ...
        │   ├── Image0/
        │   │   ├── 00000.png
        │   │   ├── ...
        │   ├── Image1/
        │   │   ├── 00000.png
        │   │   ├── ...
        ├── ...
        ├── 00001
        ├── ...
        ├── 00008
    """
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)
    wai_frames = []

    # Get all file names from Image0 directory
    image0_dir = scene_root / "Image0"
    file_names = [f.stem for f in image0_dir.glob("*.png")]
    file_names = natsorted(file_names)

    # Loop over all files and process both cameras
    for file_name in file_names:
        # Load camera parameters
        cam0_path = scene_root / "Extrinsics0" / f"{file_name}.txt"
        cam1_path = scene_root / "Extrinsics1" / f"{file_name}.txt"

        # Parse camera 0 parameters
        with open(cam0_path, "r") as f:
            intrinsics0_line, extrinsics0_line = f.read().strip().splitlines()
        intrinsics0 = np.fromstring(
            intrinsics0_line, sep=" ", dtype=np.float32
        ).reshape(3, 3)
        H_w2c0 = np.eye(4, dtype=np.float32)
        H_w2c0[:3, :] = np.fromstring(
            extrinsics0_line, sep=" ", dtype=np.float32
        ).reshape(3, 4)
        H_c2w0 = np.linalg.inv(H_w2c0)

        # Parse camera 1 parameters
        with open(cam1_path, "r") as f:
            intrinsics1_line, extrinsics1_line = f.read().strip().splitlines()
        intrinsics1 = np.fromstring(
            intrinsics1_line, sep=" ", dtype=np.float32
        ).reshape(3, 3)
        H_w2c1 = np.eye(4, dtype=np.float32)
        H_w2c1[:3, :] = np.fromstring(
            extrinsics1_line, sep=" ", dtype=np.float32
        ).reshape(3, 4)
        H_c2w1 = np.linalg.inv(H_w2c1)

        # Calculate stereo baseline
        stereo_baseline = np.linalg.norm((H_w2c0 @ H_c2w1)[:3, 3])

        # Convert left-handed system to right-handed one
        # RUF -> RDF
        flip_y = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        H_c2w0 = flip_y @ H_c2w0
        H_c2w1 = flip_y @ H_c2w1

        # Process Camera 0 (original images are RGBA)
        image0_path = scene_root / "Image0" / f"{file_name}.png"
        image0_name = f"{file_name}_cam0.png"
        rel_target_image0_path = Path("images") / image0_name
        os.symlink(image0_path, target_scene_root / rel_target_image0_path)

        # Load disparity map from .npy file for camera 0
        disp0_path = scene_root / "Disp0" / f"{file_name}.npy"
        disp0 = np.load(disp0_path)

        # Convert disparity to depth for camera 0
        depth0 = stereo_baseline * intrinsics0[0, 0] / disp0
        depth0_validity_mask = depth0 < 10000
        depth0 = np.where(depth0_validity_mask, depth0, 0)
        depth0 = depth0.copy()

        # Save depth map to EXR file using WAI for camera 0
        rel_depth0_out_path = Path("depth") / f"{file_name}_cam0.exr"
        store_data(
            target_scene_root / rel_depth0_out_path,
            torch.tensor(depth0),
            "depth",
        )

        # Store WAI frame metadata for camera 0
        wai_frame0 = {
            "frame_name": f"{file_name}_cam0",
            "image": str(rel_target_image0_path),
            "file_path": str(rel_target_image0_path),
            "depth": str(rel_depth0_out_path),
            "transform_matrix": H_c2w0.tolist(),
            "h": depth0.shape[0],
            "w": depth0.shape[1],
            "fl_x": float(intrinsics0[0, 0]),
            "fl_y": float(intrinsics0[1, 1]),
            "cx": float(intrinsics0[0, 2]),
            "cy": float(intrinsics0[1, 2]),
        }
        wai_frames.append(wai_frame0)

        # Process Camera 1 (original images are RGBA)
        image1_path = scene_root / "Image1" / f"{file_name}.png"
        image1_name = f"{file_name}_cam1.png"
        rel_target_image1_path = Path("images") / image1_name
        os.symlink(image1_path, target_scene_root / rel_target_image1_path)

        # Load disparity map from .npy file for camera 1
        disp1_path = scene_root / "Disp1" / f"{file_name}.npy"
        disp1 = np.load(disp1_path)

        # Convert disparity to depth for camera 1
        depth1 = stereo_baseline * intrinsics1[0, 0] / disp1
        depth1_validity_mask = depth1 < 10000
        depth1 = np.where(depth1_validity_mask, depth1, 0)
        depth1 = depth1.copy()

        # Save depth map to EXR file using WAI for camera 1
        rel_depth1_out_path = Path("depth") / f"{file_name}_cam1.exr"
        store_data(
            target_scene_root / rel_depth1_out_path,
            torch.tensor(depth1),
            "depth",
        )

        # Store WAI frame metadata for camera 1
        wai_frame1 = {
            "frame_name": f"{file_name}_cam1",
            "image": str(rel_target_image1_path),
            "file_path": str(rel_target_image1_path),
            "depth": str(rel_depth1_out_path),
            "transform_matrix": H_c2w1.tolist(),
            "h": depth1.shape[0],
            "w": depth1.shape[1],
            "fl_x": float(intrinsics1[0, 0]),
            "fl_y": float(intrinsics1[1, 1]),
            "cx": float(intrinsics1[0, 2]),
            "cy": float(intrinsics1[1, 2]),
        }
        wai_frames.append(wai_frame1)

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
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/unrealstereo4k.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_unrealstereo4k_scene, cfg)
