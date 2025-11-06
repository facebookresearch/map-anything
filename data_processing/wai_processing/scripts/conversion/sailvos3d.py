# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
from shutil import rmtree

import cv2
import numpy as np
import torch
import yaml
from argconf import argconf_parse
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

from morphcloud.utils.wai.camera import gl2cv
from morphcloud.utils.wai.core import store_data

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logger = logging.getLogger("sailvos3d")


def pixels_to_ndcs(xx, yy, size):
    """
    Converts pixel coordinates to Normalized Device Coordinates (NDC).

    Reference: https://github.com/nianticlabs/mvsanywhere/blob/main/src/mvsanywhere/datasets/sailvos3d.py

    Parameters
    ----------
    xx : A 1D numpy array of x pixel coordinates.
    yy : A 1D numpy array of y pixel coordinates.
    size : A tuple containing the height (H) and width (W) of the image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] -> Two 1D numpy arrays representing the x and y coordinates in NDC space.
    """
    s_y, s_x = size
    s_x -= 1  # so 1 is being mapped into (n-1)th pixel
    s_y -= 1  # so 1 is being mapped into (n-1)th pixel
    x_ndc = (2.0 / s_x) * xx - 1.0
    y_ndc = (-2.0 / s_y) * yy + 1.0
    return x_ndc, y_ndc


def convert_ndc_depth_to_cam(depth, P_inverse, depth_h, depth_w):
    """
    Converts a depth map from Normalized Device Coordinates (NDC) to camera space.

    Reference: https://github.com/nianticlabs/mvsanywhere/blob/main/src/mvsanywhere/datasets/sailvos3d.py

    Parameters
    ----------
    depth : A 2D numpy array of shape (H, W) representing the depth map in NDC. Depth values are
            assumed to be in a normalized range and will be scaled accordingly.
    P_inverse : A 4x4 numpy array representing the inverse of the projection matrix. This matrix is used
                to transform NDC coordinates to camera coordinates.
    depth_h : The height of the depth map in pixels.
    depth_w : The width of the depth map in pixels.

    Returns
    -------
    np.ndarray -> A 2D numpy array of shape (H, W) containing the depth values in camera space.
    """
    # Apply depth scaling based on dataset specification
    depth_scaled = (depth / 6.0) - 4e-5

    # Generate pixel coordinates
    px = np.arange(depth_w)
    py = np.arange(depth_h)
    px_grid, py_grid = np.meshgrid(px, py, sparse=False)
    px_flat = px_grid.reshape(-1)
    py_flat = py_grid.reshape(-1)

    # Retrieve depth values at each pixel
    ndcz = depth_scaled[py_flat, px_flat]  # Depth in NDC

    # Convert pixel coordinates to NDC
    ndcx, ndcy = pixels_to_ndcs(px_flat, py_flat, (depth_h, depth_w))

    # Stack NDC coordinates with depth and homogeneous coordinate
    ndc_coord = np.stack(
        [ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1
    )  # Shape: (N, 4)

    # Transform NDC coordinates to camera space
    camera_coord = ndc_coord @ P_inverse  # Shape: (N, 4)
    camera_coord /= camera_coord[:, -1:]

    # Extract and negate the Z-component to align with camera forward direction
    depth_cam = -camera_coord[:, 2].reshape(depth_h, depth_w)  # Shape: (H, W)

    return depth_cam


def process_sailvos3d_scene(cfg, scene_name):
    """
    Process a SAIL-VOS 3D scene into the WAI format.
    The original RGBA images are symlinked.
    Depths, camera params and poses are processed to WAI format.

    Expected root directory structure for the raw SAIL-VOS 3D dataset:
    .
    └── sailvos3d/
        ├── ah_3a_ext/
        │   ├── depth/
        │   │   |── 000000000000000000.npy
        │   │   |── 000000000000000001.npy
        │   │   |── ...
        │   │   ├── ...
        │   ├── camera/
        │   │   ├── 000000000000000000.yaml
        │   │   ├── 000000000000000001.yaml
        │   │   ├── ...
        │   ├── rage_matrices/
        │   │   ├── 000000000000000000.npz
        │   │   ├── 000000000000000001.npz
        │   │   ├── ...
        │   ├── other modalities ...
        │   ├── localization_1675829175834000_1675829180734000.json
        │   ├── scene_6e420602a6de7a4fb1ed4671775efa83424c85c5.json
        ├── ...
        ├── fam_6_mcs_1/
        ├── ...
    """
    # Get the scene paths
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)
    wai_frames = []

    # Check if images directory exists
    if not (scene_root / "images").exists():
        logger.error(
            f"[sailvos3d] Scene '{scene_name}' does not have an 'images' directory. Skipping."
        )
        # Clean up created directories
        if image_dir.exists():
            rmtree(image_dir)
        if depth_dir.exists():
            rmtree(depth_dir)
        raise FileNotFoundError("Images folder not found")

    # Check correspondence between cameras, images, depths
    cam_names = sorted((scene_root / "camera").glob("*.yaml"))
    img_names = sorted((scene_root / "images").glob("*.bmp"))
    cam_set = set(cam_name.stem for cam_name in cam_names)
    img_set = set(img_name.stem for img_name in img_names)

    # Find common files between cameras and images
    common_files = cam_set & img_set
    cam_extra = cam_set - img_set
    img_extra = img_set - cam_set

    if cam_extra:
        logger.warning(
            f"[sailvos3d] Scene '{scene_name}', files in 'camera' not found in 'images': {cam_extra}. These will be skipped."
        )
    if img_extra:
        logger.warning(
            f"[sailvos3d] Scene '{scene_name}', files in 'images' not found in 'camera': {img_extra}. These will be skipped."
        )

    # Check if we have any common files to process
    if not common_files:
        logger.error(
            f"[sailvos3d] Scene '{scene_name}' has no common files between cameras and images. Cleaning up and skipping."
        )
        # Clean up created directories
        if image_dir.exists():
            rmtree(image_dir)
        if depth_dir.exists():
            rmtree(depth_dir)
        raise ValueError(
            f"No common files found between cameras and images in scene '{scene_name}'"
        )

    logger.info(
        f"[sailvos3d] Scene '{scene_name}' processing {len(common_files)} common files out of {len(cam_set)} camera files and {len(img_set)} image files."
    )

    # Filter camera files to only include common ones
    cam_names = [cam_file for cam_file in cam_names if cam_file.stem in common_files]

    # Process each image and its corresponding data
    for cam_file in cam_names:
        cam_name = cam_file.stem
        rgb_path = scene_root / f"images/{cam_name}.bmp"
        depth_path = scene_root / f"depth/{cam_name}.npy"
        rage_path = scene_root / f"rage_matrices/{cam_name}.npz"

        # Original format not supported (bmp) so convert to png
        rgb_target_path = image_dir / f"{cam_name}.png"
        img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(rgb_target_path), img)

        # Load depth
        depth = np.load(depth_path)
        skymask = depth == 24e-5

        # Load camera intrinsics and pose from YAML
        with open(cam_file, "r") as f:
            cam_yaml = yaml.safe_load(f)
        intrinsics = np.array(cam_yaml["K"], dtype=np.float32)

        # Adjust intrinsics by NDC matrix
        depth_h, depth_w = depth.shape
        intrinsics[0, 2] += depth_w / 2.0
        intrinsics[1, 2] += depth_h / 2.0

        # Extract intrinsics in the format [fx, fy, cx, cy]
        intrinsics = intrinsics[(0, 1, 0, 1), (0, 1, 2, 2)]
        fl_x, fl_y, cx, cy = intrinsics

        # Load pose (world to camera) and invert to camera to world
        H_w2c = np.eye(4, dtype=np.float32)
        H_w2c[:3, :] = np.array(cam_yaml["Rt"], dtype=np.float32)
        H_c2w = np.linalg.inv(H_w2c)
        H_c2w, gl2cv_cmat = gl2cv(H_c2w, return_cmat=True)  # OpenGL -> OpenCV

        # Load rage matrices and convert NDC depth to metric
        rage_matrices = np.load(rage_path)
        depth_metric = convert_ndc_depth_to_cam(
            depth, rage_matrices["P_inv"], depth_h, depth_w
        )
        depth_metric = np.where(skymask, 0, depth_metric)
        depth_metric = depth_metric.copy()

        # Save depth as EXR using WAI store_data
        depth_target_path = depth_dir / f"{cam_name}.exr"
        depth_tensor = torch.from_numpy(depth_metric.astype(np.float32))
        store_data(depth_target_path, depth_tensor, "depth")

        # Append frame metadata
        wai_frames.append(
            {
                "frame_name": cam_name,
                "file_path": f"images/{cam_name}.png",
                "image": f"images/{cam_name}.png",
                "depth": f"depth/{cam_name}.exr",
                "transform_matrix": H_c2w.tolist(),
                "h": int(depth_metric.shape[0]),
                "w": int(depth_metric.shape[1]),
                "fl_x": float(fl_x),
                "fl_y": float(fl_y),
                "cx": float(cx),
                "cy": float(cy),
            }
        )

    # Construct scene_meta.json metadata
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "last_modified": None,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "frames": wai_frames,
        "scene_modalities": {},
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {"frame_key": "depth", "format": "depth"},
        },
    }
    # Store scene metadata JSON
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/sailvos3d.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_sailvos3d_scene, cfg)
