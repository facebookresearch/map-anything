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

logger = logging.getLogger(__name__)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _load_kpts_and_poses(root, scene_name, subscene, z_only=False, intrinsics=True):
    """
    Load camera parameters from MegaDepth dataset.

    Args:
        root: Root directory of the MegaDepth dataset
        scene_name: Scene Name
        subscene: Subscene ID
        z_only: If True, only return the principal axis
        intrinsics: If True, also return camera intrinsics

    Returns:
        points3D_idxs: Dictionary mapping image IDs to 3D point indices
        poses: Dictionary mapping image IDs to camera poses
        image_intrinsics: Dictionary mapping image IDs to camera intrinsics (if intrinsics=True)
    """
    if intrinsics:
        with open(
            os.path.join(
                root, scene_name, "sparse", "manhattan", subscene, "cameras.txt"
            ),
            "r",
        ) as f:
            raw = f.readlines()[3:]  # skip the header

        camera_intrinsics = {}
        for camera in raw:
            camera = camera.split(" ")
            width, height, focal, cx, cy, k0 = [float(elem) for elem in camera[2:]]
            K = np.eye(3)
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = cx
            K[1, 2] = cy
            camera_intrinsics[int(camera[0])] = (
                (int(width), int(height)),
                K,
                (k0, 0, 0, 0),
            )

    with open(
        os.path.join(root, scene_name, "sparse", "manhattan", subscene, "images.txt"),
        "r",
    ) as f:
        raw = f.read().splitlines()[4:]  # skip the header

    extract_pose = (
        colmap_raw_pose_to_principal_axis if z_only else colmap_raw_pose_to_RT
    )

    poses = {}
    points3D_idxs = {}
    camera = []

    for image, points in zip(raw[::2], raw[1::2]):
        image = image.split(" ")
        points = points.split(" ")

        image_id = image[-1]
        camera.append(int(image[-2]))

        # find the principal axis
        raw_pose = [float(elem) for elem in image[1:-2]]
        poses[image_id] = extract_pose(raw_pose)

        current_points3D_idxs = {int(i) for i in points[2::3] if i != "-1"}
        assert -1 not in current_points3D_idxs
        points3D_idxs[image_id] = current_points3D_idxs

    if intrinsics:
        image_intrinsics = {
            im_id: camera_intrinsics[cam] for im_id, cam in zip(poses, camera)
        }
        return points3D_idxs, poses, image_intrinsics
    else:
        return points3D_idxs, poses


def colmap_raw_pose_to_principal_axis(image_pose):
    """Convert COLMAP quaternion to principal axis."""
    qvec = image_pose[:4]
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    z_axis = np.float32(
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    )
    return z_axis


def colmap_raw_pose_to_RT(image_pose):
    """Convert COLMAP quaternion to rotation matrix and translation vector."""
    qvec = image_pose[:4]
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    t = image_pose[4:7]
    # World-to-Camera pose
    current_pose = np.eye(4)
    current_pose[:3, :3] = R
    current_pose[:3, 3] = t
    return current_pose


def process_megadepth_scene(cfg, scene_subscene_name):
    """
    Process a MegaDepth scene into the WAI format.
    Convert the H5 format depth maps to default WAI depth format (exr).
    Load the distorted intrinsics (pre rectification in the sparse reconstruction metadata) and convert them to undistorted rectified intrinsics.
    Save the images, depth maps, intrinsics and poses (for the rectified version) to the WAI format.
    Only processes images that are in the megadepth_pairs.npz file.

    Expected root directory structure for the raw MegaDepth dataset:
    .
    └── megadepth/
        ├── 0000/
        │   ├── dense0/
        │   │   ├── depths/
        │   │   |   ├── 5978745_520771b1cb_o.h5
        │   │   |   ├── ...
        │   │   ├── imgs/
        │   │   |   ├── 5978745_520771b1cb_o.jpg
        │   │   |   ├── ...
        │   ├── dense1/
        │   │   ├── depths/
        │   │   |   ├── 9616344_27950454e0_o.h5
        │   │   |   ├── ...
        │   │   ├── imgs/
        │   │   |   ├── 9616344_27950454e0_o.jpg
        │   │   |   ├── ...
        │   ├── images/
        │   │   ├── 1416978_a6f1f7a16b_o.jpg
        │   │   ├── ...
        │   ├── sparse/manhattan/
        │   │   ├── 0/
        │   │   |   ├── ...
        │   │   ├── 1/
        │   │   |   ├── ...
        ├── 0001/
        │   ├── dense0/
        │   ├── images/
        │   ├── sparse/manhattan/
        │   │   ├── 0/
        ├── ...
        ├── megadepth_pairs.npz
    """

    # Split the scene_subscene_name back into scene_name and subscene
    parts = scene_subscene_name.split("_")
    subscene = parts[-1]
    scene_name = "_".join(
        parts[:-1]
    )  # Handle scene names that might contain underscores

    scene_outpath = Path(cfg.root) / scene_subscene_name
    scene_outpath.mkdir(parents=True, exist_ok=True)

    # Create target directories for this subscene
    target_scene_root = Path(cfg.root) / scene_subscene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)

    # Initialize frames list for this subscene
    wai_frames = []

    # Load camera parameters
    _, pose_w2cam, intrinsics = _load_kpts_and_poses(
        cfg.original_root, scene_name, subscene, intrinsics=True
    )

    # Get the scene path and dense directory for this subscene
    scene_path = Path(cfg.original_root) / scene_name
    dense_dir = scene_path / f"dense{subscene}"

    # Load megadepth_pairs.npz to filter images
    pairs_path = Path(cfg.original_root) / "megadepth_pairs.npz"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"megadepth_pairs.npz not found at {pairs_path}. Cannot proceed without pairs file."
        )

    # Load pairs data
    data = np.load(pairs_path, allow_pickle=True)
    scenes = data["scenes"]
    images = data["images"]
    pairs = data["pairs"]

    # Find images for this scene
    images_to_process = set()
    scene_found = False

    # Current scene identifier
    current_scene = f"{scene_name} {subscene}"

    # Collect all images for this scene from the pairs
    for scene_id, im1, im2, _ in pairs:
        # Check if this pair belongs to our scene
        if isinstance(scenes[scene_id], str) and scenes[scene_id] == current_scene:
            scene_found = True
            images_to_process.add(images[im1])
            images_to_process.add(images[im2])

    if not scene_found:
        logger.warning(
            f"Scene {scene_name}_{subscene} not found in pairs file. Skipping this scene."
        )
        return "skipped", f"Scene {scene_name}_{subscene} not found in pairs file"

    logger.info(
        f"Found {len(images_to_process)} images to process for scene {scene_name}_{subscene}"
    )

    # Process each image in the subscene in natural sorted order
    for image_id in tqdm(natsorted(images_to_process)):
        # Get intrinsic data for this image
        intrinsic_data = intrinsics[image_id]

        # Get image filename
        img_path = dense_dir / "imgs" / image_id

        # Skip if image doesn't exist
        if not img_path.exists():
            continue

        # Check if depth file exists
        depth_filename = Path(image_id).stem + ".h5"
        depth_path = dense_dir / "depths" / depth_filename

        if not depth_path.exists():
            continue

        # Symlink original image to WAI path
        rel_target_image_path = Path("images") / image_id
        os.symlink(img_path, target_scene_root / rel_target_image_path)

        # Load depth map from H5 file
        with h5py.File(depth_path, "r") as hd5:
            depthmap = np.asarray(hd5["depth"])

        # Get the dimensions of the depth map
        H, W = depthmap.shape

        # Save depth map to EXR file using WAI
        rel_depth_out_path = Path("depth") / (Path(image_id).stem + ".exr")
        store_data(
            target_scene_root / rel_depth_out_path,
            torch.tensor(depthmap),
            "depth",
        )

        # Get intrinsics
        imsize_pre, K_pre, distortion = intrinsic_data

        # Rectify = undistort the intrinsics
        imsize_post = (W, H)
        K_post = cv2.getOptimalNewCameraMatrix(
            K_pre,
            distortion,
            imsize_pre,
            alpha=0,
            newImgSize=imsize_post,
            centerPrincipalPoint=True,
        )[0]

        # Get camera pose (world to camera)
        w2cam_pose = pose_w2cam[image_id]

        # Convert to camera to world pose
        cam2world_pose = np.linalg.inv(w2cam_pose)

        # Store WAI frame metadata
        wai_frame = {
            "frame_name": Path(image_id).stem,
            "image": str(rel_target_image_path),
            "file_path": str(rel_target_image_path),
            "depth": str(rel_depth_out_path),
            "transform_matrix": cam2world_pose.tolist(),
            "h": H,
            "w": W,
            "fl_x": float(K_post[0, 0]),
            "fl_y": float(K_post[1, 1]),
            "cx": float(K_post[0, 2]),
            "cy": float(K_post[1, 2]),
        }
        wai_frames.append(wai_frame)

    # Construct scene metadata for this subscene
    scene_meta = {
        "scene_name": scene_subscene_name,
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


def get_original_scene_names(
    cfg,
):
    # Get all scene names to process
    original_scene_names = sorted(os.listdir(cfg.original_root))

    # Create a list of all scene_subscene combinations
    all_scene_subscene_names = []
    # First pass: collect all subscenes for each scene
    for scene_name in original_scene_names:
        scene_path = Path(cfg.original_root) / scene_name
        subscenes = []
        for item in scene_path.glob("dense*"):
            if item.is_dir():
                subscene = item.name.replace("dense", "")
                subscenes.append(subscene)
                scene_subscene_name = f"{scene_name}_{subscene}"
                all_scene_subscene_names.append(scene_subscene_name)
    # scene filter for batch processing
    all_scene_subscene_names = _filter_scenes(
        cfg.root, all_scene_subscene_names, cfg.get("scene_filters")
    )
    return all_scene_subscene_names


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/megadepth.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(
        process_megadepth_scene,
        cfg,
        get_original_scene_names_func=get_original_scene_names,
    )
