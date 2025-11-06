# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import gzip
import json
import logging
import os
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

from morphcloud.utils.wai.core import store_data
from morphcloud.utils.wai.scene_frame import _filter_scenes

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logger = logging.getLogger(__name__)


def get_annotated_split_map(cfg):
    """
    Build a map from DynamicReplica scene names to their relative paths.
    Checks for existence of frame_annotations_*.jgz files and image/depth folders.
    """
    scene_to_split_map = {}
    root = Path(cfg.original_root)
    splits = ["train", "valid", "test"]
    for split in splits:
        # Load frame annotations
        with gzip.open(
            f"{root}/frame_annotations_{split}.jgz",
            "rt",
            encoding="utf-8",
        ) as f:
            frames_annot = json.load(f)

        for frame_annot in frames_annot:
            # retrieve frame name annotation in the format <scene_name>_<camera>-<frame_number> for example 90ac3c-3_obj_source_left-0
            frame_id = (
                str(frame_annot["sequence_name"])
                + "_source_"
                + str(frame_annot["camera_name"])
                + "-"
                + str(frame_annot["frame_number"])
            )
            scene_to_split_map[frame_id] = frame_annot

    return scene_to_split_map


def get_original_scene_names(cfg):
    """
    Discover all valid DynamicReplica scenes by checking splits and scene folders.
    """
    root = Path(cfg.original_root)
    all_scenes = []
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        # Check if it's a directory and ends with _left or _right
        if os.path.isdir(item_path) and (
            item.endswith("_left") or item.endswith("_right")
        ):
            # scenes form a stereo pair (left and right) so we only add it once
            scene_path = item.replace("_left", "").replace("_right", "")
            if scene_path not in all_scenes:
                all_scenes.append(scene_path)
    # Optionally apply scene filters from config if you have such logic
    all_scenes = _filter_scenes(cfg.original_root, all_scenes, cfg.get("scene_filters"))
    return all_scenes


def get_intrinsics_matrix(viewpoint, image_width, image_height):
    """
    Convert NDC isotropic intrinsics to 3x3 pixel-space intrinsics matrix.
    """
    f_x_ndc, f_y_ndc = viewpoint["focal_length"]
    c_x_ndc, c_y_ndc = viewpoint["principal_point"]
    half_image_size = np.array([image_width, image_height]) / 2.0
    rescale = np.min(half_image_size)
    focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale
    principal_point_px = half_image_size - np.array([c_x_ndc, c_y_ndc]) * rescale
    K = np.array(
        [
            [focal_length_px[0], 0, principal_point_px[0]],
            [0, focal_length_px[1], principal_point_px[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return K


def get_extrinsics_matrix(viewpoint):
    """
    Construct 4x4 camera-to-world extrinsics matrix (opencv convention).
    """
    R = np.array(viewpoint["R"], dtype=np.float32)
    t = np.array(viewpoint["T"], dtype=np.float32)
    # Apply opencv convention: flip x/y axes
    R[:, :2] *= -1
    t[:2] *= -1
    H_c2w = np.eye(4, dtype=np.float32)
    H_c2w[:3, :3] = R
    H_c2w[:3, 3] = -R @ t
    return H_c2w


def load_16bit_png_depth(depth_png_path):
    """Load 16-bit PNG depth, convert to float32."""
    with Image.open(depth_png_path) as depth_pil:
        arr = np.array(depth_pil, dtype=np.uint16)
        depth = arr.view(np.float16).astype(np.float32)
        # Reshape to (H, W)
        depth = depth.reshape((depth_pil.size[1], depth_pil.size[0]))
    return depth


def process_dynamicreplica_scene(cfg, scene_name, annotated_split_map):
    """
    Process a Dynamic Replica scene into the WAI format.
    Depths, camera params and poses are processed to WAI format.

    Both camera left and camera right images are processed as separate frames.

    Expected root directory structure for the raw Dynamic Replica dataset:
    .
    └── dynamicrange/
        ├── 009850-3_obj_source_left/
            ├── depths/
                ├── 009850-3_obj_source_left_0000.geometric.png
                ├── ...
                ├── 009850-3_obj_source_left_0299.geometric.png
            ├── flow_backward/
                ├── 009850-3_obj_source_left_0000.png
                ├── ...
                ├── 009850-3_obj_source_left_0299.png
            ├── flow_backward_mask/
                ├── 009850-3_obj_source_left_0000.png
                ├── ...
                ├── 009850-3_obj_source_left_0299.png
            ├── flow_forward/
                ├── 009850-3_obj_source_left_0000.png
                ├── ...
                ├── 009850-3_obj_source_left_0298.png
            ├── flow_forward_mask/
                ├── 009850-3_obj_source_left_0000.png
                ├── ...
                ├── 009850-3_obj_source_left_0298.png
            ├── images/
                ├── 009850-3_obj_source_left-0000.png
                ├── ...
                ├── 009850-3_obj_source_left-0299.png
                └── done.ok
            ├── instance_id_maps/
                ├── 009850-3_obj_source_left_0000.pkl
                ├── 009850-3_obj_source_left_0000.png
                ├── ...
                ├── 009850-3_obj_source_left_0298.pkl
                ├── 009850-3_obj_source_left_0298.png
            ├── masks/
                ├── 009850-3_obj_source_left_0000.png
                ├── ...
                └── 009850-3_obj_source_left_0299.png
            └── trajectories/
                ├── 000000.pth
                ├── ...
                └── 000299.pth
        ├── 009850-3_obj_source_right/
        ├── 13a144-3_obj_source_left/
        ├── ...
        ├── frame_annotations_test.jgz
        ├── frame_annotations_train.jgz
        ├── frame_annotations_valid.jgz
        ├── ignacio_waving
        ├── nikita_reading
        └── teddy_static
    """
    # Dynamic replica has stereo pair (left and right). We treat them separately in the same wai scene.
    scene_root = f"{cfg.original_root}/{scene_name}"
    rgb_root_left = Path(f"{scene_root}_left") / "images"
    rgb_root_right = Path(f"{scene_root}_right") / "images"

    # Validate that both directories exist
    if not rgb_root_left.exists():
        raise RuntimeError(f"Left RGB directory does not exist: {rgb_root_left}")
    if not rgb_root_right.exists():
        raise RuntimeError(f"Right RGB directory does not exist: {rgb_root_right}")

    # Get file lists and extract frame identifiers
    left_files = [f for f in os.listdir(rgb_root_left) if f != "done.ok"]
    right_files = [f for f in os.listdir(rgb_root_right) if f != "done.ok"]

    if len(left_files) == 0 and len(right_files) == 0:
        raise RuntimeError(
            f"No valid image files found in either {rgb_root_left} or {rgb_root_right}"
        )

    # Extract frame numbers/identifiers from filenames to match pairs
    def extract_frame_id(filename):
        # Extract the frame number from filenames like "scene_left-0000.png" or "scene_right-0000.png"
        # Split by '-' and take the last part before the extension
        return filename.split("-")[-1].split(".")[0]

    # Create mappings from frame_id to filename
    left_frame_map = {extract_frame_id(f): f for f in left_files}
    right_frame_map = {extract_frame_id(f): f for f in right_files}

    # Find common frame IDs that exist in both directories
    common_frame_ids = set(left_frame_map.keys()) & set(right_frame_map.keys())

    if len(common_frame_ids) == 0:
        logger.warning(
            f"No matching frame pairs found for scene {scene_name}. "
            f"Left has {len(left_files)} files, right has {len(right_files)} files"
        )
        raise RuntimeError(f"No matching frame pairs found for scene {scene_name}")

    # Sort the common frame IDs to ensure consistent processing order
    sorted_frame_ids = natsorted(list(common_frame_ids))

    # Warn if there's a mismatch in file counts
    if len(left_files) != len(right_files) or len(common_frame_ids) != len(left_files):
        logger.warning(
            f"File count mismatch for scene {scene_name}: "
            f"left has {len(left_files)} files, right has {len(right_files)} files, "
            f"processing {len(sorted_frame_ids)} common frames"
        )

    # Loop over frames and process
    wai_frames = []
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)

    # Process only the frames that exist in both directories
    for frame_id in tqdm(sorted_frame_ids):
        rgb_path_left = left_frame_map[frame_id]
        rgb_path_right = right_frame_map[frame_id]

        # Process stereo pairs
        frame_annot_left = annotated_split_map[f"{scene_name}_left-{int(frame_id)}"]
        frame_annot_right = annotated_split_map[f"{scene_name}_right-{int(frame_id)}"]

        # Symlink RGB image
        # Left
        rgb_name_left = Path(frame_annot_left["image"]["path"]).name
        rgb_target_left = image_dir / rgb_name_left
        os.symlink((rgb_root_left / rgb_path_left).resolve(), rgb_target_left)
        # Right
        rgb_name_right = Path(frame_annot_right["image"]["path"]).name
        rgb_target_right = image_dir / rgb_name_right
        os.symlink((rgb_root_right / rgb_path_right).resolve(), rgb_target_right)

        # Load depth and save in WAI format
        # Left
        depth_path_left = Path(cfg.original_root) / frame_annot_left["depth"]["path"]
        depth_left = load_16bit_png_depth(depth_path_left)
        depth_target_left = depth_dir / f"{rgb_name_left[:-4]}.exr"
        store_data(depth_target_left, depth_left.astype(np.float32), "depth")
        # Right
        depth_path_right = Path(cfg.original_root) / frame_annot_right["depth"]["path"]
        depth_right = load_16bit_png_depth(depth_path_right)
        depth_target_right = depth_dir / f"{rgb_name_right[:-4]}.exr"
        store_data(depth_target_right, depth_right.astype(np.float32), "depth")

        # Get intrinsics and extrinsics of left frame
        img_height, img_width = frame_annot_left["image"]["size"]
        viewpoint = frame_annot_left["viewpoint"]
        K = get_intrinsics_matrix(viewpoint, img_width, img_height)
        H_c2w = get_extrinsics_matrix(viewpoint)

        # Add frame data for left frame
        wai_frames.append(
            {
                "frame_name": rgb_name_left[:-4],
                "file_path": f"images/{rgb_name_left}",
                "image": f"images/{rgb_name_left}",
                "depth": f"depth/{rgb_name_left[:-4]}.exr",
                "transform_matrix": H_c2w.tolist(),
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "h": int(img_height),
                "w": int(img_width),
            }
        )

        # Get intrinsics and extrinsics of right frame
        img_height, img_width = frame_annot_right["image"]["size"]
        viewpoint = frame_annot_right["viewpoint"]
        K = get_intrinsics_matrix(viewpoint, img_width, img_height)
        H_c2w = get_extrinsics_matrix(viewpoint)

        # Add frame data for right frame
        wai_frames.append(
            {
                "frame_name": rgb_name_right[:-4],
                "file_path": f"images/{rgb_name_right}",
                "image": f"images/{rgb_name_right}",
                "depth": f"depth/{rgb_name_right[:-4]}.exr",
                "transform_matrix": H_c2w.tolist(),
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "h": int(img_height),
                "w": int(img_width),
            }
        )

    # Build the overall scene metadata
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
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
    # Save scene_meta.json in wai format
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/dynamicreplica.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    annotated_split_map = get_annotated_split_map(cfg)
    convert_scenes_wrapper(
        process_dynamicreplica_scene,
        cfg,
        get_original_scene_names_func=get_original_scene_names,
        annotated_split_map=annotated_split_map,
    )
