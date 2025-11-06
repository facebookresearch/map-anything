# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

from morphcloud.utils.wai.camera import CAMERA_KEYS, gl2cv
from morphcloud.utils.wai.core import load_data, store_data
from morphcloud.utils.wai.scene_frame import get_scene_names

logger = logging.getLogger(__name__)


def get_original_scene_names(cfg):
    all_scene_names = []
    # hard-coding as DL3DV has exactly 11 splits from 1K to 11K
    for split_idx in range(1, 12):
        data_split = f"{split_idx}K"
        split_root = Path(cfg.original_root, data_split)
        scene_names_one_split = sorted(
            [
                f"{data_split}_{orig_scene_id}"
                for orig_scene_id in os.listdir(split_root)
                if os.path.isdir(os.path.join(split_root, orig_scene_id))
            ]
        )
        all_scene_names.extend(scene_names_one_split)
    scene_names = get_scene_names(cfg, scene_names=all_scene_names)
    return scene_names


def convert_scene(cfg, scene_name) -> None | tuple[str, str]:
    # scene_name is f"{split_name}_{scene_id}"
    dataset_name = cfg.get("dataset_name", "dl3dv")
    version = cfg.get("version", "0.1")
    source_scene_root = Path(cfg.original_root, scene_name.replace("_", "/"))
    if any(
        [
            not Path(source_scene_root, "transforms.json").exists(),
            not Path(source_scene_root, "colmap").exists(),
            not Path(source_scene_root, "images").exists(),
        ]
    ):
        raise RuntimeError(
            f"Expected 'transforms.json', 'images', and 'colmap' to exist in {source_scene_root}"
        )
    logger.info(f"Processing: {scene_name}")
    transforms_fn = Path(source_scene_root, "transforms.json")
    out_path = Path(cfg.root) / scene_name
    meta = load_data(transforms_fn)
    frames = meta["frames"]

    # skip portrait images for now
    if meta["h"] > meta["w"]:
        # return state, error_message
        return "data_issue", "Images are in portrait, not supported for now."

    image_out_path = out_path / "images_distorted"
    os.makedirs(image_out_path)
    wai_frames = []
    for frame in frames:
        frame_name = Path(frame["file_path"]).stem
        wai_frame = {"frame_name": frame_name}
        org_transform_matrix = np.array(frame["transform_matrix"]).astype(np.float32)
        opencv_pose, gl2cv_cmat = gl2cv(org_transform_matrix, return_cmat=True)
        # link distorted images
        source_image_path = Path(source_scene_root, frame["file_path"])
        target_image_path = f"images_distorted/{frame_name}.png"
        os.symlink(source_image_path, out_path / target_image_path)
        wai_frame["image_distorted"] = target_image_path
        wai_frame["file_path"] = target_image_path
        wai_frame["transform_matrix"] = opencv_pose.tolist()
        other_keys = ["colmap_im_id"]

        for other_key in other_keys:
            if other_key in frame:
                wai_frame[other_key] = frame[other_key]
        wai_frames.append(wai_frame)

    # link colmap cache
    os.symlink(Path(source_scene_root, "colmap"), out_path / "colmap")
    # atm no native support for colmap as a format - we can this later if needed
    scene_modalities = {"colmap": {"scene_key": "colmap"}}

    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "version": version,
        "shared_intrinsics": True,
        "camera_model": meta["camera_model"],
        "camera_convention": "opencv",
        "scale_type": "colmap",
    }
    # dl3dv applied an additional transform on the colmap poses
    # store it to retrieve the original colmap poses
    for camera_key in CAMERA_KEYS:
        if camera_key in meta:
            scene_meta[camera_key] = meta[camera_key]
    scene_meta["frames"] = wai_frames
    scene_meta["frame_modalities"] = {
        "image_distorted": {"frame_key": "image_distorted", "format": "image"},
    }
    scene_meta["scene_modalities"] = scene_modalities
    applied_transform = np.array(meta["applied_transform"]).reshape(3, 4)
    applied_transform = np.concatenate([applied_transform, np.array([[0, 0, 0, 1.0]])])
    scene_meta["_applied_transform"] = (
        applied_transform.tolist()
    )  # transform from colmap poses to opencv poses
    scene_meta["_applied_transforms"] = {
        "opengl2opencv": gl2cv_cmat.tolist()
    }  # transforms raw poses to opencv poses
    store_data(out_path / "scene_meta_distorted.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/dl3dv.yaml")
    convert_scenes_wrapper(
        convert_scene,
        cfg,
        # Need to use the dl3dv func to get original scenes names
        # as the subsets with 1K, 2K, 3K etc are DL3DV specific
        get_original_scene_names_func=get_original_scene_names,
    )
