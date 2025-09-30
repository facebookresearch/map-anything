# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import traceback
from copy import deepcopy
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import torch
from argconf import argconf_parse
from torch.utils.data import DataLoader
from tqdm import tqdm
from wai_processing.utils.state import SceneProcessLock, set_processing_state

from mapanything.utils.wai.basic_dataset import BasicSceneframeDataset
from mapanything.utils.wai.camera import DISTORTION_PARAM_KEYS
from mapanything.utils.wai.core import get_frame, load_data, set_frame, store_data
from mapanything.utils.wai.scene_frame import get_scene_names

logger = getLogger(__name__)


def compute_undistort_intrinsic(
    K: np.ndarray,
    width: int,
    height: int,
    distortion_params: np.ndarray,
    center_principal_point: bool = True,
) -> np.ndarray:
    """
    Computes new camera intrinsic matrix for undistortion of fisheye images.

    Args:
        K: Original camera intrinsic matrix (3x3)
        width: Image width in pixels
        height: Image height in pixels
        distortion_params: Fisheye distortion parameters [k1, k2, k3, k4]
        center_principal_point: Set principal point at image center

    Returns:
        New camera intrinsic matrix with principal point centered in the image
    """
    assert len(distortion_params.shape) == 1
    assert distortion_params.shape[0] == 4  # OPENCV_FISHEYE has k1, k2, k3, k4

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        distortion_params,
        (width, height),
        R=np.eye(3),
        balance=0.0,
    )

    # Move principal point (cx, cy) to the center of the image
    if center_principal_point:
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

    return new_K


def update_camera_meta(
    cam_meta: dict,
    new_K: np.ndarray,
    new_width: int,
    new_height: int,
) -> dict:
    """
    Updates camera metadata with new intrinsic parameters after undistortion.

    Args:
        meta: Original camera metadata dictionary, can be a scene meta or a frame meta
        new_K: New camera intrinsic matrix (3x3)
        new_width: Width of the undistorted image in pixels
        new_height: Height of the undistorted image in pixels

    Returns:
        Updated camera metadata dictionary with PINHOLE camera model and no distortion parameters
    """
    new_meta = deepcopy(cam_meta)
    new_meta["w"] = new_width
    new_meta["h"] = new_height
    new_meta["fl_x"] = float(new_K[0, 0])
    new_meta["fl_y"] = float(new_K[1, 1])
    new_meta["cx"] = float(new_K[0, 2])
    new_meta["cy"] = float(new_K[1, 2])
    new_meta["camera_model"] = "PINHOLE"

    # Remove distortion parameters
    for key in DISTORTION_PARAM_KEYS:
        if key in new_meta:
            del new_meta[key]

    return new_meta


def undistort_precompute(cam_meta: dict):
    intrinsics = np.array(
        [
            [cam_meta["fl_x"], 0, cam_meta["cx"]],
            [0, cam_meta["fl_y"], cam_meta["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    w, h = cam_meta["w"], cam_meta["h"]

    if cam_meta["camera_model"] == "OPENCV_FISHEYE":
        distortion_params = np.array(
            [cam_meta.get(cam_coeff, 0) for cam_coeff in ["k1", "k2", "k3", "k4"]]
        ).astype(np.float32)
        new_K = compute_undistort_intrinsic(
            intrinsics,
            w,
            h,
            distortion_params,
            center_principal_point=cfg.get("center_principal_point", True),
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsics,
            distortion_params,
            np.eye(3),
            new_K,
            (w, h),
            cv2.CV_32FC1,
        )
        roi = None  # fisheye does not require cropping
        new_w, new_h = w, h
    elif cam_meta["camera_model"] == "OPENCV":
        distortion_params = np.array(
            [cam_meta.get(cam_coeff, 0) for cam_coeff in ["k1", "k2", "p1", "p2", "k3"]]
        ).astype(np.float32)
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion_params, (w, h), 1, (w, h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            intrinsics, distortion_params, None, new_K, (w, h), cv2.CV_16SC2
        )
        _, _, new_w, new_h = roi
    else:
        raise NotImplementedError(
            f"Camera model not yet supported: {cam_meta['camera_model']}"
        )
    return new_K, new_w, new_h, map1, map2, roi


def undistort_scene(cfg, scene_name):
    # Create a dataloader that only parses a single scene.
    # This ensures that every frame loaded belongs to this scene.
    cfg.scene_filters = [scene_name]
    scene_meta = load_data(
        Path(cfg.root) / scene_name / cfg.scene_meta_path, "scene_meta"
    )
    single_scene_dataset = BasicSceneframeDataset(cfg)
    dataloader = DataLoader(
        single_scene_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    if scene_meta["shared_intrinsics"]:
        # --- precompute the undistortion map ---
        new_K, new_w, new_h, map1, map2, roi = undistort_precompute(cam_meta=scene_meta)
        undistorted_scene_meta = update_camera_meta(scene_meta, new_K, new_w, new_h)
    else:
        # For shared intrinsic case, the cameras are overwritten per frame
        undistorted_scene_meta = deepcopy(scene_meta)
        logger.warning(
            "Assuming per frame intrinsics, if any frame does not contain per "
            "frame intrinsics, undistortion will fail."
        )

    modality_mapping = {}  # mapping to replace the names with undistorted version
    for frame_modality in cfg.frame_modalities:
        if not frame_modality.endswith("_distorted"):
            raise ValueError("Only supports undistortion for modalities '*_distorted'")
        new_modality_name = frame_modality.replace("_distorted", "")
        modality_mapping[frame_modality] = new_modality_name

    # --- Iterate over all images and perform undistortion ---
    for batch in tqdm(dataloader, f"Undistorting images ({scene_name})"):
        for index, frame_name in enumerate(batch["frame_name"]):
            new_frame = get_frame(undistorted_scene_meta, frame_name)
            if not scene_meta["shared_intrinsics"]:
                # If the camera model is  defined only globally, fetch it
                if "camera_model" not in new_frame:
                    new_frame["camera_model"] = scene_meta["camera_model"]
                new_K, new_w, new_h, map1, map2, roi = undistort_precompute(
                    cam_meta=new_frame
                )
                new_frame = update_camera_meta(new_frame, new_K, new_w, new_h)
            for frame_modality in cfg.frame_modalities:
                modality = batch[frame_modality][index]
                # --- undistort modality ---
                if "mask" in frame_modality:
                    if torch.all(modality > 0):
                        # No invalid pixels. Just use empty mask
                        undistorted_modality = (
                            np.zeros((new_h, new_w), dtype=np.uint8) + 255
                        )
                    else:
                        undistorted_modality = cv2.remap(
                            modality.numpy().astype(np.uint8),
                            map1,
                            map2,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255,
                        )
                        undistorted_modality[undistorted_modality < 255] = 0
                else:
                    # Specific interpolation settings for depth / other modalities
                    if "depth" in frame_modality:
                        undistortion_args = {
                            "interpolation": cv2.INTER_NEAREST,
                            "borderMode": cv2.BORDER_CONSTANT,
                            "borderValue": -1,
                        }
                        modality = modality.unsqueeze(0)
                    else:
                        undistortion_args = {
                            "interpolation": cv2.INTER_LINEAR,
                            "borderMode": cv2.BORDER_REFLECT_101,
                        }

                    # Undistort image or map
                    undistorted_modality = cv2.remap(
                        modality.numpy().transpose(1, 2, 0),
                        map1,
                        map2,
                        **undistortion_args,
                    )

                    # Crop after undistort
                    if roi is not None:
                        x, y, new_w, new_h = roi
                        undistorted_modality = undistorted_modality[
                            y : y + new_h, x : x + new_w
                        ]

                # --- Store meta info for undistorted modality ---

                # New modality named/stored without "_distorted"
                new_modality_name = modality_mapping[frame_modality]
                new_modality_path = new_frame[frame_modality].replace("_distorted", "")
                if cfg.get("store_as_jpg", True):
                    # by default store jpgs
                    new_modality_path = str(Path(new_modality_path).with_suffix(".jpg"))

                # Add new modality to frame and remove distorted
                new_frame[new_modality_name] = new_modality_path
                del new_frame[frame_modality]
                if new_modality_name == "image":
                    # for nerfstudio compatibility
                    new_frame["file_path"] = new_modality_path
                set_frame(undistorted_scene_meta, frame_name, new_frame, sort=True)

                # Store undistorted modality
                new_path = Path(cfg.root) / scene_name / new_modality_path
                if "mask" in frame_modality:
                    format_type = "binary"
                else:
                    format_type = None
                store_data(new_path, undistorted_modality, format_type=format_type)

    # --- Update frame_modalities with undistorted modalities ---
    frame_modalities = {}
    for frame_modality in cfg.frame_modalities:
        undistorted_frame_modality = modality_mapping[frame_modality]
        frame_modalities[undistorted_frame_modality] = {
            "frame_key": undistorted_frame_modality,
            "format": scene_meta["frame_modalities"][frame_modality]["format"],
        }
    undistorted_scene_meta["frame_modalities"] = frame_modalities

    # Store new scene_meta
    scene_meta_path = Path(cfg.root) / scene_name / "scene_meta.json"
    store_data(scene_meta_path, undistorted_scene_meta, "scene_meta")


if __name__ == "__main__":
    logger.info("Undistorting using config:")
    cfg = argconf_parse()  # e.g. configs/undistortion/scannetppv2.yaml
    logger.info(dict(cfg))

    scene_names = get_scene_names(
        cfg, shuffle=cfg.get("random_scene_processing_order", True)
    )
    for scene_name in tqdm(scene_names, "Undistorting scenes"):
        try:
            scene_root = Path(cfg.root) / scene_name
            with SceneProcessLock(scene_root):
                logger.info(f"Processing: {scene_name}")
                set_processing_state(scene_root, "undistortion", "running")
                undistort_scene(cfg, scene_name)
                set_processing_state(scene_root, "undistortion", "finished")
        except Exception:
            trace_message = traceback.format_exc()
            logger.error(
                f"Undistortion failed on scene: {scene_name} \n{trace_message}"
            )
            set_processing_state(
                scene_root,
                "undistortion",
                "failed",
                message=trace_message,
            )
            continue
