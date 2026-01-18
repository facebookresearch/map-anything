# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Note: We use an internal version of the ASE dataset (ASE-V2) and it's format/data might not
# necessarily align with the official public [ASE-V1](https://www.projectaria.com/datasets/ase/#download-dataset).
# Hence, unfortunately, it might not be possible to reproduce the WAI version of ASE.

import logging
import os
import re
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from argconf import argconf_parse
from PIL import Image
from projectaria_tools.core import calibration
from projectaria_tools.core.calibration import device_calibration_from_json_string
from projectaria_tools.core.image import InterpolationMethod
from scipy.spatial.transform import Rotation
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.mapper import DistanceToDepthConverter
from wai_processing.utils.state import set_processing_state
from wai_processing.utils.wrapper import convert_scenes_wrapper

from mapanything.utils.wai.camera import PINHOLE_CAM_KEYS, rotate_pinhole_90degcw
from mapanything.utils.wai.core import load_data, store_data
from mapanything.utils.wai.scene_frame import _filter_scenes

MAX_UINT_16: int = np.iinfo(np.uint16).max
RGB_IMAGE_SIZE = 704
SUPPORTED_SENSORS = ["camera-rgb", "camera-slam-left", "camera-slam-right"]

sensor_name_to_render_dir = {
    "camera-slam-left": "0",
    "camera-slam-right": "1",
    "camera-rgb": "2",
}

rot90 = np.array(
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    dtype=np.float32,
)

logger = logging.getLogger("ase")


def get_original_scene_names(cfg):
    """
    Get all scenes that should be converted:
    1. Get all available scenes from the source / original dataset
    2. Filter these scenes with the scene filters set in the config
        yaml, e.g. usually filter out scenes with conversion status
        'finished'.
    3. Special case for ASE: Filter out scenes with corrupted source
        scenes, e.g. the renderings in the source dataset are missing.
    """
    # Get the filtered scene names
    scene_config_root = cfg.get("original_root_scene_config_path") or cfg.original_root
    original_root = Path(cfg.original_root)
    root = Path(cfg.root)
    original_scene_names = [
        path.name
        for path in original_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    scene_names = _filter_scenes(root, original_scene_names, cfg.get("scene_filters"))
    scene_names = list(scene_names)
    scene_names_filtered = []
    for scene_name in scene_names:
        render_path = Path(cfg.original_root) / scene_name / "render" / "images"
        scene_config_path = Path(scene_config_root) / scene_name / "scene_config.json"
        if render_path.exists() and scene_config_path.exists():
            scene_names_filtered.append(scene_name)
        else:
            # TODO: silently skip scenes without rendering data for now
            set_processing_state(
                Path(cfg.root) / scene_name, "conversion", "skipped", create=True
            )
    return scene_names_filtered


def rt_transformation_matrix(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """
    Computes a transformation matrix from a rotation-translation pair.
    Args:
        rotation: a rotation matrix with shape (3,3)
        translation: a translation vector with shape (3,) or (3,1)
    Returns:
        a transformation matrix with shape (4x4)
    """
    if translation.shape[-1] == 1:
        translation = translation.squeeze(axis=-1)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def _read_trajectory_line(line):
    """Parses a line from the ground truth trajectory file."""
    line = line.rstrip().split(",")
    pose = {}
    pose["timestamp"] = int(line[1])
    translation = np.array([float(p) for p in line[3:6]])
    quat_xyzw = np.array([float(o) for o in line[6:10]])
    rotation = Rotation.from_quat(quat_xyzw).as_matrix()
    pose["transform"] = rt_transformation_matrix(rotation, translation)
    return pose


def read_trajectory_file(fname: Path) -> dict[str, np.ndarray]:
    """Reads a ground truth trajectory file."""
    if not fname.exists():
        raise IOError(f"Could not find trajectory file: {str(fname)}")

    with open(fname, "r") as f:
        _ = f.readline()  # skip header
        transforms = []
        timestamps = []
        for line in f.readlines():
            pose = _read_trajectory_line(line)
            transforms.append(pose["transform"])
            timestamps.append(pose["timestamp"])
        transforms = np.stack(transforms)
        timestamps = np.array(timestamps)
    logger.info(f"Loaded trajectory with {len(timestamps)} device poses.")
    return {
        "Ts_world_from_device": transforms,
        "timestamps": timestamps,
    }


def process_sensor(
    trajectory: dict[str, np.ndarray],
    sensors_calib: calibration.DeviceCalibration,
    sensor_name: str,
    render_path: Path,
    rotate_to_portrait: bool,
    out_path: Path,
):
    # Relative pose of the sensor w.r.t. the Aria device coordinate system
    cam_calib = sensors_calib.get_camera_calib(sensor_name)
    T_device_from_camera = cam_calib.get_transform_device_camera().to_matrix()
    if rotate_to_portrait:
        T_device_from_camera[:3, :3] = T_device_from_camera[:3, :3] @ rot90

    # Cam-to-world transform for N images with shape (N,4,4)
    cam2worlds = np.array(
        [ts @ T_device_from_camera for ts in trajectory["Ts_world_from_device"]]
    )

    # BUGFIX: image size is wrongly stored as (2880, 2880) for camera-rgb
    image_size = (
        (RGB_IMAGE_SIZE, RGB_IMAGE_SIZE)
        if sensor_name == "camera-rgb"
        else cam_calib.get_image_size().tolist()
    )

    # Target intrinsics (before optional camera rotation)
    pinhole = calibration.get_linear_camera_calibration(
        image_width=image_size[0],
        image_height=image_size[1],
        focal_length=cam_calib.get_focal_lengths()[0],
    )

    # Utility to convert range images to depth maps
    fx, fy = pinhole.get_focal_lengths().tolist()
    cx, cy = pinhole.get_principal_point().tolist()
    W, H = pinhole.get_image_size().tolist()
    dist2depth_converter = DistanceToDepthConverter(W, H, fx, fy, cx, cy, "PINHOLE")

    # Intrinsics of the final wai frames
    wai_intrinsics = {}
    if rotate_to_portrait:
        W, H, fx, fy, cx, cy = rotate_pinhole_90degcw(W, H, fx, fy, cx, cy)

    wai_intrinsics["w"] = W
    wai_intrinsics["h"] = H
    wai_intrinsics["fl_x"] = fx
    wai_intrinsics["fl_y"] = fy
    wai_intrinsics["cx"] = cx
    wai_intrinsics["cy"] = cy

    # Sanity check: Do we have a pose for every image?
    sensor_render_path = render_path / sensor_name_to_render_dir[sensor_name]
    n_poses = cam2worlds.shape[0]
    rgb_paths = sorted(list(sensor_render_path.glob("rgb*")))
    if n_poses != len(rgb_paths):
        logger.error(
            f"{sensor_name}: found {len(rgb_paths)} RGB images and {n_poses} camera poses."
        )

    # Start conversion and undistortion
    logger.info(f"{sensor_name}: processing {n_poses} images")
    fname_prefix = sensor_name.replace("camera-", "").replace("-", "_")
    wai_frames = []

    for idx, rgb_path in enumerate(rgb_paths):
        frame_idx = re.match(r"rgb(\d+)", rgb_path.stem)[1]  # noqa
        range_path = sensor_render_path / f"depth{frame_idx}.png"

        if not range_path.exists():
            raise IOError(f"{sensor_name}: could not find range image {range_path}")

        # Load data
        img = load_data(rgb_path)
        range_img = np.array(Image.open(range_path)).astype(np.float32)

        # Generate image mask
        mask = np.ones_like(range_img).astype(np.uint8)
        mask[np.logical_or(range_img == 0, range_img == MAX_UINT_16)] = 0
        range_img[mask == 0] = 0

        # Undistort data
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img_undistorted = calibration.distort_by_calibration(
            img, pinhole, cam_calib, InterpolationMethod.BILINEAR
        )
        range_img_undistorted = calibration.distort_by_calibration(
            range_img,
            pinhole,
            cam_calib,
            InterpolationMethod.NEAREST_NEIGHBOR,
        )
        mask_undistorted = calibration.distort_by_calibration(
            mask,
            pinhole,
            cam_calib,
            InterpolationMethod.NEAREST_NEIGHBOR,
        )

        # Convert range image from millimeters to meters
        range_img_undistorted = range_img_undistorted.astype(np.float32) / 1000.0

        # Convert range image (depth along the pixel’s ray direction) to a depth map (depth in the camera’s Z-axis)
        depth_undistorted = dist2depth_converter.distance_to_depth(
            range_img_undistorted
        )

        if rotate_to_portrait:
            # 90 degrees clockwise rotation
            img_undistorted = np.rot90(img_undistorted, axes=(1, 0))
            depth_undistorted = np.rot90(depth_undistorted, axes=(1, 0))
            mask_undistorted = np.rot90(mask_undistorted, axes=(1, 0))

        # Export data
        frame_name = f"{fname_prefix}_{frame_idx}"
        target_image_path = f"images/{frame_name}.jpg"
        target_depth_path = f"depth/{frame_name}.exr"
        target_mask_path = f"masks/{frame_name}.jpg"
        img_undistorted = img_undistorted.astype(np.float32) / 255.0
        store_data(out_path / target_image_path, img_undistorted, "image")
        store_data(out_path / target_depth_path, depth_undistorted, "depth")
        store_data(out_path / target_mask_path, mask_undistorted, "binary")

        # wai frame
        wai_frame = {"frame_name": frame_name}
        wai_frame["image"] = target_image_path
        wai_frame["file_path"] = target_image_path
        wai_frame["depth"] = target_depth_path
        wai_frame["mask_path"] = target_mask_path
        wai_frame["transform_matrix"] = cam2worlds[idx].tolist()
        for camera_key in PINHOLE_CAM_KEYS:
            wai_frame[camera_key] = wai_intrinsics[camera_key]
        wai_frames.append(wai_frame)

    return wai_frames


def convert_ase_scene(
    cfg,
    scene_name: str,
    sensors_calib: calibration.DeviceCalibration,
):
    dataset_name = cfg.get("dataset_name", "ASEv2")
    version = cfg.get("version", "0.1")
    rotate_to_portrait = cfg.get("rotate_to_portrait", True)
    sensor_names = sorted(cfg.get("sensor_names", SUPPORTED_SENSORS))

    root = Path(cfg.original_root) / scene_name
    render_path = root / "render" / "images"
    scene_config_path = (
        Path(cfg.get("original_root_scene_config_path", cfg.original_root))
        / scene_name
        / "scene_config.json"
    )

    if not render_path.exists():
        raise IOError(f"The rendering directory does not exist: {render_path}")
    if not scene_config_path.exists():
        raise IOError(
            f"The scene configuration file does not exist: {scene_config_path}"
        )

    logger.info(f"Started conversion of scene {scene_name}")
    out_path = Path(cfg.root, scene_name)
    for subdir in [
        "images",
        "depth",
        "masks",
    ]:
        os.makedirs(out_path / subdir)

    # Load device extrinsics
    trajectory_file = root / "gt_trajectory_mps.csv"
    trajectory = read_trajectory_file(trajectory_file)

    wai_frames = []

    for sensor_name in sensor_names:
        senor_frames = process_sensor(
            trajectory,
            sensors_calib,
            sensor_name,
            render_path,
            rotate_to_portrait,
            out_path,
        )
        wai_frames = wai_frames + senor_frames

    if len(wai_frames) == 0:
        raise RuntimeError("Processed 0 wai frames")

    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "version": version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
    }

    # Simplification: store intrinsics at the scene_meta level if a single sensor has been processed
    if len(sensor_names) == 1:
        scene_meta["shared_intrinsics"] = True
        for camera_key in PINHOLE_CAM_KEYS:
            scene_meta[camera_key] = wai_frames[0][camera_key]
        for frame in wai_frames:
            for camera_key in PINHOLE_CAM_KEYS:
                del frame[camera_key]

    scene_meta["frames"] = wai_frames
    scene_meta["frame_modalities"] = {
        "image": {"frame_key": "image", "format": "image"},
        "depth": {"frame_key": "depth", "format": "depth"},
        "mask": {"frame_key": "mask_path", "format": "binary"},
    }

    if rotate_to_portrait:
        scene_meta["_applied_transform"] = rot90.tolist()
        scene_meta["_applied_transforms"] = {"image_rotation": rot90.tolist()}
    else:
        # Original data already in opencv convention
        scene_meta["_applied_transform"] = np.eye(4).tolist()
        scene_meta["_applied_transforms"] = {}
    store_data(out_path / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/ase.yaml")

    if "sensor_names" in cfg and any(
        [name not in SUPPORTED_SENSORS for name in cfg.sensor_names]
    ):
        raise RuntimeError(
            f'Sensors "{[cfg.sensor_names]}" is currently not supported.'
        )

    if "original_root_scene_config_path" in cfg:
        if cfg.original_root_scene_config_path is None:
            del cfg.original_root_scene_config_path
        elif not Path(cfg.original_root_scene_config_path).exists():
            raise RuntimeError(
                f'Cannot find "original_root_scene_config_path": {cfg.original_root_scene_config_path}'
            )

    # Preload sensor calibrations
    sensors_calib = device_calibration_from_json_string(
        load_data(cfg.ase_calib_json_path, load_as_string=True)
    )
    assert sensors_calib is not None
    sensor_labels = sensors_calib.get_all_labels()
    sensor_names = sorted(cfg.get("sensor_names", SUPPORTED_SENSORS))
    for name in sensor_names:
        if name not in sensor_labels:
            raise RuntimeError(
                f'Cannot find the sensor calibration of "{name}" in {cfg.ase_calib_json_path}'
            )

    convert_scenes_wrapper(
        convert_ase_scene,
        cfg,
        get_original_scene_names_func=get_original_scene_names,
        sensors_calib=sensors_calib,
    )
