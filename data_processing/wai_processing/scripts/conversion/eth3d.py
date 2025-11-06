# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import numpy as np
import pycolmap
import torch
from argconf import argconf_parse
from natsort import natsorted
from PIL import Image
from scipy.spatial.transform import Rotation
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,  # noqa: F401, Needed for launch_slurm.py
)

from morphcloud.utils.wai.core import load_data, store_data

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def load_eth3d_raw_depth(path):
    """Function to load ETH3D raw depth"""
    height, width = 4032, 6048
    depth = np.fromfile(path, dtype=np.float32).reshape(height, width)
    depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0, nan=0.0)
    return depth


def pose_matrix_from_quaternion(pvec):
    """
    Get 4x4 pose matrix from quaternion (t, q)
    t = (tx, ty, tz)
    q = (qw, qx, qy, qz)
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_quat(pvec[3:], scalar_first=True).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose


def adjust_camera_params_for_rotation(camera_params, original_size, k):
    """
    Adjust camera parameters for rotation.

    Args:
        camera_params: Camera parameters [fx, fy, cx, cy, ...]
        original_size: Original image size as (width, height)
        k: Number of 90-degree rotations counter-clockwise (k=3 means 90 degrees clockwise)

    Returns:
        Adjusted camera parameters
    """
    fx, fy, cx, cy = camera_params[:4]
    width, height = original_size

    if k % 4 == 1:  # 90 degrees counter-clockwise
        new_fx, new_fy = fy, fx
        new_cx, new_cy = height - cy, cx
    elif k % 4 == 2:  # 180 degrees
        new_fx, new_fy = fx, fy
        new_cx, new_cy = width - cx, height - cy
    elif k % 4 == 3:  # 90 degrees clockwise (270 counter-clockwise)
        new_fx, new_fy = fy, fx
        new_cx, new_cy = cy, width - cx
    else:  # No rotation
        return camera_params

    adjusted_params = [new_fx, new_fy, new_cx, new_cy]
    if len(camera_params) > 4:
        adjusted_params.extend(camera_params[4:])

    return adjusted_params


def adjust_pose_for_rotation(pose, k):
    """
    Adjust camera pose for rotation.

    Args:
        pose: 4x4 camera pose matrix (camera-to-world, OpenCV convention - X right, Y down, Z forward)
        k: Number of 90-degree rotations counter-clockwise (k=3 means 90 degrees clockwise)

    Returns:
        Adjusted 4x4 camera pose matrix
    """
    # Create rotation matrices for different rotations
    if k % 4 == 1:  # 90 degrees counter-clockwise
        rot_transform = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    elif k % 4 == 2:  # 180 degrees
        rot_transform = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif k % 4 == 3:  # 90 degrees clockwise (270 counter-clockwise)
        rot_transform = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    else:  # No rotation
        return pose

    # Apply the transformation to the pose
    adjusted_pose = pose.copy()
    adjusted_pose[:3, :3] = adjusted_pose[:3, :3] @ rot_transform.T

    return adjusted_pose


def find_non_gravity_aligned_poses(cam2world_poses, threshold_angle=30):
    """
    Find camera poses that are not aligned with gravity.
    Assumes that majority of the cameras in a scene are gravity aligned.

    Parameters:
    -----------
    cam2world_poses : numpy.ndarray
        Array of 4x4 camera-to-world transformation matrices in OpenCV convention
        (X right, Y down, Z forward).
    threshold_angle : float, optional
        Threshold angle in degrees. Poses with Y-axis deviating from the
        estimated gravity direction by more than this angle are considered
        non-gravity-aligned. Default is 15 degrees.

    Returns:
    --------
    numpy.ndarray
        Indices of poses that are not gravity-aligned.
    numpy.ndarray
        Estimated gravity direction in the world frame.
    numpy.ndarray
        Angles (in degrees) between each pose's Y-axis and the estimated gravity.
    """
    # Extract Y-axes from all poses (in world coordinates)
    y_axes = np.array([pose[:3, 1] for pose in cam2world_poses])

    # Normalize Y-axes
    norm = np.linalg.norm(y_axes, axis=1, keepdims=True)
    y_axes = y_axes / norm

    # Compute the pairwise dot products between all y-axes
    dot_products = np.abs(y_axes @ y_axes.T)

    # For each y-axis, count how many other y-axes are close to it
    close_counts = np.sum(dot_products > np.cos(np.radians(threshold_angle)), axis=1)

    # The y-axis with the most "close" neighbors is likely the gravity direction
    best_idx = np.argmax(close_counts)
    gravity_direction = y_axes[best_idx]

    # Calculate angle between each Y-axis and the estimated gravity direction
    dot_products = np.clip(np.abs(np.dot(y_axes, gravity_direction)), -1.0, 1.0)
    angles = np.degrees(np.arccos(dot_products))

    # Find poses with angle greater than threshold
    non_gravity_aligned_indices = np.where(angles > threshold_angle)[0]

    return non_gravity_aligned_indices, gravity_direction, angles


def detect_non_gravity_aligned_poses(scene_folder):
    """
    Detect non-gravity-aligned poses in a scene.

    Args:
        scene_folder: Path to the scene folder
    """
    # Read images.txt for distorted images
    images_txt_path = os.path.join(
        scene_folder, "dslr_calibration_undistorted", "images.txt"
    )
    with open(images_txt_path, "r") as f:
        image_lines = f.readlines()[4:]  # Skip header

    # Load the Opencv Camera2World poses of all the images
    all_opencv_c2w_poses = []
    names = []
    for i in range(0, len(image_lines), 2):
        parts = image_lines[i].strip().split()
        _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts[:10]
        names.append(name)
        pose = pose_matrix_from_quaternion([tx, ty, tz, qw, qx, qy, qz])
        c2w_pose = np.linalg.inv(pose)
        all_opencv_c2w_poses.append(c2w_pose)

    # Stack the poses
    all_opencv_c2w_poses = np.stack(all_opencv_c2w_poses, axis=0)

    # Detect non-gravity-aligned poses
    non_gravity_aligned_indices, gravity_direction, angles = (
        find_non_gravity_aligned_poses(all_opencv_c2w_poses)
    )

    # Print the names of the non-gravity-aligned poses
    print(f"Scene: {scene_folder}")
    if len(non_gravity_aligned_indices) > 0:
        print(
            f"Found {len(non_gravity_aligned_indices)} images that are not gravity aligned in scene"
        )
        non_gravity_aligned_names = [
            name
            for name_idx, name in enumerate(names)
            if name_idx in non_gravity_aligned_indices
        ]
        non_gravity_aligned_names = natsorted(non_gravity_aligned_names)
        print(f"Non-gravity aligned images: {non_gravity_aligned_names}")
    else:
        print("All images are gravity aligned")


# Dictionary of images that were originally portrait but are now landscape in the ETH3D dataset
originally_portrait_imgs_in_eth3d_dataset = {
    "delivery_area": ["DSC_0711.JPG", "DSC_0712.JPG", "DSC_0713.JPG", "DSC_0714.JPG"],
    "playground": [
        "DSC_0587.JPG",
        "DSC_0588.JPG",
        "DSC_0589.JPG",
        "DSC_0590.JPG",
        "DSC_0591.JPG",
        "DSC_0592.JPG",
    ],
    "relief": [
        "DSC_0427.JPG",
        "DSC_0428.JPG",
        "DSC_0429.JPG",
        "DSC_0430.JPG",
        "DSC_0431.JPG",
        "DSC_0432.JPG",
        "DSC_0433.JPG",
        "DSC_0434.JPG",
        "DSC_0435.JPG",
        "DSC_0436.JPG",
        "DSC_0437.JPG",
        "DSC_0438.JPG",
        "DSC_0439.JPG",
    ],
    "relief_2": [
        "DSC_0458.JPG",
        "DSC_0459.JPG",
        "DSC_0460.JPG",
        "DSC_0461.JPG",
        "DSC_0462.JPG",
        "DSC_0463.JPG",
        "DSC_0464.JPG",
        "DSC_0465.JPG",
        "DSC_0466.JPG",
        "DSC_0467.JPG",
        "DSC_0468.JPG",
    ],
}


# TODO: Move this functionaliy into the undistortion stage
def undistort_depth_maps(scene_folder):
    """
    Undistort depth maps using camera calibration data:

    1) For a scene (<scene_folder> = <root>/<scene_name>), read the images.txt file under the 'dslr_calibration_jpg' folder.
       The format of the images.txt file is as follows (after 4 lines of header):
       IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
       POINTS2D[] as (X, Y, POINT3D_ID)
       We only care about the IMAGE_ID based rows in the file (i.e., lines[::2]).

    2) Load the cameras.txt file under the 'dslr_calibration_jpg' folder.
       The format of the cameras.txt file is as follows (after 3 lines of header):
       CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
       Create a dictionary of camera_id to pycolmap.Camera THIN_PRISM_FISHEYE model using the camera_params mapping.

    3) Load the cameras.txt file under the 'dslr_calibration_undistorted' folder.
       The format of the cameras.txt file is as follows (after 3 lines of header):
       CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
       Create a dictionary of camera_id to pycolmap.Camera PINHOLE model using the camera_params mapping.

    4) For the PINHOLE pycolmap models, compute the point in image plane to world by using:
       pinhole_world_pts = cam_from_img(self: pycolmap.Camera, image_points: numpy.ndarray[numpy.float64[m, 2]])
       → numpy.ndarray[numpy.float64[m, 2]].

    5) Using the THIN_PRISM_FISHEYE model, compute the projection of pinhole_world_pts to distorted image plane:
       distorted_img_coords = img_from_cam(self: pycolmap.Camera, cam_points: numpy.ndarray[numpy.float64[m, 2]])
       → numpy.ndarray[numpy.float64[m, 2]]

    6) Using the distorted_img_coords, find the undistorted depth values:
       - For each IMAGE_ID based row in the images.txt file, read the corresponding depth map
       - Get corresponding distorted_img_coords using the respective camera_id
       - The path to the depth map is <scene_folder>/ground_truth_depth/NAME
       - Get the undistorted_depth as depth[distorted_img_coords]
       - Save the depth to <scene_folder>/undistorted_depth/NAME

    Args:
        scene_folder: Path to the scene folder
    """
    print(f"Undistorting depth maps for {scene_folder.name}...")

    # Check if undistorted depth directory exists
    undistorted_depth_dir = (
        scene_folder / "ground_truth_depth" / "dslr_images_undistorted"
    )
    undistorted_depth_dir.mkdir(parents=True, exist_ok=True)

    # Read cameras.txt for distorted cameras
    distorted_cameras_txt_path = scene_folder / "dslr_calibration_jpg" / "cameras.txt"
    with open(distorted_cameras_txt_path, "r") as f:
        distorted_camera_lines = f.readlines()[3:]  # Skip header

    # Create camera_id to camera_params mapping for distorted cameras
    distorted_camera_params_dict = {}
    for line in distorted_camera_lines:
        parts = line.strip().split()
        distorted_camera_id = int(parts[0])
        distorted_params = list(map(float, parts[4:]))
        distorted_camera_params_dict[distorted_camera_id] = pycolmap.Camera(
            model="THIN_PRISM_FISHEYE",
            width=int(parts[2]),
            height=int(parts[3]),
            params=distorted_params,
        )

    # Read cameras.txt for undistorted cameras
    undistorted_cameras_txt_path = (
        scene_folder / "dslr_calibration_undistorted" / "cameras.txt"
    )
    with open(undistorted_cameras_txt_path, "r") as f:
        undistorted_camera_lines = f.readlines()[3:]  # Skip header

    # Create camera_id to camera_params mapping for undistorted cameras
    undistorted_camera_params_dict = {}
    for line in undistorted_camera_lines:
        parts = line.strip().split()
        undistorted_camera_id = int(parts[0])
        undistorted_params = list(map(float, parts[4:]))
        undistorted_camera_params_dict[undistorted_camera_id] = pycolmap.Camera(
            model="PINHOLE",
            width=int(parts[2]),
            height=int(parts[3]),
            params=undistorted_params,
        )

    # Precompute distorted image coordinates for each camera ID
    distorted_img_coords_dict = {}
    for camera_id, undistorted_camera in undistorted_camera_params_dict.items():
        # Generate image points grid
        height, width = undistorted_camera.height, undistorted_camera.width
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        image_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        # Compute world points from image points using the undistorted camera
        pinhole_world_pts = undistorted_camera.cam_from_img(image_points)

        # Get the distorted camera
        distorted_camera = distorted_camera_params_dict[camera_id]

        # Project world points to distorted image plane
        distorted_img_coords = distorted_camera.img_from_cam(pinhole_world_pts)
        distorted_img_coords = np.clip(distorted_img_coords, 0, [width - 1, height - 1])
        distorted_img_coords = distorted_img_coords.astype(int)

        # Store the precomputed coordinates
        distorted_img_coords_dict[camera_id] = distorted_img_coords

    # Read images.txt for distorted images
    distorted_images_txt_path = scene_folder / "dslr_calibration_jpg" / "images.txt"
    with open(distorted_images_txt_path, "r") as f:
        distorted_image_lines = f.readlines()[4:]  # Skip header

    # Process each image
    for i in range(0, len(distorted_image_lines), 2):
        parts = distorted_image_lines[i].strip().split()
        _, _, _, _, _, _, _, _, camera_id, image_name = parts[:10]
        camera_id = int(camera_id)
        base_name = os.path.basename(image_name)

        # Check if undistorted depth already exists
        undistorted_depth_path = undistorted_depth_dir / base_name.replace(
            ".JPG", ".exr"
        )
        if undistorted_depth_path.exists():
            continue

        # Load the corresponding depth map
        depth_map_path = scene_folder / "ground_truth_depth" / "dslr_images" / base_name
        if not depth_map_path.exists():
            print(f"Warning: Depth map not found for {base_name}, skipping...")
            continue

        depth_map = load_eth3d_raw_depth(depth_map_path)

        # Retrieve precomputed distorted image coordinates
        if camera_id not in distorted_img_coords_dict:
            print(
                f"Warning: Camera ID {camera_id} not found in distorted_img_coords_dict, skipping..."
            )
            continue
        distorted_img_coords = distorted_img_coords_dict[camera_id]

        # Get undistorted depth values
        undistorted_depth = depth_map[
            distorted_img_coords[:, 1], distorted_img_coords[:, 0]
        ]

        # Fetch the height and width specific to the current camera_id
        undistorted_camera = undistorted_camera_params_dict[camera_id]
        height, width = undistorted_camera.height, undistorted_camera.width

        # Reshape the undistorted depth map to match the original image size
        undistorted_depth = undistorted_depth.reshape(height, width)
        undistorted_depth = undistorted_depth.copy()

        # Save the undistorted depth map as .exr using WAI
        store_data(
            undistorted_depth_path,
            torch.tensor(undistorted_depth),
            "depth",
        )

    print(f"Completed undistorting depthmaps for {scene_folder.name}")


def process_eth3d_scene(cfg, scene_name):
    """
    Process a ETH3D scene into the WAI format:

    1) Undistort depth and save to original raw data folder in WAI default format (.exr)
        - Save to: <scene_folder>/ground_truth_depth/dslr_images_undistorted/

    2) Read the images.txt file under the 'dslr_calibration_undistorted' folder.
       The format of the images.txt file is as follows (after 4 lines of header):
       IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
       POINTS2D[] as (X, Y, POINT3D_ID)
       We only care about the IMAGE_ID based rows in the file (i.e., lines[::2]).

       Then load the cameras.txt files under the 'dslr_calibration_undistorted' folders.
       The format of the cameras.txt file is as follows (after 3 lines of header):
       CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
       We create a dictionary of camera_id to camera_params mapping.

    3) For each IMAGE_ID based row in the images.txt file:
        a) Get the undistorted RGB image, depthmap, camera parameters and pose
            - Undistorted image path: <scene_folder>/images/dslr_images_undistorted/IMG_NAME
            - Undistorted depth map path: <scene_folder>/ground_truth_depth/dslr_images_undistorted/DEPTH_NAME
        b) Determine if the image is originally portrait. If it is, rotate the image and depth map
           by 90 degrees clockwise and adjust the camera parameters and pose accordingly.
        c) Save the image and depth in WAI format. Add camera parameters and pose to the scene meta.

    Expected root directory structure for the raw ETH3D dataset:
    .
    └── eth3d/
        ├── courtyard/
        │   ├── dslr_calibration_jpg/
        │   │   ├── cameras.txt
        │   │   ├── images.txt
        │   │   ├── points3D.txt
        │   ├── dslr_calibration_undistorted/
        │   │   ├── cameras.txt
        │   │   ├── images.txt
        │   │   ├── points3D.txt
        │   ├── ground_truth_depth/
        │   │   ├── dslr_images/
        │   │   |   ├── DSC_0286.JPG
        │   │   |   ├── ...
        │   ├── images/
        │   │   ├── dslr_images/
        │   │   |   ├── DSC_0286.JPG
        │   │   |   ├── ...
        │   │   ├── dslr_images_undistorted/
        │   │   |   ├── DSC_0286.JPG
        │   │   |   ├── ...
        ├── delivery_area/
        │   ├── ...
        ├── ...
    """
    # Set up paths
    scene_root = Path(cfg.original_root) / scene_name
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    depth_dir = target_scene_root / "depth"

    # Create directories
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Check if all undistorted depth maps exist, if not, generate them
    undistorted_depth_dir = (
        scene_root / "ground_truth_depth" / "dslr_images_undistorted"
    )
    undistorted_depth_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all original depth images
    original_depth_dir = scene_root / "ground_truth_depth" / "dslr_images"
    if original_depth_dir.exists():
        original_depth_files = list(original_depth_dir.glob("*.JPG"))

        # Check if all depth maps have been undistorted
        all_undistorted = True
        for orig_depth_file in original_depth_files:
            undistorted_depth_file = (
                undistorted_depth_dir / orig_depth_file.name.replace(".JPG", ".exr")
            )
            if not undistorted_depth_file.exists():
                all_undistorted = False
                break

        # If any depth maps are missing, run undistortion
        if not all_undistorted:
            undistort_depth_maps(scene_root)
    else:
        print(f"Warning: Original depth directory not found for {scene_name}")

    # Initialize list to store frame metadata
    print(f"Processing {scene_name} data to WAI format ...")
    wai_frames = []

    # Read cameras.txt for undistorted cameras
    cameras_txt_path = scene_root / "dslr_calibration_undistorted" / "cameras.txt"
    camera_params_dict = {}
    with open(cameras_txt_path, "r") as f:
        camera_lines = f.readlines()[3:]  # Skip header

    # Create camera_id to camera_params mapping
    for line in camera_lines:
        parts = line.strip().split()
        camera_id = int(parts[0])
        width = int(parts[2])
        height = int(parts[3])
        params = list(map(float, parts[4:]))
        camera_params_dict[camera_id] = {
            "width": width,
            "height": height,
            "params": params,
        }

    # Read images.txt for undistorted images
    images_txt_path = scene_root / "dslr_calibration_undistorted" / "images.txt"
    with open(images_txt_path, "r") as f:
        image_lines = f.readlines()[4:]  # Skip header

    # Process each image
    for i in range(0, len(image_lines), 2):  # Skip POINTS2D lines
        parts = image_lines[i].strip().split()
        _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts[:10]
        camera_id = int(camera_id)
        base_name = os.path.basename(name)

        # Get camera parameters
        camera_info = camera_params_dict[camera_id]
        width = camera_info["width"]
        height = camera_info["height"]
        fx, fy, cx, cy = camera_info["params"]

        # Get paths for undistorted image and depth
        image_path = scene_root / "images" / "dslr_images_undistorted" / base_name
        depth_path = (
            scene_root
            / "ground_truth_depth"
            / "dslr_images_undistorted"
            / base_name.replace(".JPG", ".exr")
        )

        # Skip if image or depth doesn't exist
        if not image_path.exists() or not depth_path.exists():
            print(f"Warning: Missing image or depth for {base_name}, skipping...")
            continue

        # Get camera pose (world to camera)
        w2c_pose = pose_matrix_from_quaternion(
            [
                float(tx),
                float(ty),
                float(tz),
                float(qw),
                float(qx),
                float(qy),
                float(qz),
            ]
        )
        # Convert to OpenCV convention (X right, Y down, Z forward) Camera2World
        c2w_pose = np.linalg.inv(w2c_pose)

        # Use original file name without extension as frame name
        frame_name = os.path.splitext(base_name)[0]
        target_image_name = f"{frame_name}.png"
        rel_target_image_path = Path("images") / target_image_name
        rel_depth_out_path = Path("depth") / f"{frame_name}.exr"

        # Check if image is originally portrait
        is_portrait = False
        if (
            scene_name in originally_portrait_imgs_in_eth3d_dataset
            and base_name in originally_portrait_imgs_in_eth3d_dataset[scene_name]
        ):
            is_portrait = True

            # Load image and depth for rotation
            img = Image.open(image_path)
            depth = load_data(depth_path, "depth").numpy()

            # Rotate image 90 degrees clockwise
            img = img.rotate(-90, expand=True)

            # Rotate depth 90 degrees clockwise
            depth = np.rot90(depth, k=3)
            depth = depth.copy()

            # Adjust camera parameters and pose for rotation (Need to counter-rotate, i.e., rotate 90 degrees counter-clockwise)
            camera_params = [fx, fy, cx, cy]
            adjusted_params = adjust_camera_params_for_rotation(
                camera_params, (width, height), k=1
            )
            fx, fy, cx, cy = adjusted_params
            c2w_pose = adjust_pose_for_rotation(c2w_pose, k=1)

            # Save rotated image to target location
            img.save(target_scene_root / rel_target_image_path)

            # Save rotated depth to target location
            store_data(
                target_scene_root / rel_depth_out_path,
                torch.tensor(depth),
                "depth",
            )

            # Update the height & width
            height = depth.shape[0]
            width = depth.shape[1]
        else:
            # For non-rotated images, create symlinks to the original files
            os.symlink(image_path, target_scene_root / rel_target_image_path)
            os.symlink(depth_path, target_scene_root / rel_depth_out_path)

        # Store WAI frame metadata
        wai_frame = {
            "frame_name": frame_name,
            "image": str(rel_target_image_path),
            "file_path": str(rel_target_image_path),
            "depth": str(rel_depth_out_path),
            "transform_matrix": c2w_pose.tolist(),
            "h": height,
            "w": width,
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "is_portrait": str(is_portrait),
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

    # Save scene metadata
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/eth3d.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(
        process_eth3d_scene,
        cfg,
    )
