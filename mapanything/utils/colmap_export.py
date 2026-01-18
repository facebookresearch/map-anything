# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
COLMAP export utilities for MapAnything.

This module provides functions to export MapAnything predictions to COLMAP format
with scene-adaptive voxel downsampling and efficient backprojection for Point2D mapping.
"""

import os

import numpy as np
import trimesh
from PIL import Image

from mapanything.utils.geometry import closed_form_pose_inverse

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import pycolmap

    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False


def voxel_downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_fraction: float = 0.01,
    voxel_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud with scene-adaptive or explicit voxel size.

    If voxel_size is provided, it is used directly. Otherwise, the voxel size
    is computed adaptively using the interquartile range (IQR) of point positions:
        voxel_size = iqr_extent * voxel_fraction

    Using IQR instead of full bounding box extent makes the method robust to
    outliers and large depth variations (e.g., landscape scenes with 1m to 1000m depth).

    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255 uint8 or 0-1 float)
        voxel_fraction: Fraction of IQR extent to use as voxel size (default: 0.01 = 1%)
        voxel_size: Explicit voxel size in meters (overrides voxel_fraction if provided)

    Returns:
        tuple: (downsampled_points, downsampled_colors)
            - downsampled_points: (M, 3) array of downsampled 3D points
            - downsampled_colors: (M, 3) array of corresponding colors (uint8)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError(
            "open3d is required for voxel downsampling. "
            "Install it with: pip install open3d"
        )

    if len(points) == 0:
        return points, colors

    if voxel_size is not None:
        # Use explicit voxel size
        print(f"Using explicit voxel size: {voxel_size:.4f}m")
    else:
        # Compute scene extent using IQR (robust to outliers)
        # This handles landscape scenes with large depth variations better
        q25 = np.percentile(points, 25, axis=0)
        q75 = np.percentile(points, 75, axis=0)
        iqr_extent = (q75 - q25).max()

        # Also compute full extent for reference
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        full_extent = (bbox_max - bbox_min).max()

        # Use IQR-based extent if valid, otherwise fall back to full extent
        if iqr_extent > 0:
            # Scale up IQR to approximate useful scene range
            # IQR covers ~50% of data, so multiply by 2 for better coverage
            scene_extent = iqr_extent * 2
        else:
            scene_extent = full_extent

        # Compute adaptive voxel size
        voxel_size = scene_extent * voxel_fraction

        # Ensure voxel size is positive
        if voxel_size <= 0:
            voxel_size = 0.01  # Fallback to 1cm if extent is zero

        print(
            f"Scene extent (IQR-based): {scene_extent:.3f}m, full extent: {full_extent:.3f}m"
        )
        print(f"Adaptive voxel size: {voxel_size:.4f}m")

    # Normalize colors to [0, 1] if needed
    if colors.dtype == np.uint8:
        colors_normalized = colors.astype(np.float64) / 255.0
    else:
        colors_normalized = colors.astype(np.float64)
        if colors_normalized.max() > 1.0:
            colors_normalized = colors_normalized / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    # Voxel downsample
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    # Extract downsampled points and colors
    downsampled_points = np.asarray(pcd_downsampled.points)
    downsampled_colors = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)

    print(f"Downsampled from {len(points)} to {len(downsampled_points)} points")

    return downsampled_points, downsampled_colors


def backproject_points_to_frames(
    points_3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[list[tuple[int, float, float]]]:
    """
    Backproject 3D points to all frames to find Point2D observations.

    For each frame, this function:
    1. Transforms points to camera space (vectorized)
    2. Frustum culls points with z <= 0
    3. Projects to 2D with intrinsics
    4. Filters points outside image bounds

    Args:
        points_3d: (P, 3) array of 3D points in world coordinates
        extrinsics: (N, 3, 4) array of world2cam transforms [R|t]
        intrinsics: (N, 3, 3) array of camera intrinsic matrices
        image_width: Width of images in pixels
        image_height: Height of images in pixels

    Returns:
        List of length N, where each element is a list of (point3D_id, u, v) tuples
        for points that project into that frame. point3D_id is 1-indexed for COLMAP.
    """
    num_frames = extrinsics.shape[0]
    num_points = points_3d.shape[0]

    # Convert points to homogeneous coordinates (P, 4)
    points_homo = np.hstack([points_3d, np.ones((num_points, 1))])

    observations_per_frame = []

    for frame_idx in range(num_frames):
        # Get camera parameters
        ext = extrinsics[frame_idx]  # (3, 4)
        K = intrinsics[frame_idx]  # (3, 3)

        # Transform to camera space: X_cam = [R|t] @ X_world_homo
        # ext is (3, 4), points_homo.T is (4, P) -> result is (3, P)
        points_cam = ext @ points_homo.T  # (3, P)

        # Frustum cull: keep points in front of camera (z > 0)
        z = points_cam[2, :]
        in_front_mask = z > 0

        if not np.any(in_front_mask):
            observations_per_frame.append([])
            continue

        # Get valid points and their indices
        valid_indices = np.where(in_front_mask)[0]
        points_cam_valid = points_cam[:, in_front_mask]  # (3, M)
        z_valid = z[in_front_mask]

        # Project to 2D: uv_homo = K @ X_cam
        uv_homo = K @ points_cam_valid  # (3, M)

        # Normalize by z to get pixel coordinates
        u = uv_homo[0, :] / z_valid
        v = uv_homo[1, :] / z_valid

        # Bounds check
        in_bounds_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

        # Collect valid observations
        frame_observations = []
        for i, (idx, uu, vv) in enumerate(
            zip(valid_indices[in_bounds_mask], u[in_bounds_mask], v[in_bounds_mask])
        ):
            # point3D_id is 1-indexed for COLMAP
            point3d_id = int(idx) + 1
            frame_observations.append((point3d_id, float(uu), float(vv)))

        observations_per_frame.append(frame_observations)

    return observations_per_frame


def _build_pycolmap_intrinsics(
    intrinsics: np.ndarray,
    camera_type: str = "PINHOLE",
) -> np.ndarray:
    """
    Build pycolmap camera intrinsics array from 3x3 intrinsic matrix.

    Args:
        intrinsics: (3, 3) camera intrinsic matrix
        camera_type: Camera model type ("PINHOLE" or "SIMPLE_PINHOLE")

    Returns:
        numpy array of camera parameters for pycolmap
    """
    if camera_type == "PINHOLE":
        # [fx, fy, cx, cy]
        return np.array(
            [
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            ]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        # [f, cx, cy] - use average of fx and fy
        focal = (intrinsics[0, 0] + intrinsics[1, 1]) / 2
        return np.array(
            [
                focal,
                intrinsics[0, 2],
                intrinsics[1, 2],
            ]
        )
    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")


def build_colmap_reconstruction(
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: list[str] | None = None,
    camera_type: str = "PINHOLE",
    skip_point2d: bool = False,
) -> "pycolmap.Reconstruction":
    """
    Build a pycolmap Reconstruction from MapAnything outputs.

    This function:
    1. Creates cameras and images with poses
    2. Adds 3D points
    3. Backprojects points to get Point2D observations (unless skip_point2d=True)
    4. Links Point2D observations to 3D points via tracks

    Args:
        points_3d: (P, 3) array of 3D points in world coordinates
        points_rgb: (P, 3) array of RGB colors (uint8)
        extrinsics: (N, 3, 4) array of world2cam transforms [R|t]
        intrinsics: (N, 3, 3) array of camera intrinsic matrices
        image_width: Width of images in pixels
        image_height: Height of images in pixels
        image_names: Optional list of image names. If None, uses "image_N.jpg"
        camera_type: Camera model type ("PINHOLE" or "SIMPLE_PINHOLE")
        skip_point2d: If True, skip Point2D backprojection for faster export

    Returns:
        pycolmap.Reconstruction object
    """
    if not PYCOLMAP_AVAILABLE:
        raise ImportError(
            "pycolmap is required for COLMAP export. "
            "Install it with: pip install pycolmap"
        )

    num_frames = extrinsics.shape[0]
    num_points = points_3d.shape[0]

    # Generate default image names if not provided
    if image_names is None:
        image_names = [f"image_{i + 1}.jpg" for i in range(num_frames)]

    # Backproject to get Point2D observations (unless skipped)
    if skip_point2d:
        print("Skipping Point2D backprojection...")
        observations_per_frame = [[] for _ in range(num_frames)]
    else:
        print("Backprojecting points to frames...")
        observations_per_frame = backproject_points_to_frames(
            points_3d, extrinsics, intrinsics, image_width, image_height
        )

    # Create reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add 3D points with empty tracks (will be populated later)
    for point_idx in range(num_points):
        point3d_id = point_idx + 1  # 1-indexed
        reconstruction.add_point3D(
            points_3d[point_idx],
            pycolmap.Track(),
            points_rgb[point_idx],
        )

    # Add cameras and images
    for frame_idx in range(num_frames):
        # Build camera intrinsics
        cam_params = _build_pycolmap_intrinsics(intrinsics[frame_idx], camera_type)

        # Create camera
        camera = pycolmap.Camera(
            model=camera_type,
            width=image_width,
            height=image_height,
            params=cam_params,
            camera_id=frame_idx + 1,
        )
        reconstruction.add_camera(camera)

        # Create image with pose
        ext = extrinsics[frame_idx]  # (3, 4)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(ext[:3, :3]),
            ext[:3, 3],
        )

        image = pycolmap.Image(
            id=frame_idx + 1,
            name=image_names[frame_idx],
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        # Build Point2D list and update tracks
        points2d_list = []
        frame_observations = observations_per_frame[frame_idx]

        for point2d_idx, (point3d_id, u, v) in enumerate(frame_observations):
            # Create Point2D
            points2d_list.append(pycolmap.Point2D(np.array([u, v]), point3d_id))

            # Update track for this 3D point
            track = reconstruction.points3D[point3d_id].track
            track.add_element(frame_idx + 1, point2d_idx)

        # Set points2D on image
        if points2d_list:
            try:
                image.points2D = pycolmap.ListPoint2D(points2d_list)
                image.registered = True
            except Exception as e:
                print(f"Warning: Failed to set points2D for frame {frame_idx}: {e}")
                image.registered = False
        else:
            image.registered = True  # Still registered, just no observations

        reconstruction.add_image(image)

    # Print summary
    total_observations = sum(len(obs) for obs in observations_per_frame)
    print("Built COLMAP reconstruction:")
    print(f"  - {num_frames} images")
    print(f"  - {num_points} 3D points")
    print(f"  - {total_observations} Point2D observations")

    return reconstruction


def export_predictions_to_colmap(
    outputs: list[dict],
    processed_views: list[dict],
    image_names: list[str],
    output_dir: str,
    voxel_fraction: float = 0.01,
    voxel_size: float | None = None,
    data_norm_type: str = "dinov2",
    save_ply: bool = True,
    save_images: bool = True,
    skip_point2d: bool = False,
) -> "pycolmap.Reconstruction":
    """
    High-level function to export MapAnything predictions to COLMAP format.

    This is the main entry point for COLMAP export. It:
    1. Collects 3D points and colors from all views
    2. Applies scene-adaptive voxel downsampling
    3. Builds COLMAP reconstruction with proper Point2D observations (unless skip_point2d=True)
    4. Saves to disk (including processed images if requested)

    Args:
        outputs: List of prediction dictionaries from model.infer()
        processed_views: List of preprocessed view dictionaries
        image_names: List of original image file names
        output_dir: Directory to save COLMAP outputs
        voxel_fraction: Fraction of IQR-based scene extent for voxel size (default: 0.01 = 1%)
        voxel_size: Explicit voxel size in meters (overrides voxel_fraction if provided)
        data_norm_type: Data normalization type for denormalizing images
        save_ply: Whether to save a PLY file of the point cloud
        save_images: Whether to save processed images to output_dir/images/
        skip_point2d: If True, skip Point2D backprojection for faster export

    Returns:
        pycolmap.Reconstruction object
    """
    num_frames = len(outputs)

    # Collect data from outputs
    all_points = []
    all_colors = []
    all_masks = []

    intrinsics_list = []
    extrinsics_list = []

    for i in range(num_frames):
        pred = outputs[i]

        # Get 3D points and mask
        pts3d = pred["pts3d"][0].cpu().numpy()  # (H, W, 3)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)

        # Filter by valid depth (camera Z, not world Z)
        depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
        valid_depth_mask = depth_z > 0
        combined_mask = mask & valid_depth_mask

        # Get colors from denormalized image
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3) in [0, 1]
        colors = (img_no_norm * 255).astype(np.uint8)

        # Collect valid points and colors
        all_points.append(pts3d[combined_mask])
        all_colors.append(colors[combined_mask])
        all_masks.append(combined_mask)

        # Collect camera parameters
        intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())
        # Convert cam2world to world2cam for COLMAP
        cam2world = pred["camera_poses"][0].cpu().numpy()
        world2cam = closed_form_pose_inverse(cam2world[None])[0]
        extrinsics_list.append(world2cam[:3, :4])

    # Stack camera parameters
    intrinsics = np.stack(intrinsics_list)  # (N, 3, 3)
    extrinsics = np.stack(extrinsics_list)  # (N, 3, 4)

    # Get image size from first output
    h, w = outputs[0]["pts3d"][0].shape[:2]

    # Concatenate all points and colors
    all_points_concat = np.concatenate(all_points, axis=0)
    all_colors_concat = np.concatenate(all_colors, axis=0)

    print(f"Total points before downsampling: {len(all_points_concat)}")

    # Voxel downsample
    downsampled_points, downsampled_colors = voxel_downsample_point_cloud(
        all_points_concat, all_colors_concat, voxel_fraction, voxel_size
    )

    # Build COLMAP reconstruction
    reconstruction = build_colmap_reconstruction(
        points_3d=downsampled_points,
        points_rgb=downsampled_colors,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_width=w,
        image_height=h,
        image_names=image_names,
        camera_type="PINHOLE",
        skip_point2d=skip_point2d,
    )

    # Save reconstruction
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)
    print(f"Saved COLMAP reconstruction to: {sparse_dir}")

    # Optionally save PLY file
    if save_ply:
        ply_path = os.path.join(sparse_dir, "points.ply")
        trimesh.PointCloud(downsampled_points, colors=downsampled_colors).export(
            ply_path
        )
        print(f"Saved point cloud PLY to: {ply_path}")

    # Optionally save processed images
    if save_images:
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        for i in range(num_frames):
            img_no_norm = (
                outputs[i]["img_no_norm"][0].cpu().numpy()
            )  # (H, W, 3) in [0, 1]
            img_uint8 = (img_no_norm * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)

            # Save with original image name
            img_path = os.path.join(images_dir, image_names[i])
            img_pil.save(img_path, quality=95)

        print(f"Saved {num_frames} processed images to: {images_dir}")

    return reconstruction
