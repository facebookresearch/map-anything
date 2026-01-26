# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Visualize COLMAP sparse reconstruction in Rerun.

This script loads a COLMAP reconstruction (from sparse/ folder) and visualizes
the 3D point cloud, camera poses, and optionally images in the Rerun viewer.

Usage:
    # Start Rerun server first
    rerun --serve --port 2004 --web-viewer-port 2006

    # Then run the visualization
    python visualize_colmap_format_in_rerun.py --scene_dir /path/to/scene/

    # Open web viewer at http://127.0.0.1:2006

Expected folder structure:
    scene_dir/
        images/          # Optional - for displaying camera images
            img1.jpg
            img2.jpg
            ...
        sparse/          # Required - COLMAP reconstruction
            cameras.bin (or .txt)
            images.bin (or .txt)
            points3D.bin (or .txt)
"""

import argparse
from pathlib import Path

import numpy as np
import rerun as rr

from mapanything.utils.colmap import read_model
from mapanything.utils.viz import script_add_rerun_args

# Default minimum track length for filtering points
# Note: For MapAnything outputs, most points have low track length (1-2) due to voxel downsampling
# Track length filtering is primarily useful for traditional SfM outputs with matching noise
FILTER_MIN_TRACK_LENGTH = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize COLMAP reconstruction in Rerun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Scene directory containing sparse/ and optionally images/",
    )
    parser.add_argument(
        "--sparse_subdir",
        type=str,
        default="sparse",
        help="Subdirectory name for sparse reconstruction (default: sparse)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        default=False,
        help="Filter points based on track length and color (primarily useful for traditional SfM outputs)",
    )
    parser.add_argument(
        "--min_track_length",
        type=int,
        default=FILTER_MIN_TRACK_LENGTH,
        help="Minimum track length for filtering (default: 1; for SfM outputs try 3-5)",
    )
    parser.add_argument(
        "--show_images",
        action="store_true",
        default=False,
        help="Display camera images in the viewer",
    )
    parser.add_argument(
        "--show_keypoints",
        action="store_true",
        default=False,
        help="Display 2D keypoints on images (requires --show_images)",
    )
    parser.add_argument(
        "--image_plane_distance",
        type=float,
        default=0.5,
        help="Distance of image plane from camera origin for visualization",
    )

    # Add Rerun arguments
    script_add_rerun_args(parser)

    return parser.parse_args()


def log_colmap_reconstruction(
    scene_dir: str,
    sparse_subdir: str = "sparse",
    filter_points: bool = False,
    min_track_length: int = 3,
    show_images: bool = False,
    show_keypoints: bool = False,
    image_plane_distance: float = 0.5,
) -> None:
    """
    Load and log COLMAP reconstruction to Rerun.

    Args:
        scene_dir: Path to scene directory
        sparse_subdir: Name of sparse reconstruction subdirectory
        filter_points: Whether to filter noisy points
        min_track_length: Minimum track length for filtering
        show_images: Whether to display camera images
        show_keypoints: Whether to display 2D keypoints
        image_plane_distance: Distance of image plane from camera
    """
    scene_path = Path(scene_dir)
    sparse_path = scene_path / sparse_subdir
    images_path = scene_path / "images"

    # Check if sparse folder exists
    if not sparse_path.exists():
        raise FileNotFoundError(f"Sparse reconstruction not found: {sparse_path}")

    # Detect format and read model
    print(f"Reading COLMAP reconstruction from: {sparse_path}")

    # Try binary format first, then text
    if (sparse_path / "cameras.bin").exists():
        ext = ".bin"
    elif (sparse_path / "cameras.txt").exists():
        ext = ".txt"
    else:
        raise FileNotFoundError(
            f"No COLMAP model files found in {sparse_path}. "
            "Expected cameras.bin/txt, images.bin/txt, points3D.bin/txt"
        )

    cameras, images, points3D = read_model(sparse_path, ext=ext)

    print(
        f"Loaded: {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points"
    )

    # Filter points if requested
    if filter_points:
        original_count = len(points3D)
        points3D = {
            id: point
            for id, point in points3D.items()
            if point.rgb.any() and len(point.image_ids) >= min_track_length
        }
        print(
            f"Filtered points: {original_count} -> {len(points3D)} (min track length: {min_track_length})"
        )

    # Set up coordinate system (COLMAP uses right-handed Y-down)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Log all 3D points
    if points3D:
        points_xyz = np.array([p.xyz for p in points3D.values()])
        points_rgb = np.array([p.rgb for p in points3D.values()])
        points_error = np.array([p.error for p in points3D.values()])

        rr.log(
            "points",
            rr.Points3D(
                positions=points_xyz,
                colors=points_rgb,
            ),
            rr.AnyValues(reprojection_error=points_error),
            static=True,
        )
        print(f"Logged {len(points_xyz)} 3D points")

    # Log cameras and images
    for _, image in sorted(images.items(), key=lambda x: x[1].name):
        camera = cameras[image.camera_id]

        # Camera pose: COLMAP stores "camera from world" transform
        # Convert quaternion from wxyz (COLMAP) to xyzw (Rerun)
        quat_xyzw = image.qvec[[1, 2, 3, 0]]

        # Log camera transform
        rr.log(
            f"camera/{image.name}",
            rr.Transform3D(
                translation=image.tvec,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                relation=rr.TransformRelation.ChildFromParent,
            ),
            static=True,
        )

        # Log camera view coordinates (Right-Down-Forward)
        rr.log(f"camera/{image.name}", rr.ViewCoordinates.RDF, static=True)

        # Build camera intrinsics based on model type
        if camera.model == "PINHOLE":
            # [fx, fy, cx, cy]
            focal_length = camera.params[:2]
            principal_point = camera.params[2:4]
        elif camera.model == "SIMPLE_PINHOLE":
            # [f, cx, cy]
            focal_length = [camera.params[0], camera.params[0]]
            principal_point = camera.params[1:3]
        elif camera.model in ("SIMPLE_RADIAL", "RADIAL"):
            # [f, cx, cy, k1, ...]
            focal_length = [camera.params[0], camera.params[0]]
            principal_point = camera.params[1:3]
        elif camera.model == "OPENCV":
            # [fx, fy, cx, cy, k1, k2, p1, p2]
            focal_length = camera.params[:2]
            principal_point = camera.params[2:4]
        else:
            # Fallback for other models - use first params
            print(
                f"Warning: Camera model '{camera.model}' not fully supported, using approximate intrinsics"
            )
            focal_length = [camera.params[0], camera.params[0]]
            principal_point = [camera.width / 2, camera.height / 2]

        # Log pinhole camera
        rr.log(
            f"camera/{image.name}/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=focal_length,
                principal_point=principal_point,
                image_plane_distance=image_plane_distance,
            ),
            static=True,
        )

        # Log image if requested and available
        if show_images:
            image_file = images_path / image.name
            if image_file.exists():
                rr.log(
                    f"camera/{image.name}/image",
                    rr.EncodedImage(path=str(image_file)),
                    static=True,
                )

                # Log 2D keypoints if requested
                if show_keypoints and len(image.xys) > 0:
                    # Filter to only show keypoints with valid 3D points
                    valid_mask = np.array(
                        [pid != -1 and pid in points3D for pid in image.point3D_ids]
                    )
                    if valid_mask.any():
                        valid_xys = image.xys[valid_mask]
                        rr.log(
                            f"camera/{image.name}/image/keypoints",
                            rr.Points2D(
                                positions=valid_xys,
                                colors=[34, 138, 167],  # Teal color
                                radii=3.0,
                            ),
                            static=True,
                        )

    print(f"Logged {len(images)} camera poses")


def main():
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("COLMAP Rerun Visualization")
    print("=" * 60)
    print(f"Scene directory: {args.scene_dir}")
    print(f"Sparse subdirectory: {args.sparse_subdir}")
    print(f"Filter points: {args.filter}")
    if args.filter:
        print(f"Min track length: {args.min_track_length}")
    print(f"Show images: {args.show_images}")
    print(f"Show keypoints: {args.show_keypoints}")

    # Initialize Rerun
    rr.script_setup(args, "colmap_visualization")

    # Log the reconstruction
    log_colmap_reconstruction(
        scene_dir=args.scene_dir,
        sparse_subdir=args.sparse_subdir,
        filter_points=args.filter,
        min_track_length=args.min_track_length,
        show_images=args.show_images,
        show_keypoints=args.show_keypoints,
        image_plane_distance=args.image_plane_distance,
    )

    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
