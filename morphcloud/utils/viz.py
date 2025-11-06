# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for visualization
"""

from argparse import ArgumentParser, Namespace
from distutils.util import strtobool

import numpy as np
import rerun as rr
import trimesh

from morphcloud.utils.hf_utils.viz import image_mesh


def log_posed_rgbd_data_to_rerun(
    image, depthmap, pose, intrinsics, base_name, mask=None
):
    """
    Log camera and image data to Rerun visualization tool.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image to be logged
    depthmap : numpy.ndarray
        Depth map corresponding to the image
    pose : numpy.ndarray
        4x4 camera pose matrix with rotation (3x3) and translation (3x1)
    intrinsics : numpy.ndarray
        Camera intrinsic matrix
    base_name : str
        Base name for the logged entities in Rerun
    mask : numpy.ndarray, optional
        Optional segmentation mask for the depth image
    """
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if mask is not None:
        rr.log(
            f"{base_name}/pinhole/depth_mask",
            rr.SegmentationImage(mask),
        )


def str2bool(v):
    return bool(strtobool(v))


def script_add_rerun_args(parser: ArgumentParser) -> None:
    """
    Add common Rerun script arguments to `parser`.

    Change Log from https://github.com/rerun-io/rerun/blob/29eb8954b08e59ff96943dc0677f46f7ea4ea734/rerun_py/rerun_sdk/rerun/script_helpers.py#L65:
        - Added default portforwarding url for ease of use
        - Update parser types

    Parameters
    ----------
    parser : ArgumentParser
        The parser to add arguments to.

    Returns
    -------
    None
    """
    parser.add_argument(
        "--headless",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Don't show GUI",
    )
    parser.add_argument(
        "--connect",
        dest="connect",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Connect to an external viewer",
    )
    parser.add_argument(
        "--serve",
        dest="serve",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Serve a web viewer (WARNING: experimental feature)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="rerun+http://127.0.0.1:2004/proxy",
        help="Connect to this HTTP(S) URL",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save data to a .rrd file at this path"
    )
    parser.add_argument(
        "-o",
        "--stdout",
        dest="stdout",
        action="store_true",
        help="Log data to standard output, to be piped into a Rerun Viewer",
    )


def init_rerun_args(
    headless=True,
    connect=True,
    serve=False,
    url="rerun+http://127.0.0.1:2004/proxy",
    save=None,
    stdout=False,
) -> Namespace:
    """
    Initialize common Rerun script arguments.

    Parameters
    ----------
    headless : bool, optional
        Don't show GUI, by default True
    connect : bool, optional
        Connect to an external viewer, by default True
    serve : bool, optional
        Serve a web viewer (WARNING: experimental feature), by default False
    url : str, optional
        Connect to this HTTP(S) URL, by default rerun+http://127.0.0.1:2004/proxy
    save : str, optional
        Save data to a .rrd file at this path, by default None
    stdout : bool, optional
        Log data to standard output, to be piped into a Rerun Viewer, by default False

    Returns
    -------
    Namespace
        The parsed arguments.
    """
    rerun_args = Namespace()
    rerun_args.headless = headless
    rerun_args.connect = connect
    rerun_args.serve = serve
    rerun_args.url = url
    rerun_args.save = save
    rerun_args.stdout = stdout

    return rerun_args


def predictions_to_glb(
    predictions,
    as_mesh=True,
) -> trimesh.Scene:
    """
    Converts predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (V, H, W, 3)
            - images: Input images (V, H, W, 3)
            - final_masks: Validity masks (V, H, W)
        as_mesh (bool): Represent the data as a mesh instead of point cloud (default: True)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud/mesh and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # Get the world frame points and images from the predictions
    pred_world_points = predictions["world_points"]
    images = predictions["images"]

    # Get the points and rgb
    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    if as_mesh:
        # Multi-frame case - create separate meshes for each frame
        for frame_idx in range(pred_world_points.shape[0]):
            H, W = pred_world_points.shape[1:3]

            # Get data for this frame
            frame_points = pred_world_points[frame_idx]
            frame_final_mask = predictions["final_masks"][frame_idx]

            # Get frame image
            if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
                frame_image = np.transpose(images[frame_idx], (1, 2, 0))
            else:  # Assume already in HWC format
                frame_image = images[frame_idx]
            frame_image *= 255

            # Create mesh for this frame
            faces, vertices, vertex_colors = image_mesh(
                frame_points * np.array([1, -1, 1], dtype=np.float32),
                frame_image / 255.0,
                mask=frame_final_mask,
                tri=True,
                return_indices=False,
            )
            vertices = vertices * np.array([1, -1, 1], dtype=np.float32)

            # Create trimesh object for this frame
            frame_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=(vertex_colors * 255).astype(np.uint8),
                process=False,
            )
            scene_3d.add_geometry(frame_mesh)
    else:
        final_masks = predictions["final_masks"].reshape(-1)
        vertices_3d = vertices_3d[final_masks].copy()
        colors_rgb = colors_rgb[final_masks].copy()
        point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
        scene_3d.add_geometry(point_cloud_data)

    # Apply 180° rotation around X-axis to fix orientation (upside-down issue)
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    scene_3d.apply_transform(rotation_matrix_x)

    return scene_3d
