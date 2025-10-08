# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
This utils script contains PORTAGE of wai-core camera methods for MapAnything.
"""

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from mapanything.utils.wai.ops import get_dtype_device

# constants regarding camera models
PINHOLE_CAM_KEYS = ["fl_x", "fl_y", "cx", "cy", "h", "w"]
DISTORTION_PARAM_KEYS = [
    "k1",
    "k2",
    "k3",
    "k4",
    "p1",
    "p2",
]  # order corresponds to the OpenCV convention
CAMERA_KEYS = PINHOLE_CAM_KEYS + DISTORTION_PARAM_KEYS

# Retrieve all available camera models and their associated parameters using pycolmap
CAM_STRS_TO_PARAMS = {
    # For PINHOLE we use separate focal length for x and y, even though almost always we
    # will have fx=fy.
    "PINHOLE": ["fl_x", "fl_y", "cx", "cy"],
    # Undistortion supported by OPenCV
    "OPENCV": ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"],
    "OPENCV_FISHEYE": ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4"],
    # Undistortion supported by pycolmap
    "FULL_OPENCV": [
        "fl_x",
        "fl_y",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "k4",
        "k5",
        "k6",
    ],
    "FOV": ["fl_x", "fl_y", "cx", "cy", "omega"],
    "THIN_PRISM_FISHEYE": [
        "fl_x",
        "fl_y",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "k4",
        "sx1",
        "sy1",
    ],
    "RAD_TAN_THIN_PRISM_FISHEYE": [
        "fl_x",
        "fl_y",
        "cx",
        "cy",
        "k0",
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "p0",
        "p1",
        "s0",
        "s1",
        "s2",
        "s3",
    ],
    # Non-OpenCV and non-pycolmap camera models
    "EQUIRECTANGULAR": [],  # Only width and height needed
}
# This is just an unordered helper list for all cam params and for distortion parameters
# which should never occur for a pinhole camera
ALL_CAM_PARAMS = list(set().union(*CAM_STRS_TO_PARAMS.values())) + ["w", "h"]


def interpolate_intrinsics(
    frame1: dict[str, Any],
    frame2: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """
    Interpolate camera intrinsics linearly.
    Args:
        frame1: The first frame dictionary.
        frame2: The second frame dictionary.
        alpha: Interpolation parameter. alpha = 0 for frame1, alpha = 1 for frame2.
    Returns:
        frame_inter: dictionary with new intrinsics.
    """
    frame_inter = {}
    for key in CAMERA_KEYS:
        if key in frame1 and key in frame2:
            p1 = frame1[key]
            p2 = frame2[key]
            frame_inter[key] = (1 - alpha) * p1 + alpha * p2
    return frame_inter


def interpolate_extrinsics(
    matrix1: list | np.ndarray | torch.Tensor,
    matrix2: list | np.ndarray | torch.Tensor,
    alpha: float,
) -> list | np.ndarray | torch.Tensor:
    """
    Interpolate camera extrinsics 4x4 matrices using SLERP.
    Args:
        matrix1: The first matrix.
        matrix2: The second matrix.
        alpha: Interpolation parameter. alpha = 0 for matrix1, alpha = 1 for matrix2.
    Returns:
        matrix: 4x4 interpolated matrix, same type.
    Raises:
        ValueError: If different type.
    """
    if not isinstance(matrix1, type(matrix2)):
        raise ValueError("Both matrices should have the same type.")

    dtype, device = get_dtype_device(matrix1)
    if isinstance(matrix1, list):
        mtype = "list"
        matrix1 = np.array(matrix1)
        matrix2 = np.array(matrix2)
    elif isinstance(matrix1, np.ndarray):
        mtype = "numpy"
    elif isinstance(matrix1, torch.Tensor):
        mtype = "torch"
        matrix1 = matrix1.numpy()
        matrix2 = matrix2.numpy()
    else:
        raise ValueError(
            "Only list, numpy array and torch tensors are supported as inputs."
        )

    R1 = matrix1[:3, :3]
    t1 = matrix1[:3, 3]
    R2 = matrix2[:3, :3]
    t2 = matrix2[:3, 3]

    # interpolate translation
    t = (1 - alpha) * t1 + alpha * t2

    # interpolate rotations with SLERP
    R1_quat = Rotation.from_matrix(R1).as_quat()
    R2_quat = Rotation.from_matrix(R2).as_quat()
    rotation_slerp = Slerp([0, 1], Rotation(np.stack([R1_quat, R2_quat])))
    R = rotation_slerp(alpha).as_matrix()
    matrix_inter = np.eye(4)

    # combine together
    matrix_inter[:3, :3] = R
    matrix_inter[:3, 3] = t

    if mtype == "list":
        matrix_inter = matrix_inter.tolist()
    elif mtype == "torch":
        matrix_inter = torch.from_numpy(matrix_inter).to(dtype).to(device)
    elif mtype == "numpy":
        matrix_inter = matrix_inter.astype(dtype)

    return matrix_inter


def convert_camera_coeffs_to_pinhole_matrix(
    scene_meta, frame, fmt="torch"
) -> torch.Tensor | np.ndarray | list:
    """
    Convert camera intrinsics from NeRFStudio format to a 3x3 intrinsics matrix.

    Args:
        scene_meta: Scene metadata containing camera parameters
        frame: Frame-specific camera parameters that override scene_meta

    Returns:
        torch.Tensor: 3x3 camera intrinsics matrix

    Raises:
        ValueError: If camera model is not PINHOLE or if distortion coefficients are present
    """
    # Check if camera model is supported
    camera_model = frame.get("camera_model", scene_meta.get("camera_model"))
    if camera_model != "PINHOLE":
        raise ValueError("Only PINHOLE camera model supported")

    # Check for unsupported distortion coefficients
    if any(
        (frame.get(coeff, 0) != 0) or (scene_meta.get(coeff, 0) != 0)
        for coeff in DISTORTION_PARAM_KEYS
    ):
        raise ValueError(
            "Pinhole camera does not support radial/tangential distortion -> Undistort first"
        )

    # Extract camera intrinsic parameters
    camera_coeffs = {}
    for coeff in ["fl_x", "fl_y", "cx", "cy"]:
        camera_coeffs[coeff] = frame.get(coeff, scene_meta.get(coeff))
        if camera_coeffs[coeff] is None:
            raise ValueError(f"Missing required camera parameter: {coeff}")

    # Create intrinsics matrix
    intrinsics = [
        [camera_coeffs["fl_x"], 0.0, camera_coeffs["cx"]],
        [0.0, camera_coeffs["fl_y"], camera_coeffs["cy"]],
        [0.0, 0.0, 1.0],
    ]
    if fmt == "torch":
        intrinsics = torch.tensor(intrinsics)
    elif fmt == "np":
        intrinsics = np.array(intrinsics)

    return intrinsics


def rotate_pinhole_90degcw(
    W: int, H: int, fx: float, fy: float, cx: float, cy: float
) -> tuple[int, int, float, float, float, float]:
    """Rotates the intrinsics of a pinhole camera model by 90 degrees clockwise."""
    W_new = H
    H_new = W
    fx_new = fy
    fy_new = fx
    cy_new = cx
    cx_new = H - 1 - cy
    return W_new, H_new, fx_new, fy_new, cx_new, cy_new


def _gl_cv_cmat() -> np.ndarray:
    cmat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return cmat


def _apply_transformation(
    c2ws: torch.Tensor | np.ndarray, cmat: np.ndarray
) -> torch.Tensor | np.ndarray:
    """
    Convert camera poses using a provided conversion matrix.

    Args:
        c2ws (torch.Tensor or np.ndarray): Camera poses (batch_size, 4, 4) or (4, 4)
        cmat (torch.Tensor or np.ndarray): Conversion matrix (4, 4)

    Returns:
        torch.Tensor or np.ndarray: Transformed camera poses (batch_size, 4, 4) or (4, 4)
    """
    if isinstance(c2ws, torch.Tensor):
        # Clone the input tensor to avoid modifying it in-place
        c2ws_transformed = c2ws.clone()
        # Apply the conversion matrix to the rotation part of the camera poses
        if len(c2ws.shape) == 3:
            c2ws_transformed[:, :3, :3] = c2ws_transformed[
                :, :3, :3
            ] @ torch.from_numpy(cmat[:3, :3]).to(c2ws).unsqueeze(0)
        else:
            c2ws_transformed[:3, :3] = c2ws_transformed[:3, :3] @ torch.from_numpy(
                cmat[:3, :3]
            ).to(c2ws)

    elif isinstance(c2ws, np.ndarray):
        # Clone the input array to avoid modifying it in-place
        c2ws_transformed = c2ws.copy()
        if len(c2ws.shape) == 3:  # batched
            # Apply the conversion matrix to the rotation part of the camera poses
            c2ws_transformed[:, :3, :3] = np.einsum(
                "ijk,lk->ijl", c2ws_transformed[:, :3, :3], cmat[:3, :3]
            )
        else:  # single 4x4 matrix
            # Apply the conversion matrix to the rotation part of the camera pose
            c2ws_transformed[:3, :3] = np.dot(c2ws_transformed[:3, :3], cmat[:3, :3])

    else:
        raise ValueError("Input data type not supported.")

    return c2ws_transformed


def gl2cv(
    c2ws: torch.Tensor | np.ndarray,
    return_cmat: bool = False,
) -> torch.Tensor | np.ndarray | tuple[torch.Tensor | np.ndarray, np.ndarray]:
    """
    Convert camera poses from OpenGL to OpenCV coordinate system.

    Args:
        c2ws (torch.Tensor or np.ndarray): Camera poses (batch_size, 4, 4) or (4, 4)
        return_cmat (bool): If True, return the conversion matrix along with the transformed poses

    Returns:
        torch.Tensor or np.ndarray: Transformed camera poses (batch_size, 4, 4) or (4, 4)
        np.ndarray (optional): Conversion matrix if return_cmat is True
    """
    cmat = _gl_cv_cmat()
    if return_cmat:
        return _apply_transformation(c2ws, cmat), cmat
    return _apply_transformation(c2ws, cmat)


def intrinsics_to_fov(
    fx: torch.Tensor, fy: torch.Tensor, h: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the horizontal and vertical fields of view in radians from camera intrinsics.

    Args:
        fx (torch.Tensor): focal x
        fy (torch.Tensor): focal y
        h (torch.Tensor): Image height(s) with shape (B,).
        w (torch.Tensor): Image width(s) with shape (B,).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the horizontal and vertical fields
        of view in radians, both with shape (N,).
    """
    return 2 * torch.atan((w / 2) / fx), 2 * torch.atan((h / 2) / fy)


def cv2gl(
    c2ws: torch.Tensor | np.ndarray,
    return_cmat: bool = False,
) -> torch.Tensor | np.ndarray | tuple[torch.Tensor | np.ndarray, np.ndarray]:
    """
    Convert camera poses from OpenCV to OpenGL coordinate system.

    Args:
        c2ws (torch.Tensor or np.ndarray): Camera poses (batch_size, 4, 4) or (4, 4)
        return_cmat (bool): If True, return the conversion matrix along with the transformed poses

    Returns:
        torch.Tensor or np.ndarray: Transformed camera poses (batch_size, 4, 4) or (4, 4)
        np.ndarray (optional): Conversion matrix if return_cmat is True
    """
    cmat = _gl_cv_cmat()
    if return_cmat:
        return _apply_transformation(c2ws, cmat), cmat
    return _apply_transformation(c2ws, cmat)
