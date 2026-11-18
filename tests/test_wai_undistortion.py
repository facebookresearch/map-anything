# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Regression tests for the public WAI processor's OpenCV pixel bounds.

Requires the WAI processing dependencies in addition to MapAnything's runtime.
Tests construct maps only; they do not process scenes or write dataset files.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def undistortion_module(monkeypatch):
    processing_root = Path(__file__).resolve().parents[1] / "data_processing"
    monkeypatch.syspath_prepend(str(processing_root))
    return importlib.import_module("wai_processing.scripts.undistort")


def _camera(width, height, **overrides):
    return {
        "camera_model": "OPENCV",
        "w": width,
        "h": height,
        "fl_x": 900.5,
        "fl_y": 905.25,
        "cx": (width - 1) / 2,
        "cy": (height - 1) / 2,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
        **overrides,
    }


@pytest.mark.parametrize("width,height", [(1280, 720), (720, 1280), (1001, 777)])
def test_zero_distortion_preserves_camera(undistortion_module, width, height):
    camera = _camera(width, height)
    matrix, new_width, new_height, map1, map2, roi = (
        undistortion_module.undistort_precompute(camera)
    )
    expected = np.array(
        [
            [camera["fl_x"], 0, camera["cx"]],
            [0, camera["fl_y"], camera["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(matrix, expected)
    assert (new_width, new_height) == (width - 1, height - 1)
    assert roi == (0, 0, width - 1, height - 1)
    assert map1.shape[:2] == map2.shape == (height, width)


@pytest.mark.parametrize("width,height", [(1280, 720), (720, 1280)])
def test_radial_camera_keeps_centered_principal_point(
    undistortion_module, width, height
):
    matrix, *_ = undistortion_module.undistort_precompute(
        _camera(width, height, fl_x=900.0, fl_y=900.0, k1=-0.1, k2=0.01)
    )
    np.testing.assert_allclose(
        matrix[:2, 2], ((width - 1) / 2, (height - 1) / 2), rtol=0, atol=1e-5
    )


def test_tangential_camera_matches_opencv_4_10_reference(undistortion_module):
    camera = _camera(
        1280,
        720,
        cx=641.3,
        cy=357.9,
        k1=0.12,
        k2=-0.05,
        p1=0.001,
        p2=-0.0015,
        k3=0.02,
    )
    matrix, width, height, _, _, roi = undistortion_module.undistort_precompute(camera)
    # Generated with opencv-python-headless 4.10.0.84 and float32 calibration.
    expected = tuple(
        float.fromhex(value)
        for value in (
            "0x1.d73bdap+9",
            "0x1.cc6a8p+9",
            "0x1.3fb82ep+9",
            "0x1.6659bap+8",
        )
    )
    assert tuple(matrix[[0, 1, 0, 1], [0, 1, 2, 2]]) == expected
    assert roi == (7, 14, 1265, 690)
    assert (width, height) == roi[2:]
    assert matrix.dtype == np.float32


def test_precompute_does_not_mutate_calibration(undistortion_module):
    camera = _camera(103, 81, k1=-0.05)
    original = dict(camera)
    undistortion_module.undistort_precompute(camera)
    assert camera == original
