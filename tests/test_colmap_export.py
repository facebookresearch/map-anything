# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from mapanything.utils.colmap_export import build_colmap_reconstruction


def test_build_colmap_reconstruction_round_trip(tmp_path):
    reconstruction = build_colmap_reconstruction(
        points_3d=np.array([[0.0, 0.0, 2.0]]),
        points_rgb=np.array([[255, 0, 0]], dtype=np.uint8),
        extrinsics=np.array([np.column_stack((np.eye(3), np.zeros(3)))]),
        intrinsics=np.array(
            [[[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]]]
        ),
        image_width=64,
        image_height=48,
        image_names=["image.jpg"],
    )

    output_dir = tmp_path / "colmap"
    output_dir.mkdir()
    reconstruction.write(output_dir)
    loaded = pycolmap.Reconstruction(output_dir)

    assert loaded.num_cameras() == 1
    assert loaded.num_images() == 1
    assert loaded.num_reg_images() == 1
    assert loaded.num_points3D() == 1
    assert loaded.compute_num_observations() == 1
