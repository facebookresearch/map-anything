# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os

import pytest
import torch

pycolmap = pytest.importorskip("pycolmap")

from mapanything.utils.colmap_export import export_predictions_to_colmap

H, W = 8, 10


def _make_view(tx):
    ys, xs = torch.meshgrid(
        torch.arange(H).float(), torch.arange(W).float(), indexing="ij"
    )
    z = torch.full((H, W), 2.0)
    pts3d = torch.stack([(xs - W / 2) * z / 10, (ys - H / 2) * z / 10, z], dim=-1)
    pose = torch.eye(4)
    pose[0, 3] = tx
    intrinsics = torch.tensor([[10.0, 0.0, W / 2], [0.0, 10.0, H / 2], [0.0, 0.0, 1.0]])
    return {
        "pts3d": (pts3d + torch.tensor([tx, 0.0, 0.0]))[None],
        "mask": torch.ones(1, H, W, 1),
        "depth_z": z[None, ..., None],
        "img_no_norm": torch.rand(1, H, W, 3),
        "intrinsics": intrinsics[None],
        "camera_poses": pose[None],
    }


def test_export_keeps_image_subfolders(tmp_path):
    image_names = ["cam0/frame_0001.jpg", "cam1/frame_0001.jpg"]

    export_predictions_to_colmap(
        [_make_view(0.0), _make_view(0.1)],
        None,
        image_names,
        str(tmp_path),
        voxel_size=0.05,
    )

    for name in image_names:
        assert os.path.isfile(os.path.join(tmp_path, "images", name))
    reconstruction = pycolmap.Reconstruction(str(tmp_path / "sparse"))
    assert sorted(img.name for img in reconstruction.images.values()) == image_names
