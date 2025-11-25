"""Synthetic sanity check for PhotometricReprojectionLoss.

This script builds a tiny forward pass with fabricated inputs to verify the
photometric reprojection loss behaves as expected without running training.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morphcloud.train.losses import PhotometricReprojectionLoss


def make_pinhole_ray_directions(
    height: int, width: int, fx: float, fy: float
) -> torch.Tensor:
    """Generate unit ray directions for a pinhole camera.

    Args:
        height: Image height.
        width: Image width.
        fx: Focal length in pixels along x.
        fy: Focal length in pixels along y.

    Returns:
        Tensor shaped ``(H, W, 3)`` with rays pointing into the scene.
    """

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    dirs = torch.stack(((x - cx) / fx, (y - cy) / fy, torch.ones_like(x)), dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return dirs


def build_synthetic_batch(
    batch_size: int,
    num_views: int,
    height: int,
    width: int,
    base_depth: float = 2.0,
) -> Tuple[dict, dict]:
    """Create a synthetic batch and predictions for the loss."""

    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    base_image = torch.stack(
        [x, y, 0.5 * (x + y)], dim=0
    )  # (3, H, W) smooth colour gradients
    images = base_image.unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1, -1)
    timestamps = torch.zeros(batch_size, num_views)
    non_ambiguous_mask = torch.ones(batch_size, num_views, height, width)

    depth = torch.full((batch_size, num_views, 1, height, width), base_depth)
    opacity = torch.ones_like(depth)

    ray_dirs = make_pinhole_ray_directions(height, width, fx=width, fy=height)
    ray_dirs = ray_dirs.expand(batch_size, num_views, -1, -1, -1).contiguous()

    cam_trans = torch.zeros(batch_size, num_views, 3)
    cam_quats = torch.zeros(batch_size, num_views, 4)
    cam_quats[..., -1] = 1.0  # Identity rotation in (x, y, z, w) format

    batch = {
        "img": images,
        "timestamp": timestamps,
        "non_ambiguous_mask": non_ambiguous_mask,
    }
    preds = {
        "depth_along_ray": depth,
        "opacity": opacity,
        "ray_directions": ray_dirs,
        "cam_trans": cam_trans,
        "cam_quats": cam_quats,
    }
    return batch, preds


def aggregate_loss(
    loss_output: torch.Tensor
    | list[tuple[torch.Tensor, torch.Tensor, str]]
    | tuple[tuple[torch.Tensor, torch.Tensor, str], ...]
):
    """Reduce the custom loss structure returned by MultiLoss into a scalar."""

    if isinstance(loss_output, (list, tuple)):
        view_losses = []
        for loss_map, valid_mask, _ in loss_output:
            if valid_mask is None:
                view_losses.append(loss_map.mean())
            elif torch.any(valid_mask):
                view_losses.append(loss_map[valid_mask].mean())
        if not view_losses:
            return torch.tensor(0.0)
        return torch.stack(view_losses).mean()
    return loss_output


def run_forward_tests():
    loss_fn = PhotometricReprojectionLoss()

    batch, preds = build_synthetic_batch(batch_size=1, num_views=2, height=4, width=4)
    loss_raw, details = loss_fn.compute_loss(batch, preds)
    loss = aggregate_loss(loss_raw)
    print("=== Baseline (aligned cameras, identical views) ===")
    print("Loss:", loss.item())
    print("Details:", details)

    # Translating the second camera creates parallax and should increase the loss.
    translated_preds = {k: v.clone() for k, v in preds.items()}
    translated_preds["cam_trans"][0, 1, 0] = 0.5
    translated_raw, translated_details = loss_fn.compute_loss(batch, translated_preds)
    translated_loss = aggregate_loss(translated_raw)
    print("\n=== Camera translated by 0.5m on view 2 ===")
    print("Loss:", translated_loss.item())
    print("Details:", translated_details)

    # Small pose rotation on view 1 via a quaternion around y-axis.
    angle_rad = math.radians(5)
    half_angle = angle_rad / 2
    sin_half = math.sin(half_angle)
    rotated_preds = {k: v.clone() for k, v in preds.items()}
    rotated_preds["cam_quats"][0, 1] = torch.tensor([0.0, sin_half, 0.0, math.cos(half_angle)])
    rotated_raw, rotated_details = loss_fn.compute_loss(batch, rotated_preds)
    rotated_loss = aggregate_loss(rotated_raw)
    print("\n=== Camera rotated by 5 degrees (view 2) ===")
    print("Loss:", rotated_loss.item())
    print("Details:", rotated_details)


if __name__ == "__main__":
    run_forward_tests()
