# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Inference wrapper for VGGT-Omega."""

from pathlib import Path

import torch

from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


class VGGTOmegaWrapper(torch.nn.Module):
    """Run VGGT-Omega through the MapAnything external model contract.

    Args:
        name: Human-readable model name from the Hydra model config.
        torch_hub_force_reload: Kept for API consistency with other wrappers.
        checkpoint_path: Local VGGT-Omega checkpoint path.

    Input:
        A list of per-view dictionaries with identity-normalized image tensors
        under ``img``, each shaped ``(B, 3, H, W)``.

    Output:
        A list of per-view prediction dictionaries containing point maps,
        camera poses, ray directions, depth-along-ray, and confidence.
    """

    def __init__(
        self,
        name,
        torch_hub_force_reload,
        checkpoint_path,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload

        try:
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:
            raise ImportError(
                "VGGT-Omega is not importable. Install it with "
                '`pip install -e ".[vggt-omega]"` from the MapAnything repo.'
            ) from exc

        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"VGGT-Omega checkpoint does not exist: {checkpoint_path}"
            )

        self.model = VGGTOmega()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        print(self.model.load_state_dict(state_dict, strict=True))

    def forward(self, views):
        """Forward pass wrapper for VGGT-Omega.

        Args:
            views (List[dict]): Input views. Each view must contain ``img`` with
                shape ``(B, 3, H, W)`` and ``data_norm_type`` equal to
                ``["identity"]``.

        Returns:
            List[dict]: Dense geometry predictions for all input views.
        """
        if len(views) == 0:
            raise ValueError("VGGT-Omega requires at least one input view.")

        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "VGGT-Omega expects identity-normalized images"
        )

        images = torch.stack([view["img"] for view in views], dim=1)
        predictions = self.model(images)
        return self._format_predictions(
            predictions,
            image_shape=images.shape[-2:],
            num_views=len(views),
        )

    def _format_predictions(self, predictions, image_shape, num_views):
        """Convert raw VGGT-Omega outputs into MapAnything format."""
        from vggt_omega.utils.geometry import closed_form_inverse_se3
        from vggt_omega.utils.pose_enc import encoding_to_camera
        from vggt_omega.utils.rotation import mat_to_quat

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            image_shape,
        )

        output = []
        for view_idx in range(num_views):
            world_from_camera = closed_form_inverse_se3(extrinsics[:, view_idx])
            view_intrinsics = intrinsics[:, view_idx]
            depth_z = predictions["depth"][:, view_idx].squeeze(-1)
            confidence = predictions["depth_conf"][:, view_idx]
            height, width = depth_z.shape[-2:]

            pts3d_cam, _ = depthmap_to_camera_frame(depth_z, view_intrinsics)
            depth_along_ray = convert_z_depth_to_depth_along_ray(
                depth_z,
                view_intrinsics,
            ).unsqueeze(-1)
            _, ray_directions = get_rays_in_camera_frame(
                view_intrinsics,
                height,
                width,
                normalize_to_unit_sphere=True,
            )

            cam_trans = world_from_camera[..., :3, 3]
            cam_quats = mat_to_quat(world_from_camera[..., :3, :3])
            pts3d = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                ray_directions,
                depth_along_ray,
                cam_trans,
                cam_quats,
            )

            output.append(
                {
                    "pts3d": pts3d,
                    "pts3d_cam": pts3d_cam,
                    "ray_directions": ray_directions,
                    "depth_along_ray": depth_along_ray,
                    "cam_trans": cam_trans,
                    "cam_quats": cam_quats,
                    "conf": confidence,
                }
            )

        return output
