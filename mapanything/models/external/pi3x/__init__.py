# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for Pi3X
"""

import torch
from pi3.models.pi3x import Pi3X

from mapanything.models.external.vggt.utils.rotation import mat_to_quat


class Pi3XWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        geometric_input_config,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.geometric_input_config = geometric_input_config

        # Load pre-trained weights
        if not torch_hub_force_reload:
            # Initialize the Pi3X model from huggingface hub cache
            print("Loading Pi3X from huggingface cache ...")
            self.model = Pi3X.from_pretrained(
                "yyfz233/Pi3X",
            )
        else:
            # Initialize the Pi3X model
            self.model = Pi3X.from_pretrained("yyfz233/Pi3X", force_download=True)

        # Get the dtype for Pi3X inference
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    def forward(self, views):
        """
        Forward pass wrapper for Pi3X

        Assumption:
        - All the input views have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        device = views[0]["img"].device
        num_views = len(views)

        # Check the data norm type
        # Pi3 expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "Pi3X expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        # Decide if we need to use the geometric inputs
        conditions = {}
        if torch.rand(1, device=device) < self.geometric_input_config["overall_prob"]:
            # Decide if we need to use the camera intrinsics
            if (
                torch.rand(1, device=device)
                < self.geometric_input_config["ray_dirs_prob"]
            ):
                intrinsics_list = [
                    view["camera_intrinsics"].to(device) for view in views
                ]
                intrinsics = torch.stack(intrinsics_list, dim=1)
                conditions["intrinsics"] = intrinsics

            # Decide if we need to use the depth map
            if torch.rand(1, device=device) < self.geometric_input_config["depth_prob"]:
                depthmap_list = [
                    view["depthmap"].squeeze(-1).to(device) for view in views
                ]
                depths = torch.stack(depthmap_list, dim=1)
                conditions["depths"] = depths

            # Decide if we need to use the camera poses
            if torch.rand(1, device=device) < self.geometric_input_config["cam_prob"]:
                poses_list = [view["camera_pose"].to(device) for view in views]
                poses = torch.stack(poses_list, dim=1)
                conditions["poses"] = poses

        # Run the Pi3X aggregator
        with torch.autocast("cuda", dtype=self.dtype):
            results = self.model(
                imgs=images,
                **conditions,
            )

        # Need high precision for transformations
        with torch.autocast("cuda", enabled=False):
            # Convert the output to MapAnything format
            res = []
            for view_idx in range(num_views):
                # Get the extrinsics
                curr_view_extrinsic = results["camera_poses"][:, view_idx, ...]
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Get the depth along ray, ray directions, local point cloud & global point cloud
                curr_view_pts3d_cam = results["local_points"][:, view_idx, ...]
                curr_view_depth_along_ray = torch.norm(
                    curr_view_pts3d_cam, dim=-1, keepdim=True
                )
                curr_view_ray_dirs = curr_view_pts3d_cam / curr_view_depth_along_ray
                curr_view_pts3d = results["points"][:, view_idx, ...]

                # Get the confidence
                curr_view_confidence = results["conf"][:, view_idx, ...]

                # Append the outputs to the result list
                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                        "conf": curr_view_confidence,
                    }
                )

        return res
