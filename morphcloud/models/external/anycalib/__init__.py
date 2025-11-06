# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for AnyCalib
"""

import torch
from anycalib import AnyCalib

from morphcloud.utils.geometry import get_rays_in_camera_frame


class AnyCalibWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        model_id="anycalib_pinhole",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.model_id = model_id

        # Initialize the model
        self.model = AnyCalib(model_id=self.model_id)

    def forward(self, views):
        """
        Forward pass wrapper for AnyCalib.

        Assumption:
        - The number of input views is 1.
        - The output camera model is pinhole (fx, fy, cx, cy).
          This can be relaxed by not hardcoding the cam_id.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Length of the list should be 1.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]

        Returns:
            List[dict]: A list containing the final outputs for the single view. Length of the list will be 1.
        """
        # Check that the number of input views is 1
        assert len(views) == 1, "AnyCalib only supports 1 input view."

        # Get input shape of the images and batch size per view
        _, _, height, width = views[0]["img"].shape

        # Check the data norm type
        # AnyCalib expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "AnyCalib expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Run AnyCalib inference
        # Corresponding batched output dictionary:
        # {
        #      "intrinsics": List[(D_i,) tensors] for each camera model "i" at the original input resolution,
        #      "fov_field": (B, N, 2) tensor with the regressed FoV field by the network. N≈320^2 (resolution close to the one seen during training),
        #      "tangent_coords": alias for "fov_field",
        #      "rays": (B, N, 3) tensor with the corresponding (via the exponential map) ray directions in the camera frame (x right, y down, z forward),
        #      "pred_size": (H, W) tuple with the image size used by the network. It can be used e.g. for resizing the FoV/ray fields to the original image size.
        # }
        # For "pinhole" camera model, the intrinsics are (fx, fy, cx, cy).
        model_outputs = self.model.predict(views[0]["img"], cam_id="pinhole")

        # Convert the list of intrinsics to a tensor
        intrinsics = []
        for intrinsics_per_sample in model_outputs["intrinsics"]:
            pred_fx, pred_fy, pred_cx, pred_cy = intrinsics_per_sample
            intrinsics_per_sample = torch.tensor(
                [
                    [pred_fx, 0, pred_cx],
                    [0, pred_fy, pred_cy],
                    [0, 0, 1],
                ],
                device=views[0]["img"].device,
            )
            intrinsics.append(intrinsics_per_sample)

        # Convert the list of intrinsics to a tensor of size (batch_size_per_view, 3, 3)
        intrinsics = torch.stack(intrinsics)

        # Get the ray directions
        with torch.autocast("cuda", enabled=False):
            _, ray_directions = get_rays_in_camera_frame(
                intrinsics, height, width, normalize_to_unit_sphere=True
            )

        # Return the output in MapAnything format
        res = [{"ray_directions": ray_directions, "intrinsics": intrinsics}]

        return res
