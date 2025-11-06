# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for MoGe
"""

import torch

from morphcloud.models.external.moge.models.v1 import MoGeModel as MoGeModelV1
from morphcloud.models.external.moge.models.v2 import MoGeModel as MoGeModelV2


class MoGeWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        model_string="Ruicheng/moge-2-vitl",
        torch_hub_force_reload=False,
        load_custom_ckpt=False,
        custom_ckpt_path=None,
    ):
        super().__init__()
        self.name = name
        self.model_string = model_string
        self.torch_hub_force_reload = torch_hub_force_reload
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path

        # Mapping of MoGe model version to checkpoint strings
        self.moge_model_map = {
            "v1": ["Ruicheng/moge-vitl"],
            "v2": [
                "Ruicheng/moge-2-vits-normal",
                "Ruicheng/moge-2-vitb-normal",
                "Ruicheng/moge-2-vitl-normal",
                "Ruicheng/moge-2-vitl",
            ],
        }

        # Initialize the model
        if self.model_string in self.moge_model_map["v1"]:
            self.model = MoGeModelV1.from_pretrained(self.model_string)
        elif self.model_string in self.moge_model_map["v2"]:
            self.model = MoGeModelV2.from_pretrained(self.model_string)
        else:
            raise ValueError(
                f"Invalid model string: {self.model_string}. Valid strings are: {self.moge_model_map}"
            )

        # Load custom checkpoint if requested
        if self.load_custom_ckpt:
            print(f"Loading checkpoint from {self.custom_ckpt_path} ...")
            assert self.custom_ckpt_path is not None, (
                "custom_ckpt_path must be provided if load_custom_ckpt is set to True"
            )
            custom_ckpt = torch.load(self.custom_ckpt_path, weights_only=False)
            print(self.model.load_state_dict(custom_ckpt, strict=True))
            del custom_ckpt  # in case it occupies memory

    def forward(self, views):
        """
        Forward pass wrapper for MoGe-2.
        The predicted MoGe-2 mask is not applied to the outputs.
        The number of tokens for inference is determined by the image shape.

        Assumption:
        - The number of input views is 1.

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
        assert len(views) == 1, "MoGe only supports 1 input view."

        # Get input shape of the images, number of tokens for inference, and batch size per view
        _, _, height, width = views[0]["img"].shape
        num_tokens = int(height // 14) * int(width // 14)

        # Check the data norm type
        # MoGe expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "MoGe expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Run MoGe inference
        # Output dict contains: "points", "depth", "mask", "intrinsics", "normal" (based on model config)
        model_outputs = self.model.infer(
            image=views[0]["img"], num_tokens=num_tokens, apply_mask=False
        )

        # Get the ray directions and depth along ray
        with torch.autocast("cuda", enabled=False):
            depth_along_ray = torch.norm(model_outputs["points"], dim=-1, keepdim=True)
            ray_directions = model_outputs["points"] / depth_along_ray

        # Convert the output to MapAnything format
        result_dict = {
            "pts3d": model_outputs["points"],
            "pts3d_cam": model_outputs["points"],
            "depth_z": model_outputs["depth"].unsqueeze(-1),
            "intrinsics": model_outputs["intrinsics"],
            "non_ambiguous_mask": model_outputs["mask"],
            "ray_directions": ray_directions,
            "depth_along_ray": depth_along_ray,
        }
        res = [result_dict]

        return res
