# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Wrapper for using MapAnything format models with the RMVD framework
"""

import warnings

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from morphcloud.utils.geometry import get_rays_in_camera_frame
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


class RMVD_MAPA_Wrapper(torch.nn.Module):
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        data_norm_type: str,
        use_amp=True,
        amp_dtype="bf16",
        inference_conditioning="image",
        evaluate_single_view=False,
    ):
        super().__init__()
        self.name = name
        self.model = model.eval()
        self.use_amp = use_amp
        self.inference_conditioning = inference_conditioning
        self.evaluate_single_view = evaluate_single_view

        # Determine the mixed precision floating point type
        if use_amp:
            if amp_dtype == "fp16":
                self.amp_dtype = torch.float16
            elif amp_dtype == "bf16":
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                else:
                    warnings.warn(
                        "bf16 is not supported on this device. Using fp16 instead."
                    )
                    self.amp_dtype = torch.float16
            elif amp_dtype == "fp32":
                self.amp_dtype = torch.float32
        else:
            self.amp_dtype = torch.float32

        if self.inference_conditioning == "image":
            pass
        elif self.inference_conditioning == "image+intrinsics":
            self.model.geometric_input_config["ray_dirs_prob"] = 1.0
            self.model.geometric_input_config["overall_prob"] = 1.0
            self.model.geometric_input_config["dropout_prob"] = 0.0
        elif self.inference_conditioning == "image+intrinsics+pose":
            self.model.geometric_input_config["ray_dirs_prob"] = 1.0
            self.model.geometric_input_config["cam_prob"] = 1.0
            self.model.geometric_input_config["overall_prob"] = 1.0
            self.model.geometric_input_config["dropout_prob"] = 0.0
        else:
            raise ValueError(
                f"Unknown inference_conditioning: {self.inference_conditioning}"
            )

        self.data_norm_type = data_norm_type

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        Args:
            args: List of tensors, each tensor is a view.
            kwargs: Additional keyword arguments.
        Returns:
            Dictionary containing the predictions and auxiliary information.
        """

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                return self.model(*args, **kwargs)

    # RMVD adaptor methods
    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
    ):
        # construct sample dict that contains all inputs in the model-specific format: sample = {..}

        assert len(images) >= 1, "At least one image is required"
        device = "cuda"

        img_mean = IMAGE_NORMALIZATION_DICT[self.data_norm_type].mean.to(device)
        img_std = IMAGE_NORMALIZATION_DICT[self.data_norm_type].std.to(device)

        if not (keyview_idx == [0]):
            assert len(keyview_idx) == 1, "Keyview index should contain only one index"

            # switch the image, pose, intrinsics between keyview_idx and 0 so that 0 is the keyview
            images[0], images[keyview_idx[0]] = (
                images[keyview_idx[0]].copy(),
                images[0].copy(),
            )
            if poses is not None:
                poses[0], poses[keyview_idx[0]] = (
                    poses[keyview_idx[0]].copy(),
                    poses[0].copy(),
                )

                assert np.allclose(poses[0], np.eye(4), rtol=1e-03, atol=1e-04), (
                    "The first pose should be the identity matrix, but it is not."
                )
            if intrinsics is not None:
                intrinsics[0], intrinsics[keyview_idx[0]] = (
                    intrinsics[keyview_idx[0]].copy(),
                    intrinsics[0].copy(),
                )

        views = []
        for i, img in enumerate(images):
            # Normalize the image
            img = (
                torch.from_numpy(img).to(device).float() / 255.0
                - img_mean.view(1, 3, 1, 1)
            ) / img_std.view(1, 3, 1, 1)

            view = {
                "img": img,
                "true_shape": torch.tensor(
                    [img.shape[2], img.shape[3]], device=device
                ).view(1, 2),
                "idx": i,  # TODO: double check this i
                "instance": str(i),
                "data_norm_type": [self.data_norm_type],
            }

            # Transform intrinsics into Rays for MAPA conditioning, if available
            if intrinsics is not None:
                assert "intrinsics" in self.inference_conditioning, (
                    "Data should not contain intrinsics if inference_conditioning should not include them"
                )

                current_intrinsics = torch.from_numpy(intrinsics[i]).to(device).float()

                ray_origins, ray_directions = get_rays_in_camera_frame(
                    intrinsics=current_intrinsics,
                    height=img.shape[2],
                    width=img.shape[3],
                    normalize_to_unit_sphere=True,
                )

                view["ray_directions_cam"] = ray_directions

            if poses is not None:
                assert "pose" in self.inference_conditioning, (
                    "Data should not contain poses if inference_conditioning should not include them"
                )

                current_pose_np = poses[i].copy()
                current_pose_np = np.linalg.inv(current_pose_np)
                # The RMVD doc is very confusing about this, taking the inverse here yields much smaller error.

                current_R = current_pose_np[..., :3, :3]  # 3x3 rotation matrix
                current_pose_key_T_current = (
                    torch.from_numpy(current_pose_np).to(device).float()
                )

                # transform from 4x4 SE3 representation to translation and quaternion
                camera_pose_trans = current_pose_key_T_current[..., :3, 3]  # XYZ
                camera_pose_quats = (
                    torch.from_numpy(R.from_matrix(current_R).as_quat())
                    .to(device)
                    .float()
                )  # XYZW

                view["camera_pose_trans"] = camera_pose_trans
                view["camera_pose_quats"] = camera_pose_quats
                view["is_metric_scale"] = torch.tensor([True], device=device)

            views.append(view)

            if self.evaluate_single_view:
                # In single view inference, we only provide the reference view to the model.
                break

        return {"views": views}

    def output_adapter(self, model_output):
        # construct pred and aux dicts from model_output
        # pred needs to have an item with key "depth" and value of type np.ndarray and shape N1HW

        pred = {}
        pts3d = model_output[0]["pts3d_cam"]

        pred["depth"] = pts3d[..., -1].unsqueeze(1).cpu().numpy()
        _, _, H, W = pred["depth"].shape

        if "conf" in model_output[0]:
            conf = model_output[0]["conf"]
            conf = conf.reshape(1, 1, H, W)
        else:
            conf = torch.ones_like(pts3d[..., -1].unsqueeze(1))

        pred["depth_uncertainty"] = 1.0 / conf.cpu().numpy()

        aux = {}
        return pred, aux
