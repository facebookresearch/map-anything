# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything model class defined using UniCeption modules.
"""

import random
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from mapanything.models.mapanything.model import MapAnything
from mapanything.models.mapanything_adapter.alternating_attention_adapter import (
    MVAATAdapterIFR,
)
from mapanything.models.mapanything_adapter.predicter import Predicter
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    normalize_pose_translations,
)
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from uniception.models.encoders import (
    encoder_factory,
    EncoderGlobalRepInput,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
)
from uniception.models.info_sharing.base import MultiViewTransformerInput

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True



class MapAnythingAdapter(MapAnything, PyTorchModelHubMixin):
    
    def __init__(
        self,
        predicter_config: Dict,
        intermediates_only: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        
        self.intermediates_only = intermediates_only
        self.context_mode = False
        super().__init__(*args, **kwargs)

        self.predicter = Predicter(
            self.encoder.enc_embed_dim,
            self.encoder.patch_size,
            predicter_config,
            kwargs["geometric_input_config"],
            # kwargs["fusion_norm_layer"],
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _initialize_info_sharing(self, info_sharing_config):
        """
        Initialize the information sharing module based on the configuration.

        This method sets up the custom positional encoding if specified and initializes
        the appropriate multi-view transformer based on the configuration type.

        Args:
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
                Should contain 'custom_positional_encoding', 'model_type', and 'model_return_type'.

        Returns:
            None

        Raises:
            ValueError: If invalid configuration options are provided.
        """
        # Initialize Custom Positional Encoding if required
        custom_positional_encoding = info_sharing_config["custom_positional_encoding"]
        if custom_positional_encoding is not None:
            if isinstance(custom_positional_encoding, str):
                print(
                    f"Using custom positional encoding for multi-view attention transformer: {custom_positional_encoding}"
                )
                raise ValueError(
                    f"Invalid custom_positional_encoding: {custom_positional_encoding}. None implemented."
                )
            elif isinstance(custom_positional_encoding, Callable):
                print(
                    "Using callable function as custom positional encoding for multi-view attention transformer."
                )
                self.custom_positional_encoding = custom_positional_encoding
        else:
            self.custom_positional_encoding = None

        # Add dependencies to info_sharing_config
        info_sharing_config["module_args"]["input_embed_dim"] = (
            self.encoder.enc_embed_dim
        )
        info_sharing_config["module_args"]["custom_positional_encoding"] = (
            self.custom_positional_encoding
        )

        # Initialize Multi-View Transformer
        if self.info_sharing_return_type == "intermediate_features":
            # Returns intermediate features and normalized last layer features
            # Initialize mulit-view transformer based on type
            if self.info_sharing_type == "alternating_attention_adapter":
                self.info_sharing = MVAATAdapterIFR(
                    **info_sharing_config["module_args"],
                )
            else:
                raise ValueError(
                    f"Invalid info_sharing_type: {self.info_sharing_type}. Valid options: ['alternating_attention_adapter']"
                )
            # Assess if the DPT needs to use encoder features
            if len(self.info_sharing.indices) == 2:
                self.use_encoder_features_for_dpt = True
            elif len(self.info_sharing.indices) == 3:
                self.use_encoder_features_for_dpt = False
            else:
                raise ValueError(
                    "Invalid number of indices provided for info sharing feature returner. Please provide 2 or 3 indices."
                )
        else:
            raise ValueError(
                f"Invalid info_sharing_return_type: {self.info_sharing_return_type}. Valid options: ['intermediate_features']"
            )
        
    def contextualize(self):
        """
        Context mode on. Mask out all inputs not belonging to the context views
        """
        self.context_mode = True
    
    def decontextualize(self):
        """
        Context mode off. Standard operation with all image and geometric inputs.
        """
        self.context_mode = False

    def stop_non_adapter_gradient_flow(self):
        """
        Stop gradient flow through all non-adapter modules.
        """
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.predicter.parameters():
            param.requires_grad = True

        self.info_sharing.stop_non_adapter_gradient_flow()

    def _encode_and_fuse_optional_geometric_inputs(
        self, views, all_encoder_features_across_views_list
    ):
        """
        Encode all the input optional geometric modalities and fuses it with the image encoder features in a single forward pass.
        Assumes all the input views have the same shape and batch size.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            all_encoder_features_across_views (List[torch.Tensor]): List of tensors containing the encoded image features for all N views.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        num_views = len(views)
        batch_size_per_view, _, _, _ = views[0]["img"].shape
        device = all_encoder_features_across_views_list[0].device
        dtype = all_encoder_features_across_views_list[0].dtype
        all_encoder_features_across_views = torch.cat(
            all_encoder_features_across_views_list, dim=0
        )  # (V * B)

        # Get target mask, which if True will block out image and geometric inputs.
        target_mask = None
        per_sample_target_mask = None
        if self.context_mode:
            target_mask = torch.zeros(
                num_views, device=device, dtype=torch.bool
            )
            target_ids = random.sample(range(num_views), k=max(1, int(self.geometric_input_config["target_ratio"] * num_views)))
            target_mask[target_ids] = True
            per_sample_target_mask = target_mask.repeat_interleave(batch_size_per_view)  # (V * B)

        # Get the overall input mask for all the views
        overall_geometric_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["overall_prob"]
        )
        overall_geometric_input_mask = overall_geometric_input_mask.repeat(num_views)

        # Get the per sample input mask after dropout
        # Per sample input mask is in view-major order so that index v*B + b in each mask corresponds to sample b of view v: (B * V)
        per_sample_geometric_input_mask = torch.rand(
            batch_size_per_view * num_views, device=device
        ) < (1 - self.geometric_input_config["dropout_prob"])
        per_sample_geometric_input_mask = (
            per_sample_geometric_input_mask & overall_geometric_input_mask
        )

        # Get the depth input mask
        per_sample_depth_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["depth_prob"]
        )
        per_sample_depth_input_mask = per_sample_depth_input_mask.repeat(num_views)
        per_sample_depth_input_mask = (
            per_sample_depth_input_mask & per_sample_geometric_input_mask
        )

        # Get the ray direction input mask
        per_sample_ray_dirs_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["ray_dirs_prob"]
        )
        per_sample_ray_dirs_input_mask = per_sample_ray_dirs_input_mask.repeat(
            num_views
        )
        per_sample_ray_dirs_input_mask = (
            per_sample_ray_dirs_input_mask & per_sample_geometric_input_mask
        )

        # Get the camera input mask
        per_sample_cam_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["cam_prob"]
        )
        per_sample_cam_input_mask = per_sample_cam_input_mask.repeat(num_views)
        per_sample_cam_input_mask = (
            per_sample_cam_input_mask & per_sample_geometric_input_mask
        )

        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        # Returned pose quats and trans represent identity pose for views/samples where the camera input mask is False
        pose_quats_across_views, pose_trans_across_views, per_sample_cam_input_mask = (
            self._compute_pose_quats_and_trans_for_across_views_in_ref_view(
                views,
                num_views,
                device,
                dtype,
                batch_size_per_view,
                per_sample_cam_input_mask,
            )
        )

        # Encode the depths and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_depths(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_depth_input_mask,
        )

        # Encode the ray directions and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_ray_dirs(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_ray_dirs_input_mask,
        )

        # Encode the cam quat and trans and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_cam_quats_and_trans(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            pose_quats_across_views,
            pose_trans_across_views,
            per_sample_cam_input_mask,
        )

        # Normalize the fused features (permute -> normalize -> permute)
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 2, 3, 1
        ).contiguous()
        all_encoder_features_across_views = self.fusion_norm_layer(
            all_encoder_features_across_views
        )
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 3, 1, 2
        ).contiguous()


        # Zero out all inputs for non-target views if in context mode
        if self.context_mode:
            all_encoder_features_across_views = (
                all_encoder_features_across_views
                * (~per_sample_target_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )

        # Split the batched views into individual views
        fused_all_encoder_features_across_views = (
            all_encoder_features_across_views.chunk(num_views, dim=0)
        )

        return fused_all_encoder_features_across_views, target_mask

    def forward(self, views, memory_efficient_inference=False):
        """
        Forward pass performing the following operations:
        1. Encodes the N input views (images).
        2. Encodes the optional geometric inputs (ray directions, depths, camera rotations, camera translations).
            2.1. In context mode, masks out all image and geometric inputs not belonging to the context views.
        3. Fuses the encoded features from the N input views and the optional geometric inputs using addition and normalization.
        4. Information sharing across the encoded features and a scale token using a multi-view attention transformer.
            4.1. In context mode, make target predictions from masked multi-view features.
        5. Passes the final features from transformer through the prediction heads.
        6. Returns the processed final outputs for N views.

        Assumption:
        - All the input views and dense geometric inputs have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W). Input images must be normalized based on the data norm type of image encoder.
                                    "data_norm_type" (list): [model.encoder.data_norm_type]
                                Optionally, each dictionary can also contain the following keys for the respective optional geometric inputs:
                                    "ray_directions_cam" (tensor): Ray directions in the local camera frame. Tensor of shape (B, H, W, 3).
                                    "depth_along_ray" (tensor): Depth along the ray. Tensor of shape (B, H, W, 1).
                                    "camera_pose_quats" (tensor): Camera pose quaternions. Tensor of shape (B, 4). Camera pose is opencv (RDF) cam2world transformation.
                                    "camera_pose_trans" (tensor): Camera pose translations. Tensor of shape (B, 3). Camera pose is opencv (RDF) cam2world transformation.
                                    "is_metric_scale" (tensor): Boolean tensor indicating whether the geometric inputs are in metric scale or not. Tensor of shape (B, 1).
            memory_efficient_inference (bool): Whether to use memory efficient inference or not. This runs the dense prediction head (the memory bottleneck) in a memory efficient manner. Default is False.

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))
        num_views = len(views)

        # Run the image encoder on all the input views
        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self._encode_n_views(views)
        )

        # Encode the optional geometric inputs and fuse with the encoded features from the N input views
        # Use high precision to prevent NaN values after layer norm in dense representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views, target_mask = (
                self._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,  # V * (B, C, H, W)
            additional_input_tokens_per_view=all_encoder_registers_across_views,
            additional_input_tokens=input_scale_token,
        )

        (
            final_info_sharing_multi_view_feat,
            intermediate_info_sharing_multi_view_feat,
            intermediate_info_sharing_branch_feat,
        ) = self.info_sharing(info_sharing_input)

        # Make target predictions from masked multi-view features if in context mode
        pred_target_feat = None
        if self.context_mode:
            pred_target_feat = self.predicter(
                views,
                target_mask,
                intermediate_info_sharing_branch_feat[-1],
            )

        # If only intermediate features are needed, return them directly
        if self.intermediates_only:
            return intermediate_info_sharing_branch_feat, pred_target_feat, target_mask
        
        if self.pred_head_type == "linear":
            # Stack the features for all views
            dense_head_inputs = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
        elif self.pred_head_type in ["dpt", "dpt+pose"]:
            # Get the list of features for all views
            dense_head_inputs_list = []
            if self.use_encoder_features_for_dpt:
                # Stack all the image encoder features for all views
                stacked_encoder_features = torch.cat(
                    all_encoder_features_across_views, dim=0
                )
                dense_head_inputs_list.append(stacked_encoder_features)
                # Stack the first intermediate features for all views
                stacked_intermediate_features_1 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                # Stack the second intermediate features for all views
                stacked_intermediate_features_2 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                # Stack the last layer features for all views
                stacked_final_features = torch.cat(
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                dense_head_inputs_list.append(stacked_final_features)
            else:
                # Stack the first intermediate features for all views
                stacked_intermediate_features_1 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                # Stack the second intermediate features for all views
                stacked_intermediate_features_2 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                # Stack the third intermediate features for all views
                stacked_intermediate_features_3 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[2].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_3)
                # Stack the last layer
                stacked_final_features = torch.cat(
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                dense_head_inputs_list.append(stacked_final_features)
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )

        with torch.autocast("cuda", enabled=False):
            # Prepare inputs for the downstream heads
            if self.pred_head_type == "linear":
                dense_head_inputs = dense_head_inputs
            elif self.pred_head_type in ["dpt", "dpt+pose"]:
                dense_head_inputs = dense_head_inputs_list
            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features
            )

            # Run the downstream heads
            dense_final_outputs, pose_final_outputs, scale_final_output = (
                self.downstream_head(
                    dense_head_inputs=dense_head_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                    memory_efficient_inference=memory_efficient_inference,
                )
            )

            # Prepare the final scene representation for all views
            if self.scene_rep_type in [
                "pointmap",
                "pointmap+confidence",
                "pointmap+mask",
                "pointmap+confidence+mask",
            ]:
                output_pts3d = dense_final_outputs.value
                # Reshape final scene representation to (B * V, H, W, C)
                output_pts3d = output_pts3d.permute(0, 2, 3, 1).contiguous()
                # Split the predicted pointmaps back to their respective views
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self.scene_rep_type in [
                "raymap+depth",
                "raymap+depth+confidence",
                "raymap+depth+mask",
                "raymap+depth+confidence+mask",
            ]:
                # Reshape final scene representation to (B * V, H, W, C)
                output_scene_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted ray origins, directions, and depths along rays
                output_ray_origins, output_ray_directions, output_depth_along_ray = (
                    output_scene_rep.split([3, 3, 1], dim=-1)
                )
                # Get the predicted pointmaps
                output_pts3d = (
                    output_ray_origins + output_ray_directions * output_depth_along_ray
                )
                # Split the predicted quantities back to their respective views
                output_ray_origins_per_view = output_ray_origins.chunk(num_views, dim=0)
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_origins": output_ray_origins_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self.scene_rep_type in [
                "raydirs+depth+pose",
                "raydirs+depth+pose+confidence",
                "raydirs+depth+pose+mask",
                "raydirs+depth+pose+confidence+mask",
            ]:
                # Reshape output dense rep to (B * V, H, W, C)
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted ray directions and depths along rays
                output_ray_directions, output_depth_along_ray = output_dense_rep.split(
                    [3, 1], dim=-1
                )
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the predicted pointmaps in world frame and camera frame
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self.scene_rep_type in [
                "campointmap+pose",
                "campointmap+pose+confidence",
                "campointmap+pose+mask",
                "campointmap+pose+confidence+mask",
            ]:
                # Get the predicted camera frame pointmaps
                output_pts3d_cam = dense_final_outputs.value
                # Reshape final scene representation to (B * V, H, W, C)
                output_pts3d_cam = output_pts3d_cam.permute(0, 2, 3, 1).contiguous()
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the ray directions and depths along rays
                output_depth_along_ray = torch.norm(
                    output_pts3d_cam, dim=-1, keepdim=True
                )
                output_ray_directions = output_pts3d_cam / output_depth_along_ray
                # Get the predicted pointmaps in world frame
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self.scene_rep_type in [
                "pointmap+raydirs+depth+pose",
                "pointmap+raydirs+depth+pose+confidence",
                "pointmap+raydirs+depth+pose+mask",
                "pointmap+raydirs+depth+pose+confidence+mask",
            ]:
                # Reshape final scene representation to (B * V, H, W, C)
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted pointmaps, ray directions and depths along rays
                output_pts3d, output_ray_directions, output_depth_along_ray = (
                    output_dense_rep.split([3, 3, 1], dim=-1)
                )
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the predicted pointmaps in camera frame
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # Replace the predicted world-frame pointmaps if required
                if self.pred_head_config["adaptor_config"][
                    "use_factored_predictions_for_global_pointmaps"
                ]:
                    output_pts3d = (
                        convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                            output_ray_directions,
                            output_depth_along_ray,
                            output_cam_translations,
                            output_cam_quats,
                        )
                    )
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            else:
                raise ValueError(
                    f"Invalid scene_rep_type: {self.scene_rep_type}. \
                    Valid options: ['pointmap', 'raymap+depth', 'raydirs+depth+pose', 'campointmap+pose', 'pointmap+raydirs+depth+pose' \
                                    'pointmap+confidence', 'raymap+depth+confidence', 'raydirs+depth+pose+confidence', 'campointmap+pose+confidence', 'pointmap+raydirs+depth+pose+confidence' \
                                    'pointmap+mask', 'raymap+depth+mask', 'raydirs+depth+pose+mask', 'campointmap+pose+mask', 'pointmap+raydirs+depth+pose+mask' \
                                    'pointmap+confidence+mask', 'raymap+depth+confidence+mask', 'raydirs+depth+pose+confidence+mask', 'campointmap+pose+confidence+mask', 'pointmap+raydirs+depth+pose+confidence+mask']"
                )

            # Get the output confidences for all views (if available) and add them to the result
            if "confidence" in self.scene_rep_type:
                output_confidences = dense_final_outputs.confidence
                # Reshape confidences to (B * V, H, W)
                output_confidences = (
                    output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                )
                # Split the predicted confidences back to their respective views
                output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
                # Add the confidences to the result
                for i in range(num_views):
                    res[i]["conf"] = output_confidences_per_view[i]

            # Get the output masks (and logits) for all views (if available) and add them to the result
            if "mask" in self.scene_rep_type:
                # Get the output masks
                output_masks = dense_final_outputs.mask
                # Reshape masks to (B * V, H, W)
                output_masks = output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                # Threshold the masks at 0.5 to get binary masks (0: ambiguous, 1: non-ambiguous)
                output_masks = output_masks > 0.5
                # Split the predicted masks back to their respective views
                output_masks_per_view = output_masks.chunk(num_views, dim=0)
                # Get the output mask logits (for loss)
                output_mask_logits = dense_final_outputs.logits
                # Reshape mask logits to (B * V, H, W)
                output_mask_logits = (
                    output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                )
                # Split the predicted mask logits back to their respective views
                output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)
                # Add the masks and logits to the result
                for i in range(num_views):
                    res[i]["non_ambiguous_mask"] = output_masks_per_view[i]
                    res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]

        return res

    @torch.inference_mode()
    def infer(
        self,
        views: List[Dict[str, Any]],
        memory_efficient_inference: bool = False,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        apply_mask: bool = True,
        mask_edges: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10,
        ignore_calibration_inputs: bool = False,
        ignore_depth_inputs: bool = False,
        ignore_pose_inputs: bool = False,
        ignore_depth_scale_inputs: bool = False,
        ignore_pose_scale_inputs: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        User-friendly inference with strict input validation and automatic conversion.

        Args:
            views: List of view dictionaries. Each dict can contain:
                Required:
                - 'img': torch.Tensor of shape (B, 3, H, W) - normalized RGB images
                - 'data_norm_type': str - normalization type used to normalize the images (must be equal to self.model.encoder.data_norm_type)

                Optional Geometric Inputs (only one of intrinsics OR ray_directions):
                - 'intrinsics': torch.Tensor of shape (B, 3, 3) - will be converted to ray directions
                - 'ray_directions': torch.Tensor of shape (B, H, W, 3) - ray directions in camera frame
                - 'depth_z': torch.Tensor of shape (B, H, W, 1) - Z depth in camera frame (intrinsics or ray_directions must be provided)
                - 'camera_poses': torch.Tensor of shape (B, 4, 4) or tuple of (quats - (B, 4), trans - (B, 3)) - can be any world frame
                - 'is_metric_scale': bool or torch.Tensor of shape (B,) - if not provided, defaults to True

                Optional Additional Info:
                - 'instance': List[str] where length of list is B - instance info for each view
                - 'idx': List[int] where length of list is B - index info for each view
                - 'true_shape': List[tuple] where length of list is B - true shape info (H, W) for each view

            memory_efficient_inference: Whether to use memory-efficient inference for dense prediction heads (trades off speed). Defaults to False.
            use_amp: Whether to use automatic mixed precision for faster inference. Defaults to True.
            amp_dtype: The dtype to use for mixed precision. Defaults to "bf16" (bfloat16). Options: "fp16", "bf16", "fp32".
            apply_mask: Whether to apply the non-ambiguous mask to the output. Defaults to True.
            mask_edges: Whether to compute an edge mask based on normals and depth and apply it to the output. Defaults to True.
            edge_normal_threshold: Tolerance threshold for normals-based edge detection. Defaults to 5.0.
            edge_depth_threshold: Relative tolerance threshold for depth-based edge detection. Defaults to 0.03.
            apply_confidence_mask: Whether to apply the confidence mask to the output. Defaults to False.
            confidence_percentile: The percentile to use for the confidence threshold. Defaults to 10.
            ignore_calibration_inputs: Whether to ignore the calibration inputs (intrinsics and ray_directions). Defaults to False.
            ignore_depth_inputs: Whether to ignore the depth inputs. Defaults to False.
            ignore_pose_inputs: Whether to ignore the pose inputs. Defaults to False.
            ignore_depth_scale_inputs: Whether to ignore the depth scale inputs. Defaults to False.
            ignore_pose_scale_inputs: Whether to ignore the pose scale inputs. Defaults to False.

        IMPORTANT CONSTRAINTS:
        - Cannot provide both 'intrinsics' and 'ray_directions' (they represent the same information)
        - If 'depth' is provided, then 'intrinsics' or 'ray_directions' must also be provided
        - If ANY view has 'camera_poses', then view 0 (first view) MUST also have 'camera_poses'

        Returns:
            List of prediction dictionaries, one per view. Each dict contains:
                - 'img_no_norm': torch.Tensor of shape (B, H, W, 3) - denormalized rgb images
                - 'pts3d': torch.Tensor of shape (B, H, W, 3) - predicted points in world frame
                - 'pts3d_cam': torch.Tensor of shape (B, H, W, 3) - predicted points in camera frame
                - 'ray_directions': torch.Tensor of shape (B, H, W, 3) - ray directions in camera frame
                - 'intrinsics': torch.Tensor of shape (B, 3, 3) - pinhole camera intrinsics recovered from ray directions
                - 'depth_along_ray': torch.Tensor of shape (B, H, W, 1) - depth along ray in camera frame
                - 'depth_z': torch.Tensor of shape (B, H, W, 1) - Z depth in camera frame
                - 'cam_trans': torch.Tensor of shape (B, 3) - camera translation in world frame
                - 'cam_quats': torch.Tensor of shape (B, 4) - camera quaternion in world frame
                - 'camera_poses': torch.Tensor of shape (B, 4, 4) - camera pose in world frame
                - 'metric_scaling_factor': torch.Tensor of shape (B,) - applied metric scaling factor
                - 'mask': torch.Tensor of shape (B, H, W, 1) - combo of non-ambiguous mask, edge mask and confidence-based mask if used
                - 'non_ambiguous_mask': torch.Tensor of shape (B, H, W) - non-ambiguous mask
                - 'non_ambiguous_mask_logits': torch.Tensor of shape (B, H, W) - non-ambiguous mask logits
                - 'conf': torch.Tensor of shape (B, H, W) - confidence

        Raises:
            ValueError: For invalid inputs, missing required keys, conflicting modalities, or constraint violations
        """
        # Determine the mixed precision floating point type
        if use_amp:
            if amp_dtype == "fp16":
                amp_dtype = torch.float16
            elif amp_dtype == "bf16":
                if torch.cuda.is_bf16_supported():
                    amp_dtype = torch.bfloat16
                else:
                    warnings.warn(
                        "bf16 is not supported on this device. Using fp16 instead."
                    )
                    amp_dtype = torch.float16
            elif amp_dtype == "fp32":
                amp_dtype = torch.float32
        else:
            amp_dtype = torch.float32

        # Validate the input views
        validated_views = validate_input_views_for_inference(views)

        # Transfer the views to the same device as the model
        ignore_keys = set(
            [
                "instance",
                "idx",
                "true_shape",
                "data_norm_type",
            ]
        )
        for view in validated_views:
            for name in view.keys():
                if name in ignore_keys:
                    continue
                val = view[name]
                if name == "camera_poses" and isinstance(val, tuple):
                    view[name] = tuple(
                        x.to(self.device, non_blocking=True) for x in val
                    )
                elif hasattr(val, "to"):
                    view[name] = val.to(self.device, non_blocking=True)

        # Pre-process the input views
        processed_views = preprocess_input_views_for_inference(validated_views)

        # Set the model input probabilities based on input args for ignoring inputs
        self._configure_geometric_input_config(
            use_calibration=not ignore_calibration_inputs,
            use_depth=not ignore_depth_inputs,
            use_pose=not ignore_pose_inputs,
            use_depth_scale=not ignore_depth_scale_inputs,
            use_pose_scale=not ignore_pose_scale_inputs,
        )

        # Run the model
        with torch.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):
            preds = self.forward(
                processed_views, memory_efficient_inference=memory_efficient_inference
            )

        if self.intermediates_only:
            return preds

        # Post-process the model outputs
        preds = postprocess_model_outputs_for_inference(
            raw_outputs=preds,
            input_views=processed_views,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
        )

        # Restore the original configuration
        self._restore_original_geometric_input_config()

        return preds
