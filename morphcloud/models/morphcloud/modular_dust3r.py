# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Modular DUSt3R class defined using UniCeption modules.
"""

from typing import Callable, Dict

import torch
import torch.nn as nn

from uniception.models.encoders import encoder_factory, ViTEncoderInput
from uniception.models.info_sharing.alternating_attention_transformer import (
    MultiViewAlternatingAttentionTransformer,
    MultiViewAlternatingAttentionTransformerIFR,
)
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformer,
    MultiViewCrossAttentionTransformerIFR,
)
from uniception.models.info_sharing.global_attention_transformer import (
    MultiViewGlobalAttentionTransformer,
    MultiViewGlobalAttentionTransformerIFR,
)
from uniception.models.libs.croco.pos_embed import RoPE2D
from uniception.models.prediction_heads.adaptors import PointMapWithConfidenceAdaptor
from uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadInput,
    PredictionHeadLayeredInput,
)
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.linear import LinearFeature

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


class ModularDUSt3R(nn.Module):
    "Modular DUSt3R model class."

    def __init__(
        self,
        name: str,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        pretrained_checkpoint_path: str = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: list = [],
        torch_hub_force_reload: bool = False,
        *args,
        **kwargs,
    ):
        """
        Two-view model containing siamese encoders followed by a two-view attention transformer and respective downstream heads.
        The goal is to output scene representation directly, both outputs in view1's frame (hence the asymmetry).

        Args:
            name (str): Name of the model.
            encoder_config (Dict): Configuration for the encoder.
            info_sharing_config (Dict): Configuration for the two-view attention transformer.
            pred_head_config (Dict): Configuration for the prediction heads.
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            load_specific_pretrained_submodules (bool): Whether to load specific pretrained submodules. (default: False)
            specific_pretrained_submodules (list): List of specific pretrained submodules to load. Must be provided when load_specific_pretrained_submodules is True. (default: [])
            torch_hub_force_reload (bool): Whether to force reload the encoder from torch hub. (default: False)
        """
        super().__init__(*args, **kwargs)

        # Initialize the attributes
        self.name = name
        self.encoder_config = encoder_config
        self.info_sharing_config = info_sharing_config
        self.pred_head_config = pred_head_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules
        self.torch_hub_force_reload = torch_hub_force_reload
        self.class_init_args = {
            "name": self.name,
            "encoder_config": self.encoder_config,
            "info_sharing_config": self.info_sharing_config,
            "pred_head_config": self.pred_head_config,
            "pretrained_checkpoint_path": self.pretrained_checkpoint_path,
            "load_specific_pretrained_submodules": self.load_specific_pretrained_submodules,
            "specific_pretrained_submodules": self.specific_pretrained_submodules,
            "torch_hub_force_reload": self.torch_hub_force_reload,
        }

        # Get relevant parameters from the configs
        custom_positional_encoding = info_sharing_config["custom_positional_encoding"]
        self.info_sharing_type = info_sharing_config["model_type"]
        self.info_sharing_return_type = info_sharing_config["model_return_type"]
        self.pred_head_type = pred_head_config["type"]

        # Initialize Encoder
        if self.encoder_config["uses_torch_hub"]:
            self.encoder_config["torch_hub_force_reload"] = torch_hub_force_reload
        # Create a copy of the config before deleting the key to preserve it for serialization
        encoder_config_copy = self.encoder_config.copy()
        del encoder_config_copy["uses_torch_hub"]
        self.encoder = encoder_factory(**encoder_config_copy)

        # Initialize Custom Positional Encoding if required
        if custom_positional_encoding is not None:
            if isinstance(custom_positional_encoding, str):
                print(
                    f"Using custom positional encoding for multi-view cross attention transformer: {custom_positional_encoding}"
                )
                if custom_positional_encoding.startswith("RoPE"):
                    rope_freq = float(custom_positional_encoding[len("RoPE") :])
                    print(f"RoPE frequency: {rope_freq}")
                    self.custom_positional_encoding = RoPE2D(freq=rope_freq)
                else:
                    raise ValueError(
                        f"Invalid custom_positional_encoding: {custom_positional_encoding}."
                    )
            elif isinstance(custom_positional_encoding, Callable):
                print(
                    "Using callable function as custom positional encoding for multi-view cross attention transformer."
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
        if self.info_sharing_return_type == "no_intermediate_features":
            # Returns only normalized last layer features
            # Initialize multi-view transformer based on type
            if self.info_sharing_type == "cross_attention":
                self.info_sharing = MultiViewCrossAttentionTransformer(
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "global_attention":
                self.info_sharing = MultiViewGlobalAttentionTransformer(
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "alternating_attention":
                self.info_sharing = MultiViewAlternatingAttentionTransformer(
                    **info_sharing_config["module_args"]
                )
            else:
                raise ValueError(
                    f"Invalid info_sharing_type: {self.info_sharing_type}. Valid options: ['cross_attention', 'global_attention', 'alternating_attention']"
                )
        elif self.info_sharing_return_type == "intermediate_features":
            # Returns intermediate features and normalized last layer features
            # Initialize mulit-view transformer based on type
            if self.info_sharing_type == "cross_attention":
                self.info_sharing = MultiViewCrossAttentionTransformerIFR(
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "global_attention":
                self.info_sharing = MultiViewGlobalAttentionTransformerIFR(
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "alternating_attention":
                self.info_sharing = MultiViewAlternatingAttentionTransformerIFR(
                    **info_sharing_config["module_args"]
                )
            else:
                raise ValueError(
                    f"Invalid info_sharing_type: {self.info_sharing_type}. Valid options: ['cross_attention', 'global_attention', 'alternating_attention']"
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
                f"Invalid info_sharing_return_type: {self.info_sharing_return_type}. Valid options: ['no_intermediate_features', 'intermediate_features']"
            )

        # Add dependencies to prediction head config
        pred_head_config["feature_head"]["patch_size"] = self.encoder.patch_size
        if self.pred_head_type == "linear":
            pred_head_config["feature_head"]["input_feature_dim"] = (
                self.info_sharing.dim
            )
        elif self.pred_head_type == "dpt":
            if self.use_encoder_features_for_dpt:
                pred_head_config["feature_head"]["input_feature_dims"] = [
                    self.encoder.enc_embed_dim
                ] + [self.info_sharing.dim] * 3
            else:
                pred_head_config["feature_head"]["input_feature_dims"] = [
                    self.info_sharing.dim
                ] * 4
            pred_head_config["regressor_head"]["input_feature_dim"] = pred_head_config[
                "feature_head"
            ]["feature_dim"]
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt']"
            )

        # Initialize Prediction Heads
        if self.pred_head_type == "linear":
            # Initialize Prediction Head 1
            self.head1 = LinearFeature(**pred_head_config["feature_head"])
            # Initialize Prediction Head 2
            self.head2 = LinearFeature(**pred_head_config["feature_head"])
        elif self.pred_head_type == "dpt":
            # Initialize Prediction Head 1
            self.dpt_feature_head1 = DPTFeature(**pred_head_config["feature_head"])
            self.dpt_regressor_head1 = DPTRegressionProcessor(
                **pred_head_config["regressor_head"]
            )
            self.head1 = nn.Sequential(self.dpt_feature_head1, self.dpt_regressor_head1)
            # Initialize Prediction Head 2
            self.dpt_feature_head2 = DPTFeature(**pred_head_config["feature_head"])
            self.dpt_regressor_head2 = DPTRegressionProcessor(
                **pred_head_config["regressor_head"]
            )
            self.head2 = nn.Sequential(self.dpt_feature_head2, self.dpt_regressor_head2)
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt']"
            )

        # Initialize Final Output Adaptor
        if pred_head_config["adaptor_type"] == "pointmap+confidence":
            self.adaptor = PointMapWithConfidenceAdaptor(**pred_head_config["adaptor"])
            self.scene_rep_type = "pointmap"
        else:
            raise ValueError(
                f"Invalid adaptor_type: {pred_head_config['adaptor_type']}. Valid options: ['pointmap+confidence']"
            )

        # Load pretrained weights
        if self.pretrained_checkpoint_path is not None:
            if not self.load_specific_pretrained_submodules:
                print(
                    f"Loading pretrained weights from {self.pretrained_checkpoint_path} ..."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                print(self.load_state_dict(ckpt["model"]))
            else:
                print(
                    f"Loading pretrained weights from {self.pretrained_checkpoint_path} for specific submodules: {specific_pretrained_submodules} ..."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                filtered_ckpt = {}
                for ckpt_key, ckpt_value in ckpt["model"].items():
                    for submodule in specific_pretrained_submodules:
                        if ckpt_key.startswith(submodule):
                            filtered_ckpt[ckpt_key] = ckpt_value
                print(self.load_state_dict(filtered_ckpt, strict=False))

    def _encode_image_pairs(self, img1, img2, data_norm_type):
        "Encode two different batches of images (each batch can have different image shape)"
        if img1.shape[-2:] == img2.shape[-2:]:
            encoder_input = ViTEncoderInput(
                image=torch.cat((img1, img2), dim=0), data_norm_type=data_norm_type
            )
            encoder_output = self.encoder(encoder_input)
            out, out2 = encoder_output.features.chunk(2, dim=0)
        else:
            encoder_input = ViTEncoderInput(image=img1, data_norm_type=data_norm_type)
            out = self.encoder(encoder_input)
            out = out.features
            encoder_input2 = ViTEncoderInput(image=img2, data_norm_type=data_norm_type)
            out2 = self.encoder(encoder_input2)
            out2 = out2.features

        return out, out2

    def _encode_symmetrized(self, view1, view2):
        "Encode image pairs accounting for symmetrization, i.e., (a, b) and (b, a) always exist in the input"
        img1 = view1["img"]
        img2 = view2["img"]
        if isinstance(view1["data_norm_type"], list):
            assert all(
                [x == view1["data_norm_type"][0] for x in view1["data_norm_type"]]
            ), "All data_norm_type values should be the same in the list."
            data_norm_type = view1["data_norm_type"][0]
        elif isinstance(view1["data_norm_type"], str):
            data_norm_type = view1["data_norm_type"]
        else:
            raise ValueError(
                f"Invalid data_norm_type: {view1['data_norm_type']}. Should be either a list with all same values or a string."
            )
        feat1, feat2 = self._encode_image_pairs(
            img1, img2, data_norm_type=data_norm_type
        )

        return feat1, feat2

    def _downstream_head(self, head_num, decout, img_shape):
        "Run the respective prediction heads"
        head = getattr(self, f"head{head_num}")
        if self.pred_head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.pred_head_type == "dpt":
            head_input = PredictionHeadLayeredInput(
                list_features=decout[f"{head_num}"], target_output_shape=img_shape
            )

        return head(head_input)

    def forward(self, views):
        """
        Forward pass performing the following operations:
        1. Encodes the two input views (images).
        2. Combines the encoded features using a two-view attention transformer.
        3. Passes the combined features through the respective prediction heads.
        4. Returns the processed final outputs for both views.

        Args:
            views (List(dict)): A list of size two whose elements are:
                view1 (dict): Dictionary containing the first view's images and instance information.
                            "img" is a required key and value is a tensor of shape (B, C, H, W).
                view2 (dict): Dictionary containing the second view's images and instance information.
                            "img" is a required key and value is a tensor of shape (B, C, H, W).

        Returns:
            List[dict, dict]: A list containing the final outputs for both views.
        """
        # Get input shapes
        view1 = views[0]
        view2 = views[1]
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        if "img_encoder_feats" in view1 and "img_encoder_feats" in view2:
            # Reuse the pre-computed image features for the two views
            feat1 = view1["img_encoder_feats"]
            feat2 = view2["img_encoder_feats"]
        else:
            # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
            feat1, feat2 = self._encode_symmetrized(view1, view2)

        # Combine all images into view-centric representation
        info_sharing_input = MultiViewTransformerInput(features=[feat1, feat2])
        if self.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
        elif self.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self.info_sharing(info_sharing_input)

        if self.pred_head_type == "linear":
            # Define feature dictionary for linear head
            info_sharing_outputs = {
                "1": final_info_sharing_multi_view_feat.features[0].float(),
                "2": final_info_sharing_multi_view_feat.features[1].float(),
            }
        elif self.pred_head_type == "dpt":
            # Define feature dictionary for DPT head
            if self.use_encoder_features_for_dpt:
                info_sharing_outputs = {
                    "1": [
                        feat1.float(),
                        intermediate_info_sharing_multi_view_feat[0]
                        .features[0]
                        .float(),
                        intermediate_info_sharing_multi_view_feat[1]
                        .features[0]
                        .float(),
                        final_info_sharing_multi_view_feat.features[0].float(),
                    ],
                    "2": [
                        feat2.float(),
                        intermediate_info_sharing_multi_view_feat[0]
                        .features[1]
                        .float(),
                        intermediate_info_sharing_multi_view_feat[1]
                        .features[1]
                        .float(),
                        final_info_sharing_multi_view_feat.features[1].float(),
                    ],
                }
            else:
                info_sharing_outputs = {
                    "1": [
                        intermediate_info_sharing_multi_view_feat[0]
                        .features[0]
                        .float(),
                        intermediate_info_sharing_multi_view_feat[1]
                        .features[0]
                        .float(),
                        intermediate_info_sharing_multi_view_feat[2]
                        .features[0]
                        .float(),
                        final_info_sharing_multi_view_feat.features[0].float(),
                    ],
                    "2": [
                        intermediate_info_sharing_multi_view_feat[0]
                        .features[1]
                        .float(),
                        intermediate_info_sharing_multi_view_feat[1]
                        .features[1]
                        .float(),
                        intermediate_info_sharing_multi_view_feat[2]
                        .features[1]
                        .float(),
                        final_info_sharing_multi_view_feat.features[1].float(),
                    ],
                }

        # Downstream task prediction
        with torch.autocast("cuda", enabled=False):
            # Prediction heads
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)
            head_output2 = self._downstream_head(2, info_sharing_outputs, shape2)

            # Post-process outputs
            final_output1 = self.adaptor(
                AdaptorInput(
                    adaptor_feature=head_output1.decoded_channels,
                    output_shape_hw=shape1,
                )
            )
            final_output2 = self.adaptor(
                AdaptorInput(
                    adaptor_feature=head_output2.decoded_channels,
                    output_shape_hw=shape2,
                )
            )

            # Reshape final scene representation to (B, H, W, C)
            final_scene_rep1 = final_output1.value.permute(0, 2, 3, 1).contiguous()
            final_scene_rep2 = final_output2.value.permute(0, 2, 3, 1).contiguous()

            # Convert output scene representation to pointmaps
            if self.scene_rep_type == "pointmap":
                output_pts3d1 = final_scene_rep1
                output_pts3d2 = final_scene_rep2
            else:
                raise ValueError(f"Invalid scene_rep_type: {self.scene_rep_type}.")

            # Reshape confidence to (B, H, W, 1)
            output_conf1 = (
                final_output1.confidence.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )
            output_conf2 = (
                final_output2.confidence.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )

            # Convert outputs to dictionary
            res1 = {
                "pts3d": output_pts3d1,
                "conf": output_conf1,
            }
            res2 = {
                "pts3d": output_pts3d2,
                "conf": output_conf2,
            }
            res = [res1, res2]

        return res
