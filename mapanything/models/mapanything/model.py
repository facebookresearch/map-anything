# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything model class defined using UniCeption modules.
"""

import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from mapanything.utils.geometry import (
    apply_log_to_norm,
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    normalize_depth_using_non_zero_pixels,
    normalize_pose_translations,
    transform_pose_using_quats_and_trans_2_to_1,
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
from uniception.models.prediction_heads.adaptors import (
    CamTranslationPlusQuatsAdaptor,
    PointMapAdaptor,
    PointMapPlusRayDirectionsPlusDepthAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithMaskAdaptor,
    PointMapWithConfidenceAdaptor,
    PointMapWithConfidenceAndMaskAdaptor,
    PointMapWithMaskAdaptor,
    RayDirectionsPlusDepthAdaptor,
    RayDirectionsPlusDepthWithConfidenceAdaptor,
    RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor,
    RayDirectionsPlusDepthWithMaskAdaptor,
    RayMapPlusDepthAdaptor,
    RayMapPlusDepthWithConfidenceAdaptor,
    RayMapPlusDepthWithConfidenceAndMaskAdaptor,
    RayMapPlusDepthWithMaskAdaptor,
    ScaleAdaptor,
)
from uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadInput,
    PredictionHeadLayeredInput,
    PredictionHeadTokenInput,
)
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.linear import LinearFeature
from uniception.models.prediction_heads.mlp_head import MLPHead
from uniception.models.prediction_heads.pose_head import PoseHead

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


class MapAnything(nn.Module, PyTorchModelHubMixin):
    "Modular MapAnything model class that supports input of images & optional geometric modalities (multiple reconstruction tasks)."

    def __init__(
        self,
        name: str,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        geometric_input_config: Dict,
        fusion_norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: str = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: list = None,
        torch_hub_force_reload: bool = False,
    ):
        """
        Multi-view model containing an image encoder fused with optional geometric modalities followed by a multi-view attention transformer and respective downstream heads.
        The goal is to output scene representation.
        The multi-view attention transformer also takes as input a scale token to predict the metric scaling factor for the predicted scene representation.

        Args:
            name (str): Name of the model.
            encoder_config (Dict): Configuration for the encoder.
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
            pred_head_config (Dict): Configuration for the prediction heads.
            geometric_input_config (Dict): Configuration for the input of optional geometric modalities.
            fusion_norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Normalization layer to use after fusion (addition) of encoder and geometric modalities. (default: partial(nn.LayerNorm, eps=1e-6))
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            load_specific_pretrained_submodules (bool): Whether to load specific pretrained submodules. (default: False)
            specific_pretrained_submodules (list): List of specific pretrained submodules to load. Must be provided when load_specific_pretrained_submodules is True. (default: None)
            torch_hub_force_reload (bool): Whether to force reload the encoder from torch hub. (default: False)
        """
        super().__init__()

        # Initialize the attributes
        self.name = name
        self.encoder_config = encoder_config
        self.info_sharing_config = info_sharing_config
        self.pred_head_config = pred_head_config
        self.geometric_input_config = geometric_input_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules
        self.torch_hub_force_reload = torch_hub_force_reload
        self.class_init_args = {
            "name": self.name,
            "encoder_config": self.encoder_config,
            "info_sharing_config": self.info_sharing_config,
            "pred_head_config": self.pred_head_config,
            "geometric_input_config": self.geometric_input_config,
            "pretrained_checkpoint_path": self.pretrained_checkpoint_path,
            "load_specific_pretrained_submodules": self.load_specific_pretrained_submodules,
            "specific_pretrained_submodules": self.specific_pretrained_submodules,
            "torch_hub_force_reload": self.torch_hub_force_reload,
        }

        # Get relevant parameters from the configs
        self.info_sharing_type = info_sharing_config["model_type"]
        self.info_sharing_return_type = info_sharing_config["model_return_type"]
        self.pred_head_type = pred_head_config["type"]

        # Initialize image encoder
        if self.encoder_config["uses_torch_hub"]:
            self.encoder_config["torch_hub_force_reload"] = torch_hub_force_reload
        # Create a copy of the config before deleting the key to preserve it for serialization
        encoder_config_copy = self.encoder_config.copy()
        del encoder_config_copy["uses_torch_hub"]
        self.encoder = encoder_factory(**encoder_config_copy)

        # Initialize the encoder for ray directions
        ray_dirs_encoder_config = self.geometric_input_config["ray_dirs_encoder_config"]
        ray_dirs_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        ray_dirs_encoder_config["patch_size"] = self.encoder.patch_size
        self.ray_dirs_encoder = encoder_factory(**ray_dirs_encoder_config)

        # Initialize the encoder for depth (normalized per view and values after normalization are scaled logarithmically)
        depth_encoder_config = self.geometric_input_config["depth_encoder_config"]
        depth_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        depth_encoder_config["patch_size"] = self.encoder.patch_size
        self.depth_encoder = encoder_factory(**depth_encoder_config)

        # Initialize the encoder for log scale factor of depth
        depth_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        depth_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.depth_scale_encoder = encoder_factory(**depth_scale_encoder_config)

        # Initialize the encoder for camera rotation
        cam_rot_encoder_config = self.geometric_input_config["cam_rot_encoder_config"]
        cam_rot_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_rot_encoder = encoder_factory(**cam_rot_encoder_config)

        # Initialize the encoder for camera translation (normalized across all provided camera translations)
        cam_trans_encoder_config = self.geometric_input_config[
            "cam_trans_encoder_config"
        ]
        cam_trans_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_trans_encoder = encoder_factory(**cam_trans_encoder_config)

        # Initialize the encoder for log scale factor of camera translation
        cam_trans_scale_encoder_config = self.geometric_input_config[
            "scale_encoder_config"
        ]
        cam_trans_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_trans_scale_encoder = encoder_factory(**cam_trans_scale_encoder_config)

        # Initialize the fusion norm layer
        self.fusion_norm_layer = fusion_norm_layer(self.encoder.enc_embed_dim)

        # Initialize the Scale Token
        # Used to scale the final scene predictions to metric scale
        # During inference extended to (B, C, T), where T is the number of tokens (i.e., 1)
        self.scale_token = nn.Parameter(torch.zeros(self.encoder.enc_embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)

        # Initialize the info sharing module (multi-view transformer)
        self._initialize_info_sharing(info_sharing_config)

        # Initialize the prediction heads
        self._initialize_prediction_heads(pred_head_config)

        # Initialize the final adaptors
        self._initialize_adaptors(pred_head_config)

        # Load pretrained weights
        self._load_pretrained_weights()

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

    def _initialize_prediction_heads(self, pred_head_config):
        """
        Initialize the prediction heads based on the prediction head configuration.

        This method configures and initializes the appropriate prediction heads based on the
        specified prediction head type (linear, DPT, or DPT+pose). It sets up the necessary
        dependencies and creates the required model components.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads.

        Returns:
            None

        Raises:
            ValueError: If an invalid pred_head_type is provided.
        """
        # Add dependencies to prediction head config
        pred_head_config["feature_head"]["patch_size"] = self.encoder.patch_size
        if self.pred_head_type == "linear":
            pred_head_config["feature_head"]["input_feature_dim"] = (
                self.info_sharing.dim
            )
        elif "dpt" in self.pred_head_type:
            # Add dependencies for DPT & Regressor head
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
            # Add dependencies for Pose head if required
            if "pose" in self.pred_head_type:
                pred_head_config["pose_head"]["patch_size"] = self.encoder.patch_size
                pred_head_config["pose_head"]["input_feature_dim"] = (
                    self.info_sharing.dim
                )
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )
        pred_head_config["scale_head"]["input_feature_dim"] = self.info_sharing.dim

        # Initialize Prediction Heads
        if self.pred_head_type == "linear":
            # Initialize Dense Prediction Head for all views
            self.dense_head = LinearFeature(**pred_head_config["feature_head"])
        elif "dpt" in self.pred_head_type:
            # Initialize Dense Prediction Head for all views
            self.dpt_feature_head = DPTFeature(**pred_head_config["feature_head"])
            self.dpt_regressor_head = DPTRegressionProcessor(
                **pred_head_config["regressor_head"]
            )
            self.dense_head = nn.Sequential(
                self.dpt_feature_head, self.dpt_regressor_head
            )
            # Initialize Pose Head for all views if required
            if "pose" in self.pred_head_type:
                self.pose_head = PoseHead(**pred_head_config["pose_head"])
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )
        self.scale_head = MLPHead(**pred_head_config["scale_head"])

    def _initialize_adaptors(self, pred_head_config):
        """
        Initialize the adaptors based on the prediction head configuration.

        This method sets up the appropriate adaptors for different scene representation types,
        such as pointmaps, ray maps with depth, or ray directions with depth and pose.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads including adaptor type.

        Returns:
            None

        Raises:
            ValueError: If an invalid adaptor_type is provided.
            AssertionError: If ray directions + depth + pose is used with an incompatible head type.
        """
        if pred_head_config["adaptor_type"] == "pointmap":
            self.dense_adaptor = PointMapAdaptor(**pred_head_config["adaptor"])
            self.scene_rep_type = "pointmap"
        elif pred_head_config["adaptor_type"] == "pointmap+confidence":
            self.dense_adaptor = PointMapWithConfidenceAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "pointmap+confidence"
        elif pred_head_config["adaptor_type"] == "pointmap+mask":
            self.dense_adaptor = PointMapWithMaskAdaptor(**pred_head_config["adaptor"])
            self.scene_rep_type = "pointmap+mask"
        elif pred_head_config["adaptor_type"] == "pointmap+confidence+mask":
            self.dense_adaptor = PointMapWithConfidenceAndMaskAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "pointmap+confidence+mask"
        elif pred_head_config["adaptor_type"] == "raymap+depth":
            self.dense_adaptor = RayMapPlusDepthAdaptor(**pred_head_config["adaptor"])
            self.scene_rep_type = "raymap+depth"
        elif pred_head_config["adaptor_type"] == "raymap+depth+confidence":
            self.dense_adaptor = RayMapPlusDepthWithConfidenceAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+confidence"
        elif pred_head_config["adaptor_type"] == "raymap+depth+mask":
            self.dense_adaptor = RayMapPlusDepthWithMaskAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+mask"
        elif pred_head_config["adaptor_type"] == "raymap+depth+confidence+mask":
            self.dense_adaptor = RayMapPlusDepthWithConfidenceAndMaskAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+confidence+mask"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+confidence":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithConfidenceAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+confidence"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+mask"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+confidence+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+confidence+mask"
        elif pred_head_config["adaptor_type"] == "campointmap+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapAdaptor(**pred_head_config["dpt_adaptor"])
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+confidence":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithConfidenceAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+confidence"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+mask"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+confidence+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithConfidenceAndMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+confidence+mask"
        elif pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapPlusRayDirectionsPlusDepthAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose"
        elif (
            pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose+confidence"
        ):
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = (
                PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor(
                    **pred_head_config["dpt_adaptor"]
                )
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+confidence"
        elif pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapPlusRayDirectionsPlusDepthWithMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+mask"
        elif (
            pred_head_config["adaptor_type"]
            == "pointmap+raydirs+depth+pose+confidence+mask"
        ):
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = (
                PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(
                    **pred_head_config["dpt_adaptor"]
                )
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+confidence+mask"
        else:
            raise ValueError(
                f"Invalid adaptor_type: {pred_head_config['adaptor_type']}. \
                Valid options: ['pointmap', 'raymap+depth', 'raydirs+depth+pose', 'campointmap+pose', 'pointmap+raydirs+depth+pose' \
                                'pointmap+confidence', 'raymap+depth+confidence', 'raydirs+depth+pose+confidence', 'campointmap+pose+confidence', 'pointmap+raydirs+depth+pose+confidence' \
                                'pointmap+mask', 'raymap+depth+mask', 'raydirs+depth+pose+mask', 'campointmap+pose+mask', 'pointmap+raydirs+depth+pose+mask' \
                                'pointmap+confidence+mask', 'raymap+depth+confidence+mask', 'raydirs+depth+pose+confidence+mask', 'campointmap+pose+confidence+mask', 'pointmap+raydirs+depth+pose+confidence+mask']"
            )
        self.scale_adaptor = ScaleAdaptor(**pred_head_config["scale_adaptor"])

    def _load_pretrained_weights(self):
        """
        Load pretrained weights from a checkpoint file.

        If load_specific_pretrained_submodules is True, only loads weights for the specified submodules.
        Otherwise, loads all weights from the checkpoint.

        Returns:
            None
        """
        if self.pretrained_checkpoint_path is not None:
            if not self.load_specific_pretrained_submodules:
                print(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} ..."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                print(self.load_state_dict(ckpt["model"]))
            else:
                print(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} for specific submodules: {self.specific_pretrained_submodules} ..."
                )
                assert self.pred_head_type is not None, (
                    "Specific submodules to load cannot be None."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                filtered_ckpt = {}
                for ckpt_key, ckpt_value in ckpt["model"].items():
                    for submodule in self.specific_pretrained_submodules:
                        if ckpt_key.startswith(submodule):
                            filtered_ckpt[ckpt_key] = ckpt_value
                print(self.load_state_dict(filtered_ckpt, strict=False))

    def _encode_n_views(self, views):
        """
        Encode all the input views (batch of images) in a single forward pass.
        Assumes all the input views have the same image shape, batch size, and data normalization type.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        num_views = len(views)
        data_norm_type = views[0]["data_norm_type"][0]
        imgs_list = [view["img"] for view in views]
        all_imgs_across_views = torch.cat(imgs_list, dim=0)
        encoder_input = ViTEncoderInput(
            image=all_imgs_across_views, data_norm_type=data_norm_type
        )
        encoder_output = self.encoder(encoder_input)
        all_encoder_features_across_views = encoder_output.features.chunk(
            num_views, dim=0
        )

        return all_encoder_features_across_views

    def _compute_pose_quats_and_trans_for_across_views_in_ref_view(
        self,
        views,
        num_views,
        device,
        dtype,
        batch_size_per_view,
        per_sample_cam_input_mask,
    ):
        """
        Compute the pose quats and trans for all the views in the frame of the reference view 0.
        Returns identity pose for views where the camera input mask is False or the pose is not provided.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            device (torch.device): Device to use for the computation.
            dtype (torch.dtype): Data type to use for the computation.
            per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

        Returns:
            torch.Tensor: A tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
            torch.Tensor: A tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
            torch.Tensor: A tensor containing the per sample camera input mask.
        """
        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        pose_quats_non_ref_views = []
        pose_trans_non_ref_views = []
        pose_quats_ref_view_0 = []
        pose_trans_ref_view_0 = []
        for view_idx in range(num_views):
            per_sample_cam_input_mask_for_curr_view = per_sample_cam_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ]
            if (
                "camera_pose_quats" in views[view_idx]
                and "camera_pose_trans" in views[view_idx]
                and per_sample_cam_input_mask_for_curr_view.any()
            ):
                # Get the camera pose quats and trans for the current view
                cam_pose_quats = views[view_idx]["camera_pose_quats"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                cam_pose_trans = views[view_idx]["camera_pose_trans"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                # Append to the list
                pose_quats_non_ref_views.append(cam_pose_quats)
                pose_trans_non_ref_views.append(cam_pose_trans)
                # Get the camera pose quats and trans for the reference view 0
                cam_pose_quats = views[0]["camera_pose_quats"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                cam_pose_trans = views[0]["camera_pose_trans"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                # Append to the list
                pose_quats_ref_view_0.append(cam_pose_quats)
                pose_trans_ref_view_0.append(cam_pose_trans)
            else:
                per_sample_cam_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False

        # Initialize the pose quats and trans for all views as identity
        pose_quats_across_views = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device
        ).repeat(batch_size_per_view * num_views, 1)  # (q_x, q_y, q_z, q_w)
        pose_trans_across_views = torch.zeros(
            (batch_size_per_view * num_views, 3), dtype=dtype, device=device
        )

        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        if len(pose_quats_non_ref_views) > 0:
            # Stack the pose quats and trans for all the non-reference views and reference view 0
            pose_quats_non_ref_views = torch.cat(pose_quats_non_ref_views, dim=0)
            pose_trans_non_ref_views = torch.cat(pose_trans_non_ref_views, dim=0)
            pose_quats_ref_view_0 = torch.cat(pose_quats_ref_view_0, dim=0)
            pose_trans_ref_view_0 = torch.cat(pose_trans_ref_view_0, dim=0)

            # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
            (
                pose_quats_non_ref_views_in_ref_view_0,
                pose_trans_non_ref_views_in_ref_view_0,
            ) = transform_pose_using_quats_and_trans_2_to_1(
                pose_quats_ref_view_0,
                pose_trans_ref_view_0,
                pose_quats_non_ref_views,
                pose_trans_non_ref_views,
            )

            # Update the pose quats and trans for all the non-reference views
            pose_quats_across_views[per_sample_cam_input_mask] = (
                pose_quats_non_ref_views_in_ref_view_0.to(dtype=dtype)
            )
            pose_trans_across_views[per_sample_cam_input_mask] = (
                pose_trans_non_ref_views_in_ref_view_0.to(dtype=dtype)
            )

        return (
            pose_quats_across_views,
            pose_trans_across_views,
            per_sample_cam_input_mask,
        )

    def _encode_and_fuse_ray_dirs(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        per_sample_ray_dirs_input_mask,
    ):
        """
        Encode the ray directions for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_ray_dirs_input_mask (torch.Tensor): Tensor containing the per sample ray direction input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Get the height and width of the images
        _, _, height, width = views[0]["img"].shape

        # Get the ray directions for all the views where info is provided and the ray direction input mask is True
        ray_dirs_list = []
        for view_idx in range(num_views):
            per_sample_ray_dirs_input_mask_for_curr_view = (
                per_sample_ray_dirs_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ]
            )
            ray_dirs_for_curr_view = torch.zeros(
                (batch_size_per_view, height, width, 3),
                dtype=all_encoder_features_across_views.dtype,
                device=all_encoder_features_across_views.device,
            )
            if (
                "ray_directions_cam" in views[view_idx]
                and per_sample_ray_dirs_input_mask_for_curr_view.any()
            ):
                ray_dirs_for_curr_view[per_sample_ray_dirs_input_mask_for_curr_view] = (
                    views[view_idx]["ray_directions_cam"][
                        per_sample_ray_dirs_input_mask_for_curr_view
                    ]
                )
            else:
                per_sample_ray_dirs_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False
            ray_dirs_list.append(ray_dirs_for_curr_view)

        # Stack the ray directions for all the views and permute to (B * V, C, H, W)
        ray_dirs = torch.cat(ray_dirs_list, dim=0)  # (B * V, H, W, 3)
        ray_dirs = ray_dirs.permute(0, 3, 1, 2).contiguous()  # (B * V, 3, H, W)

        # Encode the ray directions
        ray_dirs_features_across_views = self.ray_dirs_encoder(
            ViTEncoderNonImageInput(data=ray_dirs)
        ).features

        # Fuse the ray direction features with the other encoder features (zero out the features where the ray direction input mask is False)
        ray_dirs_features_across_views = (
            ray_dirs_features_across_views
            * per_sample_ray_dirs_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        all_encoder_features_across_views = (
            all_encoder_features_across_views + ray_dirs_features_across_views
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_depths(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        per_sample_depth_input_mask,
    ):
        """
        Encode the z depths for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_depth_input_mask (torch.Tensor): Tensor containing the per sample depth input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Get the device and height and width of the images
        device = all_encoder_features_across_views.device
        _, _, height, width = views[0]["img"].shape

        # Decide to use randomly sampled sparse depth or dense depth
        if torch.rand(1) < self.geometric_input_config["sparse_depth_prob"]:
            use_sparse_depth = True
        else:
            use_sparse_depth = False

        # Get the depths for all the views
        depth_list = []
        depth_norm_factors_list = []
        metric_scale_depth_mask_list = []
        for view_idx in range(num_views):
            # Get the input mask for current view
            per_sample_depth_input_mask_for_curr_view = per_sample_depth_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ]
            depth_for_curr_view = torch.zeros(
                (batch_size_per_view, height, width, 1),
                dtype=all_encoder_features_across_views.dtype,
                device=device,
            )
            depth_norm_factor_for_curr_view = torch.zeros(
                (batch_size_per_view),
                dtype=all_encoder_features_across_views.dtype,
                device=device,
            )
            metric_scale_mask_for_curr_view = torch.zeros(
                (batch_size_per_view),
                dtype=torch.bool,
                device=device,
            )
            if (
                "depth_along_ray" in views[view_idx]
            ) and per_sample_depth_input_mask_for_curr_view.any():
                # Get depth for current view
                depth_for_curr_view_input = views[view_idx]["depth_along_ray"][
                    per_sample_depth_input_mask_for_curr_view
                ]
                # Get the metric scale mask
                if "is_metric_scale" in views[view_idx]:
                    metric_scale_mask = views[view_idx]["is_metric_scale"][
                        per_sample_depth_input_mask_for_curr_view
                    ]
                else:
                    metric_scale_mask = torch.zeros(
                        depth_for_curr_view_input.shape[0],
                        dtype=torch.bool,
                        device=device,
                    )
                # Turn off indication of metric scale samples based on the depth_scale_norm_all_prob
                depth_scale_norm_all_mask = (
                    torch.rand(metric_scale_mask.shape[0])
                    < self.geometric_input_config["depth_scale_norm_all_prob"]
                )
                if depth_scale_norm_all_mask.any():
                    metric_scale_mask[depth_scale_norm_all_mask] = False
                # Assign the metric scale mask to the respective indices
                metric_scale_mask_for_curr_view[
                    per_sample_depth_input_mask_for_curr_view
                ] = metric_scale_mask
                # Sparsely sample the depth if required
                if use_sparse_depth:
                    # Create a mask of ones
                    sparsification_mask = torch.ones_like(
                        depth_for_curr_view_input, device=device
                    )
                    # Create a mask for valid pixels (depth > 0)
                    valid_pixel_mask = depth_for_curr_view_input > 0
                    # Calculate the number of valid pixels
                    num_valid_pixels = valid_pixel_mask.sum().item()
                    # Calculate the number of valid pixels to set to zero
                    num_to_zero = int(
                        num_valid_pixels
                        * self.geometric_input_config["sparsification_removal_percent"]
                    )
                    if num_to_zero > 0:
                        # Get the indices of valid pixels
                        valid_indices = valid_pixel_mask.nonzero(as_tuple=True)
                        # Randomly select indices to zero out
                        indices_to_zero = torch.randperm(num_valid_pixels)[:num_to_zero]
                        # Set selected valid indices to zero in the mask
                        sparsification_mask[
                            valid_indices[0][indices_to_zero],
                            valid_indices[1][indices_to_zero],
                            valid_indices[2][indices_to_zero],
                            valid_indices[3][indices_to_zero],
                        ] = 0
                    # Apply the mask on the depth
                    depth_for_curr_view_input = (
                        depth_for_curr_view_input * sparsification_mask
                    )
                # Normalize the depth
                scaled_depth_for_curr_view_input, depth_norm_factor = (
                    normalize_depth_using_non_zero_pixels(
                        depth_for_curr_view_input, return_norm_factor=True
                    )
                )
                # Assign the depth and depth norm factor to the respective indices
                depth_for_curr_view[per_sample_depth_input_mask_for_curr_view] = (
                    scaled_depth_for_curr_view_input
                )
                depth_norm_factor_for_curr_view[
                    per_sample_depth_input_mask_for_curr_view
                ] = depth_norm_factor
            else:
                per_sample_depth_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False
            # Append the depths, depth norm factor and metric scale mask for the current view
            depth_list.append(depth_for_curr_view)
            depth_norm_factors_list.append(depth_norm_factor_for_curr_view)
            metric_scale_depth_mask_list.append(metric_scale_mask_for_curr_view)

        # Stack the depths for all the views and permute to (B * V, C, H, W)
        depths = torch.cat(depth_list, dim=0)  # (B * V, H, W, 1)
        depths = apply_log_to_norm(
            depths
        )  # Scale logarithimically (norm is computed along last dim)
        depths = depths.permute(0, 3, 1, 2).contiguous()  # (B * V, 1, H, W)
        # Encode the depths using the depth encoder
        depth_features_across_views = self.depth_encoder(
            ViTEncoderNonImageInput(data=depths)
        ).features
        # Zero out the depth features where the depth input mask is False
        depth_features_across_views = (
            depth_features_across_views
            * per_sample_depth_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        # Stack the depth norm factors for all the views
        depth_norm_factors = torch.cat(depth_norm_factors_list, dim=0)  # (B * V, )
        # Encode the depth norm factors using the log scale encoder for depth
        log_depth_norm_factors = torch.log(depth_norm_factors + 1e-8)  # (B * V, )
        depth_scale_features_across_views = self.depth_scale_encoder(
            EncoderGlobalRepInput(data=log_depth_norm_factors.unsqueeze(-1))
        ).features
        # Zero out the depth scale features where the depth input mask is False
        depth_scale_features_across_views = (
            depth_scale_features_across_views
            * per_sample_depth_input_mask.unsqueeze(-1)
        )
        # Stack the metric scale mask for all the views
        metric_scale_depth_mask = torch.cat(
            metric_scale_depth_mask_list, dim=0
        )  # (B * V, )
        # Zero out the depth scale features where the metric scale mask is False
        # Scale encoding is only provided for metric scale samples
        depth_scale_features_across_views = (
            depth_scale_features_across_views * metric_scale_depth_mask.unsqueeze(-1)
        )

        # Fuse the depth features & depth scale features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + depth_features_across_views
            + depth_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_cam_quats_and_trans(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        pose_quats_across_views,
        pose_trans_across_views,
        per_sample_cam_input_mask,
    ):
        """
        Encode the camera quats and trans for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            pose_quats_across_views (torch.Tensor): Tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
            pose_trans_across_views (torch.Tensor): Tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
            per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Encode the pose quats
        pose_quats_features_across_views = self.cam_rot_encoder(
            EncoderGlobalRepInput(data=pose_quats_across_views)
        ).features
        # Zero out the pose quat features where the camera input mask is False
        pose_quats_features_across_views = (
            pose_quats_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # Get the metric scale mask for all samples
        device = all_encoder_features_across_views.device
        metric_scale_pose_trans_mask = torch.zeros(
            (batch_size_per_view * num_views), dtype=torch.bool, device=device
        )
        for view_idx in range(num_views):
            if "is_metric_scale" in views[view_idx]:
                # Get the metric scale mask for the input pose priors
                metric_scale_mask = views[view_idx]["is_metric_scale"]
            else:
                metric_scale_mask = torch.zeros(
                    batch_size_per_view, dtype=torch.bool, device=device
                )
            metric_scale_pose_trans_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ] = metric_scale_mask

        # Turn off indication of metric scale samples based on the pose_scale_norm_all_prob
        pose_norm_all_mask = (
            torch.rand(batch_size_per_view * num_views)
            < self.geometric_input_config["pose_scale_norm_all_prob"]
        )
        if pose_norm_all_mask.any():
            metric_scale_pose_trans_mask[pose_norm_all_mask] = False

        # Get the scale norm factor for all the samples and scale the pose translations
        pose_trans_across_views = torch.split(
            pose_trans_across_views, batch_size_per_view, dim=0
        )  # Split into num_views chunks
        pose_trans_across_views = torch.stack(
            pose_trans_across_views, dim=1
        )  # Stack the views along a new dimension (batch_size_per_view, num_views, 3)
        scaled_pose_trans_across_views, pose_trans_norm_factors = (
            normalize_pose_translations(
                pose_trans_across_views, return_norm_factor=True
            )
        )

        # Resize the pose translation back to (batch_size_per_view * num_views, 3) and extend the norm factor to (batch_size_per_view * num_views, 1)
        scaled_pose_trans_across_views = scaled_pose_trans_across_views.unbind(
            dim=1
        )  # Convert back to list of views, where each view has batch_size_per_view tensor
        scaled_pose_trans_across_views = torch.cat(
            scaled_pose_trans_across_views, dim=0
        )  # Concatenate back to (batch_size_per_view * num_views, 3)
        pose_trans_norm_factors_across_views = pose_trans_norm_factors.unsqueeze(
            -1
        ).repeat(num_views, 1)  # (B, ) -> (B * V, 1)

        # Encode the pose trans
        pose_trans_features_across_views = self.cam_trans_encoder(
            EncoderGlobalRepInput(data=scaled_pose_trans_across_views)
        ).features
        # Zero out the pose trans features where the camera input mask is False
        pose_trans_features_across_views = (
            pose_trans_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # Encode the pose translation norm factors using the log scale encoder for pose trans
        log_pose_trans_norm_factors_across_views = torch.log(
            pose_trans_norm_factors_across_views + 1e-8
        )
        pose_trans_scale_features_across_views = self.cam_trans_scale_encoder(
            EncoderGlobalRepInput(data=log_pose_trans_norm_factors_across_views)
        ).features
        # Zero out the pose trans scale features where the camera input mask is False
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * per_sample_cam_input_mask.unsqueeze(-1)
        )
        # Zero out the pose trans scale features where the metric scale mask is False
        # Scale encoding is only provided for metric scale samples
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * metric_scale_pose_trans_mask.unsqueeze(-1)
        )

        # Fuse the pose quat features, pose trans features, pose trans scale features and pose trans type PE features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + pose_quats_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

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
        )

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

        # Get the depth input mask
        per_sample_depth_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["depth_prob"]
        )
        per_sample_depth_input_mask = per_sample_depth_input_mask.repeat(num_views)
        per_sample_depth_input_mask = (
            per_sample_depth_input_mask & per_sample_geometric_input_mask
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

        # Encode the ray directions and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_ray_dirs(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_ray_dirs_input_mask,
        )

        # Encode the depths and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_depths(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_depth_input_mask,
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

        # Split the batched views into individual views
        fused_all_encoder_features_across_views = (
            all_encoder_features_across_views.chunk(num_views, dim=0)
        )

        return fused_all_encoder_features_across_views

    def _compute_adaptive_minibatch_size(
        self,
        memory_safety_factor: float = 0.95,
    ) -> int:
        """
        Compute adaptive minibatch size based on available PyTorch memory.

        Args:
            memory_safety_factor: Safety factor to avoid OOM (0.95 = use 95% of available memory)

        Returns:
            Computed minibatch size
        """
        device = self.device

        if device.type == "cuda":
            # Get available GPU memory
            torch.cuda.empty_cache()
            available_memory = torch.cuda.mem_get_info()[0]  # Free memory in bytes
            usable_memory = (
                available_memory * memory_safety_factor
            )  # Use safety factor to avoid OOM
        else:
            # For non-CUDA devices, use conservative default
            print(
                "Non-CUDA device detected. Using conservative default minibatch size of 1 for memory efficient dense prediction head inference."
            )
            return 1

        # Determine minibatch size based on available memory
        max_estimated_memory_per_sample = (
            680 * 1024 * 1024
        )  # 680 MB per sample (upper bound profiling using a 518 x 518 input)
        computed_minibatch_size = int(usable_memory / max_estimated_memory_per_sample)
        if computed_minibatch_size < 1:
            computed_minibatch_size = 1

        return computed_minibatch_size

    def downstream_dense_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        img_shape: Tuple[int, int],
    ):
        """
        Run the downstream dense prediction head
        """
        if self.pred_head_type == "linear":
            dense_head_outputs = self.dense_head(
                PredictionHeadInput(last_feature=dense_head_inputs)
            )
            dense_final_outputs = self.dense_adaptor(
                AdaptorInput(
                    adaptor_feature=dense_head_outputs.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )
        elif self.pred_head_type in ["dpt", "dpt+pose"]:
            dense_head_outputs = self.dense_head(
                PredictionHeadLayeredInput(
                    list_features=dense_head_inputs,
                    target_output_shape=img_shape,
                )
            )
            dense_final_outputs = self.dense_adaptor(
                AdaptorInput(
                    adaptor_feature=dense_head_outputs.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )

        return dense_final_outputs

    def downstream_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        scale_head_inputs: torch.Tensor,
        img_shape: Tuple[int, int],
        memory_efficient_inference: bool = False,
    ):
        """
        Run Prediction Heads & Post-Process Outputs
        """
        # Get device
        device = self.device

        # Use mini-batch inference to run the dense prediction head (the memory bottleneck)
        # This saves memory and is slower than running the dense prediction head in one go
        if memory_efficient_inference:
            # Obtain the batch size of the dense head inputs
            if self.pred_head_type == "linear":
                batch_size = dense_head_inputs.shape[0]
            elif self.pred_head_type in ["dpt", "dpt+pose"]:
                batch_size = dense_head_inputs[0].shape[0]
            else:
                raise ValueError(
                    f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
                )

            # Compute the mini batch size and number of mini batches adaptively based on available memory
            minibatch = self._compute_adaptive_minibatch_size()
            num_batches = (batch_size + minibatch - 1) // minibatch

            # Run prediction for each mini-batch
            dense_final_outputs_list = []
            pose_final_outputs_list = [] if self.pred_head_type == "dpt+pose" else None
            for batch_idx in range(num_batches):
                start_idx = batch_idx * minibatch
                end_idx = min((batch_idx + 1) * minibatch, batch_size)

                # Get the inputs for the current mini-batch
                if self.pred_head_type == "linear":
                    dense_head_inputs_batch = dense_head_inputs[start_idx:end_idx]
                elif self.pred_head_type in ["dpt", "dpt+pose"]:
                    dense_head_inputs_batch = [
                        x[start_idx:end_idx] for x in dense_head_inputs
                    ]
                else:
                    raise ValueError(
                        f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
                    )

                # Dense prediction (mini-batched)
                dense_final_outputs_batch = self.downstream_dense_head(
                    dense_head_inputs_batch, img_shape
                )
                dense_final_outputs_list.append(dense_final_outputs_batch)

                # Pose prediction (mini-batched)
                if self.pred_head_type == "dpt+pose":
                    pose_head_inputs_batch = dense_head_inputs[-1][start_idx:end_idx]
                    pose_head_outputs_batch = self.pose_head(
                        PredictionHeadInput(last_feature=pose_head_inputs_batch)
                    )
                    pose_final_outputs_batch = self.pose_adaptor(
                        AdaptorInput(
                            adaptor_feature=pose_head_outputs_batch.decoded_channels,
                            output_shape_hw=img_shape,
                        )
                    )
                    pose_final_outputs_list.append(pose_final_outputs_batch)

            # Concatenate the dense prediction head outputs from all mini-batches
            available_keys = dense_final_outputs_batch.__dict__.keys()
            dense_pred_data_dict = {
                key: torch.cat(
                    [getattr(output, key) for output in dense_final_outputs_list], dim=0
                )
                for key in available_keys
            }
            dense_final_outputs = dense_final_outputs_batch.__class__(
                **dense_pred_data_dict
            )

            # Concatenate the pose prediction head outputs from all mini-batches
            pose_final_outputs = None
            if self.pred_head_type == "dpt+pose":
                available_keys = pose_final_outputs_batch.__dict__.keys()
                pose_pred_data_dict = {
                    key: torch.cat(
                        [getattr(output, key) for output in pose_final_outputs_list],
                        dim=0,
                    )
                    for key in available_keys
                }
                pose_final_outputs = pose_final_outputs_batch.__class__(
                    **pose_pred_data_dict
                )

            # Clear CUDA cache for better memory efficiency
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            # Run prediction for all (batch_size * num_views) in one go
            # Dense prediction
            dense_final_outputs = self.downstream_dense_head(
                dense_head_inputs, img_shape
            )

            # Pose prediction
            pose_final_outputs = None
            if self.pred_head_type == "dpt+pose":
                pose_head_outputs = self.pose_head(
                    PredictionHeadInput(last_feature=dense_head_inputs[-1])
                )
                pose_final_outputs = self.pose_adaptor(
                    AdaptorInput(
                        adaptor_feature=pose_head_outputs.decoded_channels,
                        output_shape_hw=img_shape,
                    )
                )

        # Scale prediction is lightweight, so we can run it in one go
        scale_head_output = self.scale_head(
            PredictionHeadTokenInput(last_feature=scale_head_inputs)
        )
        scale_final_output = self.scale_adaptor(
            AdaptorInput(
                adaptor_feature=scale_head_output.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        scale_final_output = scale_final_output.value.squeeze(-1)  # (B, 1, 1) -> (B, 1)

        # Clear CUDA cache for better memory efficiency
        if memory_efficient_inference and device.type == "cuda":
            torch.cuda.empty_cache()

        return dense_final_outputs, pose_final_outputs, scale_final_output

    def forward(self, views, memory_efficient_inference=False):
        """
        Forward pass performing the following operations:
        1. Encodes the N input views (images).
        2. Encodes the optional geometric inputs (ray directions, depths, camera rotations, camera translations).
        3. Fuses the encoded features from the N input views and the optional geometric inputs using addition and normalization.
        4. Information sharing across the encoded features and a scale token using a multi-view attention transformer.
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
        all_encoder_features_across_views = self._encode_n_views(views)

        # Encode the optional geometric inputs and fuse with the encoded features from the N input views
        # Use high precision to prevent NaN values after layer norm in dense representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
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
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )
        if self.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
        elif self.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self.info_sharing(info_sharing_input)

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

    def _configure_geometric_input_config(
        self,
        use_calibration: bool,
        use_depth: bool,
        use_pose: bool,
        use_depth_scale: bool,
        use_pose_scale: bool,
    ):
        """
        Configure the geometric input configuration
        """
        # Store original config for restoration
        if not hasattr(self, "_original_geometric_config"):
            self._original_geometric_config = dict(self.geometric_input_config)

        # Set the geometric input configuration
        if not (use_calibration or use_depth or use_pose):
            # No geometric inputs (images-only mode)
            self.geometric_input_config.update(
                {
                    "overall_prob": 0.0,
                    "dropout_prob": 1.0,
                    "ray_dirs_prob": 0.0,
                    "depth_prob": 0.0,
                    "cam_prob": 0.0,
                    "sparse_depth_prob": 0.0,
                    "depth_scale_norm_all_prob": 0.0,
                    "pose_scale_norm_all_prob": 0.0,
                }
            )
        else:
            # Enable geometric inputs with deterministic behavior
            self.geometric_input_config.update(
                {
                    "overall_prob": 1.0,
                    "dropout_prob": 0.0,
                    "ray_dirs_prob": 1.0 if use_calibration else 0.0,
                    "depth_prob": 1.0 if use_depth else 0.0,
                    "cam_prob": 1.0 if use_pose else 0.0,
                    "sparse_depth_prob": 0.0,
                    "depth_scale_norm_all_prob": 0.0 if use_depth_scale else 1.0,
                    "pose_scale_norm_all_prob": 0.0 if use_pose_scale else 1.0,
                }
            )

    def _restore_original_geometric_input_config(self):
        """
        Restore original geometric input configuration
        """
        if hasattr(self, "_original_geometric_config"):
            self.geometric_input_config.update(self._original_geometric_config)

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
                view[name] = view[name].to(self.device, non_blocking=True)

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
