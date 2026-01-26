import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn

from mapanything.models.mapanything_adapter.alternating_attention_adapter import (
    MVAATAdapterIFR,
)
from mapanything.utils.geometry import (
    normalize_pose_translations,
    transform_pose_using_quats_and_trans_2_to_1,
)
from uniception.models.encoders import (
    encoder_factory,
    EncoderGlobalRepInput,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
)
from uniception.models.info_sharing.alternating_attention_transformer import (
    MultiViewAlternatingAttentionTransformer,
)
from uniception.models.info_sharing.base import MultiViewTransformerInput

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


class Predicter(nn.Module):
    """
    Predicter module for I-JEPA like architecture with geometric conditioning.
    Combines multi-view alternating attention transformer with geometric condition encoders.
    Args:
        enc_embed_dim (int): Embedding dimension for the encoders.
        enc_patch_size (int): Patch size for the encoders.
        predicter_config (Dict): Configuration dictionary for the predicter module.
        geometric_input_config (Dict): Configuration dictionary for the geometric input encoders.
        fusion_norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Normalization layer to use for feature fusion. Default is LayerNorm.
    """
    def __init__(
            self,
            enc_embed_dim: int,
            enc_patch_size: int,
            predicter_config: Dict,
            geometric_input_config: Dict,
            fusion_norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
                nn.LayerNorm, eps=1e-6
            ),
            *args,
            **kwargs,
        ):
        super().__init__()

        self.enc_embed_dim = enc_embed_dim
        self.enc_patch_size = enc_patch_size
        self.predicter_config = predicter_config
        self.geometric_input_config = geometric_input_config

        # Add dependencies to info_sharing_config
        self.predicter_config["module_args"]["input_embed_dim"] = (
            self.enc_embed_dim
        )
        self.predicter = MultiViewAlternatingAttentionTransformer(
            **self.predicter_config["module_args"],
        )

        # Initialize the encoder for ray directions
        ray_dirs_condition_encoder_config = self.geometric_input_config["ray_dirs_encoder_config"]
        ray_dirs_condition_encoder_config["enc_embed_dim"] = self.enc_embed_dim
        ray_dirs_condition_encoder_config["patch_size"] = self.enc_patch_size
        self.ray_dirs_condition_encoder = encoder_factory(**ray_dirs_condition_encoder_config)

        # Initialize the encoder for camera rotation
        cam_rot_encoder_config = self.geometric_input_config["cam_rot_encoder_config"]
        cam_rot_encoder_config["enc_embed_dim"] = self.enc_embed_dim
        self.cam_rot_condition_encoder = encoder_factory(**cam_rot_encoder_config)

        # Initialize the encoder for camera translation (normalized across all provided camera translations)
        cam_trans_encoder_config = self.geometric_input_config[
            "cam_trans_encoder_config"
        ]
        cam_trans_encoder_config["enc_embed_dim"] = self.enc_embed_dim
        self.cam_trans_condition_encoder = encoder_factory(**cam_trans_encoder_config)

        # Initialize the encoder for log scale factor of camera translation
        cam_trans_scale_encoder_config = self.geometric_input_config[
            "scale_encoder_config"
        ]
        cam_trans_scale_encoder_config["enc_embed_dim"] = self.enc_embed_dim
        self.cam_trans_scale_condition_encoder = encoder_factory(**cam_trans_scale_encoder_config)

        # Initialize the fusion norm layer
        self.fusion_norm_layer = fusion_norm_layer(self.enc_embed_dim)
    
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

    def _condition_ray_dirs(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_features_across_views,
        per_sample_ray_dirs_input_mask,
    ):
        """
        Encode the ray directions for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
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
                dtype=all_features_across_views.dtype,
                device=all_features_across_views.device,
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
        ray_dirs_features_across_views = self.ray_dirs_condition_encoder(
            ViTEncoderNonImageInput(data=ray_dirs)
        ).features

        # Fuse the ray direction features with the other encoder features (zero out the features where the ray direction input mask is False)
        ray_dirs_features_across_views = (
            ray_dirs_features_across_views
            * per_sample_ray_dirs_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        all_features_across_views = (
            all_features_across_views + ray_dirs_features_across_views
        )

        return all_features_across_views
    
    def _condition_cam_quats_and_trans(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_features_across_views,
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
            all_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            pose_quats_across_views (torch.Tensor): Tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
            pose_trans_across_views (torch.Tensor): Tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
            per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Encode the pose quats
        pose_quats_features_across_views = self.cam_rot_condition_encoder(
            EncoderGlobalRepInput(data=pose_quats_across_views)
        ).features
        # Zero out the pose quat features where the camera input mask is False
        pose_quats_features_across_views = (
            pose_quats_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # Get the metric scale mask for all samples
        device = all_features_across_views.device
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
        pose_trans_features_across_views = self.cam_trans_condition_encoder(
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
        pose_trans_scale_features_across_views = self.cam_trans_scale_condition_encoder(
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
        all_features_across_views = (
            all_features_across_views
            + pose_quats_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_features_across_views
    
    def forward(
        self, views, target_mask, intermediate_features_across_views
    ):
        num_views = len(views)
        batch_size_per_view, _, _, _ = views[0]["img"].shape
        device = intermediate_features_across_views.features[0].device
        dtype = intermediate_features_across_views.features[0].dtype
        all_features_across_views = torch.cat(
            intermediate_features_across_views.features, dim=0
        )  # (V * B)

        per_sample_target_mask = target_mask.repeat_interleave(batch_size_per_view)  # (V * B)

        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        # Returned pose quats and trans represent identity pose for views/samples where the camera input mask is False
        pose_quats_across_views, pose_trans_across_views, per_sample_cam_input_mask = (
            self._compute_pose_quats_and_trans_for_across_views_in_ref_view(
                views,
                num_views,
                device,
                dtype,
                batch_size_per_view,
                per_sample_target_mask,
            )
        )

        # Encode the ray directions and fuse with the image encoder features
        all_features_across_views = self._condition_ray_dirs(
            views,
            num_views,
            batch_size_per_view,
            all_features_across_views,
            per_sample_target_mask,
        )

        # Encode the cam quat and trans and fuse with the image encoder features
        all_features_across_views = self._condition_cam_quats_and_trans(
            views,
            num_views,
            batch_size_per_view,
            all_features_across_views,
            pose_quats_across_views,
            pose_trans_across_views,
            per_sample_cam_input_mask,
        )

        # Normalize the fused features (permute -> normalize -> permute)
        all_features_across_views = all_features_across_views.permute(
            0, 2, 3, 1
        ).contiguous()
        all_features_across_views = self.fusion_norm_layer(
            all_features_across_views
        )
        all_features_across_views = all_features_across_views.permute(
            0, 3, 1, 2
        ).contiguous()

        # Split the batched views into individual views
        fused_all_features_across_views = (
            all_features_across_views.chunk(num_views, dim=0)
        )

        predicter_input = MultiViewTransformerInput(
            features=fused_all_features_across_views,
        )

        pred_target_features = self.predicter(predicter_input)

        return pred_target_features