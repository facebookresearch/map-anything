"""
UniCeption Alternating-Attention Transformer for Information Sharing
"""

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

from mapanything.models.mapanything_adapter.adapter_module import (
    deform_inputs,
    Extractor,
)
from uniception.models.encoders.dense_rep_encoder import ResidualBlock
from uniception.models.info_sharing import MultiViewAlternatingAttentionTransformerIFR
from uniception.models.info_sharing.base import (
    MultiViewTransformerInput,
    MultiViewTransformerOutput,
    UniCeptionInfoSharingBase,
)
from uniception.models.utils.intermediate_feature_return import (
    feature_take_indices,
    IntermediateFeatureReturner,
)
from uniception.models.utils.positional_encoding import PositionGetter
from uniception.models.utils.transformer_blocks import Mlp, SelfAttentionBlock


class MVAATAdapterIFR(MultiViewAlternatingAttentionTransformerIFR):
    """
    Adapter version of the Multi-View Alternating-Attention Transformer with Intermediate Feature Return.
    The additional branch integrates information from a separate feature stream using interaction modules
    to extract additional semantic information.
    Args:
        adapter_indices (List[int]): List of layer indices where adapter interaction modules are inserted.
    """

    def __init__(
        self,
        adapter: Dict,
        *args,
        **kwargs,
    ):
        # Init the base classes
        super().__init__(*args, **kwargs)

        self.adapter_dim = adapter["dim"]
        self.adapter_num_heads = adapter["num_heads"]
        self.adapter_mlp_ratio = adapter["mlp_ratio"]
        self.adapter_indices = adapter["indices"]

        self.branch_projector = ResidualBlock(self.input_embed_dim, self.input_embed_dim)

        if self.input_embed_dim != self.adapter_dim:
            adapter_proj_embeds = [nn.Linear(self.input_embed_dim, self.adapter_dim) for _ in range(1 + len(self.adapter_indices))]
        else:
            adapter_proj_embeds = [nn.Identity() for _ in range(1 + len(self.adapter_indices))]
        self.adapter_proj_embeds = nn.ModuleList(adapter_proj_embeds)

        self.adapter_interactions = nn.ModuleList(
            [
                Extractor(
                    dim=self.adapter_dim,
                    num_heads=8,
                    n_points=4,
                    cffn_ratio=0.25,
                )
                for _ in range(len(self.adapter_indices))
            ]
        )

        self.adapter_attentions = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.adapter_dim,
                    num_heads=self.adapter_num_heads,
                    mlp_ratio=self.adapter_mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    proj_drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    init_values=self.init_values,
                    drop_path=self.drop_path,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    mlp_layer=self.mlp_layer,
                    custom_positional_encoding=self.custom_positional_encoding,
                    use_scalable_softmax=self.use_scalable_softmax,
                    use_entropy_scaling=self.use_entropy_scaling,
                    base_token_count_for_entropy_scaling=self.base_token_count_for_entropy_scaling,
                    entropy_scaling_growth_factor=self.entropy_scaling_growth_factor,
                )
                for _ in range(2 * len(self.adapter_indices))
            ]
        )

        self.adapter_norm = self.norm_layer(self.adapter_dim)

    def stop_non_adapter_gradient_flow(self):
        """
        Stop gradient flow through all non-adapter modules.
        """
        for param in self.parameters():
            param.requires_grad = False

        for param in self.branch_projector.parameters():
            param.requires_grad = True

        for param in self.adapter_proj_embeds.parameters():
            param.requires_grad = True

        for param in self.adapter_interactions.parameters():
            param.requires_grad = True

        for param in self.adapter_attentions.parameters():
            param.requires_grad = True

        for param in self.adapter_norm.parameters():
            param.requires_grad = True

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput], List[MultiViewTransformerOutput]]:
        """
        Forward interface for the Multi-View Alternating-Attention Transformer with Intermediate Feature Return.
        This Adapter version includes additional interaction modules to fuse information from a separate branch.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.
                Optionally, the input can also include:
                - additional_input_tokens: Global additional tokens (e.g., scale token)
                  which are appended to the token set from all multi-view features and only participate in global-level attention.
                  Shape: (batch, input_embed_dim, num_of_additional_tokens).
                - additional_input_tokens_per_view: Per-view additional tokens (e.g., view-specific tokens like per-view registers)
                  which are appended to each view's token set and participate in both frame-level and global-level attention.
                  List of tensors, each of shape (batch, input_embed_dim, num_of_additional_tokens_per_view).

        Returns:
            Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]]]:
                Output of the model post information sharing.
                    - Final output (MultiViewTransformerOutput) with features, additional_token_features, and additional_token_features_per_view
                    - List of intermediate outputs (MultiViewTransformerOutput) from specified layers
        """
        # Check that the number of views matches the input and the features are of expected shape
        if self.use_pe_for_non_reference_views:
            assert (
                len(model_input.features) <= self.max_num_views_for_pe
            ), f"Expected less than {self.max_num_views_for_pe} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Get the indices of the intermediate features to return
        intermediate_multi_view_features = []
        adapter_multi_view_features = []
        take_indices, _ = feature_take_indices(self.depth, self.indices)

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Prepare branch input
        branch_input = self.branch_projector(torch.cat(multi_view_features, dim=0))  # (V * B, C, H, W)
        branch_input = branch_input.split(batch_size, dim=0)  # List of N tensors of shape (B, C, H, W)
        branch_input = MultiViewTransformerInput(
            features=branch_input,  # V * (B, C, H, W)
        )

        # Process per-view additional tokens if provided
        num_of_additional_tokens_per_view = 0
        if model_input.additional_input_tokens_per_view is not None:
            additional_tokens_per_view = model_input.additional_input_tokens_per_view
            assert len(additional_tokens_per_view) == num_of_views, (
                f"Number of additional token tensors ({len(additional_tokens_per_view)}) "
                f"must match number of views ({num_of_views})"
            )
            assert all(
                tokens.ndim == 3 for tokens in additional_tokens_per_view
            ), "Additional tokens per view must have 3 dimensions (N, C, T)"
            assert all(
                tokens.shape[1] == self.input_embed_dim for tokens in additional_tokens_per_view
            ), f"Additional tokens per view must have input dimension {self.input_embed_dim}"
            assert all(
                tokens.shape[0] == batch_size for tokens in additional_tokens_per_view
            ), "Batch size mismatch for additional tokens per view"

            num_of_additional_tokens_per_view = additional_tokens_per_view[0].shape[2]

            # Concatenate per-view additional tokens to each view's features
            multi_view_features_with_tokens = []
            for view_idx, (view_features, view_tokens) in enumerate(
                zip(multi_view_features, additional_tokens_per_view)
            ):
                # view_features: (N, C, H, W)
                # view_tokens: (N, C, T)
                # Flatten spatial dimensions: (N, C, H, W) -> (N, C, H*W)
                view_features_flat = view_features.reshape(batch_size, self.input_embed_dim, height * width)
                # Concatenate tokens: (N, C, H*W + T)
                view_with_tokens = torch.cat([view_features_flat, view_tokens], dim=2)
                multi_view_features_with_tokens.append(view_with_tokens)

            # Stack all views: (N, V, C, H*W + T)
            multi_view_features = torch.stack(multi_view_features_with_tokens, dim=1)
            # Permute to (N, V, H*W + T, C)
            multi_view_features = multi_view_features.permute(0, 1, 3, 2)
            # Reshape to (N, V * (H*W + T), C)
            multi_view_features = multi_view_features.reshape(
                batch_size,
                num_of_views * (height * width + num_of_additional_tokens_per_view),
                self.input_embed_dim,
            ).contiguous()

            # Update tokens per view to include additional tokens
            num_of_tokens_per_view = height * width + num_of_additional_tokens_per_view
        else:
            # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
            multi_view_features = torch.stack(multi_view_features, dim=1)

            # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
            multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
            multi_view_features = multi_view_features.reshape(
                batch_size, num_of_views * height * width, self.input_embed_dim
            ).contiguous()

        # @NEW, Stack the branch features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
        branch_features = branch_input.features
        deform_inps = deform_inputs(branch_features[0])
        branch_features = torch.stack(branch_features, dim=1)

        # Resize the branch features from NVCHW to NLC, where L = V * H * W
        branch_features = branch_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
        branch_features = branch_features.reshape(
            batch_size * num_of_views, height * width, self.input_embed_dim
        ).contiguous()  # (N * V, H * W, C)

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:
            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)
        # @NEW, Project branch features to the transformer dimension
        branch_features = self.adapter_proj_embeds[0](branch_features)

        # Raise error if custom positional encoding is used with additional tokens
        if self.custom_positional_encoding is not None:
            if (
                model_input.additional_input_tokens is not None
                or model_input.additional_input_tokens_per_view is not None
            ):
                raise ValueError(
                    "Custom positional encoding is not supported when additional_input_tokens or "
                    "additional_input_tokens_per_view are provided. Please set custom_positional_encoding=None "
                    "or remove additional tokens from the input."
                )

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:
            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[1]
            multi_view_positions = multi_view_positions + additional_tokens_positions

        if self.distinguish_ref_and_non_ref_views:
            # Add positional encoding for reference view (idx 0)
            ref_view_pe = self.view_pos_table[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
        else:
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]

        if self.distinguish_ref_and_non_ref_views and self.use_pe_for_non_reference_views:
            # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
            if self.use_rand_idx_pe_for_non_reference_views:
                non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views_for_pe, size=(num_of_views - 1,))
            else:
                non_ref_view_pe_indices = torch.arange(1, num_of_views)
            non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
            non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
            non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
            non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            non_ref_view_features = non_ref_view_features + non_ref_view_pe
        else:
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]

        # Concatenate the reference and non-reference view features
        # Handle additional tokens (no view-based positional encoding for them)
        if model_input.additional_input_tokens is not None:
            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features, additional_features], dim=1)
        else:
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)


        def unflatten_mv_features(current_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Extract view features (excluding global additional tokens)
            view_features_flat = current_features[:, : num_of_views * num_of_tokens_per_view, :]

            # Extract per-view additional tokens if they were provided
            per_view_additional = None
            if model_input.additional_input_tokens_per_view is not None:
                # Reshape to (N, V, H*W + T_per_view, C)
                view_features_with_tokens = view_features_flat.reshape(
                    batch_size, num_of_views, num_of_tokens_per_view, self.dim
                )

                # Split into spatial features and per-view additional tokens
                spatial_tokens_per_view = height * width
                view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
                per_view_additional = view_features_with_tokens[
                    :, :, spatial_tokens_per_view:, :
                ]  # (N, V, T_per_view, C)

                # Reshape view features to (N, V, H, W, C)
                view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
                view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            else:
                # Reshape the intermediate multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
                view_features = view_features_flat.reshape(
                    batch_size, num_of_views, height, width, self.dim
                )  # (N, V, H, W, C)
                view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            return view_features, per_view_additional  # (N, V, C, H, W), (N, V, T_per_view, C)


        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            if depth_idx % 2 == 0:
                # Apply the self-attention block and update the multi-view features
                # Global attention across all views
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)
            else:
                # Handle additional tokens separately for frame-level attention
                additional_features = None
                additional_positions = None
                if model_input.additional_input_tokens is not None:
                    # Extract additional token features
                    additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
                    # Keep only view features for frame-level attention
                    multi_view_features = multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

                    # Handle positions for additional tokens if custom positional encoding is used
                    if self.custom_positional_encoding is not None:
                        additional_positions = multi_view_positions[:, num_of_views * num_of_tokens_per_view :, :]
                        multi_view_positions = multi_view_positions[:, : num_of_views * num_of_tokens_per_view, :]

                # Reshape the multi-view features from (N, V * (H*W + T_per_view), C) to (N * V, (H*W + T_per_view), C)
                # Note: When T_per_view = 0 (no per-view tokens), this is (N, V * H*W, C) to (N * V, H*W, C)
                multi_view_features = multi_view_features.reshape(
                    batch_size * num_of_views, num_of_tokens_per_view, self.dim
                ).contiguous()  # (N * V, (H*W + T_per_view), C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size * num_of_views, num_of_tokens_per_view, 2
                    ).contiguous()  # (N * V, (H*W + T_per_view), 2)

                # Apply the self-attention block and update the multi-view features
                # Frame-level attention within each view (including per-view additional tokens if provided)
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)

                # Reshape the multi-view features from (N * V, (H*W + T_per_view), C) back to (N, V * (H*W + T_per_view), C)
                multi_view_features = multi_view_features.reshape(
                    batch_size, num_of_views * num_of_tokens_per_view, self.dim
                ).contiguous()  # (N, V * (H*W + T_per_view), C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size, num_of_views * num_of_tokens_per_view, 2
                    ).contiguous()  # (N, V * (H*W + T_per_view), 2)

                # Reattach additional tokens if they exist
                if additional_features is not None:
                    multi_view_features = torch.cat([multi_view_features, additional_features], dim=1)
                    # Reattach positions for additional tokens if they exist
                    if additional_positions is not None:
                        multi_view_positions = torch.cat([multi_view_positions, additional_positions], dim=1)
            if depth_idx in take_indices:
                # Normalize the intermediate features with final norm layer if enabled
                intermediate_multi_view_features.append(
                    self.norm(multi_view_features) if self.norm_intermediate else multi_view_features
                )
            
            # @NEW, Apply adapter interaction module at specified steps
            if depth_idx in self.adapter_indices:
                deform_values, _ = unflatten_mv_features(multi_view_features)  # (N, V, C, H, W)
                deform_values = deform_values.permute(0, 1, 3, 4, 2).contiguous()  # (N, V, H, W, C)
                deform_values = deform_values.reshape(
                    batch_size * num_of_views, height * width, self.dim
                ).contiguous()  # (N * V, H * W, C)

                deform_values = self.adapter_proj_embeds[self.adapter_indices.index(depth_idx) + 1](deform_values)

                # Apply the interaction block to fuse information from the branch features
                extractor = self.adapter_interactions[self.adapter_indices.index(depth_idx)]
                branch_features = extractor(
                    query=branch_features, reference_points=deform_inps[0],
                    feat=deform_values, spatial_shapes=deform_inps[1],
                    level_start_index=deform_inps[2],
                    H=height, W=width,
                )  # (N * V, H * W, C)

                branch_features = self.adapter_attentions[2 * self.adapter_indices.index(depth_idx)](
                    branch_features, multi_view_positions if multi_view_positions[0] is not None else None
                )  # (N * V, H * W, C)

                # Extra global attention
                branch_features = branch_features.reshape(
                    batch_size, num_of_views * height * width, self.adapter_dim
                ).contiguous()  # (N, V * H * W, C)
                
                branch_features = self.adapter_attentions[2 * self.adapter_indices.index(depth_idx) + 1](
                    branch_features, multi_view_positions if multi_view_positions[0] is not None else None
                )  # (N, V * H * W, C)

                adapter_multi_view_features.append(
                    self.adapter_norm(branch_features) if self.norm_intermediate else branch_features
                )
                branch_features = branch_features.reshape(
                    batch_size * num_of_views, height * width, self.adapter_dim
                ).contiguous()  # (N * V, H * W, C)

        # Reshape the intermediate features and convert to MultiViewTransformerOutput class
        for idx in range(len(intermediate_multi_view_features)):
            # Get the current intermediate features
            current_features = intermediate_multi_view_features[idx]

            view_features, per_view_additional = unflatten_mv_features(current_features)
            # Split the intermediate multi-view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [
                intermediate_view_features.squeeze(dim=1) for intermediate_view_features in view_features
            ]

            additional_token_features_per_view = None
            if per_view_additional is not None:
                # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
                per_view_additional = per_view_additional.split(1, dim=1)
                additional_token_features_per_view = [
                    tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                    for tokens in per_view_additional
                ]

            # Extract and return additional token features (global) if provided
            additional_token_features = None
            if model_input.additional_input_tokens is not None:
                additional_token_features = current_features[:, num_of_views * num_of_tokens_per_view :, :]
                additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

            intermediate_multi_view_features[idx] = MultiViewTransformerOutput(
                features=view_features,
                additional_token_features=additional_token_features,
                additional_token_features_per_view=additional_token_features_per_view,
            )

        # @NEW, Reshape the intermediate btranch features and convert to MultiViewTransformerOutput class
        for idx in range(len(adapter_multi_view_features)):
            # Get the current intermediate features
            current_features = adapter_multi_view_features[idx]

            view_features = current_features.reshape(
                batch_size, num_of_views, height, width, self.adapter_dim
            )  # (N, V, H, W, C)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)
            # Split the intermediate multi-view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [
                intermediate_view_features.squeeze(dim=1) for intermediate_view_features in view_features
            ]
            adapter_multi_view_features[idx] = MultiViewTransformerOutput(features=view_features)

        # # Return only the intermediate features if enabled
        # if self.intermediates_only:
        #     return intermediate_multi_view_features, adapter_multi_view_features

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Extract view features (excluding global additional tokens)
        view_features_flat = output_multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

        # Extract per-view additional tokens if they were provided
        additional_token_features_per_view = None
        if model_input.additional_input_tokens_per_view is not None:
            # Reshape to (N, V, H*W + T_per_view, C)
            view_features_with_tokens = view_features_flat.reshape(
                batch_size, num_of_views, num_of_tokens_per_view, self.dim
            )

            # Split into spatial features and per-view additional tokens
            spatial_tokens_per_view = height * width
            view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
            per_view_additional = view_features_with_tokens[:, :, spatial_tokens_per_view:, :]  # (N, V, T_per_view, C)

            # Reshape view features to (N, V, H, W, C)
            view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

            # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
            per_view_additional = per_view_additional.split(1, dim=1)
            additional_token_features_per_view = [
                tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                for tokens in per_view_additional
            ]
        else:
            # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
            view_features = view_features_flat.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split the output multi-view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

        # Extract and return additional token features (global) if provided
        additional_token_features = None
        if model_input.additional_input_tokens is not None:
            additional_token_features = output_multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

        output_multi_view_features = MultiViewTransformerOutput(
            features=view_features,
            additional_token_features=additional_token_features,
            additional_token_features_per_view=additional_token_features_per_view,
        )

        return output_multi_view_features, intermediate_multi_view_features, adapter_multi_view_features
