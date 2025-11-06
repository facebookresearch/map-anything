import functools
import importlib
import itertools
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphcloud.models.external.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)
from morphcloud.models.external.moge.models.utils import (
    wrap_dinov2_attention_with_sdpa,
    wrap_module_with_gradient_checkpointing,
)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        in_norm: Literal[
            "group_norm", "layer_norm", "instance_norm", "none"
        ] = "layer_norm",
        hidden_norm: Literal[
            "group_norm", "layer_norm", "instance_norm"
        ] = "group_norm",
    ):
        super(ResidualConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = nn.ReLU
        elif activation == "leaky_relu":
            activation_cls = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation == "silu":
            activation_cls = nn.SiLU
        elif activation == "elu":
            activation_cls = nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.Sequential(
            (
                nn.GroupNorm(in_channels // 32, in_channels)
                if in_norm == "group_norm"
                else (
                    nn.GroupNorm(1, in_channels)
                    if in_norm == "layer_norm"
                    else (
                        nn.InstanceNorm2d(in_channels)
                        if in_norm == "instance_norm"
                        else nn.Identity()
                    )
                )
            ),
            activation_cls(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            (
                nn.GroupNorm(hidden_channels // 32, hidden_channels)
                if hidden_norm == "group_norm"
                else (
                    nn.GroupNorm(1, hidden_channels)
                    if hidden_norm == "layer_norm"
                    else (
                        nn.InstanceNorm2d(hidden_channels)
                        if hidden_norm == "instance_norm"
                        else nn.Identity()
                    )
                )
            ),
            activation_cls(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class DINOv2Encoder(nn.Module):
    "Wrapped DINOv2 encoder supporting gradient checkpointing. Input is RGB image in range [0, 1]."

    backbone: DinoVisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(
        self,
        backbone: str,
        intermediate_layers: Union[int, List[int]],
        dim_out: int,
        **deprecated_kwargs,
    ):
        super(DINOv2Encoder, self).__init__()

        self.intermediate_layers = intermediate_layers

        # Load the backbone
        self.hub_loader = getattr(
            importlib.import_module(
                "morphcloud.models.external.dinov2.hub.backbones", __package__
            ),
            backbone,
        )
        self.backbone_name = backbone
        self.backbone = self.hub_loader(pretrained=False)

        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = (
            intermediate_layers
            if isinstance(intermediate_layers, int)
            else len(intermediate_layers)
        )

        self.output_projections = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.dim_features,
                    out_channels=dim_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for _ in range(self.num_features)
            ]
        )

        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    def init_weights(self):
        pretrained_backbone_state_dict = self.hub_loader(pretrained=True).state_dict()
        self.backbone.load_state_dict(pretrained_backbone_state_dict)

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(
        self,
        image: torch.Tensor,
        token_rows: Union[int, torch.LongTensor],
        token_cols: Union[int, torch.LongTensor],
        return_class_token: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_14 = F.interpolate(
            image,
            (token_rows * 14, token_cols * 14),
            mode="bilinear",
            align_corners=False,
            antialias=not self.onnx_compatible_mode,
        )
        image_14 = (image_14 - self.image_mean) / self.image_std

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(
            image_14, n=self.intermediate_layers, return_class_token=True
        )

        # Project features to the desired dimensionality
        x = torch.stack(
            [
                proj(
                    feat.permute(0, 2, 1)
                    .unflatten(2, (token_rows, token_cols))
                    .contiguous()
                )
                for proj, (feat, clstoken) in zip(self.output_projections, features)
            ],
            dim=1,
        ).sum(dim=1)

        if return_class_token:
            return x, features[-1][1]
        else:
            return x


class Resampler(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_: Literal[
            "pixel_shuffle",
            "nearest",
            "bilinear",
            "conv_transpose",
            "pixel_unshuffle",
            "avg_pool",
            "max_pool",
        ],
        scale_factor: int = 2,
    ):
        if type_ == "pixel_shuffle":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels * (scale_factor**2),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            for i in range(1, scale_factor**2):
                self[0].weight.data[i :: scale_factor**2] = self[0].weight.data[
                    0 :: scale_factor**2
                ]
                self[0].bias.data[i :: scale_factor**2] = self[0].bias.data[
                    0 :: scale_factor**2
                ]
        elif type_ in ["nearest", "bilinear"]:
            nn.Sequential.__init__(
                self,
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode=type_,
                    align_corners=False if type_ == "bilinear" else None,
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "conv_transpose":
            nn.Sequential.__init__(
                self,
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                ),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        elif type_ == "pixel_unshuffle":
            nn.Sequential.__init__(
                self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(
                    in_channels * (scale_factor**2),
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "avg_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == "max_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f"Unsupported resampler type: {type_}")


class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int]):
        nn.Sequential.__init__(
            self,
            *itertools.chain(
                *[
                    (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
                ]
            ),
            nn.Linear(dims[-2], dims[-1]),
        )


class ConvStack(nn.Module):
    def __init__(
        self,
        dim_in: List[Optional[int]],
        dim_res_blocks: List[int],
        dim_out: List[Optional[int]],
        resamplers: Union[
            Literal[
                "pixel_shuffle",
                "nearest",
                "bilinear",
                "conv_transpose",
                "pixel_unshuffle",
                "avg_pool",
                "max_pool",
            ],
            List,
        ],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_in_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "layer_norm",
        res_block_hidden_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "group_norm",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList(
            [
                (
                    nn.Conv2d(
                        dim_in_, dim_res_block_, kernel_size=1, stride=1, padding=0
                    )
                    if dim_in_ is not None
                    else nn.Identity()
                )
                for dim_in_, dim_res_block_ in zip(
                    (
                        dim_in
                        if isinstance(dim_in, Sequence)
                        else itertools.repeat(dim_in)
                    ),
                    dim_res_blocks,
                )
            ]
        )
        self.resamplers = nn.ModuleList(
            [
                Resampler(dim_prev, dim_succ, scale_factor=2, type_=resampler)
                for i, (dim_prev, dim_succ, resampler) in enumerate(
                    zip(
                        dim_res_blocks[:-1],
                        dim_res_blocks[1:],
                        (
                            resamplers
                            if isinstance(resamplers, Sequence)
                            else itertools.repeat(resamplers)
                        ),
                    )
                )
            ]
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        ResidualConvBlock(
                            dim_res_block_,
                            dim_res_block_,
                            dim_times_res_block_hidden * dim_res_block_,
                            activation=activation,
                            in_norm=res_block_in_norm,
                            hidden_norm=res_block_hidden_norm,
                        )
                        for _ in range(
                            num_res_blocks[i]
                            if isinstance(num_res_blocks, list)
                            else num_res_blocks
                        )
                    )
                )
                for i, dim_res_block_ in enumerate(dim_res_blocks)
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                (
                    nn.Conv2d(
                        dim_res_block_, dim_out_, kernel_size=1, stride=1, padding=0
                    )
                    if dim_out_ is not None
                    else nn.Identity()
                )
                for dim_out_, dim_res_block_ in zip(
                    (
                        dim_out
                        if isinstance(dim_out, Sequence)
                        else itertools.repeat(dim_out)
                    ),
                    dim_res_blocks,
                )
            ]
        )

    def enable_gradient_checkpointing(self):
        for i in range(len(self.resamplers)):
            self.resamplers[i] = wrap_module_with_gradient_checkpointing(
                self.resamplers[i]
            )
        for i in range(len(self.res_blocks)):
            for j in range(len(self.res_blocks[i])):
                self.res_blocks[i][j] = wrap_module_with_gradient_checkpointing(
                    self.res_blocks[i][j]
                )

    def forward(self, in_features: List[torch.Tensor]):
        out_features = []
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features
