from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from morphcloud.models.external.dinov2.hub.backbones import dinov2_vitl14_reg
from morphcloud.models.external.dinov2.layers import Mlp
from morphcloud.models.external.pi3.layers.attention import FlashAttentionRope
from morphcloud.models.external.pi3.layers.block import BlockRope
from morphcloud.models.external.pi3.layers.camera_head import CameraHead
from morphcloud.models.external.pi3.layers.pos_embed import PositionGetter, RoPE2D
from morphcloud.models.external.pi3.layers.transformer_head import (
    LinearPts3d,
    TransformerDecoder,
)


def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        pos_type="rope100",
        decoder_size="large",
    ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else "none"
        self.rope = None
        if self.pos_type.startswith("rope"):  # eg rope100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(self.pos_type[len("rope") :])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == "small":
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == "base":
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == "large":
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList(
            [
                BlockRope(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=0.01,
                    qk_norm=True,
                    attn_class=FlashAttentionRope,
                    rope=self.rope,
                )
                for _ in range(dec_depth)
            ]
        )
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,  # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False,
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith("rope"):
            pos = self.position_getter(
                B * N, H // self.patch_size, W // self.patch_size, hidden.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * N, self.patch_start_idx, 2)
                .to(hidden.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(
            B * N, hw, -1
        )

    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # encode by dinov2
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head(
                [conf_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx :], patch_h, patch_w
            ).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum(
                "bnij, bnhwj -> bnhwi", camera_poses, homogenize_points(local_points)
            )[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
