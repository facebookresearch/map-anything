from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from morphcloud.models.external.dinov2.layers import Mlp
from morphcloud.models.external.pi3.layers.attention import FlashAttentionRope
from morphcloud.models.external.pi3.layers.block import BlockRope


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = (
            nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        )
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
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
                    init_values=None,
                    qk_norm=False,
                    # attn_class=MemEffAttentionRope,
                    attn_class=FlashAttentionRope,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out


class LinearPts3d(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(
        self,
        patch_size,
        dec_embed_dim,
        output_dim=3,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim) * self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(
            B, -1, H // self.patch_size, W // self.patch_size
        )
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)
