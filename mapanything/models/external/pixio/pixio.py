from collections import namedtuple
from functools import partial
from typing import Callable, Type, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.checkpoint import checkpoint

from .layers.attention import SelfAttentionBlock
from .layers.mlp import Mlp
from .layers.patch_embed import PatchEmbed

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


class PixioEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        n_cls_tokens: int = 8,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        hf_model_name: str = "facebook/pixio-vith16",
        gradient_checkpointing: bool = True,
    ):
        """
        Pixio ViT Encoder.
        """
        super().__init__()

        self.n_cls_tokens = n_cls_tokens

        self.patch_size = patch_size

        self.enc_embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, self.enc_embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, self.enc_embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, self.patch_embed.num_patches + n_cls_tokens, self.enc_embed_dim
            )
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    self.enc_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    mlp_layer=Mlp,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(self.enc_embed_dim)

        ckpt_path = self.get_pth_file(repo_id=hf_model_name)
        print(f"Loading pretrained Pixio Encoder from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, weights_only=False)
        print(self.load_state_dict(ckpt, strict=False))

        if gradient_checkpointing:
            for i in range(len(self.blocks)):
                self.blocks[i] = self.wrap_module_with_gradient_checkpointing(
                    self.blocks[i]
                )

    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        class _CheckpointingWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                return checkpoint(
                    self.inner.forward, *args, use_reentrant=False, **kwargs
                )

        return _CheckpointingWrapper(module)

    def _interpolate_pos_emb(self, x):
        """
        Interpolate the positional embeddings to match the input x.
        """
        assert x.shape[-2] % self.patch_embed.patch_size[0] == 0, (
            f"height {x.shape[-2]} must be divisible by patch size {self.patch_embed.patch_size[0]}"
        )
        assert x.shape[-1] % self.patch_embed.patch_size[1] == 0, (
            f"width {x.shape[-1]} must be divisible by patch size {self.patch_embed.patch_size[1]}"
        )

        H = x.shape[-2] // self.patch_embed.patch_size[0]
        W = x.shape[-1] // self.patch_embed.patch_size[1]

        cls_pos_embed = self.pos_embed[:, : self.n_cls_tokens]
        patch_pos_embed = self.pos_embed[:, self.n_cls_tokens :]

        pt_size = int(patch_pos_embed.shape[1] ** 0.5)

        if pt_size == H == W:
            return self.pos_embed

        patch_pos_embed = patch_pos_embed.reshape(1, pt_size, pt_size, -1).permute(
            0, 3, 1, 2
        )
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed, size=(H, W), mode="bicubic", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)

        new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

        return new_pos_embed

    def forward(self, encoder_input):
        assert isinstance(encoder_input.image, torch.Tensor), (
            "Input must be a torch.Tensor"
        )
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        _, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert height % self.patch_size == 0 and width % self.patch_size == 0, (
            f"Input shape must be divisible by patch size: {self.patch_size}"
        )

        pos_embed = self._interpolate_pos_emb(encoder_input.image)

        x = self.patch_embed(encoder_input.image)

        x = x + pos_embed[:, self.n_cls_tokens :, :]

        cls_token = self.cls_token + pos_embed[:, : self.n_cls_tokens, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        layers = list(range(len(self.blocks)))
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in layers:
                x_norm = self.norm(x)
                features = x_norm[:, self.n_cls_tokens :]

        features = features.permute(0, 2, 1)

        features = features.reshape(
            -1, self.enc_embed_dim, height // self.patch_size, width // self.patch_size
        ).contiguous()

        return namedtuple("res", ["features"])(features=features)

    def get_pth_file(self, repo_id: str) -> str:
        files = list_repo_files(repo_id)
        pth_files = [f for f in files if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in {repo_id}")
        if len(pth_files) > 1:
            raise ValueError(f"Multiple .pth files found: {pth_files}")
        return hf_hub_download(repo_id=repo_id, filename=pth_files[0])


def pixio_vitb16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vitl16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vith16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vit1b16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=1536,
        depth=48,
        num_heads=24,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model


def pixio_vit5b16(pretrained=None):
    model = PixioEncoder(
        img_size=256,
        patch_size=16,
        embed_dim=3072,
        depth=48,
        num_heads=32,
        mlp_ratio=4,
        n_cls_tokens=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    return model
