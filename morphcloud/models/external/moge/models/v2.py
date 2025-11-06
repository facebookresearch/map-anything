import warnings
from numbers import Number
from pathlib import Path
from typing import Any, Dict, IO, List, Literal, Optional, Union

import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
from huggingface_hub import hf_hub_download

from morphcloud.models.external.moge.models.modules import (
    ConvStack,
    DINOv2Encoder,
    MLP,
)
from morphcloud.models.external.moge.models.utils import (
    depth_to_points,
    intrinsics_from_focal_center,
    normalized_view_plane_uv,
    recover_focal_shift,
)


class MoGeModel(nn.Module):
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    onnx_compatible_mode: bool

    def __init__(
        self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal["linear", "sinh", "exp", "sinh_exp"] = "linear",
        num_tokens_range: List[int] = [1200, 3600],
        **deprecated_kwargs,
    ):
        super(MoGeModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(
                f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}"
            )

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range

        self.encoder = DINOv2Encoder(**encoder)
        self.neck = ConvStack(**neck)
        if points_head is not None:
            self.points_head = ConvStack(**points_head)
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def onnx_compatible_mode(self) -> bool:
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        **hf_kwargs,
    ) -> "MoGeModel":
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `compiled`
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs,
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        model_config = checkpoint["model_config"]
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint["model"], strict=False)

        return model

    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ["points_head", "normal_head", "mask_head"]:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == "linear":
            pass
        elif self.remap_output == "sinh":
            points = torch.sinh(points)
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (
            int((num_tokens / aspect_ratio) ** 0.5),
            int((num_tokens * aspect_ratio) ** 0.5),
        )
        num_tokens = base_h * base_w

        # Backbones encoding
        features, cls_token = self.encoder(
            image, base_h, base_w, return_class_token=True
        )
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.neck(features)

        # Heads decoding
        points, normal, mask = (
            getattr(self, head)(features)[-1] if hasattr(self, head) else None
            for head in ["points_head", "normal_head", "mask_head"]
        )
        metric_scale = (
            self.scale_head(cls_token) if hasattr(self, "scale_head") else None
        )

        # Resize
        points, normal, mask = (
            (
                F.interpolate(
                    v,
                    (img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                if v is not None
                else None
            )
            for v in [points, normal, mask]
        )

        # Remap output
        if points is not None:
            points = points.permute(0, 2, 3, 1)
            points = self._remap_points(
                points
            )  # slightly improves the performance in case of very large output values
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            "points": points,
            "normal": normal,
            "mask": mask,
            "metric_scale": metric_scale,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    # @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: Literal[False, True, "blend"] = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `num_tokens`: the number of base ViT tokens to use for inference, `'least'` or `'most'` or an integer. Suggested range: 1200 ~ 2500.
            More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: `'most'`.
        - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True

        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(
                min_tokens + (resolution_level / 9) * (max_tokens - min_tokens)
            )

        # Forward pass
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=use_fp16 and self.dtype != torch.float16,
        ):
            output = self.forward(image, num_tokens=num_tokens)
        points, normal, mask, metric_scale = (
            output.get(k, None) for k in ["points", "normal", "mask", "metric_scale"]
        )

        # Always process the output in fp32 precision
        points, normal, mask, metric_scale, fov_x = map(
            lambda x: x.float() if isinstance(x, torch.Tensor) else x,
            [points, normal, mask, metric_scale, fov_x],
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None

            if points is not None:
                # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
                # NOTE: Focal here is the focal length relative to half the image diagonal
                if fov_x is None:
                    # Recover focal and shift from predicted point map
                    focal, shift = recover_focal_shift(points, mask_binary)
                else:
                    # Focal is known, recover shift only
                    focal = (
                        aspect_ratio
                        / (1 + aspect_ratio**2) ** 0.5
                        / torch.tan(
                            torch.deg2rad(
                                torch.as_tensor(
                                    fov_x, device=points.device, dtype=points.dtype
                                )
                                / 2
                            )
                        )
                    )
                    if focal.ndim == 0:
                        focal = focal[None].expand(points.shape[0])
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
                fx, fy = (
                    focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio,
                    focal / 2 * (1 + aspect_ratio**2) ** 0.5,
                )
                intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
                points[..., 2] += shift[..., None, None]
                if mask_binary is not None:
                    mask_binary &= (
                        points[..., 2] > 0
                    )  # in case depth is contains negative values (which should never happen in practice)
                depth = points[..., 2].clone()
            else:
                depth, intrinsics = None, None

            # If projection constraint is forced, recompute the point map using the actual depth map & intrinsics
            if force_projection and depth is not None:
                points = depth_to_points(depth, intrinsics=intrinsics)

            # Apply metric scale
            if metric_scale is not None:
                if points is not None:
                    points *= metric_scale[:, None, None, None]
                if depth is not None:
                    depth *= metric_scale[:, None, None]

            # Apply mask
            if apply_mask and mask_binary is not None:
                points = (
                    torch.where(mask_binary[..., None], points, torch.inf)
                    if points is not None
                    else None
                )
                depth = (
                    torch.where(mask_binary, depth, torch.inf)
                    if depth is not None
                    else None
                )
                normal = (
                    torch.where(
                        mask_binary[..., None], normal, torch.zeros_like(normal)
                    )
                    if normal is not None
                    else None
                )

        return_dict = {
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "normal": normal,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
