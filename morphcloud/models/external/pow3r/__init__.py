# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for Pow3R
"""

import warnings
from copy import deepcopy

import pow3r.model.blocks  # noqa
import roma
import torch
import torch.nn as nn
import tqdm
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import check_if_same_size
from dust3r.model import CroCoNet
from dust3r.patch_embed import get_patch_embed as dust3r_patch_embed
from dust3r.utils.device import collate_with_cat, to_cpu
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    interleave,
    is_symmetrized,
    transpose_to_landscape,
)
from pow3r.model.blocks import Block, BlockInject, DecoderBlock, DecoderBlockInject, Mlp
from pow3r.model.heads import head_factory
from pow3r.model.inference import (
    add_depth,
    add_intrinsics,
    add_relpose,
)
from pow3r.model.patch_embed import get_patch_embed

from morphcloud.models.external.vggt.utils.rotation import mat_to_quat
from morphcloud.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


class Pow3R(CroCoNet):
    """Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(
        self,
        mode="embed",
        head_type="linear",
        patch_embed_cls="PatchEmbedDust3R",
        freeze="none",
        landscape_only=True,
        **croco_kwargs,
    ):
        # retrieve all default arguments using python magic
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)
        del self.mask_token  # useless
        del self.prediction_head

        dec_dim, enc_dim = self.decoder_embed.weight.shape
        self.enc_embed_dim = enc_dim
        self.dec_embed_dim = dec_dim

        self.mode = mode
        # additional parameters in the encoder
        img_size = self.patch_embed.img_size
        patch_size = self.patch_embed.patch_size[0]
        self.patch_embed = dust3r_patch_embed(
            patch_embed_cls, img_size, patch_size, self.enc_embed_dim
        )
        self.patch_embed_rays = get_patch_embed(
            patch_embed_cls + "_Mlp",
            img_size,
            patch_size,
            self.enc_embed_dim,
            in_chans=3,
        )
        self.patch_embed_depth = get_patch_embed(
            patch_embed_cls + "_Mlp",
            img_size,
            patch_size,
            self.enc_embed_dim,
            in_chans=2,
        )
        self.pose_embed = Mlp(12, 4 * dec_dim, dec_dim)

        # additional parameters in the decoder
        self.dec_cls = "_cls" in self.mode
        self.dec_num_cls = 0
        if self.dec_cls:
            # use a CLS token in the decoder only
            self.mode = self.mode.replace("_cls", "")
            self.cls_token1 = nn.Parameter(torch.zeros((dec_dim,)))
            self.cls_token2 = nn.Parameter(torch.zeros((dec_dim,)))
            self.dec_num_cls = 1  # affects all blocks

        use_ln = "_ln" in self.mode  # TODO remove?
        self.patch_ln = nn.LayerNorm(enc_dim) if use_ln else nn.Identity()
        self.dec1_pre_ln = nn.LayerNorm(dec_dim) if use_ln else nn.Identity()
        self.dec2_pre_ln = nn.LayerNorm(dec_dim) if use_ln else nn.Identity()

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        # here we modify some of the blocks
        self.replace_some_blocks()

        self.set_downstream_head(head_type, landscape_only, **croco_kwargs)
        self.set_freeze(freeze)

    def replace_some_blocks(self):
        assert self.mode.startswith("inject")  # inject[0,0.5]
        NewBlock = BlockInject
        DecoderNewBlock = DecoderBlockInject

        all_layers = {
            i / n
            for i in range(len(self.enc_blocks))
            for n in [len(self.enc_blocks), len(self.dec_blocks)]
        }
        which_layers = eval(self.mode[self.mode.find("[") :]) or all_layers
        assert isinstance(which_layers, (set, list))

        n = 0
        for i, block in enumerate(self.enc_blocks):
            if i / len(self.enc_blocks) in which_layers:
                block.__class__ = NewBlock
                block.init(self.enc_embed_dim)
                n += 1
            else:
                block.__class__ = Block
        assert n == len(which_layers), breakpoint()

        n = 0
        for i in range(len(self.dec_blocks)):
            for blocks in [self.dec_blocks, self.dec_blocks2]:
                block = blocks[i]
                if i / len(self.dec_blocks) in which_layers:
                    block.__class__ = DecoderNewBlock
                    block.init(self.dec_embed_dim)
                    n += 1
                else:
                    block.__class__ = DecoderBlock
        assert n == 2 * len(which_layers), breakpoint()

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kw):
        return _load_model(pretrained_model_path, device="cpu")

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        # remove layers that have different shapes
        cur_ckpt = self.state_dict()
        for key, val in ckpt.items():
            if key.startswith("downstream_head2.proj"):
                if key in cur_ckpt and cur_ckpt[key].shape != val.shape:
                    print(f" (removing ckpt[{key}] because wrong shape)")
                    del new_ckpt[key]
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "encoder": [self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        head_type,
        landscape_only,
        patch_size,
        img_size,
        mlp_ratio,
        dec_depth,
        **kw,
    ):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, (
            f"{img_size=} must be multiple of {patch_size=}"
        )

        # split heads if different
        heads = head_type.split(";")
        assert len(heads) in (1, 2)
        head1_type, head2_type = (heads + heads)[:2]

        # allocate heads
        self.downstream_head1 = head_factory(head1_type, self)
        self.downstream_head2 = head_factory(head2_type, self)

        # magic wrapper
        self.head1 = transpose_to_landscape(
            self.downstream_head1, activate=landscape_only
        )
        self.head2 = transpose_to_landscape(
            self.downstream_head2, activate=landscape_only
        )

    def _encode_image(self, image, true_shape, rays=None, depth=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if rays is not None:  # B,3,H,W
            rays_emb, pos2 = self.patch_embed_rays(rays, true_shape=true_shape)
            assert (pos == pos2).all()
            if self.mode.startswith("embed"):
                x = x + rays_emb
        else:
            rays_emb = None

        if depth is not None:  # B,2,H,W
            depth_emb, pos2 = self.patch_embed_depth(depth, true_shape=true_shape)
            assert (pos == pos2).all()
            if self.mode.startswith("embed"):
                x = x + depth_emb
        else:
            depth_emb = None

        x = self.patch_ln(x)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos, rays=rays_emb, depth=depth_emb)

        x = self.enc_norm(x)
        return x, pos

    def encode_symmetrized(self, view1, view2):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        )
        # warning! maybe the images have different portrait/landscape orientations

        # privileged information
        rays1 = view1.get("known_rays", None)
        rays2 = view2.get("known_rays", None)
        depth1 = view1.get("known_depth", None)
        depth2 = view2.get("known_depth", None)

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            def hsub(x):
                return None if x is None else x[::2]

            feat1, pos1 = self._encode_image(
                img1[::2], shape1[::2], rays=hsub(rays1), depth=hsub(depth1)
            )
            feat2, pos2 = self._encode_image(
                img2[::2], shape2[::2], rays=hsub(rays2), depth=hsub(depth2)
            )

            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, pos1 = self._encode_image(img1, shape1, rays=rays1, depth=depth1)
            feat2, pos2 = self._encode_image(img2, shape2, rays=rays2, depth=depth2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, relpose1=None, relpose2=None):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        # add CLS token for the decoder
        if self.dec_cls:
            cls1 = self.cls_token1[None, None].expand(len(f1), 1, -1).clone()
            cls2 = self.cls_token2[None, None].expand(len(f2), 1, -1).clone()

        if relpose1 is not None:  # shape = (B, 4, 4)
            pose_emb1 = self.pose_embed(relpose1[:, :3].flatten(1)).unsqueeze(1)
            if self.mode.startswith("embed"):
                if self.dec_cls:
                    cls1 = cls1 + pose_emb1
                else:
                    f1 = f1 + pose_emb1
        else:
            pose_emb1 = None

        if relpose2 is not None:  # shape = (B, 4, 4)
            pose_emb2 = self.pose_embed(relpose2[:, :3].flatten(1)).unsqueeze(1)
            if self.mode.startswith("embed"):
                if self.dec_cls:
                    cls2 = cls2 + pose_emb2
                else:
                    f2 = f2 + pose_emb2
        else:
            pose_emb2 = None

        if self.dec_cls:
            f1, pos1 = cat_cls(cls1, f1, pos1)
            f2, pos2 = cat_cls(cls2, f2, pos2)

        f1 = self.dec1_pre_ln(f1)
        f2 = self.dec2_pre_ln(f2)

        final_output.append((f1, f2))  # to be removed later
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(
                *final_output[-1][::+1],
                pos1,
                pos2,
                relpose=pose_emb1,
                num_cls=self.dec_num_cls,
            )
            # img2 side
            f2, _ = blk2(
                *final_output[-1][::-1],
                pos2,
                pos1,
                relpose=pose_emb2,
                num_cls=self.dec_num_cls,
            )
            # store the result
            final_output.append((f1, f2))

        del final_output[1]  # duplicate with final_output[0] (after decoder proj)
        if self.dec_cls:  # remove cls token for decoder layers
            final_output[1:] = [(f1[:, 1:], f2[:, 1:]) for f1, f2 in final_output[1:]]
        # normalize last output
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encode_symmetrized(
            view1, view2
        )

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(
            feat1,
            pos1,
            feat2,
            pos2,
            relpose1=view1.get("known_pose"),
            relpose2=view2.get("known_pose"),
        )
        with torch.autocast("cuda", enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2["pts3d_in_other_view"] = res2.pop(
            "pts3d"
        )  # predict view2's pts3d in view1's frame
        return res1, res2


def convert_release_dust3r_args(args):
    args.model = (
        args.model.replace("patch_embed_cls", "patch_embed")
        .replace("AsymmetricMASt3R", "AsymmetricCroCo3DStereo")
        .replace("PatchEmbedDust3R", "convManyAR")
        .replace(
            "pos_embed='RoPE100'",
            "enc_pos_embed='cuRoPE100', dec_pos_embed='cuRoPE100'",
        )
    )
    return args


def _load_model(model_path, device):
    print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    try:
        net = eval(
            ckpt["args"].model[:-1].replace("convManyAR", "convP")
            + ", landscape_only=False)"
        )
    except Exception:
        args = convert_release_dust3r_args(ckpt["args"])
        net = eval(
            args.model[:-1].replace("convManyAR", "convP") + ", landscape_only=False)"
        )
    ckpt["model"] = {
        k.replace("_downstream_head", "downstream_head"): v
        for k, v in ckpt["model"].items()
    }
    print(net.load_state_dict(ckpt["model"], strict=False))
    return net.to(device)


def cat_cls(cls, tokens, pos):
    tokens = torch.cat((cls, tokens), dim=1)
    pos = torch.cat((-pos.new_ones(len(cls), 1, 2), pos), dim=1)
    return tokens, pos


class Pow3RWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
        geometric_input_config,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.ckpt_path = ckpt_path
        self.geometric_input_config = geometric_input_config

        # Init the model and load the checkpoint
        print(f"Loading checkpoint from {self.ckpt_path} ...")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        model = ckpt["definition"]
        print(f"Creating model = {model}")
        self.model = eval(model)
        print(self.model.load_state_dict(ckpt["weights"]))

    def forward(self, views):
        """
        Forward pass wrapper for Pow3R.

        Assumption:
        - The number of input views is 2.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Length of the list should be 2.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["dust3r"]
                                Optionally, each dictionary can also contain the following keys for the respective optional geometric inputs:
                                    "camera_intrinsics" (tensor): Camera intrinsics. Tensor of shape (B, 3, 3).
                                    "camera_pose" (tensor): Camera pose. Tensor of shape (B, 4, 4). Camera pose is opencv (RDF) cam2world transformation.
                                    "depthmap" (tensor): Z Depth map. Tensor of shape (B, H, W, 1).

        Returns:
            List[dict]: A list containing the final outputs for the two views. Length of the list will be 2.
        """
        # Check that the number of input views is 2
        assert len(views) == 2, "Pow3R requires 2 input views."

        # Check the data norm type
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "dust3r", (
            "Pow3R expects a normalized image with the DUSt3R normalization scheme applied"
        )

        # Get the batch size per view, device and two views
        batch_size_per_view = views[0]["img"].shape[0]
        device = views[0]["img"].device
        view1, view2 = views

        # Decide if we need to use the geometric inputs
        if torch.rand(1, device=device) < self.geometric_input_config["overall_prob"]:
            # Decide if we need to use the camera intrinsics
            if (
                torch.rand(1, device=device)
                < self.geometric_input_config["ray_dirs_prob"]
            ):
                add_intrinsics(view1, view1.get("camera_intrinsics"))
                add_intrinsics(view2, view2.get("camera_intrinsics"))

            # Decide if we need to use the depth map
            if torch.rand(1, device=device) < self.geometric_input_config["depth_prob"]:
                depthmap1 = view1.get("depthmap")
                depthmap2 = view2.get("depthmap")
                if depthmap1 is not None:
                    depthmap1 = depthmap1.squeeze(-1).to(device)
                if depthmap2 is not None:
                    depthmap2 = depthmap2.squeeze(-1).to(device)
                add_depth(view1, depthmap1)
                add_depth(view2, depthmap2)

            # Decide if we need to use the camera pose
            if torch.rand(1, device=device) < self.geometric_input_config["cam_prob"]:
                cam1 = view1.get("camera_pose")
                cam2 = view2.get("camera_pose")
                add_relpose(view1, cam2_to_world=cam2, cam1_to_world=cam1)
                add_relpose(view2, cam2_to_world=cam2, cam1_to_world=cam1)

        # Get the model predictions
        preds = self.model(view1, view2)

        # Convert the output to MapAnything format
        with torch.autocast("cuda", enabled=False):
            res = []
            for view_idx in range(2):
                # Get the model predictions for the current view
                curr_view_pred = preds[view_idx]

                # For the first view
                if view_idx == 0:
                    # Get the global frame and camera frame pointmaps
                    global_pts = curr_view_pred["pts3d"]
                    cam_pts = curr_view_pred["pts3d"]
                    conf = curr_view_pred["conf"]

                    # Get the ray directions and depth along ray
                    depth_along_ray = torch.norm(cam_pts, dim=-1, keepdim=True)
                    ray_directions = cam_pts / depth_along_ray

                    # Initalize identity camera pose
                    cam_rot = torch.eye(3, device=device)
                    cam_quat = mat_to_quat(cam_rot)
                    cam_trans = torch.zeros(3, device=device)
                    cam_quat = cam_quat.unsqueeze(0).repeat(batch_size_per_view, 1)
                    cam_trans = cam_trans.unsqueeze(0).repeat(batch_size_per_view, 1)
                # For the second view
                elif view_idx == 1:
                    # Get the global frame and camera frame pointmaps
                    pred_global_pts = curr_view_pred["pts3d_in_other_view"]
                    cam_pts = curr_view_pred["pts3d2"]
                    conf = (curr_view_pred["conf"] * curr_view_pred["conf2"]).sqrt()

                    # Get the ray directions and depth along ray
                    depth_along_ray = torch.norm(cam_pts, dim=-1, keepdim=True)
                    ray_directions = cam_pts / depth_along_ray

                    # Compute the camera pose using the pointmaps
                    cam_rot, cam_trans, scale = roma.rigid_points_registration(
                        cam_pts.reshape(batch_size_per_view, -1, 3),
                        pred_global_pts.reshape(batch_size_per_view, -1, 3),
                        weights=conf.reshape(batch_size_per_view, -1),
                        compute_scaling=True,
                    )
                    cam_quat = mat_to_quat(cam_rot)

                    # Scale the predicted camera frame pointmap and compute the new global frame pointmap
                    cam_pts = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * cam_pts
                    global_pts = cam_pts.reshape(
                        batch_size_per_view, -1, 3
                    ) @ cam_rot.permute(0, 2, 1) + cam_trans.unsqueeze(1)
                    global_pts = global_pts.view(pred_global_pts.shape)

                # Append the result in MapAnything format
                res.append(
                    {
                        "pts3d": global_pts,
                        "pts3d_cam": cam_pts,
                        "ray_directions": ray_directions,
                        "depth_along_ray": depth_along_ray,
                        "cam_trans": cam_trans,
                        "cam_quats": cam_quat,
                        "conf": conf,
                    }
                )

        return res


class Pow3RBAWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
        geometric_input_config,
        scene_graph="complete",
        inference_batch_size=32,
        global_optim_schedule="cosine",
        global_optim_lr=0.01,
        global_optim_niter=300,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.ckpt_path = ckpt_path
        self.geometric_input_config = geometric_input_config
        self.scene_graph = scene_graph
        self.inference_batch_size = inference_batch_size
        self.global_optim_schedule = global_optim_schedule
        self.global_optim_lr = global_optim_lr
        self.global_optim_niter = global_optim_niter

        # Init the model and load the checkpoint
        print(f"Loading checkpoint from {self.ckpt_path} ...")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        model = ckpt["definition"]
        print(f"Creating model = {model}")
        self.model = eval(model)
        print(self.model.load_state_dict(ckpt["weights"]))

        # Init the global aligner mode
        self.global_aligner_mode = GlobalAlignerMode.PointCloudOptimizer

    def infer_two_views(self, views):
        """
        Wrapper for Pow3R 2-View inference.

        Assumption:
        - The number of input views is 2.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Length of the list should be 2.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["dust3r"]
                                Optionally, each dictionary can also contain the following keys for the respective optional geometric inputs:
                                    "camera_intrinsics" (tensor): Camera intrinsics. Tensor of shape (B, 3, 3).
                                    "camera_pose" (tensor): Camera pose. Tensor of shape (B, 4, 4). Camera pose is opencv (RDF) cam2world transformation.
                                    "depthmap" (tensor): Z Depth map. Tensor of shape (B, H, W, 1).

        Returns:
            List[dict]: A list containing the final outputs for the two views. Length of the list will be 2.
        """
        # Check that the number of input views is 2
        assert len(views) == 2, "Pow3R requires 2 input views."

        # Check the data norm type
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "dust3r", (
            "Pow3R expects a normalized image with the DUSt3R normalization scheme applied"
        )

        # Get the device and two views
        device = views[0]["img"].device
        view1, view2 = views

        # Decide if we need to use the geometric inputs
        if torch.rand(1, device=device) < self.geometric_input_config["overall_prob"]:
            # Decide if we need to use the camera intrinsics
            if (
                torch.rand(1, device=device)
                < self.geometric_input_config["ray_dirs_prob"]
            ):
                add_intrinsics(view1, view1.get("camera_intrinsics"))
                add_intrinsics(view2, view2.get("camera_intrinsics"))

            # Decide if we need to use the depth map
            if torch.rand(1, device=device) < self.geometric_input_config["depth_prob"]:
                depthmap1 = view1.get("depthmap")
                depthmap2 = view2.get("depthmap")
                if depthmap1 is not None:
                    depthmap1 = depthmap1.squeeze(-1).to(device)
                if depthmap2 is not None:
                    depthmap2 = depthmap2.squeeze(-1).to(device)
                add_depth(view1, depthmap1)
                add_depth(view2, depthmap2)

            # Decide if we need to use the camera pose
            if torch.rand(1, device=device) < self.geometric_input_config["cam_prob"]:
                cam1 = view1.get("camera_pose")
                cam2 = view2.get("camera_pose")
                add_relpose(view1, cam2_to_world=cam2, cam1_to_world=cam1)
                add_relpose(view2, cam2_to_world=cam2, cam1_to_world=cam1)

        # Get the model predictions
        preds = self.model(view1, view2)

        return preds

    def loss_of_one_batch(self, batch, device):
        """
        Compute prediction for two views.
        """
        view1, view2 = batch
        ignore_keys = set(
            [
                "dataset",
                "label",
                "instance",
                "idx",
                "true_shape",
                "rng",
                "name",
                "data_norm_type",
            ]
        )
        for view in batch:
            for name in view.keys():  # pseudo_focal
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        pred1, pred2 = self.infer_two_views([view1, view2])

        result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2)

        return result

    @torch.no_grad()
    def inference(self, pairs, device, verbose=False):
        """
        Wrapper for multi-pair inference using Pow3R.
        """
        if verbose:
            print(f">> Inference with model on {len(pairs)} image pairs")
        result = []

        multiple_shapes = not (check_if_same_size(pairs))
        if multiple_shapes:
            self.inference_batch_size = 1

        for i in tqdm.trange(
            0, len(pairs), self.inference_batch_size, disable=not verbose
        ):
            res = self.loss_of_one_batch(
                collate_with_cat(pairs[i : i + self.inference_batch_size]), device
            )
            result.append(to_cpu(res))

        result = collate_with_cat(result, lists=multiple_shapes)

        return result

    def forward(self, views):
        """
        Forward pass wrapper for Pow3R using the global aligner.

        Assumption:
        - The batch size of input views is 1.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys, where B is the batch size and is 1:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["dust3r"]

        Returns:
            List[dict]: A list containing the final outputs for the input views.
        """
        # Check the batch size of input views
        batch_size_per_view, _, height, width = views[0]["img"].shape
        device = views[0]["img"].device
        num_views = len(views)
        assert batch_size_per_view == 1, (
            f"Batch size of input views should be 1, but got {batch_size_per_view}."
        )

        # Check the data norm type
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "dust3r", (
            "Pow3R-BA expects a normalized image with the DUSt3R normalization scheme applied"
        )

        # Convert the input views to the expected input format
        images = []
        for view in views:
            images.append(
                dict(
                    img=view["img"],
                    camera_intrinsics=view["camera_intrinsics"],
                    depthmap=view["depthmap"],
                    camera_pose=view["camera_pose"],
                    data_norm_type=view["data_norm_type"],
                    true_shape=view["true_shape"],
                    idx=len(images),
                    instance=str(len(images)),
                )
            )

        # Make image pairs and run inference pair-wise
        pairs = make_pairs(
            images, scene_graph=self.scene_graph, prefilter=None, symmetrize=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            output = self.inference(
                pairs,
                device,
                verbose=False,
            )

        # Global optimization
        with torch.enable_grad():
            scene = global_aligner(
                output, device=device, mode=self.global_aligner_mode, verbose=False
            )
            _ = scene.compute_global_alignment(
                init="mst",
                niter=self.global_optim_niter,
                schedule=self.global_optim_schedule,
                lr=self.global_optim_lr,
            )

        # Make sure scene is not None
        if scene is None:
            raise RuntimeError("Global optimization failed.")

        # Get the predictions
        intrinsics = scene.get_intrinsics()
        c2w_poses = scene.get_im_poses()
        depths = scene.get_depthmaps()

        # Convert the output to the MapAnything format
        with torch.autocast("cuda", enabled=False):
            res = []
            for view_idx in range(num_views):
                # Get the current view predictions
                curr_view_intrinsic = intrinsics[view_idx].unsqueeze(0)
                curr_view_pose = c2w_poses[view_idx].unsqueeze(0)
                curr_view_depth_z = depths[view_idx].unsqueeze(0)

                # Convert the pose to quaternions and translation
                curr_view_cam_translations = curr_view_pose[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_pose[..., :3, :3])

                # Get the camera frame pointmaps
                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                # Convert the z depth to depth along ray
                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                )
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

                # Get the ray directions on the unit sphere in the camera frame
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic, height, width, normalize_to_unit_sphere=True
                )

                # Get the pointmaps
                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                # Append the outputs to the result list
                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                    }
                )

        return res
