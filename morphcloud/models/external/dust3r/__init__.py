# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for DUSt3R
"""

import warnings

import torch
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo  # noqa

from morphcloud.models.external.vggt.utils.rotation import mat_to_quat
from morphcloud.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)

inf = float("inf")


def load_model(model_path, device, verbose=True):
    if verbose:
        print("Loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"Instantiating: {args}")
    try:
        net = eval(args)
    except NameError:
        net = AsymmetricCroCo3DStereo(
            enc_depth=24,
            dec_depth=12,
            enc_embed_dim=1024,
            dec_embed_dim=768,
            enc_num_heads=16,
            dec_num_heads=12,
            pos_embed="RoPE100",
            patch_embed_cls="PatchEmbedDust3R",
            img_size=(512, 512),
            head_type="dpt",
            output_mode="pts3d",
            depth_mode=("exp", -inf, inf),
            conf_mode=("exp", 1, inf),
            landscape_only=False,
        )
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class DUSt3RBAWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
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
        self.scene_graph = scene_graph
        self.inference_batch_size = inference_batch_size
        self.global_optim_schedule = global_optim_schedule
        self.global_optim_lr = global_optim_lr
        self.global_optim_niter = global_optim_niter

        # Init the model and load the checkpoint
        self.model = load_model(self.ckpt_path, device="cpu")

        # Init the global aligner mode
        self.global_aligner_mode = GlobalAlignerMode.PointCloudOptimizer

    def forward(self, views):
        """
        Forward pass wrapper for DUSt3R using the global aligner.

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
            "DUSt3R expects a normalized image with the DUSt3R normalization scheme applied"
        )

        # Convert the input views to the expected input format
        images = []
        for view in views:
            images.append(
                dict(
                    img=view["img"],
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
            output = inference(
                pairs,
                self.model,
                device,
                batch_size=self.inference_batch_size,
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
