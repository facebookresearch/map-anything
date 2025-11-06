# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for MASt3R + Sparse GA
"""

import os
import tempfile
import warnings

import torch
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import load_model

from morphcloud.models.external.vggt.utils.rotation import mat_to_quat
from morphcloud.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


class MASt3RSGAWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
        cache_dir,
        scene_graph="complete",
        sparse_ga_lr1=0.07,
        sparse_ga_niter1=300,
        sparse_ga_lr2=0.01,
        sparse_ga_niter2=300,
        sparse_ga_optim_level="refine+depth",
        sparse_ga_shared_intrinsics=False,
        sparse_ga_matching_conf_thr=5.0,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.ckpt_path = ckpt_path
        self.cache_dir = cache_dir
        self.scene_graph = scene_graph
        self.sparse_ga_lr1 = sparse_ga_lr1
        self.sparse_ga_niter1 = sparse_ga_niter1
        self.sparse_ga_lr2 = sparse_ga_lr2
        self.sparse_ga_niter2 = sparse_ga_niter2
        self.sparse_ga_optim_level = sparse_ga_optim_level
        self.sparse_ga_shared_intrinsics = sparse_ga_shared_intrinsics
        self.sparse_ga_matching_conf_thr = sparse_ga_matching_conf_thr

        # Init the model and load the checkpoint
        self.model = load_model(self.ckpt_path, device="cpu")

    def forward(self, views):
        """
        Forward pass wrapper for MASt3R using the sparse global aligner.

        Assumption:
        - The batch size of input views is 1.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys, where B is the batch size and is 1:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["dust3r"]
                                    "label" (list): ["scene_name"]
                                    "instance" (list): ["image_name"]

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
            "MASt3R expects a normalized image with the DUSt3R normalization scheme applied"
        )

        # Convert the input views to the expected input format
        images = []
        image_paths = []
        for view in views:
            images.append(
                dict(
                    img=view["img"].cpu(),
                    idx=len(images),
                    instance=str(len(images)),
                    true_shape=torch.tensor(view["img"].shape[-2:])[None]
                    .repeat(batch_size_per_view, 1)
                    .numpy(),
                )
            )
            view_name = os.path.join(view["label"][0], view["instance"][0])
            image_paths.append(view_name)

        # Make image pairs and run inference
        # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
        pairs = make_pairs(
            images, scene_graph=self.scene_graph, prefilter=None, symmetrize=True
        )
        with torch.enable_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                tempfile.mkdtemp(dir=self.cache_dir)
                scene = sparse_global_alignment(
                    image_paths,
                    pairs,
                    self.cache_dir,
                    self.model,
                    lr1=self.sparse_ga_lr1,
                    niter1=self.sparse_ga_niter1,
                    lr2=self.sparse_ga_lr2,
                    niter2=self.sparse_ga_niter2,
                    device=device,
                    opt_depth="depth" in self.sparse_ga_optim_level,
                    shared_intrinsics=self.sparse_ga_shared_intrinsics,
                    matching_conf_thr=self.sparse_ga_matching_conf_thr,
                    verbose=False,
                )

        # Make sure scene is not None
        if scene is None:
            raise RuntimeError("Global optimization failed.")

        # Get the predictions
        intrinsics = scene.intrinsics
        c2w_poses = scene.get_im_poses()
        _, depths, _ = scene.get_dense_pts3d()

        # Convert the output to the MapAnything format
        with torch.autocast("cuda", enabled=False):
            res = []
            for view_idx in range(num_views):
                # Get the current view predictions
                curr_view_intrinsic = intrinsics[view_idx].unsqueeze(0)
                curr_view_pose = c2w_poses[view_idx].unsqueeze(0)
                curr_view_depth_z = (
                    depths[view_idx].reshape((height, width)).unsqueeze(0)
                )

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
