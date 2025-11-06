# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for MUSt3R
"""

import datetime
import os

import numpy as np
import torch
from dust3r.viz import rgb
from must3r.demo.inference import SceneState
from must3r.engine.inference import inference_multi_ar, postprocess
from must3r.model import get_pointmaps_activation, load_model

from morphcloud.models.external.vggt.utils.rotation import mat_to_quat


def must3r_inference(
    views,
    filelist,
    model,
    retrieval,
    device,
    amp,
    num_mem_images,
    max_bs,
    init_num_images=2,
    batch_num_views=1,
    render_once=False,
    is_sequence=False,
    viser_server=None,
    num_refinements_iterations=2,
    verbose=True,
):
    if amp == "fp16":
        dtype = torch.float16
    elif amp == "bf16":
        assert torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16
    else:
        assert not amp
        dtype = torch.float32

    max_bs = None if max_bs == 0 else max_bs
    encoder, decoder = model
    pointmaps_activation = get_pointmaps_activation(decoder, verbose=verbose)

    def post_process_function(x):
        return postprocess(
            x, pointmaps_activation=pointmaps_activation, compute_cam=True
        )

    if verbose:
        print("loading images")
    time_start = datetime.datetime.now()
    nimgs = len(views)

    ellapsed = datetime.datetime.now() - time_start
    if verbose:
        print(f"loaded in {ellapsed}")
        print("running inference")
    time_start = datetime.datetime.now()
    if viser_server is not None:
        viser_server.reset(nimgs)

    imgs = [b["img"].to("cpu") for b in views]
    true_shape = [torch.from_numpy(b["true_shape"]).to("cpu") for b in views]
    true_shape = torch.stack(true_shape, dim=0)
    nimgs = true_shape.shape[0]

    # Use all images as keyframes
    keyframes = np.linspace(0, len(imgs) - 1, num_mem_images, dtype=int).tolist()
    encoder_precomputed_features = None

    not_keyframes = sorted(set(range(nimgs)).difference(set(keyframes)))
    assert (len(keyframes) + len(not_keyframes)) == nimgs
    # reorder images
    views = [views[i] for i in keyframes] + [views[i] for i in not_keyframes]
    imgs = [b["img"].to(device) for b in views]
    true_shape = [torch.from_numpy(b["true_shape"]).to(device) for b in views]
    filenames = [filelist[i] for i in keyframes + not_keyframes]
    img_ids = [torch.tensor(v) for v in keyframes + not_keyframes]

    if encoder_precomputed_features is not None:
        x_start, pos_start = encoder_precomputed_features
        x = [x_start[i] for i in keyframes] + [x_start[i] for i in not_keyframes]
        pos = [pos_start[i] for i in keyframes] + [pos_start[i] for i in not_keyframes]
        encoder_precomputed_features = (x, pos)

    mem_batches = [init_num_images]
    while (sum_b := sum(mem_batches)) != max(num_mem_images, init_num_images):
        size_b = min(batch_num_views, num_mem_images - sum_b)
        mem_batches.append(size_b)

    if render_once:
        to_render = list(range(num_mem_images, nimgs))
    else:
        to_render = None

    with torch.autocast("cuda", dtype=dtype):
        x_out_0, x_out = inference_multi_ar(
            encoder,
            decoder,
            imgs,
            img_ids,
            true_shape,
            mem_batches,
            max_bs=max_bs,
            verbose=verbose,
            to_render=to_render,
            encoder_precomputed_features=encoder_precomputed_features,
            device=device,
            preserve_gpu_mem=True,
            post_process_function=post_process_function,
            viser_server=viser_server,
            num_refinements_iterations=num_refinements_iterations,
        )
    if to_render is not None:
        x_out = x_out_0 + x_out

    ellapsed = datetime.datetime.now() - time_start
    if verbose:
        print(f"inference in {ellapsed}")
        try:
            print(str(int(torch.cuda.max_memory_reserved(device) / (1024**2))) + " MB")
        except Exception:
            pass

    if viser_server is not None:
        viser_server.reset_cam_visility()
        viser_server.send_message("Finished")

    if verbose:
        print("preparing pointcloud")
    time_start = datetime.datetime.now()
    focals = []
    cams2world = []
    for i in range(nimgs):
        focals.append(float(x_out[i]["focal"].cpu()))
        cams2world.append(x_out[i]["c2w"].cpu())

    # x_out to cpu
    for i in range(len(x_out)):
        for k in x_out[i].keys():
            x_out[i][k] = x_out[i][k].cpu()

    rgbimg = [rgb(imgs[i], true_shape[i]) for i in range(nimgs)]
    scene = SceneState(x_out, rgbimg, true_shape, focals, cams2world, filenames)

    ellapsed = datetime.datetime.now() - time_start
    if verbose:
        print(f"pointcloud prepared in {ellapsed}")

    return scene


class MUSt3RWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
        retrieval_ckpt_path,
        img_size=512,
        amp="bf16",
        max_bs=1,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.ckpt_path = ckpt_path
        self.retrieval_ckpt_path = retrieval_ckpt_path
        self.amp = amp
        self.max_bs = max_bs

        # Init the model and load the checkpoint
        self.model = load_model(self.ckpt_path, img_size=512)

    def forward(self, views):
        """
        Forward pass wrapper for MUSt3R.

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
            "MUSt3R expects a normalized image with the DUSt3R normalization scheme applied"
        )

        # Convert the input views to the expected input format
        images = []
        image_paths = []
        for view in views:
            images.append(
                dict(
                    img=view["img"][0].cpu(),
                    idx=len(images),
                    instance=str(len(images)),
                    true_shape=np.int32([view["img"].shape[-2], view["img"].shape[-1]]),
                )
            )
            view_name = os.path.join(view["label"][0], view["instance"][0])
            image_paths.append(view_name)

        # Run MUSt3R inference
        scene = must3r_inference(
            images,
            image_paths,
            self.model,
            self.retrieval_ckpt_path,
            device,
            self.amp,
            num_views,
            self.max_bs,
            verbose=False,
        )

        # Make sure scene is not None
        if scene is None:
            raise RuntimeError("MUSt3R failed.")

        # Get the predictions
        predictions = scene.x_out

        # Convert the output to the MapAnything format
        with torch.autocast("cuda", enabled=False):
            res = []
            for view_idx in range(num_views):
                # Get the current view predictions
                curr_view_prediction = predictions[view_idx]
                curr_view_conf = curr_view_prediction["conf"]
                curr_view_pose = curr_view_prediction["c2w"].unsqueeze(0)

                # Convert the pose to quaternions and translation
                curr_view_cam_translations = curr_view_pose[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_pose[..., :3, :3])

                # Get the camera frame pointmaps
                curr_view_pts3d_cam = curr_view_prediction["pts3d_local"].unsqueeze(0)

                # Get the depth along ray and ray directions
                curr_view_depth_along_ray = torch.norm(
                    curr_view_pts3d_cam, dim=-1, keepdim=True
                )
                curr_view_ray_dirs = curr_view_pts3d_cam / curr_view_depth_along_ray

                # Get the pointmaps
                curr_view_pts3d = curr_view_prediction["pts3d"].unsqueeze(0)

                # Append the outputs to the result list
                res.append(
                    {
                        "pts3d": curr_view_pts3d.to(device),
                        "pts3d_cam": curr_view_pts3d_cam.to(device),
                        "ray_directions": curr_view_ray_dirs.to(device),
                        "depth_along_ray": curr_view_depth_along_ray.to(device),
                        "cam_trans": curr_view_cam_translations.to(device),
                        "cam_quats": curr_view_cam_quats.to(device),
                        "conf": curr_view_conf.to(device),
                    }
                )

        return res
