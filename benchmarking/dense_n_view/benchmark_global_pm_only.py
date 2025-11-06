# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to benchmark the predicted metric global pointmaps
Assumes global pointmaps are predicted in view0 frame
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

from morphcloud.datasets import get_test_data_loader
from morphcloud.models import init_model
from morphcloud.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
)
from morphcloud.utils.metrics import (
    m_rel_ae,
    thresh_inliers,
)
from morphcloud.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


def get_all_info_for_metric_computation(batch, preds, norm_mode="avg_dis"):
    """
    Function to get all the information needed to compute the metrics.
    Returns all quantities normalized w.r.t. camera of view0.
    """
    n_views = len(batch)

    # Everything is normalized w.r.t. camera of view0
    # Intialize lists to store data for all views
    # Ground truth quantities
    in_camera0 = inv(batch[0]["camera_pose"])
    no_norm_gt_pts = []
    valid_masks = []
    # Predicted quantities
    no_norm_pr_pts = []
    metric_pr_pts_to_compute_scale = []

    # Get ground truth & prediction info for all views
    for i in range(n_views):
        # Get the ground truth
        no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))
        valid_masks.append(batch[i]["valid_mask"].clone())

        # Get predicted global pointmaps in view0's frame
        # WARNING: Assumes the predicted global pointmaps are in view0's frame
        pr_pts3d_in_view0 = preds[i]["pts3d"]

        # Get predictions
        if "metric_scaling_factor" in preds[i].keys():
            # Divide by the predicted metric scaling factor to get the raw predicted points, pts3d_cam, and pose_trans
            # This detaches the predicted metric scaling factor from the geometry
            curr_view_no_norm_pr_pts = pr_pts3d_in_view0 / preds[i][
                "metric_scaling_factor"
            ].unsqueeze(-1).unsqueeze(-1)
        else:
            curr_view_no_norm_pr_pts = pr_pts3d_in_view0
        no_norm_pr_pts.append(curr_view_no_norm_pr_pts)

        # Get the predicted metric scale points
        if "metric_scaling_factor" in preds[i].keys():
            curr_view_metric_pr_pts_to_compute_scale = (
                curr_view_no_norm_pr_pts.detach()
                * preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            )
        else:
            curr_view_metric_pr_pts_to_compute_scale = curr_view_no_norm_pr_pts.clone()
        metric_pr_pts_to_compute_scale.append(curr_view_metric_pr_pts_to_compute_scale)

    # Initialize normalized tensors
    gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
    pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]

    # Normalize the predicted points
    pr_normalization_output = normalize_multiple_pointclouds(
        no_norm_pr_pts,
        valid_masks,
        norm_mode,
        ret_factor=True,
    )
    pr_pts_norm = pr_normalization_output[:-1]

    # Normalize the ground truth points
    gt_normalization_output = normalize_multiple_pointclouds(
        no_norm_gt_pts, valid_masks, norm_mode, ret_factor=True
    )
    gt_pts_norm = gt_normalization_output[:-1]
    gt_norm_factor = gt_normalization_output[-1]

    for i in range(n_views):
        # Assign the normalized predictions
        pr_pts[i] = pr_pts_norm[i]
        # Assign the normalized ground truth quantities
        gt_pts[i] = gt_pts_norm[i]

    # Get the mask indicating ground truth metric scale quantities
    metric_scale_mask = batch[0]["is_metric_scale"]
    valid_gt_norm_factor_mask = (
        gt_norm_factor[:, 0, 0, 0] > 1e-8
    )  # Mask out cases where depth for all views is invalid
    valid_metric_scale_mask = metric_scale_mask & valid_gt_norm_factor_mask

    if valid_metric_scale_mask.any():
        # Compute the scale norm factor using the predicted metric scale points
        metric_pr_normalization_output = normalize_multiple_pointclouds(
            metric_pr_pts_to_compute_scale, valid_masks, norm_mode, ret_factor=True
        )
        pr_metric_norm_factor = metric_pr_normalization_output[-1]

        # Get the valid ground truth and predicted scale norm factors for the metric ground truth quantities
        gt_metric_norm_factor = gt_norm_factor[valid_metric_scale_mask]
        pr_metric_norm_factor = pr_metric_norm_factor[valid_metric_scale_mask]

        # Convert the ground truth and predicted scale norm factors to cpu
        gt_metric_norm_factor = gt_metric_norm_factor.cpu()
        pr_metric_norm_factor = pr_metric_norm_factor.cpu()
    else:
        gt_metric_norm_factor = None
        pr_metric_norm_factor = None

    # Move the other quantites to CPU
    for i in range(n_views):
        # Convert the other quantities to cpu
        gt_pts[i] = gt_pts[i].cpu()
        pr_pts[i] = pr_pts[i].cpu()
        valid_masks[i] = valid_masks[i].cpu()

    # Pack the required information into a dictionary
    gt_info = {
        "pts3d": gt_pts,
        "metric_scale": gt_metric_norm_factor,
    }
    pr_info = {
        "pts3d": pr_pts,
        "metric_scale": pr_metric_norm_factor,
    }

    return gt_info, pr_info, valid_masks


def build_dataset(dataset, batch_size, num_workers):
    """
    Builds data loaders for testing.

    Args:
        dataset: Dataset specification string.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    print("Building data loader for dataset: ", dataset)
    loader = get_test_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )

    print("Dataset length: ", len(loader))
    return loader


@torch.no_grad()
def benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # Determine the mixed precision floating point type
    if args.amp:
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif args.amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Init Test Datasets and Dataloaders
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    data_loaders = {
        dataset.split("(")[0]: build_dataset(
            dataset, args.batch_size, args.dataset.num_workers
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load Model
    model = init_model(
        args.model.model_str, args.model.model_config, torch_hub_force_reload=False
    )
    model.to(device)  # Move model to device

    # Load pretrained model
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(
            args.model.pretrained, map_location=device, weights_only=False
        )
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # in case it occupies memory

    # Warn user that the benchmarking assumes global pointmaps are predicted in view0 frame
    warnings.warn(
        "Assumes global pointmaps are predicted in view0 frame. Please ensure this is the case."
    )

    # Create dictionary to keep track of the results across different benchmarking datasets
    per_dataset_results = {}

    # Loop over the benchmarking datasets
    for benchmark_dataset_name, data_loader in data_loaders.items():
        print("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        # Create dictionary to keep track of the results across different scenes
        per_scene_results = {}

        # Init list of metrics for each scene
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                "metric_scale_abs_rel": [],
                "pointmaps_abs_rel": [],
                "pointmaps_inlier_thres_103": [],
            }

        # Loop over the batches
        for batch in data_loader:
            n_views = len(batch)
            # Remove unnecessary indices
            for view in batch:
                view["idx"] = view["idx"][2:]

            # Transfer batch to device
            ignore_keys = set(
                [
                    "depthmap",
                    "dataset",
                    "label",
                    "instance",
                    "idx",
                    "true_shape",
                    "rng",
                    "data_norm_type",
                ]
            )
            for view in batch:
                for name in view.keys():  # pseudo_focal
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            # Run model inference
            # Length of preds is equal to the number of views
            with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
                preds = model(batch)

            # Get all the information needed to compute the metrics
            gt_info, pr_info, valid_masks = get_all_info_for_metric_computation(
                batch,
                preds,
            )

            # Loop over each set in the batch and compute the metrics across all views
            batch_size = batch[0]["img"].shape[0]
            for batch_idx in range(batch_size):
                # Get the scene of the multi-view set
                scene = batch[0]["label"][batch_idx]

                # Compute the metrics across all views
                pointmaps_abs_rel_across_views = []
                pointmaps_inlier_thres_103_across_views = []
                for view_idx in range(n_views):
                    # Get the valid mask for the current view
                    valid_mask_curr_view = valid_masks[view_idx][batch_idx].numpy()
                    # Compute the metrics and append them to the respective lists
                    pointmaps_abs_rel_curr_view = m_rel_ae(
                        gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
                        pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
                        mask=valid_mask_curr_view,
                    )
                    pointmaps_inlier_thres_103_curr_view = thresh_inliers(
                        gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
                        pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
                        mask=valid_mask_curr_view,
                        thresh=1.03,
                    )
                    pointmaps_abs_rel_across_views.append(pointmaps_abs_rel_curr_view)
                    pointmaps_inlier_thres_103_across_views.append(
                        pointmaps_inlier_thres_103_curr_view
                    )

                # Compute the average across all views
                pointmaps_abs_rel_curr_set = np.mean(pointmaps_abs_rel_across_views)
                pointmaps_inlier_thres_103_curr_set = np.mean(
                    pointmaps_inlier_thres_103_across_views
                )

                # Compute the metric scale absolute relative error
                gt_metric_scale_curr_set = gt_info["metric_scale"][batch_idx].numpy()
                pr_metric_scale_curr_set = pr_info["metric_scale"][batch_idx].numpy()
                metric_scale_err_curr_set = (
                    pr_metric_scale_curr_set - gt_metric_scale_curr_set
                )
                metric_scale_abs_rel_curr_set = (
                    np.abs(metric_scale_err_curr_set) / gt_metric_scale_curr_set
                )

                # Append the metrics to the respective lists
                per_scene_results[scene]["metric_scale_abs_rel"].append(
                    metric_scale_abs_rel_curr_set.item()
                )
                per_scene_results[scene]["pointmaps_abs_rel"].append(
                    pointmaps_abs_rel_curr_set.item()
                )
                per_scene_results[scene]["pointmaps_inlier_thres_103"].append(
                    pointmaps_inlier_thres_103_curr_set.item()
                )

        # Save the per scene results to a json file
        with open(
            os.path.join(
                args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"
            ),
            "w",
        ) as f:
            json.dump(per_scene_results, f, indent=4)

        # Aggregate the per scene results
        across_dataset_results = {}
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                if metric not in across_dataset_results.keys():
                    across_dataset_results[metric] = []
                across_dataset_results[metric].extend(per_scene_results[scene][metric])

        # Compute the mean across all scenes
        for metric in across_dataset_results.keys():
            across_dataset_results[metric] = np.mean(
                across_dataset_results[metric]
            ).item()

        # Save the average results across all scenes to a json file
        with open(
            os.path.join(
                args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"
            ),
            "w",
        ) as f:
            json.dump(across_dataset_results, f, indent=4)

        # Print the average results across all scenes
        print("Average results across all scenes for dataset: ", benchmark_dataset_name)
        for metric in across_dataset_results.keys():
            print(f"{metric}: {across_dataset_results[metric]}")

        # Add the average result to the per dataset result dictionary
        per_dataset_results[benchmark_dataset_name] = across_dataset_results

    # Compute the average results across all datasets and add an average entry to the per dataset result dictionary
    average_results = {}
    for metric in per_dataset_results[next(iter(per_dataset_results))].keys():
        metric_values = [
            per_dataset_results[dataset][metric] for dataset in per_dataset_results
        ]
        average_results[metric] = np.mean(metric_values).item()
    per_dataset_results["Average"] = average_results

    # Print the average results across all datasets
    print("Benchmarking Done! ...")
    print("Average results across all datasets:")
    for metric in average_results.keys():
        print(f"{metric}: {average_results[metric]}")

    # Save the per dataset results to a json file
    with open(os.path.join(args.output_dir, "per_dataset_results.json"), "w") as f:
        json.dump(per_dataset_results, f, indent=4)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="dense_n_view_benchmark"
)
def execute_benchmarking(cfg: DictConfig):
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Run the testing
    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()  # noqa
