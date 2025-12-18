# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to visualize WAI format data
"""

import argparse
import os

import cv2
import numpy as np
import rerun as rr
from tqdm import tqdm

from mapanything.utils.cropping import (
    rescale_image_and_other_optional_info,
    resize_with_nearest_interpolation_to_match_aspect_ratio,
)
from mapanything.utils.misc import seed_everything
from mapanything.utils.viz import log_posed_rgbd_data_to_rerun, script_add_rerun_args
from mapanything.utils.wai.core import load_data, load_frame


def viz_wai_rgbd_data(
    args,
    depth_key="depth",
    local_frame=False,
    viz_string="WAI_Viz",
    load_skymask=False,
    confidence_key=None,
    confidence_thres=0,
):
    """
    Visualize all the images in the scene directory by logging it to rerun
    """
    # Setup Rerun if needed
    if args.viz:
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("gt", rr.ViewCoordinates.RDF, static=True)

    # Load the scene meta data
    scene_root = os.path.join(args.root_dir, args.scene)
    scene_meta = load_data(os.path.join(scene_root, "scene_meta.json"), "scene_meta")
    scene_frame_names = list(scene_meta["frame_names"].keys())

    # Loop over images and log to rerun
    for frame_idx, frame in enumerate(tqdm(scene_frame_names)):
        # Load the frame data
        if load_skymask:
            modalities = ["image", depth_key, "skymask"]
        else:
            modalities = ["image", depth_key]
        if confidence_key is not None:
            modalities.append(confidence_key)
        frame_data = load_frame(
            os.path.join(args.root_dir, args.scene),
            frame,
            modalities=modalities,
            scene_meta=scene_meta,
        )

        # Convert necessary data to numpy
        rgb_image = frame_data["image"].permute(1, 2, 0).numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)
        depth_data = frame_data[depth_key].numpy()
        intrinsics = frame_data["intrinsics"].numpy()

        # If depth is predicted, resize it to match the aspect ratio of the image
        # Then, resize the image and update intrinsics to match the resized predicted depth
        if "pred" in depth_key:
            # Get the dimensions of the original image
            img_h, img_w = rgb_image.shape[:2]

            # Resize depth to match image aspect ratio while ensuring that depth resolution doesn't increase
            depth_data, target_depth_h, target_depth_w = (
                resize_with_nearest_interpolation_to_match_aspect_ratio(
                    input_data=depth_data, img_h=img_h, img_w=img_w
                )
            )

            # Now resize the image and update intrinsics to match the resized depth
            rgb_image, _, intrinsics, _ = rescale_image_and_other_optional_info(
                image=rgb_image,
                output_resolution=(target_depth_w, target_depth_h),
                depthmap=None,
                camera_intrinsics=intrinsics,
            )
            rgb_image = np.array(rgb_image)

        # Mask depth if sky mask is loaded
        if load_skymask:
            mask_data = frame_data["skymask"].numpy().astype(int)
            mask_data = cv2.resize(
                mask_data,
                (depth_data.shape[1], depth_data.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            depth_data = np.where(mask_data, 0, depth_data)

        if confidence_key is not None:
            confidence_map = frame_data[confidence_key].numpy().astype(np.float32)
            confidence_mask = (confidence_map > confidence_thres).astype(int)
            confidence_mask = cv2.resize(
                confidence_mask,
                (depth_data.shape[1], depth_data.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            depth_data = np.where(confidence_mask, depth_data, 0)

        # Resize data to smaller resolution for visualization
        # Set output resolution to longest side of image to be 224 (preserving aspect ratio)
        target_longest_side = 224
        if rgb_image.shape[0] > rgb_image.shape[1]:
            # Height is longer, so set height to target and scale width
            output_resolution = (
                int(rgb_image.shape[1] * target_longest_side / rgb_image.shape[0]),
                target_longest_side,
            )
        else:
            # Width is longer or equal, so set width to target and scale height
            output_resolution = (
                target_longest_side,
                int(rgb_image.shape[0] * target_longest_side / rgb_image.shape[1]),
            )
        rgb_image, depth_data, intrinsics, _ = rescale_image_and_other_optional_info(
            image=rgb_image,
            output_resolution=output_resolution,
            depthmap=depth_data,
            camera_intrinsics=intrinsics,
        )
        rgb_image = np.array(rgb_image)

        # Init pose
        if local_frame:
            pose = np.eye(4)
            base_name = "gt/image"
        else:
            pose = frame_data["extrinsics"].numpy()
            base_name = f"gt/image_{frame_idx}"

        # Log data to rerun
        if args.viz:
            rr.set_time("stable_time", sequence=frame_idx)
            log_posed_rgbd_data_to_rerun(
                rgb_image, depth_data, pose, intrinsics, base_name
            )


def get_dataset_config(dataset_type):
    """
    Get the configuration for a specific dataset type

    Args:
        dataset_type: The type of dataset to configure

    Returns:
        dict: Configuration for the dataset including root_dir, scene, and viz parameters
    """
    configs = {
        "scannetpp": {
            "root_dir": "/ai4rl/fsx/xrtech/data/scannetppv2",
            "scene": "0a5c013435",
            "depth_key": "rendered_depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "blendedmvs": {
            "root_dir": "/ai4rl/fsx/xrtech/data/blendedmvs",
            "scene": "584b9a747072670e72bfc49d",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "eth3d": {
            "root_dir": "/ai4rl/fsx/xrtech/data/eth3d",
            "scene": "delivery_area",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "megadepth": {
            "root_dir": "/ai4rl/fsx/xrtech/data/megadepth",
            "scene": "0000_1",
            # "scene": "0086_0", # Disjoint reconstructions with different scale
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "spring": {
            "root_dir": "/ai4rl/fsx/xrtech/data/spring",
            "scene": "0004",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": True,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "mpsd": {
            "root_dir": "/ai4rl/fsx/xrtech/data/mpsd",
            "scene": "geoeven_4_2019-03-19T11_33_14.019516",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "tav2": {
            "root_dir": "/ai4rl/fsx/xrtech/data/tav2_wb",
            "scene": "PolarSciFi",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "ase": {
            "root_dir": "/ai4rl/fsx/xrtech/data/ase",
            "scene": "10000",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "dl3dv": {
            "root_dir": "/ai4rl/fsx/xrtech/data/dl3dv",
            "scene": "9K_963080e5ee7ca52ee8fabd294ad9e12220ed5064686ec9786a17aed23da8850f",
            "depth_key": "pred_depth/mvsanywhere",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": "depth_confidence/mvsanywhere",
            "confidence_thres": 0.25,
        },
        "unrealstereo4k": {
            "root_dir": "/ai4rl/fsx/xrtech/data/unrealstereo4k",
            "scene": "00000",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "mvs_synth": {
            "root_dir": "/ai4rl/fsx/xrtech/data/mvs_synth",
            "scene": "0000",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "paralleldomain4d": {
            "root_dir": "/ai4rl/fsx/xrtech/data/paralleldomain4d",
            "scene": "scene_000000",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "sailvos3d": {
            "root_dir": "/ai4rl/fsx/xrtech/data/sailvos3d",
            "scene": "fam_6_mcs_5",
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
        "dynamicreplica": {
            "root_dir": "/ai4rl/fsx/xrtech/data/dynamicreplica",
            "scene": "26dd2c-3_obj_source",
            # "scene": "009850-3_obj_source", # Part of the floor depth is wrong
            "depth_key": "depth",
            "local_frame": False,
            "viz_string": "WAI_Viz",
            "load_skymask": False,
            "confidence_key": None,
            "confidence_thres": 0.0,
        },
    }

    assert dataset_type in configs, (
        f"Dataset {dataset_type} not found, available: {list(configs.keys())}"
    )
    return configs.get(dataset_type)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Path to the root directory",
        default="/ai4rl/fsx/xrtech/data/eth3d",
    )
    parser.add_argument(
        "--scene", type=str, help="Scene to visualize", default="courtyard"
    )
    parser.add_argument("--viz", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "scannetpp",
            "blendedmvs",
            "eth3d",
            "megadepth",
            "spring",
            "mpsd",
            "ase",
            "tav2",
            "dl3dv",
            "unrealstereo4k",
            "mvs_synth",
            "paralleldomain4d",
            "sailvos3d",
            "dynamicreplica",
        ],
        default="eth3d",
        help="Dataset type to visualize",
    )
    parser.add_argument(
        "--depth_key", type=str, help="Key for depth data in the frame", default=None
    )
    parser.add_argument(
        "--load_skymask", action="store_true", help="Whether to load and apply sky mask"
    )
    parser.add_argument(
        "--local_frame",
        action="store_true",
        help="Whether to use local frame for visualization",
    )

    return parser


if __name__ == "__main__":
    # Parser for Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # Set the seed
    seed_everything(0)

    # Get dataset configuration
    config = get_dataset_config(args.dataset)

    # Override config with command line arguments if provided
    if args.root_dir != parser.get_default("root_dir"):
        config["root_dir"] = args.root_dir
    if args.scene != parser.get_default("scene"):
        config["scene"] = args.scene
    if args.depth_key is not None:
        config["depth_key"] = args.depth_key
    if args.load_skymask:
        config["load_skymask"] = True
    if args.local_frame:
        config["local_frame"] = True

    # Update args with config values
    args.root_dir = config["root_dir"]
    args.scene = config["scene"]

    # Run visualization with the configured parameters
    viz_wai_rgbd_data(
        args,
        depth_key=config["depth_key"],
        local_frame=config["local_frame"],
        viz_string=config["viz_string"],
        load_skymask=config["load_skymask"],
        confidence_key=config["confidence_key"],
        confidence_thres=config["confidence_thres"],
    )
