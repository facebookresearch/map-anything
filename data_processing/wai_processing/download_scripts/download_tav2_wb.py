# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Download the wide baseline variant of TartanAirV2 (TAv2) dataset used in UFM (https://uniflowmatch.github.io/).
Extract the images, depth, poses and calibration corresponding to the different unqiue environments from the h5s.
"""

import argparse
import concurrent.futures
import logging
import os
import re

import numpy as np
import pandas as pd
import urllib3
from minio import Minio
from minio.error import S3Error
from PIL import Image
from tqdm import tqdm
from wai_processing.utils.distributed_h5 import (
    DistributedH5Reader,
)
from wai_processing.utils.parallel import parallel_threads

from morphcloud.utils.wai.core import store_data


def download_file(client, bucket_name, obj, destination_folder):
    "Download a file from MinIO server"
    object_name = os.path.basename(obj.object_name)
    destination_file = os.path.join(destination_folder, object_name)
    if not os.path.exists(destination_file):
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        try:
            client.fget_object(bucket_name, obj.object_name, destination_file)
            logging.info(f"Download successful: {object_name}")
        except S3Error as e:
            logging.error(f"Error downloading {object_name}: {e}")
            return
    else:
        logging.info(f"File {destination_file} already exists. Skipping...")


def download_folder(folder_name, bucket_name, client, destination_folder, num_workers):
    "Download a folder from MinIO server"
    objects = list(client.list_objects(bucket_name, prefix=folder_name, recursive=True))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(download_file, client, bucket_name, obj, destination_folder)
            for obj in objects
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Downloading {folder_name}",
        ):
            future.result()


def download_tav2_wb(args, num_workers):
    """Download the TAv2 wide baseline dataset.

    Args:
        args: Parsed command line arguments
        num_workers: Number of workers for parallel download
    """
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(args.root_dir, "tav2_download.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Define download client
    access_key = "jH8UFqt4oli1lmGabHeT"
    secret_key = "SU5aUyahsXB7AlgbSEYNeshm8GL2P5iKatd6iRrt"
    http_client = urllib3.PoolManager(
        cert_reqs="CERT_NONE",  # Disable SSL certificate verification
        maxsize=20,
    )
    urllib3.disable_warnings(
        urllib3.exceptions.InsecureRequestWarning
    )  # Disable SSL warnings
    client = Minio(
        "128.237.74.10:9000",
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        http_client=http_client,
    )
    bucket_name = "tav2"

    # Define the target download directory
    target_dir = os.path.join(args.root_dir, "tav2_wb_h5")
    os.makedirs(target_dir, exist_ok=True)

    # Define mapping of folders that need to be downloaded
    download_mapping = [
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/train_camera_data/",
            os.path.join(target_dir, "train_camera_data"),
        ),
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/validation_camera_data/",
            os.path.join(target_dir, "val_camera_data"),
        ),
        (
            "TartanAir/assembled/tartanair_640_pinhole_test_good_imgdep/train_camera_data/",
            os.path.join(target_dir, "test_camera_data"),
        ),
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/validation/",
            os.path.join(target_dir, "val"),
        ),
        (
            "TartanAir/assembled/tartanair_640_pinhole_test_good_imgdep/train/",
            os.path.join(target_dir, "test"),
        ),
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/train/",
            os.path.join(target_dir, "train"),
        ),
    ]

    # Loop over the folders and download them
    for curr_download_mapping in tqdm(download_mapping):
        source_folder_name, destination_folder = curr_download_mapping
        os.makedirs(destination_folder, exist_ok=True)
        download_folder(
            source_folder_name, bucket_name, client, destination_folder, num_workers
        )


def strip_terms_from_string(input_string, terms):
    """
    Remove all terms from the input string.

    Args:
        input_string: String to remove terms from
        terms: List of terms to remove

    Returns:
        String with all terms removed
    """
    # Create a regular expression pattern that matches any of the terms
    pattern = "|".join(map(re.escape, terms))
    # Use re.sub to replace all occurrences of the terms with an empty string
    result = re.sub(pattern, "", input_string)
    return result


class TartanAirV2WideBaseline:
    """
    TartanAirV2 Wide Baseline dataset containing 360-view images of different environments across various environmental conditions.
    The dataset is a custom dataset generated by pairing images from the same environment.
    See UFM paper for more details: https://uniflowmatch.github.io/
    """

    def __init__(self, ROOT, split=None):
        """
        Initialize the TartanAirV2WideBaseline dataset.

        Args:
            ROOT: Root directory containing the dataset
            split: Dataset split ('train', 'val', or 'test')
        """
        self.ROOT = ROOT
        self._load_data(split)
        self.split = split
        self.cross_env_terms = [
            "Autumn",
            "Spring",
            "SummerNight",
            "Winter",
            "WinterNight",
            "Day",
            "Night",
            "Fall",
            "Summer",
        ]

    def _load_data(self, split):
        """Load dataset files and metadata."""
        # Check if split is valid (Options: train, val, test)
        if split is None or split not in ["train", "val", "test"]:
            raise ValueError(f"Unknown split {split}, must be train, val or test")

        # Load the corresponding h5 reader
        self.h5_reader = DistributedH5Reader(os.path.join(self.ROOT, split))

        # Load the extrinsics & intrinsics data & pair metadata
        self.cam1_extrinsics = np.load(
            os.path.join(self.ROOT, f"{split}_camera_data", "cam1_extrinsics.npz")
        )["arr_0"]
        self.cam2_extrinsics = np.load(
            os.path.join(self.ROOT, f"{split}_camera_data", "cam2_extrinsics.npz")
        )["arr_0"]
        self.cam1_intrinsics = np.load(
            os.path.join(self.ROOT, f"{split}_camera_data", "cam1_intrinsics.npz")
        )["arr_0"]
        self.cam2_intrinsics = np.load(
            os.path.join(self.ROOT, f"{split}_camera_data", "cam2_intrinsics.npz")
        )["arr_0"]
        self.pair_metadata = pd.read_json(
            os.path.join(self.ROOT, f"{split}_camera_data", "metadata.json"),
            orient="split",
        )

    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.h5_reader)

    def __getitem__(self, pair_idx):
        """
        Get a pair of views from the dataset.

        Args:
            pair_idx: Index of the pair to retrieve

        Returns:
            List of two views, each containing image, depthmap, camera pose, etc.
        """
        # Get the image & depthmap data for the pair
        data = self.h5_reader.read(
            pair_idx,
            [
                "img0",
                "img1",
                "depth0",
                "depth1",
            ],
        )

        imgs = [data["img0"], data["img1"]]
        depthmaps = [data["depth0"], data["depth1"]]
        intrinsics_list = [
            self.cam1_intrinsics[pair_idx],
            self.cam2_intrinsics[pair_idx],
        ]
        camera_poses = [self.cam1_extrinsics[pair_idx], self.cam2_extrinsics[pair_idx]]
        curr_pair_metadata = self.pair_metadata.iloc[pair_idx]

        views = []
        for view_idx in range(len(imgs)):
            image = imgs[view_idx].numpy()
            depthmap = depthmaps[view_idx].numpy()
            intrinsics = intrinsics_list[view_idx]
            camera_pose = camera_poses[view_idx]
            env = curr_pair_metadata[f"img{view_idx + 1}_env"]
            env = strip_terms_from_string(env, self.cross_env_terms)
            frame_name = f"{pair_idx:08}_{view_idx:01}"

            # Mask out sky depth based on env name
            if env in ["Prison", "OldTown"]:
                sky_depth = 80.0
            elif env == "ShoreCaves":
                sky_depth = 120.0
            elif env == "OldBrickHouse":
                sky_depth = 35.0
            else:
                sky_depth = 200.0
            non_sky_mask = depthmap < sky_depth
            depthmap = depthmap * non_sky_mask

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="TartanAirV2WideBaseline",
                    label=env,
                    instance=frame_name,
                )
            )

        return views


def process_pair(dataset, pair_idx, output_dir):
    "Get a pair of views from the dataset and save them to the output directory"
    # Get the views
    views = dataset[pair_idx]

    for view in views:
        img = view["img"]
        depth = view["depthmap"]
        camera_params = view["camera_intrinsics"]
        camera_pose = view["camera_pose"]
        scene_name = view["label"]
        frame_name = view["instance"]

        # Get the respective output directories
        output_scene_dir = os.path.join(output_dir, scene_name)
        output_img_dir = os.path.join(output_scene_dir, "images")
        output_depth_dir = os.path.join(output_scene_dir, "depth")
        output_cam_params_dir = os.path.join(output_scene_dir, "camera_params")
        output_cam_pose_dir = os.path.join(output_scene_dir, "poses")

        # Get the output paths
        output_img_path = os.path.join(output_img_dir, f"{frame_name}.png")
        output_depth_path = os.path.join(output_depth_dir, f"{frame_name}.exr")
        output_cam_params_path = os.path.join(
            output_cam_params_dir, f"{frame_name}.npy"
        )
        output_cam_pose_path = os.path.join(output_cam_pose_dir, f"{frame_name}.npy")

        # Save the data
        img = Image.fromarray(img.astype(np.uint8))
        store_data(output_img_path, img, "image")
        store_data(output_depth_path, depth, "depth")
        np.save(output_cam_params_path, camera_params)
        np.save(output_cam_pose_path, camera_pose)


def extract_h5_data(root_dir, output_dir, split, num_of_workers):
    """
    Extract the h5 data from the TartanAirV2 Wide Baseline dataset.
    """
    # Initialize the pair dataloader
    dataset = TartanAirV2WideBaseline(
        split=split,
        ROOT=root_dir,
    )

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Pre-create all necessary directories to avoid race conditions
    unique_envs = dataset.pair_metadata["img1_env"].unique()
    for env in unique_envs:
        geometric_env = strip_terms_from_string(env, dataset.cross_env_terms)
        env_output_dir = os.path.join(output_dir, geometric_env)
        os.makedirs(os.path.join(env_output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(env_output_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(env_output_dir, "camera_params"), exist_ok=True)
        os.makedirs(os.path.join(env_output_dir, "poses"), exist_ok=True)

    # Prepare arguments for parallel processing
    func_args = [(dataset, pair_idx, output_dir) for pair_idx in range(len(dataset))]

    # Use parallel_threads to process the views
    parallel_threads(process_pair, func_args, workers=num_of_workers, star_args=True)


def setup_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download & Extract TAv2 Wide Baseline Dataset (UFM Version)"
    )
    parser.add_argument(
        "-r",
        "--root_dir",
        type=str,
        help="Root directory for download, tav2_wb_h5 & tav2_wb will be created in this directory",
    )
    parser.add_argument(
        "-sd",
        "--skip_download",
        action="store_true",
        help="Skip downloading the h5 dataset",
    )
    parser.add_argument(
        "-se",
        "--skip_extract",
        action="store_true",
        help="Skip extracting the h5 dataset",
    )
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Download the h5 dataset if not already downloaded
    if not args.skip_download:
        download_tav2_wb(args, num_workers=20)

    # Extract the h5 dataset if not already extracted
    if not args.skip_extract:
        h5_root_dir = os.path.join(args.root_dir, "tav2_wb_h5")
        output_dir = os.path.join(args.root_dir, "tav2_wb")
        splits = ["val", "test", "train"]
        for split in splits:
            extract_h5_data(h5_root_dir, output_dir, split, num_of_workers=40)
