# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to aggregate the precomputed pairwise overlap matrices as adjacency lists across all scenes.
Stores the aggregated info as a npz file after splitting the scene names into train, val and test splits.
This is not used since the aggregated metadata files can be very large.
"""

import os
from glob import glob
from typing import List, Optional, Set, Union

import numpy as np
from natsort import natsorted

from morphcloud.utils.misc import seed_everything
from morphcloud.utils.parallel import parallel_processes
from morphcloud.utils.wai.scene_frame import get_scene_names


def load_overlap_data(scene_folder, covisibility_version_key):
    """Load the pairwise overlap matrix and file names for a given scene."""
    overlap_pattern = os.path.join(
        scene_folder,
        "covisibility",
        covisibility_version_key,
        "pairwise_covisibility*.npy",
    )
    overlap_files = glob(overlap_pattern)

    # Assuming there's only one valid overlap matrix file per scene
    overlap_file = overlap_files[0]
    pairwise_overlap_mat = np.load(overlap_file)

    return pairwise_overlap_mat


def preprocess_to_adjacency_list(overlap_matrix):
    """
    Converts a dense overlap matrix to an adjacency list and saves it to a file.
    Parameters:
      overlap_matrix : numpy.ndarray
          A binary adjacency matrix of shape (N, N) where overlap_matrix[i, j] == 1 indicates an overlap.

    Returns:
      adjacency_list : dict of lists
          A dictionary mapping each vertex to a list of its neighbors.
      total_number_of_edges : float
    """
    adjacency_list = {}
    total_number_of_edges = np.sum(overlap_matrix)
    N = overlap_matrix.shape[0]
    for i in range(N):
        neighbors = np.flatnonzero(overlap_matrix[i])
        # Skip adding index to adjacency list if it has no neighbors
        if len(neighbors) == 0:
            continue
        adjacency_list[i] = neighbors.tolist()

    return adjacency_list, total_number_of_edges


def process_single_scene(scene_name, root_dir, optimal_thres, covisibility_version_key):
    """Process a single scene and return its data."""
    scene_folder = os.path.join(root_dir, scene_name)
    pairwise_overlap_mat = load_overlap_data(scene_folder, covisibility_version_key)

    # Compute the final pairwise overlap matrix
    final_pairwise_overlap_mat = (pairwise_overlap_mat + pairwise_overlap_mat.T) / 2

    # Normalize the final pairwise overlap matrix using the diagonal elements
    diag_self_overlap = np.diag(final_pairwise_overlap_mat) + 1e-8
    final_pairwise_overlap_mat = final_pairwise_overlap_mat / diag_self_overlap

    # Assign overlap score of zero to self-pairs
    np.fill_diagonal(final_pairwise_overlap_mat, 0)

    # Threshold the pairwise overlap matrix
    final_pairwise_overlap_mat = (final_pairwise_overlap_mat > optimal_thres).astype(
        int
    )

    # Convert the pairwise overlap matrix to an adjacency list
    adjacency_list, total_number_of_edges = preprocess_to_adjacency_list(
        final_pairwise_overlap_mat
    )

    # If the adjacency list is empty, return None
    if len(list(adjacency_list.keys())) == 0:
        return scene_name, None

    # Return the data for this scene
    return scene_name, {
        "adjacency_list": adjacency_list,
        "total_number_of_edges": int(total_number_of_edges),
    }


def aggregate_scenes(
    root_dir,
    scene_list,
    optimal_thres,
    output_path,
    covisibility_version_key,
    num_workers=1,
):
    """Load the pairwise overlap matrices as adjacency lists for all scenes and store them as a npz metadata file."""
    # Initialize an empty dict to store the aggregated data
    aggregated_data = {}

    # Prepare arguments for parallel processing
    args_list = [
        (scene_name, root_dir, optimal_thres, covisibility_version_key)
        for scene_name in scene_list
    ]

    # Process scenes in parallel
    print(f"Processing {len(scene_list)} scenes using {num_workers} workers")
    results = parallel_processes(
        process_single_scene,
        args_list,
        workers=num_workers,
        star_args=True,
        desc="Processing scenes",
        front_num=0,
    )

    # Collect results
    for scene_name, data in results:
        if data is not None:
            aggregated_data[scene_name] = data
        else:
            print(f"Skipping {scene_name} due to no adjacency")

    # Save the aggregated data as a npz file
    np.savez(output_path, **aggregated_data)


class DatasetAggregator:
    """Base class for dataset aggregation."""

    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        output_dir: str,
        optimal_thres: float = 0.25,
        covisibility_version_key: str = "v0",
        depth_folder: str = "depth",
        raw_data_root_dir: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize the dataset aggregator.

        Args:
            dataset_name: Name of the dataset
            root_dir: Path to the root directory of the dataset
            output_dir: Path to the output directory
            optimal_thres: Threshold for the overlap matrix
            covisibility_version_key: Key for the covisibility version
            depth_folder: Name of the depth folder (usually "depth" or "rendered_depth")
            raw_data_root_dir: Path to the raw data root directory (optional)
            num_workers: Number of parallel workers to use (optional)
        """
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.optimal_thres = optimal_thres
        self.covisibility_version_key = covisibility_version_key
        self.depth_folder = depth_folder
        self.raw_data_root_dir = raw_data_root_dir
        self.num_workers = num_workers

        # Get all WAI scenes
        self.all_scenes = self._get_all_scenes()

    def _get_all_scenes(self) -> Set[str]:
        """Get all WAI scenes for the dataset."""
        cfg = {
            "root": self.root_dir,
            "scene_filters": [
                {"exists": "scene_meta.json"},
                {"exists": f"covisibility/{self.covisibility_version_key}"},
                {"exists": self.depth_folder},
            ],
        }
        return set(get_scene_names(cfg))

    def _get_split_scenes(
        self, split_name: str, split_scenes: Union[List[str], str, Set[str]]
    ) -> List[str]:
        """Get the scenes for a specific split."""
        if split_scenes == "all":
            return_scenes = natsorted(list(self.all_scenes))
            self.all_scenes -= self.all_scenes
            return return_scenes

        if isinstance(split_scenes, (list, set)):
            split_set = set(split_scenes)
            if not split_set.issubset(self.all_scenes):
                missing_scenes = split_set - self.all_scenes
                print(
                    f"{len(missing_scenes)} {split_name} scenes are missing in all WAI scenes of {self.dataset_name}."
                )
                split_set -= missing_scenes
            self.all_scenes -= split_set
            return natsorted(list(split_set))

        raise ValueError(f"Invalid split_scenes type: {type(split_scenes)}")

    def _process_split(self, split_name: str, scenes: List[str]):
        """Process a specific split."""
        print(f"Aggregating {split_name} split...")
        split_output_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        output_path = os.path.join(
            split_output_dir,
            f"{self.dataset_name}_aggregated_metadata_{split_name}.npz",
        )
        aggregate_scenes(
            root_dir=self.root_dir,
            scene_list=scenes,
            optimal_thres=self.optimal_thres,
            output_path=output_path,
            covisibility_version_key=self.covisibility_version_key,
            num_workers=self.num_workers,
        )

    def aggregate(
        self, val_split_scenes=None, test_split_scenes=None, train_split_scenes=None
    ):
        """Aggregate the dataset."""
        # Process test split if provided
        if test_split_scenes is not None:
            test_scenes = self._get_split_scenes("test", test_split_scenes)
            self._process_split("test", test_scenes)

        # Process val split if provided
        if val_split_scenes is not None:
            val_scenes = self._get_split_scenes("val", val_split_scenes)
            self._process_split("val", val_scenes)

        # Process remaining scenes as train split if not provided
        if train_split_scenes is not None:
            train_scenes = self._get_split_scenes("train", train_split_scenes)
            self._process_split("train", train_scenes)
        else:
            if self.all_scenes and (len(self.all_scenes) > 0):
                train_scenes = natsorted(list(self.all_scenes))
                self._process_split("train", train_scenes)


class RandomSplitAggregator(DatasetAggregator):
    """Aggregator for datasets with random splits."""

    def aggregate(self, val_ratio: float = 0.05):
        """Aggregate the dataset with random splits."""
        # Create a val split by randomly selecting scenes
        all_scenes_list = list(self.all_scenes)
        val_scenes = np.random.choice(
            all_scenes_list, int(len(all_scenes_list) * val_ratio), replace=False
        )
        val_scenes = set(val_scenes)

        # Remove val scenes from all scenes
        self.all_scenes -= val_scenes

        # Process val split
        self._process_split("val", natsorted(list(val_scenes)))

        # Process train split
        train_scenes = natsorted(list(self.all_scenes))
        self._process_split("train", train_scenes)


def main():
    """Main function to parse arguments and aggregate datasets."""
    # Seed the random number generator
    seed_everything(42)

    print("No datasets implemented yet")


if __name__ == "__main__":
    main()
