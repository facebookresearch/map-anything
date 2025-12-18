# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to aggregate the scene names for the different splits used in MapAnything.
The valid scenes for each dataset are organized into train, val and optionally test splits.
Scene lists are saved as numpy arrays for efficient loading.
"""

import argparse
import os
from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from mapanything.datasets.utils.data_splits import (
    BlendedMVSSplits,
    DL3DV10KSplits,
    ETH3DSplits,
    MegaDepthSplits,
    MPSDSplits,
    ScanNetPPSplits,
    SpringSplits,
    TartanAirV2Splits,
)
from mapanything.utils.misc import seed_everything
from mapanything.utils.wai.scene_frame import get_scene_names


def save_scene_lists(scene_list, output_path):
    """Save the list of scene names as a numpy array."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert scene list to numpy array and save
    scene_array = np.array(scene_list, dtype=object)
    np.save(output_path, scene_array)

    print(f"Saved {len(scene_list)} scene names to {output_path}")


def print_dataset_stats(output_dir: str, datasets: List[str]):
    """Print statistics for the number of scenes in each dataset split.

    Args:
        output_dir: Path to the output directory containing the aggregated scene lists
        datasets: List of dataset names to print stats for
    """
    print("\n" + "=" * 80)
    print("DATASET SCENE STATISTICS")
    print("=" * 80)

    # Define the possible splits
    splits = ["train", "val", "test"]

    # Track totals across all datasets
    total_stats = {split: 0 for split in splits}

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 40)

        dataset_stats = {}
        dataset_total = 0

        for split in splits:
            split_file = os.path.join(
                output_dir, split, f"{dataset}_scene_list_{split}.npy"
            )

            if os.path.exists(split_file):
                try:
                    scene_array = np.load(split_file, allow_pickle=True)
                    num_scenes = len(scene_array)
                    dataset_stats[split] = num_scenes
                    dataset_total += num_scenes
                    total_stats[split] += num_scenes
                    print(f"  {split:>5}: {num_scenes:>6} scenes")
                except Exception as e:
                    print(f"  {split:>5}: Error loading file - {e}")
                    dataset_stats[split] = 0
            else:
                dataset_stats[split] = 0

        if dataset_total > 0:
            print(f"  {'Total':>5}: {dataset_total:>6} scenes")
        else:
            print(f"  No scene files found for {dataset}")

    # Print overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    overall_total = 0
    for split in splits:
        if total_stats[split] > 0:
            print(f"  {split:>5}: {total_stats[split]:>6} scenes")
            overall_total += total_stats[split]

    print(f"  {'Total':>5}: {overall_total:>6} scenes")
    print("=" * 80)


class DatasetAggregator:
    """Base class for dataset scene list aggregation."""

    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        output_dir: str,
        depth_folder: str = "depth",
        covisibility_version_key: str = "v0",
        raw_data_root_dir: Optional[str] = None,
    ):
        """Initialize the dataset aggregator.

        Args:
            dataset_name: Name of the dataset
            root_dir: Path to the root directory of the dataset
            output_dir: Path to the output directory
            depth_folder: Name of the depth folder (usually "depth", "rendered_depth" or "pred_depth/method")
            covisibility_version_key: Key for the covisibility version
            raw_data_root_dir: Path to the raw data root directory (optional)
        """
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.depth_folder = depth_folder
        self.covisibility_version_key = covisibility_version_key
        self.raw_data_root_dir = raw_data_root_dir

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
        print(f"Saving {split_name} split scene list...")
        split_output_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        output_path = os.path.join(
            split_output_dir, f"{self.dataset_name}_scene_list_{split_name}.npy"
        )
        save_scene_lists(scenes, output_path)

    def aggregate(
        self, val_split_scenes=None, test_split_scenes=None, train_split_scenes=None
    ):
        """Aggregate the dataset scene lists."""
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
        num_val_scenes = max(1, int(len(all_scenes_list) * val_ratio))
        val_scenes = np.random.choice(all_scenes_list, num_val_scenes, replace=False)
        val_scenes = set(val_scenes)

        # Remove val scenes from all scenes
        self.all_scenes -= val_scenes

        # Process val split
        self._process_split("val", natsorted(list(val_scenes)))

        # Process train split
        train_scenes = natsorted(list(self.all_scenes))
        self._process_split("train", train_scenes)


class ASEAggregator(RandomSplitAggregator):
    """Aggregator for ASE dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="ase",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )


class BlendedMVSAggregator(DatasetAggregator):
    """Aggregator for BlendedMVS dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="blendedmvs",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = BlendedMVSSplits()

    def aggregate(self):
        """Aggregate the BlendedMVS dataset."""
        super().aggregate(
            val_split_scenes=self.dataset_split_info.val_split_scenes,
            train_split_scenes=self.dataset_split_info.train_split_scenes,
        )


class DL3DVAggregator(DatasetAggregator):
    """Aggregator for DL3DV-10K dataset."""

    def __init__(
        self,
        root_dir,
        output_dir,
        covisibility_version_key="v0_mvsa_based",
        raw_data_root_dir=None,
    ):
        super().__init__(
            dataset_name="dl3dv",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            raw_data_root_dir=raw_data_root_dir,
            depth_folder="mvsanywhere/v0/depth",
        )
        self.dataset_split_info = DL3DV10KSplits()

    def aggregate(self):
        """Aggregate the DL3DV-10K dataset."""
        # Load benchmark metadata
        benchmark_metadata = pd.read_csv(
            os.path.join(self.raw_data_root_dir, "benchmark-meta.csv")
        )
        scene_split_info = pd.read_csv(
            os.path.join(self.raw_data_root_dir, "DL3DV-valid.csv")
        )

        # Get the hash of val scenes
        val_scenes_hash = benchmark_metadata["hash"].tolist()

        # Create a mapping from hash to batch
        hash_to_batch = dict(zip(scene_split_info["hash"], scene_split_info["batch"]))

        # Create the scene names in format 'batch_hash'
        val_scenes = [f"{hash_to_batch.get(h, 'unknown')}_{h}" for h in val_scenes_hash]

        # Check if all val scenes are present in the WAI root directory and filter out missing ones
        valid_val_scenes = []
        for scene in val_scenes:
            scene_path = os.path.join(self.root_dir, scene)
            if os.path.exists(scene_path):
                valid_val_scenes.append(scene)

        # Aggregate the DL3DV-10K dataset with the constructed valid val scenes
        val_scenes = valid_val_scenes
        super().aggregate(val_split_scenes=val_scenes)


class DynamicReplicaAggregator(RandomSplitAggregator):
    """Aggregator for Dynamic Replica dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="dynamicreplica",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )


class ETH3DAggregator(DatasetAggregator):
    """Aggregator for ETH3D dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="eth3d",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = ETH3DSplits()

    def aggregate(self):
        """Aggregate the ETH3D dataset."""
        super().aggregate(test_split_scenes=self.dataset_split_info.test_split_scenes)


class MegaDepthAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="megadepth",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = MegaDepthSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class MPSDAggregator(DatasetAggregator):
    """Aggregator for MPSD dataset."""

    def __init__(
        self,
        root_dir,
        output_dir,
        covisibility_version_key="v0",
        raw_data_root_dir=None,
    ):
        super().__init__(
            dataset_name="mpsd",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            raw_data_root_dir=raw_data_root_dir,
            depth_folder="depth",
        )
        self.dataset_split_info = MPSDSplits()

    def aggregate(self):
        """Aggregate the MPSD dataset."""
        assert (
            self.dataset_split_info.val_split_scenes
            == "load_numpy_file_with_val_scenes"
        )
        assert self.raw_data_root_dir is not None, (
            "raw_data_root_dir must be provided for MPSD dataset"
        )

        # Load val scenes from numpy file
        val_scenes = np.load(
            os.path.join(self.raw_data_root_dir, "val_recon_folder_names.npy")
        )
        val_scenes = {scene.replace("/", "_") for scene in val_scenes}

        super().aggregate(val_split_scenes=val_scenes)


class MVSSynthAggregator(RandomSplitAggregator):
    """Aggregator for MVS Synth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="mvs_synth",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )


class ParallelDomain4DAggregator(RandomSplitAggregator):
    """Aggregator for Parallel Domain 4D dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="paralleldomain4d",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )


class SAILVOS3DAggregator(RandomSplitAggregator):
    """Aggregator for SAIL-VOS 3D dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="sailvos3d",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )


class ScanNetPPV2Aggregator(DatasetAggregator):
    """Aggregator for ScanNet++V2 dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="scannetppv2",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="rendered_depth",
        )
        self.dataset_split_info = ScanNetPPSplits()

    def aggregate(self):
        """Aggregate the ScanNet++V2 dataset."""
        super().aggregate(
            val_split_scenes=self.dataset_split_info.val_split_scenes,
            test_split_scenes=self.dataset_split_info.test_split_scenes,
        )


class SpringAggregator(DatasetAggregator):
    """Aggregator for Spring dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="spring",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = SpringSplits()

    def aggregate(self):
        """Aggregate the Spring dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class TartanAirV2Aggregator(DatasetAggregator):
    """Aggregator for TartanAirV2-WB dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="tav2_wb",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = TartanAirV2Splits()

    def aggregate(self):
        """Aggregate the TartanAirV2-WB dataset."""
        super().aggregate(
            val_split_scenes=self.dataset_split_info.val_split_scenes,
            test_split_scenes=self.dataset_split_info.test_split_scenes,
        )


class UnrealStereo4KAggregator(RandomSplitAggregator):
    """Aggregator for Unreal Stereo 4K dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="unrealstereo4k",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )


def main():
    """Main function to parse arguments and aggregate scene lists."""
    # Seed the random number generator
    seed_everything(42)

    # Setup args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wai_root",
        type=str,
        help="Path to the root of WAI format datasets",
        default="/ai4rl/fsx/xrtech/data",
    )
    parser.add_argument(
        "--raw_data_root",
        type=str,
        help="Path to the root of raw datasets from WAI is processed",
        default="/ai4rl/fsx/xrtech/raw_data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory",
        default="/ai4rl/fsx/nkeetha/mapanything_dataset_metadata",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Names of the datasets to process",
        default=[
            "ase",
            "blendedmvs",
            "dl3dv",
            "dynamicreplica",
            "eth3d",
            "megadepth",
            "mpsd",
            "mvs_synth",
            "paralleldomain4d",
            "sailvos3d",
            "scannetppv2",
            "spring",
            "tav2_wb",
            "unrealstereo4k",
        ],
        choices=[
            "ase",
            "blendedmvs",
            "dl3dv",
            "dynamicreplica",
            "eth3d",
            "megadepth",
            "mpsd",
            "mvs_synth",
            "paralleldomain4d",
            "sailvos3d",
            "scannetppv2",
            "spring",
            "tav2_wb",
            "unrealstereo4k",
        ],
    )
    parser.add_argument(
        "--print_stats",
        action="store_true",
        help="Print statistics for the number of scenes in each dataset split",
    )

    args = parser.parse_args()

    # Create output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # If print_stats flag is set, print statistics and exit
    if args.print_stats:
        print_dataset_stats(args.output_dir, args.datasets)
        return

    # Process each specified dataset
    for dataset in tqdm(args.datasets):
        # Get WAI Dataset Root Dir
        root_dir = os.path.join(args.wai_root, dataset)

        if dataset == "ase":
            # ASE
            aggregator = ASEAggregator(root_dir=root_dir, output_dir=args.output_dir)
            aggregator.aggregate()
        elif dataset == "blendedmvs":
            # BlendedMVS
            aggregator = BlendedMVSAggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "dl3dv":
            # DL3DV-10K. Expects the following csvs at raw data root directory:
            # https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/raw/main/benchmark-meta.csv
            # https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv
            raw_data_root_dir = os.path.join(
                args.raw_data_root, "DL3DV_10K_4K_resolution"
            )
            aggregator = DL3DVAggregator(
                root_dir=root_dir,
                output_dir=args.output_dir,
                raw_data_root_dir=raw_data_root_dir,
            )
            aggregator.aggregate()
        elif dataset == "dynamicreplica":
            # Dynamic Replica
            aggregator = DynamicReplicaAggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "eth3d":
            # ETH3D
            aggregator = ETH3DAggregator(root_dir=root_dir, output_dir=args.output_dir)
            aggregator.aggregate()
        elif dataset == "megadepth":
            # MegaDepth
            aggregator = MegaDepthAggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "mpsd":
            # MPSD
            raw_data_root_dir = os.path.join(args.raw_data_root, "mpsd")
            aggregator = MPSDAggregator(
                root_dir=root_dir,
                output_dir=args.output_dir,
                raw_data_root_dir=raw_data_root_dir,
            )
            aggregator.aggregate()
        elif dataset == "mvs_synth":
            # MVS Synth
            aggregator = MVSSynthAggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "paralleldomain4d":
            # Parallel Domain 4D
            aggregator = ParallelDomain4DAggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "sailvos3d":
            # SAIL-VOS 3D
            aggregator = SAILVOS3DAggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "scannetppv2":
            # ScanNet++V2
            aggregator = ScanNetPPV2Aggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "spring":
            # Spring
            aggregator = SpringAggregator(root_dir=root_dir, output_dir=args.output_dir)
            aggregator.aggregate()
        elif dataset == "tav2_wb":
            # TartanAirV2-WB
            aggregator = TartanAirV2Aggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        elif dataset == "unrealstereo4k":
            # Unreal Stereo 4K
            aggregator = UnrealStereo4KAggregator(
                root_dir=root_dir, output_dir=args.output_dir
            )
            aggregator.aggregate()
        else:
            print(f"Dataset {dataset} not implemented yet")


if __name__ == "__main__":
    main()
