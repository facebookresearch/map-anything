# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Base dataset class that enables easy resizing and combining

References: DUSt3R
"""

import numpy as np

from mapanything.datasets.base.batched_sampler import (
    BatchedMultiFeatureRandomSampler,
    DynamicBatchedMultiFeatureRandomSampler,
)


class EasyDataset:
    """
    Dataset that can be easily resized and combined.

    Examples:
    ---------
        2 * dataset ==> Duplicate each element 2x

        10 @ dataset ==> Set the size to 10 (random sampling, duplicates if necessary)

        Dataset1 + Dataset2 ==> Concatenate datasets
    """

    def __add__(self, other):
        """
        Concatenate this dataset with another dataset.

        Args:
            other (EasyDataset): Another dataset to concatenate with this one

        Returns:
            CatDataset: A new dataset that is the concatenation of this dataset and the other
        """
        return CatDataset([self, other])

    def __rmul__(self, factor):
        """
        Multiply the dataset by a factor, duplicating each element.

        Args:
            factor (int): Number of times to duplicate each element

        Returns:
            MulDataset: A new dataset with each element duplicated 'factor' times
        """
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        """
        Resize the dataset to a specific size using random sampling.

        Args:
            factor (int): The new size of the dataset

        Returns:
            ResizedDataset: A new dataset with the specified size
        """
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        """
        Set the current epoch for all constituent datasets.

        Args:
            epoch (int): The current epoch number
        """
        pass  # nothing to do by default

    def make_sampler(
        self,
        batch_size=None,
        shuffle=True,
        world_size=1,
        rank=0,
        drop_last=True,
        max_num_of_images_per_gpu=None,
        use_dynamic_sampler=True,
    ):
        """
        Create a sampler for this dataset.

        Args:
            batch_size (int, optional): Number of samples per batch (used for non-dynamic sampler). Defaults to None.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            world_size (int, optional): Number of distributed processes. Defaults to 1.
            rank (int, optional): Rank of the current process. Defaults to 0.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
            max_num_of_images_per_gpu (int, optional): Maximum number of images per GPU for dynamic batching. Defaults to None.
            use_dynamic_sampler (bool, optional): Whether to use the dynamic sampler. Defaults to True.

        Returns:
            DynamicBatchedMultiFeatureRandomSampler or BatchedMultiFeatureRandomSampler: A sampler for this dataset

        Raises:
            NotImplementedError: If shuffle is False
            ValueError: If num_views has an invalid type or required parameters are missing
        """
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet

        if isinstance(self.num_views, int):
            num_of_aspect_ratios = len(self._resolutions)
            feature_pool_sizes = [num_of_aspect_ratios]
            scaling_feature_idx = 0  # Use aspect ratio as scaling feature
        elif isinstance(self.num_views, list):
            num_of_aspect_ratios = len(self._resolutions)
            num_of_num_views = len(self.num_views)
            feature_pool_sizes = [num_of_aspect_ratios, num_of_num_views]
            scaling_feature_idx = 1  # Use num_views as scaling feature
        else:
            raise ValueError(
                f"Bad type for {self.num_views=}, should be int or list of ints"
            )

        if use_dynamic_sampler:
            if max_num_of_images_per_gpu is None:
                raise ValueError(
                    "max_num_of_images_per_gpu must be provided when using dynamic sampler"
                )

            # Create feature-to-batch-size mapping
            if isinstance(self.num_views, list):
                # Map num_views_idx to batch size: max(1, max_num_of_images_per_gpu // (num_views_idx + dataset.num_views_min))
                feature_to_batch_size_map = {}
                for num_views_idx, num_views in enumerate(self.num_views):
                    batch_size_for_multi_view_sets = max(
                        1, max_num_of_images_per_gpu // num_views
                    )
                    feature_to_batch_size_map[num_views_idx] = (
                        batch_size_for_multi_view_sets
                    )
            else:
                # For fixed num_views, use a simple mapping
                feature_to_batch_size_map = {
                    0: max(1, max_num_of_images_per_gpu // self.num_views)
                }

            return DynamicBatchedMultiFeatureRandomSampler(
                self,
                pool_sizes=feature_pool_sizes,
                scaling_feature_idx=scaling_feature_idx,
                feature_to_batch_size_map=feature_to_batch_size_map,
                world_size=world_size,
                rank=rank,
                drop_last=drop_last,
            )
        else:
            if batch_size is None:
                raise ValueError(
                    "batch_size must be provided when not using dynamic sampler"
                )

            return BatchedMultiFeatureRandomSampler(
                self,
                batch_size,
                feature_pool_sizes,
                world_size=world_size,
                rank=rank,
                drop_last=drop_last,
            )


class MulDataset(EasyDataset):
    """Artificially augmenting the size of a dataset."""

    multiplicator: int

    def __init__(self, multiplicator, dataset):
        """
        Initialize a dataset that artificially augments the size of another dataset.

        Args:
            multiplicator (int): Factor by which to multiply the dataset size
            dataset (EasyDataset): The dataset to augment
        """
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset
        """
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        """
        Get a string representation of the dataset.

        Returns:
            str: String representation showing the multiplication factor and the original dataset
        """
        return f"{self.multiplicator}*{repr(self.dataset)}"

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: Index or tuple of indices to retrieve

        Returns:
            The item at the specified index from the original dataset
        """
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]
            new_idx = (idx // self.multiplicator, *other)
            return self.dataset[new_idx]
        else:
            return self.dataset[idx // self.multiplicator]

    @property
    def _resolutions(self):
        """
        Get the resolutions of the dataset.

        Returns:
            The resolutions from the original dataset
        """
        return self.dataset._resolutions

    @property
    def num_views(self):
        """
        Get the number of views used for the dataset.

        Returns:
            int or list: The number of views parameter from the original dataset
        """
        return self.dataset.num_views


class ResizedDataset(EasyDataset):
    """Artificially changing the size of a dataset."""

    new_size: int

    def __init__(self, new_size, dataset):
        """
        Initialize a dataset with an artificially changed size.

        Args:
            new_size (int): The new size of the dataset
            dataset (EasyDataset): The original dataset
        """
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The new size of the dataset
        """
        return self.new_size

    def __repr__(self):
        """
        Get a string representation of the dataset.

        Returns:
            str: String representation showing the new size and the original dataset
        """
        size_str = str(self.new_size)
        for i in range((len(size_str) - 1) // 3):
            sep = -4 * i - 3
            size_str = size_str[:sep] + "_" + size_str[sep:]
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        """
        Set the current epoch and generate a new random mapping of indices.

        This method must be called before using __getitem__.

        Args:
            epoch (int): The current epoch number
        """
        # This random shuffle only depends on the epoch
        rng = np.random.default_rng(seed=epoch + 777)

        # Shuffle all indices
        perm = rng.permutation(len(self.dataset))

        # Calculate how many repetitions we need
        num_repetitions = 1 + (len(self) - 1) // len(self.dataset)

        # Rotary extension until target size is met
        shuffled_idxs = np.concatenate([perm] * num_repetitions)
        self._idxs_mapping = shuffled_idxs[: self.new_size]

        # Generate the seed offset for each repetition
        # This is needed to ensure we see unique samples when we repeat a scene
        seed_offset_per_repetition = [
            np.full(len(self.dataset), i) for i in range(num_repetitions)
        ]
        seed_offset_idxs = np.concatenate(seed_offset_per_repetition)
        self._idxs_seed_offset = seed_offset_idxs[: self.new_size]

        assert len(self._idxs_mapping) == self.new_size
        assert len(self._idxs_seed_offset) == self.new_size

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: Index or tuple of indices to retrieve

        Returns:
            The item at the mapped index from the original dataset

        Raises:
            AssertionError: If set_epoch has not been called
        """
        assert hasattr(self, "_idxs_mapping"), (
            "You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()"
        )
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]
            self.dataset._set_seed_offset(self._idxs_seed_offset[idx])
            new_idx = (self._idxs_mapping[idx], *other)
            return self.dataset[new_idx]
        else:
            self.dataset._set_seed_offset(self._idxs_seed_offset[idx])
            return self.dataset[self._idxs_mapping[idx]]

    @property
    def _resolutions(self):
        """
        Get the resolutions of the dataset.

        Returns:
            The resolutions from the original dataset
        """
        return self.dataset._resolutions

    @property
    def num_views(self):
        """
        Get the number of views used for the dataset.

        Returns:
            int or list: The number of views parameter from the original dataset
        """
        return self.dataset.num_views


class CatDataset(EasyDataset):
    """Concatenation of several datasets"""

    def __init__(self, datasets):
        """
        Initialize a dataset that is a concatenation of several datasets.

        Args:
            datasets (list): List of EasyDataset instances to concatenate
        """
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        """
        Get the length of the concatenated dataset.

        Returns:
            int: Total number of samples across all datasets
        """
        return self._cum_sizes[-1]

    def __repr__(self):
        """
        Get a string representation of the concatenated dataset.

        Returns:
            str: String representation showing all concatenated datasets joined by '+'
        """
        # Remove uselessly long transform
        return " + ".join(
            repr(dataset).replace(
                ",transform=Compose( ToTensor() Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))",
                "",
            )
            for dataset in self.datasets
        )

    def set_epoch(self, epoch):
        """
        Set the current epoch for all constituent datasets.

        Args:
            epoch (int): The current epoch number
        """
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        """
        Get an item from the concatenated dataset.

        Args:
            idx: Index or tuple of indices to retrieve

        Returns:
            The item at the specified index from the appropriate constituent dataset

        Raises:
            IndexError: If the index is out of range
        """
        other = None
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, "right")
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if other is not None:
            new_idx = (new_idx, *other)
        return dataset[new_idx]

    @property
    def _resolutions(self):
        """
        Get the resolutions of the dataset.

        Returns:
            The resolutions from the first dataset (all datasets must have the same resolutions)

        Raises:
            AssertionError: If datasets have different resolutions
        """
        resolutions = self.datasets[0]._resolutions
        for dataset in self.datasets[1:]:
            assert tuple(dataset._resolutions) == tuple(resolutions), (
                "All datasets must have the same resolutions"
            )
        return resolutions

    @property
    def num_views(self):
        """
        Get the number of views used for the dataset.

        Returns:
            int or list: The number of views parameter from the first dataset

        Raises:
            AssertionError: If datasets have different num_views
        """
        num_views = self.datasets[0].num_views
        for dataset in self.datasets[1:]:
            assert dataset.num_views == num_views, (
                "All datasets must have the same num_views and variable_num_views parameters"
            )
        return num_views
