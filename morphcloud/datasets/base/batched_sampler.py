# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utilities for random sampling under a single or multiple constraints

References: DUSt3R
"""

import numpy as np
import torch


def round_by(total, multiple, up=False):
    """
    Round a number to the nearest multiple of another number.

    Args:
        total (int): The number to round
        multiple (int): The multiple to round to
        up (bool, optional): Whether to round up. Defaults to False.

    Returns:
        int: The rounded number
    """
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple


class BatchedRandomSampler:
    """
    Random sampling under a constraint: each sample in the batch has the same feature,
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(
        self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True
    ):
        """
        Args:
            dataset: Dataset to sample from
            batch_size: Number of samples per batch
            pool_size: Integer representing the size of feature pool
            world_size: Number of distributed processes
            rank: Rank of the current process
            drop_last: Whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.pool_size = pool_size

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size * world_size) if drop_last else N
        assert world_size == 1 or drop_last, (
            "must drop the last batch in distributed mode"
        )

        # Distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        """
        Get the length of the sampler.

        Returns:
            int: The number of samples in the sampler for the current process
        """
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler.

        This should be called before each epoch to ensure proper shuffling of the data.

        Args:
            epoch (int): The current epoch number
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Iterator over the indices.

        This method generates random indices for each batch, ensuring that all samples
        within a batch have the same feature index for the given feature pool.

        Yields:
            tuple: A tuple containing (sample_idx, feat_idx)
        """
        # Prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # Random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # Random feat_idxs (same across each batch)
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches)
        feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        feat_idxs = feat_idxs.ravel()[: self.total_size]

        # Put them together
        idxs = np.c_[sample_idxs, feat_idxs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # Make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        idxs = idxs[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)


class BatchedMultiFeatureRandomSampler:
    """
    Random sampling under multiple constraints: each sample in the batch has the same features,
    which are chosen randomly from known pools of 'features' for each batch.

    For instance, the 'features' could be the image aspect-ratio and scene type.

    The index returned is a tuple (sample_idx, feat_idx_1, feat_idx_2, ...).
    This sampler ensures that each series of `batch_size` indices has the same feature indices.
    """

    def __init__(
        self, dataset, batch_size, pool_sizes, world_size=1, rank=0, drop_last=True
    ):
        """
        Args:
            dataset: Dataset to sample from
            batch_size: Number of samples per batch
            pool_sizes: List of integers representing the size of each feature pool
            world_size: Number of distributed processes
            rank: Rank of the current process
            drop_last: Whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size * world_size) if drop_last else N
        assert world_size == 1 or drop_last, (
            "must drop the last batch in distributed mode"
        )

        # Distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        """
        Get the length of the sampler.

        Returns:
            int: The number of samples in the sampler for the current process
        """
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler.

        This should be called before each epoch to ensure proper shuffling of the data.

        Args:
            epoch (int): The current epoch number
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Iterator over the indices.

        This method generates random indices for each batch, ensuring that all samples
        within a batch have the same feature indices for multiple features.

        Yields:
            tuple: A tuple containing (sample_idx, feat_idx_1, feat_idx_2, ...)
        """
        # Prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # Random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # Random feat_idxs (same across each batch)
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size

        # Generate feature indices for each feature pool
        all_feat_idxs = []
        for pool_size in self.pool_sizes:
            feat_idxs = rng.integers(pool_size, size=n_batches)
            feat_idxs = np.broadcast_to(
                feat_idxs[:, None], (n_batches, self.batch_size)
            )
            feat_idxs = feat_idxs.ravel()[: self.total_size]
            all_feat_idxs.append(feat_idxs)

        # Put them together
        idxs = np.column_stack(
            [sample_idxs] + all_feat_idxs
        )  # shape = (total_size, 1 + len(pool_sizes))

        # Distributed sampler: we select a subset of batches
        # Make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        idxs = idxs[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)


class DynamicBatchedMultiFeatureRandomSampler:
    """
    Random sampling under multiple constraints with dynamic batch size:
    each sample in the batch has the same features, which are chosen randomly
    from known pools of 'features' for each batch.

    The batch size is dynamically determined based on a specified feature index,
    using a direct mapping from feature values to batch sizes.

    For instance, if one of the features is the number of images in a multi-view set,
    you can specify different batch sizes for different numbers of images to optimize
    GPU memory usage. This is achieved by using the feature_to_batch_size_map parameter
    to directly specify what batch size to use for each feature value.

    The returned index is a list of tuples [(sample_idx, feat_idx_1, feat_idx_2, ...), ...].
    """

    def __init__(
        self,
        dataset,
        pool_sizes,
        scaling_feature_idx=0,
        feature_to_batch_size_map=None,
        world_size=1,
        rank=0,
        drop_last=True,
    ):
        """
        Args:
            dataset: Dataset to sample from
            pool_sizes: List of integers representing the size of each feature pool
            scaling_feature_idx: Index of the feature to use for determining batch size (0-based index into pool_sizes)
            feature_to_batch_size_map: Optional function or dict that maps feature values directly to batch sizes.
                                 For example, if the feature represents number of views, this maps number of views
                                 to appropriate batch size that can fit in GPU memory.
                                 If None, uses a default batch size of 1 for all feature values.
            world_size: Number of distributed processes
            rank: Rank of the current process
            drop_last: Whether to drop the last incomplete batch
        """
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]
        self.scaling_feature_idx = scaling_feature_idx

        # Ensure scaling_feature_idx is valid
        if scaling_feature_idx < 0 or scaling_feature_idx >= len(self.pool_sizes):
            raise ValueError(
                f"scaling_feature_idx must be between 0 and {len(self.pool_sizes) - 1}"
            )

        # Set up mapping from feature values to batch sizes
        self.feature_to_batch_size_map = feature_to_batch_size_map
        if self.feature_to_batch_size_map is None:
            # Default: batch size of 1 for all feature values
            self.feature_to_batch_size_map = {
                i: 1 for i in range(self.pool_sizes[scaling_feature_idx])
            }

        self.len_dataset = N = len(dataset)

        # We don't know the exact batch size yet, so we use a large number for total_size
        # This will be adjusted during iteration
        self.total_size = N

        # Distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None
        self.drop_last = drop_last

    def __len__(self):
        """
        Get the approximate length of the sampler.

        Since batch size varies, this is an estimate based on the largest batch size
        in the mapping, which provides a lower bound on the number of batches.

        Returns:
            int: The estimated minimum number of samples in the sampler for the current process
        """
        # Find the largest batch size in the mapping
        if callable(self.feature_to_batch_size_map):
            # If it's a function, sample some values to find the maximum
            batch_sizes = [
                self.feature_to_batch_size_map(i)
                for i in range(self.pool_sizes[self.scaling_feature_idx])
            ]
            max_batch_size = max(batch_sizes)
        else:
            # If it's a dict or similar, find the maximum directly
            max_batch_size = max(self.feature_to_batch_size_map.values())

        # Ensure minimum batch size of 1
        max_batch_size = max(1, max_batch_size)

        # Estimate total batches using the largest batch size
        # This gives a lower bound on the number of batches
        total_batches = self.total_size // max_batch_size
        if not self.drop_last and self.total_size % max_batch_size > 0:
            total_batches += 1

        # Distribute among processes
        return total_batches // self.world_size

    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler.

        This should be called before each epoch to ensure proper shuffling of the data.

        Args:
            epoch (int): The current epoch number
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Iterator over the indices with dynamic batch sizes.

        This method generates random indices for each batch, ensuring that all samples
        within a batch have the same feature indices for multiple features.
        The batch size is determined directly from the feature_to_batch_size_map.

        The iterator enforces the length returned by __len__() by stopping after
        exactly that many batches have been yielded for this process.

        Yields:
            list of tuples: A batch of tuples, each containing (sample_idx, feat_idx_1, feat_idx_2, ...)
        """
        # Prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # Random indices for the entire dataset
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # Get the target number of batches for this process (enforce strict length)
        target_batches_for_process = len(self)
        batches_yielded_for_process = 0

        # Process indices in batches with dynamic sizing
        idx = 0
        batch_idx = 0  # Track batch index for even distribution
        while idx < len(sample_idxs) and (
            batches_yielded_for_process < target_batches_for_process
        ):
            # Randomly select feature indices for this batch
            feat_idxs = [rng.integers(pool_size) for pool_size in self.pool_sizes]

            # Get the scaling feature value
            scaling_feat = feat_idxs[self.scaling_feature_idx]

            # Get the batch size directly from the mapping
            if callable(self.feature_to_batch_size_map):
                batch_size = self.feature_to_batch_size_map(scaling_feat)
            else:
                batch_size = self.feature_to_batch_size_map.get(scaling_feat, 1)

            # Ensure minimum batch size of 1
            batch_size = max(1, batch_size)

            # Ensure we don't go beyond available samples
            remaining = len(sample_idxs) - idx
            if remaining < batch_size:
                if self.drop_last:
                    break
                batch_size = remaining

            # Create batch with consistent feature indices
            batch = []
            for i in range(batch_size):
                if idx + i < len(sample_idxs):
                    sample_idx = sample_idxs[idx + i]
                    batch.append(tuple([sample_idx] + feat_idxs))

            # Distribute batches among processes in round-robin fashion
            if len(batch) > 0 and (batch_idx % self.world_size == self.rank):
                yield batch
                batches_yielded_for_process += 1

            batch_idx += 1  # Increment batch index
            idx += batch_size
