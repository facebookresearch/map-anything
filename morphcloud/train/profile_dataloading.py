# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Debug script to profile dataloading for MapAnything training.

This script measures and analyzes the performance of data loading operations
for MapAnything training workflows. It simulates the training process without
actual model training to isolate and profile the data loading components.
"""

import datetime
import json
import os
import time
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import morphcloud.utils.train_tools as train_tools
from morphcloud.datasets import get_test_data_loader, get_train_data_loader
from morphcloud.datasets.base.base_dataset import view_name

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


def profile_dataloading(args):
    """
    Main profiling function that simulates the training process to measure data loading performance.

    This function initializes the distributed environment, sets up datasets and data loaders,
    and runs through training epochs to profile the data loading operations. It measures
    the time taken for data loading without performing actual model training or optimization.

    In this simulation, an epoch represents a complete pass through a chunk of the dataset.

    Args:
        args: Configuration object containing all parameters including:
            - dataset: Dataset configuration (train_dataset, test_dataset, num_workers)
            - train_params: Training parameters (batch_size, epochs, seed, etc.)
            - distributed: Distributed training configuration
            - output_dir: Directory for saving logs and profiling results
    """
    # Initialize distributed training if required
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()
    world_size = train_tools.get_world_size()  # noqa

    # Init output directory and device
    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fix the seed
    seed = args.train_params.seed + train_tools.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.train_params.disable_cudnn_benchmark

    # Datasets and Dataloaders
    print("Building train dataset {:s}".format(args.dataset.train_dataset))
    data_loader_train = build_dataset(
        dataset=args.dataset.train_dataset,
        num_workers=args.dataset.num_workers,
        test=False,
        max_num_of_imgs_per_gpu=args.train_params.max_num_of_imgs_per_gpu,
    )
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    test_batch_size = 2 * (
        args.train_params.max_num_of_imgs_per_gpu // args.dataset.num_views
    )  # Since we don't have any backward overhead
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset=dataset,
            num_workers=args.dataset.num_workers,
            test=True,
            batch_size=test_batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    def write_log_stats(epoch, train_stats, test_stats):
        """
        Writes profiling statistics to log files and TensorBoard.

        This function collects metrics from the training and testing phases and writes them
        to log files and TensorBoard for visualization and analysis. It only executes on the
        main process in a distributed setting.

        Args:
            epoch: int, current epoch number
            train_stats: dict, containing training metrics and timing information
            test_stats: dict, containing testing metrics for each test dataset
        """
        if train_tools.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.train_params.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    args.train_params.start_epoch = 0
    for epoch in range(args.train_params.start_epoch, args.train_params.epochs + 1):
        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch >= args.train_params.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            data_loader_train,
            device,
            epoch,
            log_writer=log_writer,
            args=args,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def build_dataset(
    dataset, num_workers, test, batch_size=None, max_num_of_imgs_per_gpu=None
):
    """
    Builds data loaders for training or testing.

    Args:
        dataset: Dataset specification string.
        num_workers: Number of worker processes for data loading.
        test: Boolean flag indicating whether this is a test dataset.
        batch_size: Number of samples per batch. Defaults to None. Used only for testing.
        max_num_of_imgs_per_gpu: Maximum number of images per GPU. Defaults to None. Used only for training.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    if test:
        assert batch_size is not None, (
            "batch_size must be specified for testing dataloader"
        )
        loader = get_test_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=False,
            drop_last=False,
        )
    else:
        assert max_num_of_imgs_per_gpu is not None, (
            "max_num_of_imgs_per_gpu must be specified for training dataloader"
        )
        loader = get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=True,
            drop_last=True,
        )

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
):
    """
    Simulates training for one epoch to profile data loading performance.

    This function runs through a single epoch, simulating the data loading and device transfer
    operations that would occur during actual training. It measures and logs the time taken
    for these operations without performing actual model training.

    Args:
        data_loader: Sized, DataLoader providing the training data
        device: torch.device, device to transfer data to (CPU or GPU)
        epoch: int, current epoch number
        args: object, configuration object containing training parameters including:
            - train_params.print_freq: frequency of logging during the epoch
        log_writer: Optional[SummaryWriter], TensorBoard SummaryWriter for logging metrics

    Returns:
        dict: Dictionary containing profiling metrics averaged over the epoch
    """
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(epoch)

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # Simulate the device loading in loss_of_one_batch_multi_view
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
            for name in view.keys():
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        local_rank = train_tools.get_rank()
        n_views = len(batch)
        batch_shape = batch[0]["img"].shape
        first_sample_name = view_name(batch[0], batch_index=0)
        print(
            f"Rank: {local_rank}, Num views: {n_views}, Batch Shape: {batch_shape}, First Sample Name: {first_sample_name}",
            force=True,
        )

        del batch

        metric_logger.update(epoch=epoch_f)
        metric_logger.update(loss=0)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
