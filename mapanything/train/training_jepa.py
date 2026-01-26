# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Training Code for MapAnything.

References:
DUSt3R: https://github.com/naver/dust3r
"""

import datetime
import json
import math
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import mapanything.utils.train_tools as train_tools
from mapanything.datasets import get_test_data_loader, get_train_data_loader
from mapanything.models import init_model
from mapanything.train.losses import *  # noqa
from mapanything.train.losses_jepa import LeJEPALoss, OverfittingLoss, L2Loss
from mapanything.utils.inference import loss_of_one_batch_multi_view
from mapanything.utils.train_tools import NativeScalerWithGradNormCount as NativeScaler

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


def train_jepa(args):
    """
    JEPA training version of train().
    This function adds target model and EMA updates for JEPA training.

    Args:
        args: Configuration object containing all training parameters including
              dataset configs, model configs, training parameters, and loss functions.
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
    # test_batch_size = args.train_params.max_num_of_imgs_per_gpu // args.dataset.num_views
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

    # Load Model
    if global_rank == 0:
        model = init_model(
            args.model.model_str,
            args.model.model_config,
            torch_hub_force_reload=args.model.torch_hub_force_reload,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Make sure the model is initialized before proceeding
    if global_rank != 0:
        model = init_model(
            args.model.model_str, args.model.model_config, torch_hub_force_reload=False
        )

    model.to(device)  # Move model to device
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Criterion
    print(f">> Creating train criterion = {args.loss.train_criterion}")
    train_criterion = eval(args.loss.train_criterion).to(device)
    print(
        f">> Creating test criterion = {args.loss.test_criterion or args.loss.train_criterion}"
    )
    test_criterion = eval(args.loss.test_criterion or args.loss.train_criterion).to(
        device
    )

    # Load pretrained model if provided
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(
            args.model.pretrained, map_location=device, weights_only=False
        )
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # in case it occupies memory

    target_model = deepcopy(model)
    target_model_without_ddp = target_model
    
    # Init model for DDP training
    if args.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=True,
            static_graph=False,
        )
        model_without_ddp = model.module

        target_model = torch.nn.parallel.DistributedDataParallel(
            target_model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=True,
            static_graph=False,
        )
        target_model_without_ddp = target_model.module

    # Optimizer and loss scaler for gradient accumulation
    # Following timm: set wd as 0 for bias and norm layers
    param_groups, param_groups_name_to_idx_map, param_groups_idx_to_name_map = (
        train_tools.get_parameter_groups(
            model_without_ddp,
            args.train_params.lr,
            args.train_params.weight_decay,
            submodule_configs=args.train_params.submodule_configs,
            warn_not_in_submodule=args.train_params.warn_not_in_submodule,
        )
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.train_params.lr, betas=(0.9, 0.95)
    )
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        """
        Writes training and testing statistics to log files and TensorBoard.

        Args:
            epoch: Current epoch number.
            train_stats: Dictionary containing training metrics.
            test_stats: Dictionary containing testing metrics for each test dataset.
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

    def save_model(epoch, fname, best_so_far):
        """
        Saves model checkpoint to disk.

        Args:
            epoch: Current epoch number.
            fname: Filename or identifier for the checkpoint.
            best_so_far: Best validation metric achieved so far.
        """
        save_model_jepa(
            args=args,
            model_without_ddp=model_without_ddp,
            target_model_without_ddp=target_model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    # Resume from a checkpoint if needed
    last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
    if args.train_params.resume and os.path.isfile(last_ckpt_fname):
        args.train_params.resume_ckpt = last_ckpt_fname
    else:
        args.train_params.resume_ckpt = None
    best_so_far = load_model_jepa(
        train_args=args.train_params,
        model_without_ddp=model_without_ddp,
        target_model_without_ddp=target_model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if best_so_far is None:
        best_so_far = float("inf")

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # -- momentum schedule
    ema = args.train_params.target_ema
    ipe = len(data_loader_train) // args.train_params.accum_iter  # iterations per epoch
    num_epochs = args.train_params.epochs
    ipe_scale = args.train_params.ema_ipe_scale
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0]) / (ipe * num_epochs * ipe_scale)
                          for i in range(int(ipe * num_epochs * ipe_scale) + 1))

    for _ in range(args.train_params.start_epoch * ipe):
        next(momentum_scheduler)

    print(f"Start training for {args.train_params.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.train_params.start_epoch, args.train_params.epochs + 1):
        # Save immediately the last checkpoint
        if epoch > args.train_params.start_epoch:
            if (
                args.train_params.save_freq
                and epoch % args.train_params.save_freq == 0
                or epoch == args.train_params.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        test_stats = {}
        if (
            args.train_params.eval_freq > 0
            and epoch % args.train_params.eval_freq == 0
            and epoch > 0
        ):
            for test_name, testset in data_loader_test.items():
                print(f"Testing on {test_name} ...")
                stats = test_one_epoch(
                    model,
                    target_model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

            # Calculate average test loss median
            avg_test_loss_med = np.mean(
                [stats["loss_med"] for stats in test_stats.values()]
            )
            test_stats["Average Test Loss Median"] = avg_test_loss_med
            # Save best
            if avg_test_loss_med < best_so_far:
                best_so_far = avg_test_loss_med
                new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.train_params.start_epoch:
            if args.train_params.keep_freq and epoch % args.train_params.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.train_params.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            target_model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            param_groups_name_to_idx_map=param_groups_name_to_idx_map,
            param_groups_idx_to_name_map=param_groups_idx_to_name_map,
            model_without_ddp=model_without_ddp,
            target_model_without_ddp=target_model_without_ddp,
        )

        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(model_without_ddp.parameters(), target_model_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    save_final_model(
        args, args.train_params.epochs, model_without_ddp, target_model_without_ddp, best_so_far=best_so_far
    )


def save_model_jepa(
    args, epoch, model_without_ddp, target_model_without_ddp, optimizer, loss_scaler, fname=None, best_so_far=None
):
    """
    Save model checkpoint to disk.

    This function saves the model state, optimizer state, loss scaler state,
    training arguments, current epoch, and optionally the best metric value so far.
    The checkpoint is only saved on the master process in distributed training.

    Args:
        args: Arguments containing output directory information
        epoch (int): Current training epoch
        model_without_ddp (torch.nn.Module): Model without DistributedDataParallel wrapper
        optimizer (torch.optim.Optimizer): Optimizer instance
        loss_scaler: Gradient scaler for mixed precision training
        fname (str, optional): Custom filename suffix. If None, uses the epoch number. Defaults to None.
        best_so_far (float, optional): Best metric value achieved so far. Defaults to None.
    """
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / ("checkpoint-%s.pth" % fname)
    to_save = {
        "model": model_without_ddp.state_dict(),
        "target_model": target_model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": loss_scaler.state_dict(),
        "args": args,
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    train_tools.save_on_master(to_save, checkpoint_path)


def load_model_jepa(train_args, model_without_ddp, target_model_without_ddp, optimizer, loss_scaler):
    """
    Load model checkpoint from disk or URL.

    This function loads a saved checkpoint, restoring the model state, optimizer state,
    loss scaler state, and training epoch. It can load from a local file or a URL.

    Args:
        train_args: Training arguments containing resume information
        model_without_ddp (torch.nn.Module): Model without DistributedDataParallel wrapper
        optimizer (torch.optim.Optimizer): Optimizer instance
        loss_scaler: Gradient scaler for mixed precision training

    Returns:
        float or None: Best metric value from the checkpoint if available, otherwise None
    """
    train_args.start_epoch = 0
    best_so_far = None
    if train_args.resume and train_args.resume_ckpt is not None:
        if train_args.resume_ckpt.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                train_args.resume_ckpt, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(
                train_args.resume_ckpt, map_location="cpu", weights_only=False
            )
        print("Resume checkpoint %s" % train_args.resume_ckpt)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        target_model_without_ddp.load_state_dict(checkpoint["target_model"], strict=False)
        train_args.start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        if "best_so_far" in checkpoint:
            best_so_far = checkpoint["best_so_far"]
            print(" & best_so_far={:g}".format(best_so_far))
        else:
            print("")
        print(
            "With optim & sched! start_epoch={:d}".format(train_args.start_epoch),
            end="",
        )
    return best_so_far


def save_final_model(args, epoch, model_without_ddp, target_model_without_ddp, best_so_far=None):
    """
    Saves the final model checkpoint after training completion.

    Args:
        args: Configuration object containing output directory information.
        epoch: Current epoch number.
        model_without_ddp: Model state dictionary or model instance without DistributedDataParallel wrapper.
        best_so_far: Optional; Best validation metric achieved during training.
    """
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": model_without_ddp
        if isinstance(model_without_ddp, dict)
        else model_without_ddp.cpu().state_dict(),
        "target_model": target_model_without_ddp
        if isinstance(target_model_without_ddp, dict)
        else target_model_without_ddp.cpu().state_dict(),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    train_tools.save_on_master(to_save, checkpoint_path)


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


def loss_of_one_batch_jepa(
    batch,
    model,
    target_model,
    criterion,
    device,
    use_amp=False,
    amp_dtype="bf16",
    ret=None,
    ignore_keys=None,
):
    """
    Calculate loss for a batch with multiple views.
    JEPA version that handles both model and target_model.

    Args:
        batch (list): List of view dictionaries containing input data.
        model (torch.nn.Module): Model to run inference with.
        target_model (torch.nn.Module): Target model to run inference with.
        criterion (callable, optional): Loss function to compute the loss.
        device (torch.device): Device to run the computation on.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
        amp_dtype (str, optional): Floating point type to use for automatic mixed precision. Options: ["fp32", "fp16", "bf16"]. Defaults to "bf16".
        ret (str, optional): If provided, return only the specified key from the result dictionary.
        ignore_keys (set, optional): Set of keys to ignore when moving tensors to device.
                                   Defaults to {"dataset", "label", "instance",
                                   "idx", "true_shape", "rng", "data_norm_type"}.

    Returns:
        dict or Any: If ret is None, returns a dictionary containing views, predictions, and loss.
                     Otherwise, returns the value associated with the ret key.
    """
    # Move necessary tensors to device
    if ignore_keys is None:
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

    # Determine the mixed precision floating point type
    if use_amp:
        if amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Run model and compute loss
    with torch.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):
        contexts = model(batch)
        targets = target_model(batch)

        with torch.autocast("cuda", enabled=False):
            loss = criterion(batch, contexts, targets) if criterion is not None else None

    result = {f"view{i + 1}": view for i, view in enumerate(batch)}
    # result.update({f"preds{i + 1}": pred for i, pred in enumerate(targets)})
    result["loss"] = loss

    return result[ret] if ret else result


def train_one_epoch(
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    param_groups_name_to_idx_map=None,
    param_groups_idx_to_name_map=None,
    model_without_ddp=None,
    target_model_without_ddp=None,
):
    """
    Trains the model for one epoch.
    JEPA version that handles both model and target_model.

    Args:
        model: The neural network model to train.
        target_model: The target neural network model for JEPA training.
        criterion: Loss function to optimize.
        data_loader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (CPU or GPU).
        epoch: Current epoch number.
        loss_scaler: Scaler for gradient accumulation and mixed precision training.
        args: Configuration object containing training parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        param_groups_name_to_idx_map: Mapping from parameter group names to indices.
        param_groups_idx_to_name_map: Mapping from parameter group indices to names.
        model_without_ddp: Model without DistributedDataParallel wrapper for debugging.
        target_model_without_ddp: Target model without DistributedDataParallel wrapper for debugging.

    Returns:
        dict: Dictionary containing training metrics averaged over the epoch.
    """
    model.train(True)

    # Stop gradient flow for non-adapter parameters
    model_without_ddp.stop_non_adapter_gradient_flow()

    # Ensure target model parameters are not updated
    for param in target_model_without_ddp.parameters():
        param.requires_grad = False

    model_without_ddp.contextualize()
    target_model_without_ddp.decontextualize()

    metric_logger = train_tools.MetricLogger(delimiter="  ")
    for submodule_name in param_groups_name_to_idx_map:
        lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
        metric_logger.add_meter(
            lr_name, train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.train_params.accum_iter

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

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        epoch_f = epoch + data_iter_step / len(data_loader)

        # We use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            train_tools.adjust_learning_rate(
                optimizer,
                epoch_f,
                args.train_params,
                param_groups_idx_to_name_map,
                args.train_params.submodule_configs,
            )

        loss_tuple = loss_of_one_batch_jepa(
            batch,
            model,
            target_model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
        )
        loss, loss_details = loss_tuple  # criterion returns two values
        if n_views > 2:
            loss = loss * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        loss_value = float(loss)

        if not math.isfinite(loss_value) or (loss_value > 1000):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            print(f"Loss Details: {loss_details}", force=True)
            print(f"Epoch: {epoch}, Data Iteration: {data_iter_step}", force=True)
            # Save the current batch to the output folder for further inspection
            for view_idx, view in enumerate(batch):
                view_cpu = {}
                for k, v in view.items():
                    view_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
                with open(
                    os.path.join(args.output_dir, f"batch_view_{view_idx}.pkl"), "wb"
                ) as f:
                    pickle.dump(view_cpu, f)
            # Save the model to the output folder for further inspection
            checkpoint_debug_path = os.path.join(
                args.output_dir, "checkpoint-debug.pth"
            )
            to_save_debug = {
                "args": args,
                "model": (
                    model_without_ddp
                    if isinstance(model_without_ddp, dict)
                    else model_without_ddp.cpu().state_dict()
                ),
                "target_model": (
                    target_model_without_ddp
                    if isinstance(target_model_without_ddp, dict)
                    else target_model_without_ddp.cpu().state_dict()
                ),
                "epoch": epoch,
                "data_iter_step": data_iter_step,
            }
            torch.save(to_save_debug, checkpoint_debug_path)
            print(f"Saved debugging material to {args.output_dir}", force=True)
            sys.exit(1)

        # Scale the loss by the number of gradient accumulation iterations
        loss /= accum_iter

        # Compute the scaled gradients (also clip the gradients to max norm of 1)
        gradient_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=1.0,
        )

        # Zero out the gradients to prepare for the next iteration of gradient descent
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        metric_logger.update(epoch=epoch_f)
        for submodule_name in param_groups_name_to_idx_map:
            lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
            log_lr = optimizer.param_groups[
                param_groups_name_to_idx_map[submodule_name][0]
            ]["lr"]
            metric_logger.meters[lr_name].update(log_lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.train_params.print_freq)
        ) == 0:
            loss_value_reduce = train_tools.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            if gradient_norm is not None:
                log_writer.add_scalar("train_grad_norm", gradient_norm, epoch_1000x)
            for submodule_name in param_groups_name_to_idx_map:
                lr_name = (
                    f"train_lr_{submodule_name}"
                    if submodule_name != "default"
                    else "train_lr"
                )
                log_lr = optimizer.param_groups[
                    param_groups_name_to_idx_map[submodule_name][0]
                ]["lr"]
                log_writer.add_scalar(lr_name, log_lr, epoch_1000x)
            log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar("train_" + name, val, epoch_1000x)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):
    """
    Evaluates the model on a test dataset for one epoch.
    JEPA version that handles both model and target_model.

    Args:
        model: The neural network model to evaluate.
        target_model: The target neural network model for JEPA training.
        criterion: Loss function for evaluation.
        data_loader: DataLoader providing the test data.
        device: Device to run evaluation on (CPU or GPU).
        epoch: Current epoch number.
        args: Configuration object containing evaluation parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        prefix: String prefix for logging metrics.

    Returns:
        dict: Dictionary containing evaluation metrics (average and median values).
    """
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(
        lambda: train_tools.SmoothedValue(window_size=9**9)
    )
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.train_params.freeze_val_samples_across_all_epochs:
        dataloader_epoch = 0
    else:
        dataloader_epoch = epoch
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(dataloader_epoch)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        loss_tuple = loss_of_one_batch_jepa(
            batch,
            model,
            target_model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
        )
        loss_value, loss_details = loss_tuple  # criterion returns two values
        if n_views > 2:
            loss_value = loss_value * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        metric_logger.update(loss=float(loss_value), **loss_details)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    return results
