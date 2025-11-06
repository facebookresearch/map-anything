# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for training deep learning models, particularly focused on distributed training,
metric logging, and gradient handling.

This module provides tools for:
- Tracking and logging metrics during training
- Setting up distributed training environments
- Handling gradient scaling and normalization
- Managing learning rates and parameter groups
- Saving and loading model checkpoints

References: CroCo (https://github.com/naver/croco)
"""

import builtins
import datetime
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    """
    Logger for tracking and displaying training metrics.

    This class maintains a collection of metrics during training, provides
    methods to update them, and formats them for display. It also handles
    synchronization of metrics across processes in distributed training.
    """

    def __init__(self, delimiter="\t", print_per_view_stats=False):
        """
        Initialize the MetricLogger.

        Args:
            delimiter (str, optional): Delimiter for formatting output. Defaults to "\t".
            print_per_view_stats (bool, optional): Whether to print per-view statistics. Defaults to False.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print_per_view_stats = print_per_view_stats

    def update(self, **kwargs):
        """
        Update metrics with new values.

        Args:
            **kwargs: Key-value pairs where keys are metric names and values are metric values
                     Values can be tensors or numbers

        Raises:
            AssertionError: If a value is not a float or int after conversion from tensor
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        Get a meter by attribute name.

        This allows accessing meters as attributes of the logger.

        Args:
            attr (str): Name of the attribute to get

        Returns:
            SmoothedValue: The meter corresponding to the attribute name

        Raises:
            AttributeError: If the attribute doesn't exist as a meter or regular attribute
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        """
        Format all metrics as a string.

        Returns:
            str: Formatted string containing all metrics
        """
        loss_str = []
        for name, meter in self.meters.items():
            # Skip printing per-view stats if not enabled
            if not self.print_per_view_stats and "view" in name:
                continue
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Synchronize metrics across processes in distributed training.

        This method calls synchronize_between_processes on each meter to
        ensure consistent values across all processes.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Add a custom meter to the logger.

        Args:
            name (str): Name of the meter
            meter (SmoothedValue): The meter to add
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, max_iter=None):
        """
        Log metrics at regular intervals while iterating.

        This method wraps an iterable and logs metrics every print_freq iterations.
        It also tracks iteration time, data loading time, and memory usage.

        Args:
            iterable: Iterable to iterate over (typically a data loader)
            print_freq (int): How often to log metrics (in iterations)
            header (str, optional): Header string to print before metrics. Defaults to None.
            max_iter (int, optional): Maximum number of iterations. Defaults to None.

        Yields:
            object: Items from the original iterable
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ":" + str(len(str(len_iterable))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len_iterable,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len_iterable,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if max_iter and it >= max_iter:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len_iterable
            )
        )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process.

    It replaces the built-in print function with a custom version that only prints
    when the current process is the master process or when explicitly forced.

    Args:
        is_master (bool): Whether the current process is the master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed training is available and initialized, False otherwise
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Get the number of processes in the distributed training group.

    Returns:
        int: Number of processes in the distributed group, or 1 if not using distributed training
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process in the distributed training group.

    Returns:
        int: Rank of the current process, or 0 if not using distributed training
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Check if the current process is the main process (rank 0).

    Returns:
        bool: True if the current process is the main process, False otherwise
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save a PyTorch object only on the master process.

    This function is useful in distributed training to avoid multiple processes
    trying to save the same file simultaneously.

    Args:
        *args: Positional arguments to pass to torch.save()
        **kwargs: Keyword arguments to pass to torch.save()
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Initialize distributed training mode.

    This function sets up the distributed training environment based on environment
    variables and command-line arguments. It initializes the process group,
    sets the appropriate device, and configures printing for the distributed setup.

    Args:
        args: Arguments object containing distributed training configuration.
              Expected to have attributes like dist_url, and will be modified
              to include rank, world_size, gpu, and distributed flag.
    """
    nodist = args.nodist if hasattr(args, "nodist") else False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    """
    A gradient scaler that handles gradient scaling and norm computation for mixed precision training.

    This class wraps PyTorch's GradScaler to provide additional functionality for gradient norm tracking
    and clipping during mixed precision training.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        """Initialize the scaler.

        Args:
            enabled (bool): Whether to enable gradient scaling. Default: True
        """
        self._scaler = torch.GradScaler("cuda", enabled=enabled)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        """Scales loss and performs backward pass with optional gradient clipping.

        Args:
            loss: The loss to backpropagate
            optimizer: The optimizer being used
            clip_grad: Max norm for gradient clipping. None means no clipping
            parameters: Model parameters or list of parameters for gradient norm computation
            create_graph: Whether to create graph during backward pass
            update_grad: Whether to update gradients

        Returns:
            norm: The gradient norm if computed, else None. Returns list of norms if parameters is a list.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                if isinstance(parameters, (list, tuple)):
                    norm = [
                        torch.nn.utils.clip_grad_norm_(p, clip_grad) for p in parameters
                    ]
                else:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        """Returns the state dict of the underlying scaler.

        Returns:
            dict: The state dict of the gradient scaler
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the state dict into the underlying scaler.

        Args:
            state_dict: The state dict to load
        """
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Calculate the gradient norm of parameters.

    This function computes the norm of gradients for a set of parameters. It can handle
    both single parameter groups and multiple parameter groups (list/tuple of parameters).

    Args:
        parameters: A tensor or iterable of tensors or iterable of iterables of tensors
                   containing model parameters for which to compute gradient norms
        norm_type (float): Type of norm to use (e.g., 2.0 for L2 norm, inf for infinity norm)

    Returns:
        torch.Tensor: The computed gradient norm. If parameters is a list/tuple of parameter
                     groups, returns a list of norms, one for each group.
    """
    if isinstance(parameters, (list, tuple)):
        # If parameters is already a list/tuple, process each parameter group
        all_norms = []
        for params in parameters:
            if isinstance(params, torch.Tensor):
                params = [params]
            params = [p for p in params if p.grad is not None]
            if len(params) > 0:
                device = params[0].grad.device
                if norm_type == inf:
                    group_norm = max(
                        p.grad.detach().abs().max().to(device) for p in params
                    )
                else:
                    group_norm = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), norm_type).to(device)
                                for p in params
                            ]
                        ),
                        norm_type,
                    )
            else:
                group_norm = torch.tensor(0.0)
            all_norms.append(group_norm)
        return all_norms

    # Original logic for single parameter group
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(
    args, epoch, model_without_ddp, optimizer, loss_scaler, fname=None, best_so_far=None
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
        "optimizer": optimizer.state_dict(),
        "scaler": loss_scaler.state_dict(),
        "args": args,
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    save_on_master(to_save, checkpoint_path)


def load_model(train_args, model_without_ddp, optimizer, loss_scaler):
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


def all_reduce_mean(x):
    """
    Compute the mean of a value across all processes in distributed training.

    This function takes a value, reduces it across all processes using all_reduce,
    and returns the mean value.

    Args:
        x: The value to reduce (typically a scalar)

    Returns:
        float: The mean value across all processes
    """
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def _replace(text, src, tgt, rm=""):
    """
    Advanced string replacement utility.

    Given a text:
    - replace all elements in src by the corresponding element in tgt
    - remove all elements in rm

    Args:
        text (str): The input text to modify
        src (str): String of characters to replace
        tgt (str): String of replacement characters (must be same length as src or length 1)
        rm (str, optional): String of characters to remove. Defaults to "".

    Returns:
        str: The modified text after replacements and removals

    Raises:
        AssertionError: If src and tgt have different lengths (unless tgt has length 1)
    """
    if len(tgt) == 1:
        tgt = tgt * len(src)
    assert len(src) == len(tgt), f"'{src}' and '{tgt}' should have the same len"
    for s, t in zip(src, tgt):
        text = text.replace(s, t)
    for c in rm:
        text = text.replace(c, "")
    return text


def filename(obj):
    """
    Transform a Python object or command into a proper filename.

    This function converts a Python object or command string into a valid filename
    by replacing special characters and ensuring the filename is not too long.

    Special replacements:
    - \1 gets replaced by slash '/'
    - \2 gets replaced by comma ','

    Args:
        obj: The Python object or string to convert to a filename

    Returns:
        str: A valid filename derived from the input object

    Raises:
        AssertionError: If any part of the resulting path is longer than 256 characters
    """
    if not isinstance(obj, str):
        obj = repr(obj)
    obj = str(obj).replace("()", "")
    obj = _replace(obj, "_,(*/\1\2", "-__x%/,", rm=" )'\"")
    assert all(len(s) < 256 for s in obj.split(os.sep)), (
        "filename too long (>256 characters):\n" + obj
    )
    return obj


def compute_effective_lrs(train_args):
    """
    Compute the effective learning rates based on batch size scaling.

    This function calculates the effective learning rates for the main model and
    any submodules based on the effective batch size (accounting for gradient accumulation
    and distributed training) and the base learning rates.

    Args:
        train_args: Training arguments containing batch size, accumulation iterations,
                   learning rates, and submodule configurations

    Returns:
        train_args: Updated training arguments with computed effective learning rates
    """

    # Compute the effective batch size
    eff_batch_size = train_args.batch_size * train_args.accum_iter * get_world_size()
    print("Accumulate grad iterations: %d" % train_args.accum_iter)
    print("Effective batch size: %d" % eff_batch_size)
    # Compute the effective default learning rate
    if train_args.lr is None:  # only base_lr is specified
        train_args.lr = train_args.blr * math.sqrt(
            eff_batch_size / train_args.base_eff_batch_size
        )
    print(
        f"Base default lr for effective batch size {eff_batch_size}: %.2e"
        % (train_args.lr * math.sqrt(train_args.base_eff_batch_size / eff_batch_size))
    )
    print("Actual default lr: %.2e" % train_args.lr)
    for submodule, config in train_args.submodule_configs.items():
        if config.get("lr") is None:  # only base_lr is specified
            config["lr"] = config["blr"] * math.sqrt(
                eff_batch_size / train_args.base_eff_batch_size
            )
        print(
            f"Submodule {submodule} base lr for effective batch size {eff_batch_size}: %.2e"
            % (
                config["lr"]
                * math.sqrt(train_args.base_eff_batch_size / eff_batch_size)
            )
        )
        print(f"Submodule {submodule} actual lr: %.2e" % config["lr"])

    return train_args


def get_parameter_groups(
    model,
    lr,
    weight_decay,
    skip_list=[],
    submodule_configs=None,
    warn_not_in_submodule=False,
):
    """
    Get parameter groups for optimizer with customized learning rates and weight decay.

    This function organizes model parameters into groups for the optimizer, allowing
    different learning rates and weight decay values for different parts of the model.
    Parameters are grouped by:
    1. Whether they should have weight decay applied (bias terms and 1D tensors typically don't)
    2. Which submodule they belong to (if submodule_configs is provided)

    Args:
        model (torch.nn.Module): Model to get parameter groups for
        lr (float): Default learning rate for parameters not in submodule_configs
        weight_decay (float): Default weight decay for parameters not in submodule_configs
        skip_list (list): List of parameter names to skip weight decay for
        submodule_configs (dict, optional): Dictionary mapping submodule prefixes to configs
                                           with 'lr' and 'weight_decay' keys
        warn_not_in_submodule (bool, optional): Whether to warn if a parameter does not
                                               belong to any submodule. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - parameter_group_vars (list): List of parameter groups for optimizer
            - parameter_group_name_to_idx_map (dict): Mapping from submodule name to parameter group indices
            - parameter_group_idx_to_name_map (dict): Mapping from parameter group index to submodule name
    """

    if submodule_configs is None:
        submodule_configs = {}

    parameter_group_names = {}
    parameter_group_vars = {}
    parameter_group_name_to_idx_map = {}
    parameter_group_idx_to_name_map = {}
    mapping_index = 0

    for name, param in model.named_parameters():
        # Skip frozen parameters
        if not param.requires_grad:
            continue

        # Determine the submodule this parameter belongs to
        submodule_name = None
        for submodule, config in submodule_configs.items():
            if name.startswith(submodule):
                submodule_name = submodule
                break

        if submodule_name:
            config = submodule_configs[submodule_name]
            this_weight_decay = config.get("weight_decay", weight_decay)
            this_lr = config.get("lr", lr)
            # Freeze the parameters if lr is 0
            if this_lr == 0:
                param.requires_grad = False
                continue
        else:
            this_weight_decay = weight_decay
            this_lr = lr
            if warn_not_in_submodule and submodule_configs is not None:
                print(
                    f"Warning: Parameter {name} does not belong to any submodule in {submodule_configs.keys()}."
                )

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = f"{submodule_name}_no_decay" if submodule_name else "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = f"{submodule_name}_decay" if submodule_name else "decay"

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "lr": this_lr,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "lr": this_lr,
                "params": [],
            }
            submodule_name_mapping = submodule_name if submodule_name else "default"
            if submodule_name_mapping not in parameter_group_name_to_idx_map:
                parameter_group_name_to_idx_map[submodule_name_mapping] = [
                    mapping_index
                ]
            else:
                parameter_group_name_to_idx_map[submodule_name_mapping].append(
                    mapping_index
                )
            parameter_group_idx_to_name_map[mapping_index] = submodule_name_mapping
            mapping_index += 1

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # Print the parameter groups
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))

    return (
        list(parameter_group_vars.values()),
        parameter_group_name_to_idx_map,
        parameter_group_idx_to_name_map,
    )


def adjust_learning_rate(
    optimizer,
    epoch,
    train_args,
    parameter_group_idx_to_name_map,
    submodule_configs=None,
):
    """
    Adjust the learning rate based on the schedule type and current epoch.

    This function updates the learning rates for all parameter groups in the optimizer
    according to the specified learning rate schedule. Different submodules can have
    different learning rate schedules.

    Currently supported schedule types:
    - linear_warmup_half_cycle_cosine_decay: Linear warmup followed by cosine decay

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update
        epoch (int): Current training epoch
        train_args: Training arguments containing schedule type, warmup epochs, etc.
        parameter_group_idx_to_name_map (dict): Mapping from parameter group index to submodule name
        submodule_configs (dict, optional): Dictionary of submodule-specific configurations
                                          for learning rate schedules

    Raises:
        ValueError: If an unsupported schedule type is specified
    """

    if submodule_configs is None:
        submodule_configs = {}

    for group_num, param_group in enumerate(optimizer.param_groups):
        submodule_name = parameter_group_idx_to_name_map.get(group_num)

        if submodule_name in submodule_configs:
            config = submodule_configs[submodule_name]
            lr = config.get("lr", train_args.lr)
            warmup_epochs = config.get("warmup_epochs", train_args.warmup_epochs)
            min_lr = config.get("min_lr", train_args.min_lr)
            schedule_type = config.get("schedule_type", train_args.schedule_type)
        else:
            lr = train_args.lr
            warmup_epochs = train_args.warmup_epochs
            min_lr = train_args.min_lr
            schedule_type = train_args.schedule_type

        if schedule_type == "linear_warmup_half_cycle_cosine_decay":
            if epoch < warmup_epochs:
                lr = lr * epoch / warmup_epochs
            else:
                lr = min_lr + (lr - min_lr) * 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * (epoch - warmup_epochs)
                        / (train_args.epochs - warmup_epochs)
                    )
                )
        else:
            raise ValueError(f"Schedule type {schedule_type} not implemented")

        param_group["lr"] = lr


def debug_after_backward(
    model,
    check_missing_gradients=True,
    check_gradient_mismatch=False,
    target_size=(256, 256, 1, 1),
    target_stride=(256, 1, 256, 256),
):
    """
    Debugging function to check for gradient issues after backward pass.

    This function performs two types of gradient debugging:
    1. Gradient mismatch: Checks for parameters with specific gradient shapes and strides
       that might indicate incorrect gradient computation.
    2. Missing gradients: Identifies parameters that require gradients but didn't receive any.

    Args:
        model (torch.nn.Module): The model to check gradients for
        check_missing_gradients (bool, optional): Whether to check for missing gradients. Defaults to True.
        check_gradient_mismatch (bool, optional): Whether to check for gradient mismatches. Defaults to False.
        target_size (tuple, optional): Target tensor size to check for gradient mismatch. Defaults to (256, 256, 1, 1).
        target_stride (tuple, optional): Target tensor stride to check for gradient mismatch. Defaults to (256, 1, 256, 256).
    """
    # Debug for missing gradients
    if check_missing_gradients:
        missing_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                missing_grad_params.append(name)

        if missing_grad_params:
            print("Parameters requiring gradients but missing gradients:")
            for name in missing_grad_params:
                print(f"  - {name}")
        else:
            print("All parameters requiring gradients received gradients!")

    # Debug for gradient mismatch
    if check_gradient_mismatch:
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if grad.size() == target_size and grad.stride() == target_stride:
                print(f"Found parameter with incorrect gradient: '{name}'")
                print(f"Gradient shape: {grad.size()}, strides: {grad.stride()}")
