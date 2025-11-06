# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for multiprocessing
"""

import os
from multiprocessing.dummy import Pool as ThreadPool

import torch
from torch.multiprocessing import Pool as TorchPool, set_start_method
from tqdm import tqdm


def cpu_count():
    """
    Returns the number of available CPUs for the python process
    """
    return len(os.sched_getaffinity(0))


def parallel_threads(
    function,
    args,
    workers=0,
    star_args=False,
    kw_args=False,
    front_num=1,
    Pool=ThreadPool,
    ordered_res=True,
    **tqdm_kw,
):
    """tqdm but with parallel execution.

    Will essentially return
      res = [ function(arg) # default
              function(*arg) # if star_args is True
              function(**arg) # if kw_args is True
              for arg in args]

    Note:
        the <front_num> first elements of args will not be parallelized.
        This can be useful for debugging.
    """
    # Determine the number of workers
    while workers <= 0:
        workers += cpu_count()

    # Convert args to an iterable
    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    # Sequential execution for the first few elements (useful for debugging)
    front = []
    while len(front) < front_num:
        try:
            a = next(args)
        except StopIteration:
            return front  # end of the iterable
        front.append(
            function(*a) if star_args else function(**a) if kw_args else function(a)
        )

    # Parallel execution using multiprocessing.dummy
    out = []
    with Pool(workers) as pool:
        if star_args:
            map_func = pool.imap if ordered_res else pool.imap_unordered
            futures = map_func(starcall, [(function, a) for a in args])
        elif kw_args:
            map_func = pool.imap if ordered_res else pool.imap_unordered
            futures = map_func(starstarcall, [(function, a) for a in args])
        else:
            map_func = pool.imap if ordered_res else pool.imap_unordered
            futures = map_func(function, args)
        # Track progress with tqdm
        for f in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(f)
    return front + out


def cuda_parallel_threads(
    function,
    args,
    workers=0,
    star_args=False,
    kw_args=False,
    front_num=1,
    Pool=TorchPool,
    ordered_res=True,
    **tqdm_kw,
):
    """
    Parallel execution of a function using torch.multiprocessing with CUDA support.
    This is the CUDA variant of the parallel_threads function.
    """
    # Set the start method for multiprocessing
    set_start_method("spawn", force=True)

    # Determine the number of workers
    while workers <= 0:
        workers += torch.multiprocessing.cpu_count()

    # Convert args to an iterable
    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    # Sequential execution for the first few elements (useful for debugging)
    front = []
    while len(front) < front_num:
        try:
            a = next(args)
        except StopIteration:
            return front  # End of the iterable
        front.append(
            function(*a) if star_args else function(**a) if kw_args else function(a)
        )

    # Parallel execution using torch.multiprocessing
    out = []
    with Pool(workers) as pool:
        if star_args:
            map_func = pool.imap if ordered_res else pool.imap_unordered
            futures = map_func(starcall, [(function, a) for a in args])
        elif kw_args:
            map_func = pool.imap if ordered_res else pool.imap_unordered
            futures = map_func(starstarcall, [(function, a) for a in args])
        else:
            map_func = pool.imap if ordered_res else pool.imap_unordered
            futures = map_func(function, args)
        # Track progress with tqdm
        for f in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(f)
    return front + out


def parallel_processes(*args, **kwargs):
    """Same as parallel_threads, with processes"""
    import multiprocessing as mp

    kwargs["Pool"] = mp.Pool
    return parallel_threads(*args, **kwargs)


def starcall(args):
    """convenient wrapper for Process.Pool"""
    function, args = args
    return function(*args)


def starstarcall(args):
    """convenient wrapper for Process.Pool"""
    function, args = args
    return function(**args)
