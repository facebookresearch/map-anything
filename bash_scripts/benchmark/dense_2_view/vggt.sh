#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

python3 \
    benchmarking/dense_n_view/benchmark.py \
    machine=aws \
    dataset=benchmark_518_eth3d_snpp_tav2 \
    dataset.num_workers=12 \
    dataset.num_views=2 \
    batch_size=10 \
    model=vggt \
    hydra.run.dir='${root_experiments_dir}/morphcloud/two_view_benchmarking/vggt'
