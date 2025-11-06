#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

python3 \
    benchmarking/calibration/benchmark.py \
    machine=aws \
    dataset=benchmark_sv_calib_518_many_ar_eth3d_snpp_tav2 \
    dataset.num_workers=12 \
    batch_size=20 \
    model=anycalib \
    amp=0 \
    hydra.run.dir='${root_experiments_dir}/morphcloud/calibration_benchmarking/single_view/anycalib'
