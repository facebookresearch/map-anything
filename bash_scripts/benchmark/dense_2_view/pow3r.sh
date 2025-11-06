#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
tasks=(
    "images_only"
    "calibrated_sfm"
    "mvs"
    "registration"
    "pass_through"
)

# Loop through each task
for task in "${tasks[@]}"; do
    echo "Running with task=$task"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        dataset=benchmark_512_eth3d_snpp_tav2 \
        dataset.num_workers=12 \
        dataset.num_views=2 \
        batch_size=10 \
        model=pow3r \
        model/task=$task \
        hydra.run.dir='${root_experiments_dir}/morphcloud/two_view_benchmarking/pow3r_'"${task}"''

    echo "Finished running with task=$task"
done
