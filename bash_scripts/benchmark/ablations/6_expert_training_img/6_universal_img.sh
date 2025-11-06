#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "10 2"
    "10 4"
    "10 8"
    "5 16"
    "4 24"
    "2 32"
    "1 50"
    "1 100"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views <<< "$combo"

    echo "Running with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        dataset=benchmark_518_eth3d_snpp_tav2 \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=morphcloud \
        model/task=images_only \
        model.encoder.uses_torch_hub=false \
        model.pretrained='${root_experiments_dir}/morphcloud/training_ablations/a1d_mapa/checkpoint-best.pth' \
        hydra.run.dir='${root_experiments_dir}/morphcloud/benchmarking_ablations/dense_'"${num_views}"'_view/6_universal_img'

    echo "Finished running with batch_size=$batch_size and num_views=$num_views"
done
