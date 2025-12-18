#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "10 2 benchmark_518_eth3d_snpp_tav2"
    "10 4 benchmark_518_eth3d_snpp_tav2"
    "10 8 benchmark_518_eth3d_snpp_tav2"
    "5 16 benchmark_518_eth3d_snpp_tav2"
    "4 24 benchmark_518_eth3d_snpp_tav2"
    "2 32 benchmark_518_eth3d_snpp_tav2"
    "1 50 benchmark_518_eth3d_snpp_tav2"
    "1 100 benchmark_518_eth3d_snpp_tav2"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=mapanything \
        model/task=images_only \
        model.encoder.uses_torch_hub=false \
        model.pretrained='${root_experiments_dir}/mapanything/training/mapa_curri_4v_13d_36ipg_64g/checkpoint-last.pth' \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/dense_'"${num_views}"'_view/mapa_4v'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
