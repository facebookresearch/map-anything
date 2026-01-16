#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

PRETRAINED_CHECKPOINT_PATH=$1
RESULT_DIR=$2

export HYDRA_FULL_ERROR=1

echo "Running with task=images_only"

python3 \
    benchmarking/dense_n_view/benchmark.py \
    machine=aws \
    dataset=benchmark_512_eth3d_snpp_tav2 \
    dataset.num_workers=12 \
    dataset.num_views=2 \
    batch_size=10 \
    model=pixio \
    model/task=images_only \
    model.pretrained=${PRETRAINED_CHECKPOINT_PATH} \
    hydra.run.dir="${RESULT_DIR}/mapa_24v_images_only"

echo "Finished running with task=$task"

