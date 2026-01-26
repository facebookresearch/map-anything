#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1
export HYDRA_FULL_ERROR=1

## Assuming NUM_GPUS=16 so that effective batch size is 96-192 (equivalent to one node on AWS)
## If changing max_num_of_imgs_per_gpu, change the NUM_GPUS to match the target effective batch size of 96-192
torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=aws \
    dataset=megatrain_13d_518_many_ar_24ipg_16g dataset.num_workers=12 \
    dataset.num_views=4 \
    loss=overall_loss \
    model=mapanything_v1 \
    model/task=mvs_training \
    model.encoder.uses_torch_hub=false \
    model.encoder.gradient_checkpointing=true \
    model.info_sharing.module_args.gradient_checkpointing=true \
    model.pred_head.gradient_checkpointing=true \
    train_params=lower_encoder_lr \
    train_params.epochs=50 \
    train_params.warmup_epochs=5 \
    train_params.keep_freq=100 \
    train_params.max_num_of_imgs_per_gpu=24 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training_ablations/6a_mvs'
