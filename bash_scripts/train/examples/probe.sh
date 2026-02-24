#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1
export HYDRA_FULL_ERROR=1

## Assuming NUM_GPUS=8 so that effective batch size is 96-192
## If changing max_num_of_imgs_per_gpu, change the NUM_GPUS to match the target effective batch size of 96-192
## Use model.info_sharing.module_args.gradient_checkpointing=true & model.pred_head.gradient_checkpointing=true to save GPU memory when necessary
torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=cvr \
    dataset=scannet_518_many_ar_48ipg_8g dataset.num_workers=12 \
    dataset.num_views=4 \
    loss=jepa_probe_loss \
    model=mapanything_adapter_jepa \
    model.encoder.uses_torch_hub=false \
    model.encoder.gradient_checkpointing=true \
    model.info_sharing.module_args.gradient_checkpointing=true \
    model.pred_head.gradient_checkpointing=true \
    train_params=jepa_probe \
    train_params.epochs=10 \
    train_params.warmup_epochs=1 \
    train_params.keep_freq=200 \
    train_params.max_num_of_imgs_per_gpu=4 \
    model.pretrained='${root_experiments_dir}/mapanything/training/scannet_online_probe_2/checkpoint-best.pth' \
    model.model_config.pretrained_checkpoint_path=null \
    hydra.run.dir='${root_experiments_dir}/mapanything/training_examples/probe_scannet/${now:%Y-%m-%d_%H-%M-%S}'
