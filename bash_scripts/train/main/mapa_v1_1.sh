#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1
NUM_NODES=$2
NODE_RANK=$3
JOB_ID=$4
HOST_NODE_ADDR=$5
MAX_RESTARTS=$6

# Logging Configs
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
module load cuda/12.4.1

# AWS Multi-Node Configs
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288

# Print out the configuration
echo "Running training on ${NUM_NODES} nodes with ${NUM_GPUS} GPUs per node"
echo "Node rank: ${NODE_RANK}, RDZV ID: ${JOB_ID}, RDZV Endpoint: ${HOST_NODE_ADDR}"
echo "Max restarts: ${MAX_RESTARTS}"

# Use HOST_NODE_ADDR for rendezvous (format: "hostname:port" as passed from sbatch)
torchrun --nproc_per_node ${NUM_GPUS} --nnodes ${NUM_NODES} --node_rank ${NODE_RANK} \
    --rdzv_id ${JOB_ID} --rdzv-backend=c10d \
    --rdzv-endpoint ${HOST_NODE_ADDR} \
    --max-restarts ${MAX_RESTARTS} \
    scripts/train.py \
    machine=aws \
    dataset=megatrain_13d_518_many_ar_36ipg_64g dataset.num_workers=12 \
    dataset.num_views=24 \
    loss=overall_loss_highpm_plus_rel_pose \
    model=mapanything_dino_init \
    model/task=aug_training \
    model.pretrained='${root_experiments_dir}/mapanything/training/mapa_v1_1/checkpoint-last-conf-ft.pth' \
    train_params=finetune_heads_only \
    train_params.epochs=30 \
    train_params.warmup_epochs=3 \
    train_params.keep_freq=10 \
    train_params.max_num_of_imgs_per_gpu=48 \
    train_params.check_loss_instability=false \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/mapa_v1_1'
