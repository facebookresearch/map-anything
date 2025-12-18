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
module load cuda/12.4 nccl/2.18.3-cuda.12.1 nccl_efa/1.24.1-nccl.2.18.3-cuda.12.0 libfabric-aws/2.1.0amzn5.0 openmpi5/5.0.6

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
    dataset=megatrain_13d_518_many_ar_48ipg_64g dataset.num_workers=12 \
    dataset.num_views=4 \
    loss=overall_loss_weigh_pm_higher \
    model=mapanything_v1 \
    model/task=aug_training \
    model.encoder.gradient_checkpointing=true \
    train_params=lower_encoder_lr_64g \
    train_params.epochs=100 \
    train_params.warmup_epochs=10 \
    train_params.keep_freq=40 \
    train_params.max_num_of_imgs_per_gpu=48 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/mapa_curri_4v_13d_48ipg_64g'
