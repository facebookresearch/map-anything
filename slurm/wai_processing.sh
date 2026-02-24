#!/bin/bash
#SBATCH -p cvr
#SBATCH -N 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx:1
#SBATCH -J wai_proc
#SBATCH --output=./slurm/out/slurm-%j.out
#SBATCH --error=./slurm/out/slurm-%j.out

export NUMEXPR_MAX_THREADS=64
export TOKENIZERS_PARALLELISM=True
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NO_ALBUMENTATIONS_UPDATE=1
export TRITON_CACHE_DIR=/comp_robot/${USER}/tmp
export PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning

set -x

RANK="${1:?Error: rank argument is required}"
WORLD_SIZE="${2:?Error: world_size argument is required}"
# DATASET="${1:?Error: dataset argument is required}"
# STAGE="${2:?Error: stage argument is required}"
cd /home/${USER}/workspace_dgx/map-anything/
# python -m wai_processing.launch.slurm_stage \
#   data_processing/wai_processing/configs/launch/${DATASET}.yaml \
#   conda_env=wai_processing \
#   stage=${STAGE} \
#   launch_on_slurm=false

# python -m wai_processing.scripts.conversion.scannetv2 \
#   original_root=/comp_robot/cv_public_dataset/scannetv2/scans \
#   root=/comp_robot/cv_public_dataset/scannetv2/wai \
#   rank=${RANK} \
#   world_size=${WORLD_SIZE}

# python -m wai_processing.scripts.covisibility \
#   data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml \
#   root=/comp_robot/cv_public_dataset/scannetv2/wai \
#   rank=${RANK} \
#   world_size=${WORLD_SIZE}

python -m wai_processing.scripts.undistort \
  data_processing/wai_processing/configs/undistortion/scannetppv2.yaml \
  root=data/scannetppv2 \
  rank=${RANK} \
  world_size=${WORLD_SIZE}