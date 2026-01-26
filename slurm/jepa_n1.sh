#!/bin/bash
#SBATCH -p cvr
#SBATCH -N 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=380G
#SBATCH --gres=gpu:hgx:8
#SBATCH -J mapa
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

# cd /home/${USER}/workspace_dgx/map-anything/mapanything/ops
# sh ./make.sh
# python test.py

cd /home/${USER}/workspace_dgx/map-anything/
bash bash_scripts/train/jepa/mapa_curri_4v_bmvs_4ipg_18g_jepa.sh \
    8 $SLURM_NNODES $SLURM_NODEID $SLURM_JOBID localhost 1