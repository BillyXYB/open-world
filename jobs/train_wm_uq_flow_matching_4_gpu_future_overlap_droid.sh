#!/bin/bash
#
#SBATCH --job-name=ctrl-world-train-fo-droid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%j.err

echo "SLURM job started at $(date)"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# ===== WORKDIR =====
mkdir -p /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs
cd /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world
# ===== ENVIRONMENT =====
source scripts/setup.bash

# Disable online W&B
export WANDB_MODE=offline

# ===== TRAIN =====
# Fresh from base SVD (ckpt_path=None) with the corrected down_sample=3 --
# deliberately NOT warm-started from droid_flow_matching_uq_future_hist_v0
# (that checkpoint is flagged misaligned, down_sample=1 should be 3; see
# configs/training/droid_wm_uq_future_hist.py's docstring) or from v1 (never
# actually trained). See configs/training/droid_wm_uq_future_overlap.py.
uv run accelerate launch --num_processes 4 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/droid_wm_uq_future_overlap.py


echo "SLURM job finished at $(date)"
