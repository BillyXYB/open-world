#!/bin/bash
#
#SBATCH --job-name=ctrl-world-train-ff-droid
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
# Fresh from base SVD (ckpt_path=None), same convention as
# droid_flow_matching_uq_future_overlap_v1: avoids any chance that a
# warm-started checkpoint's weights already baked in the "peek == answer"
# shortcut this run's p_false_future augmentation is meant to break. See
# configs/training/droid_wm_uq_false_future_v1.py's module docstring --
# adds p_false_future=0.5 (mismatched-future injection) and
# zero_overlap_action=True (zeroed action conditioning at the overlap slot)
# on top of v1's p_history_future_overlap=0.5.
uv run accelerate launch --num_processes 4 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/droid_wm_uq_false_future_v1.py


echo "SLURM job finished at $(date)"
