#!/bin/bash
#
#SBATCH --job-name=ctrl-world-train-ff-v2-droid
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
# v2 of the false-future/UQ line -- see
# configs/training/droid_wm_uq_false_future_v2.py's module docstring for the
# full rationale. Relative to v1 (droid_flow_matching_uq_false_future_v1):
#   1. ckpt_path warm-starts from the pretrained Ctrl-World release
#      (yjguo/Ctrl-World checkpoint-10000.pt) instead of training from
#      scratch on top of raw SVD.
#   2. fixed_overlap_k=True: always peek the full num_frames-1 future
#      window instead of a random prefix, matching eval-time behavior.
#   3. Three-way future mix: p_history_future_overlap=0.5, and conditional
#      on that firing, p_shifted_future=0.25 (same-episode, temporally
#      shifted) + p_false_future=0.25 (other-episode) -- i.e. 50% real /
#      25% same-episode shift / 25% other-episode overall.
uv run accelerate launch --num_processes 8 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/droid_wm_uq_false_future_v2.py


echo "SLURM job finished at $(date)"
