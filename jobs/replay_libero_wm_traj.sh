#!/bin/bash
#
#SBATCH --job-name=replay-libero-wm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
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

# ===== REPLAY =====
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/checkpoints/wm_libero/libero_flow_matching_uq_v0/checkpoint-76000.pt \
    --data_root data/libero_collected \
    --output_dir checkpoints/wm_libero/libero_wm_uq/replay_norm_all \
    --stat_root /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_processed_5hz/ \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1

echo "SLURM job finished at $(date)"
