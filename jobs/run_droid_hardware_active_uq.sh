#!/bin/bash
#
#SBATCH --job-name=droid-hw-active-uq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:59:00
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

# ===== SERVER =====
# Robot-side launch is unchanged -- see the root CLAUDE.md:
#   python uq_data_collection/examples/droid/main.py \
#       --external_camera right --num_trajectories_to_collect 10
# This server is a drop-in replacement for
# svd_ac_video_model/vidwm/evaluators/uq_data_collection.py: it talks to the
# same comms-dir file protocol, so start this job FIRST (it blocks polling
# for the robot's first observation), then start the robot.
#
# See configs/collection/droid_hardware_active_uq.yaml's header for the
# known v0-checkpoint down_sample caveat before trusting collected data.
uv run python scripts/run_droid_hardware_active_uq.py \
    --config configs/collection/droid_hardware_active_uq.yaml

echo "SLURM job finished at $(date)"
