#!/bin/bash
#
#SBATCH --job-name=ctrl-world-train-bootstrap-only-v1-droid
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

# ===== PREREQUISITE =====
# Requires jobs/generate_bootstrap_futures_droid_array.sh to have already
# populated outputs/bootstrap_futures_cache_droid/ -- see
# configs/training/droid_wm_uq_bootstrap_only_v1.py's module docstring.
if [ ! -d "outputs/bootstrap_futures_cache_droid/droid_ctrl_world" ]; then
    echo "ERROR: outputs/bootstrap_futures_cache_droid/droid_ctrl_world not found." >&2
    echo "Run jobs/generate_bootstrap_futures_droid_array.sh first." >&2
    exit 1
fi

# ===== TRAIN =====
# Bootstrap-ONLY ablation -- warm-starts from
# droid_flow_matching_uq_false_future_v3's checkpoint-220000.pt and drops
# p_shifted_future/p_false_future to 0, so every firing peek is the
# self-sampled bootstrap future (p_bootstrap_future=1.0) -- the only two
# peek states seen during training are "zero" (overlap branch didn't fire)
# and "bootstrap". See configs/training/droid_wm_uq_bootstrap_only_v1.py's
# module docstring for the full rationale.
uv run accelerate launch --num_processes 4 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/droid_wm_uq_bootstrap_only_v1.py


echo "SLURM job finished at $(date)"
