#!/bin/bash
#
# Resume launcher for droid_wm_uq_bootstrap_v1_resume.py -- picks up the
# original bootstrap-v1 run after it was OOM-killed by the host kernel at
# step 186993/220000 on this shared, non-SLURM-managed node (not a training
# bug: rank 0 got a raw SIGKILL with no Python/CUDA traceback). See
# configs/training/droid_wm_uq_bootstrap_v1_resume.py's module docstring.
#
#   bash jobs/train_wm_uq_flow_matching_8_gpu_bootstrap_v1_resume_droid_local.sh
#
set -euo pipefail

cd /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world
mkdir -p logs

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-all}"

# ===== ENVIRONMENT =====
source scripts/setup.bash

# Disable online W&B
export WANDB_MODE=offline

# ===== PREREQUISITE =====
if [ ! -d "outputs/bootstrap_futures_cache_droid/droid_ctrl_world" ]; then
    echo "ERROR: outputs/bootstrap_futures_cache_droid/droid_ctrl_world not found." >&2
    exit 1
fi
if [ ! -f "checkpoints/wm_droid/droid_flow_matching_uq_bootstrap_v1/checkpoint-180000.pt" ]; then
    echo "ERROR: resume checkpoint checkpoint-180000.pt not found." >&2
    exit 1
fi

# ===== TRAIN (resume) =====
# --num_processes 8 to match this node's 8 GPUs.
uv run accelerate launch --num_processes 8 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/droid_wm_uq_bootstrap_v1_resume.py

echo "Job finished at $(date)"
