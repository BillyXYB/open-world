#!/bin/bash
#
# Plain (non-SLURM) launcher for an internal node with 8x H200 -- same
# filesystem layout as della-gpu. Run directly, e.g.:
#
#   bash jobs/train_wm_uq_flow_matching_8_gpu_bootstrap_only_v1_droid_local.sh
#
# or under nohup/tmux for a long-running session:
#
#   nohup bash jobs/train_wm_uq_flow_matching_8_gpu_bootstrap_only_v1_droid_local.sh \
#       > logs/bootstrap-only-v1-local-$(date +%Y%m%d-%H%M%S).log 2>&1 &
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
# and "bootstrap". Short, Phase-B-style budget (max_train_steps=40_000). See
# configs/training/droid_wm_uq_bootstrap_only_v1.py's module docstring for
# the full rationale.
#
# --num_processes 8 to match this node's 8 GPUs (della's SLURM job for this
# same config uses --num_processes 4 to match its --gres=gpu:4 -- don't
# reuse that value here).
uv run accelerate launch --num_processes 8 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/droid_wm_uq_bootstrap_only_v1.py

echo "Job finished at $(date)"
