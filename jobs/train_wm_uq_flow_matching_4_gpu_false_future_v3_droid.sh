#!/bin/bash
#
#SBATCH --job-name=ctrl-world-train-ff-v3-droid
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
# v3 of the false-future/UQ line -- see
# configs/training/droid_wm_uq_false_future_v3.py's module docstring for the
# full rationale. Relative to v2 (droid_flow_matching_uq_false_future_v2):
#   mask_history_from_peek=True: always reserve num_frames-1 peek slots in
#   the frame sequence (whether or not the history-future-overlap branch
#   fires this step) and mask temporal attention so true-history and peek
#   are each sealed to their own block (neither attends to the other) --
#   fixes two confounds (attention renormalization + a positional-embedding
#   shift) that otherwise ride along with the pass-1/pass-2 comparison
#   uq_epi_mode=future_overlap is built on. See
#   openworld/world_models/ctrl_world/attention_masks.py's module docstring
#   for the full design.
#
#   Also trains from scratch (ckpt_path=None) rather than warm-starting from
#   v2 or the released Ctrl-World checkpoint -- see the training config's
#   module docstring for why (masking is a distribution shift for
#   temporal-attention weights that have never seen it; isolates masking as
#   a standalone change).
#
# --num_processes 4 to match --gres=gpu:4 above (the v2 job script for this
# line uses --num_processes 8, a known mismatch bug -- don't propagate it).
uv run accelerate launch --num_processes 4 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/droid_wm_uq_false_future_v3.py


echo "SLURM job finished at $(date)"
