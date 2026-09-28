#!/bin/bash
#
#SBATCH --job-name=replay-droid-manual-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
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

# Same checkpoint as jobs/replay_droid_wm_traj_epi_false_future.sh -- this job
# only differs in WHICH episodes get replayed: hand-collected trajectories,
# imported via scripts/import_manual_droid_episode.py, instead of
# droid_ctrl_world's held-out val split.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_false_future_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

DATA_ROOT=/scratch/gpfs/AM43/yy4041/data
SUITE=droid_manual_uq_check

# ===== IMPORT (one-time per new bundle; safe to re-run, skips existing) =====
# BUNDLE_DIR should point at the wm_uq_export/ dir you rsynced over from the
# local DROID collection machine (droid/scripts/convert/export_episode_for_wm_uq.py
# output) -- either a single episode dir or a parent dir of several.
: "${BUNDLE_DIR:?Set BUNDLE_DIR to the rsynced wm_uq_export/<episode_id>/ (or parent) directory}"
uv run scripts/import_manual_droid_episode.py \
    --bundle_dir "${BUNDLE_DIR}" \
    --data_root "${DATA_ROOT}" \
    --suite "${SUITE}" \
    --split val \
    --svd_model_path external/svd_weights \
    --height 192 --width 320

# ===== REPLAY (epistemic UQ: future_overlap / same-range self-consistency) =====
# Same args as jobs/replay_droid_wm_traj_epi_false_future.sh EXCEPT:
#   --suites droid_manual_uq_check --split val  (new hand-collected suite)
#   --down_sample 1  (NOT 3 -- these episodes have camera/state captured at the
#     SAME native rate, unlike droid_ctrl_world's pre-decimated training data;
#     see openworld/utils/droid_export.py's module docstring CAUTION section)
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root "${DATA_ROOT}" \
    --suites "${SUITE}" \
    --split val \
    --stat_root "${DATA_ROOT}/dataset_meta_info" \
    --output_dir "${CKPT_DIR}/replay_manual_check" \
    --num_cams 3 --height 192 --width 320 --down_sample 1 \
    --native_fps_default 15 \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode future_overlap \
    --epi_overlap_k 0 \
    --overlap_zero_action

echo "SLURM job finished at $(date)"
