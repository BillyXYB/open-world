#!/bin/bash
#
#SBATCH --job-name=replay-droid-manual-check-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
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

# Uses the v1 checkpoint (see jobs/replay_droid_wm_traj_manual_check.sh for
# the v3 variant, jobs/replay_droid_wm_traj_manual_check_v2.sh for v2) --
# this job also differs in WHICH episodes get replayed: hand-collected
# trajectories, imported via scripts/import_manual_droid_episode.py, instead
# of droid_ctrl_world's held-out val split.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_false_future_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

OUT_DIR="${CKPT_DIR}/replay_manual_check"

# DATA_ROOT / STAT_REFERENCE_ROOT / SUITE / SPLIT / BUNDLE_DIR / QUADRANT_CELLS,
# the SVD+CLIP weight detection, and the import/rotate/aggregate helpers.
source jobs/_common_manual_check.sh

rotate_chunk_metrics

# ===== IMPORT (one-time per new bundle; safe to re-run, skips existing) =====
# BUNDLE_DIR points at the wm_uq_export/ dir rsynced from the local DROID collection
# machine by droid/scripts/convert/export_and_sync_uq_eval.py -- either a single episode
# dir or a parent dir of several. It defaults to the path that script syncs to.
run_manual_check_import

# ===== REPLAY (epistemic UQ: future_overlap / same-range self-consistency) =====
# Same args as jobs/replay_droid_wm_traj_epi_false_future.sh EXCEPT:
#   --suites droid_manual_uq_check --split val  (new hand-collected suite)
#   --down_sample 1  (NOT 3 -- these episodes have camera/state captured at the
#     SAME native rate, unlike droid_ctrl_world's pre-decimated training data;
#     see openworld/utils/droid_export.py's module docstring CAUTION section)
#   --history_source gt  (NOT the default "rolled" -- these hand-collected
#     episodes are few and short, so we want every chunk graded against the
#     TRUE history/future instead of a closed-loop rollout of the model's own
#     prior predictions. This also turns each episode into many independent
#     (true-history, true-future) windows rather than one long autoregressive
#     chain, which is what we actually want for a per-window fidelity/UQ check
#     against hand-picked quadrant labels. Does not change future_overlap's
#     pass-2 peek, which stays pass-1's own predicted future by design.)
#
# NOTE -- v1 eval compatibility (see configs/training/droid_wm_uq_false_future_v1.py's
# module docstring): --overlap_zero_action IS required (v1 was trained with
# zero_overlap_action=True); --mask_history_from_peek must NOT be set (v1 was
# trained with a variable-length, unmasked sequence -- that flag is v3-only).
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root "${DATA_ROOT}" \
    --suites "${SUITE}" \
    --split "${SPLIT}" \
    --stat_root "${DATA_ROOT}/dataset_meta_info" \
    --output_dir "${OUT_DIR}" \
    --svd_model_path "${SVD_PATH}" \
    --clip_model_path "${CLIP_PATH}" \
    --num_cams 3 --height 192 --width 320 --down_sample 1 \
    --native_fps_default 15 \
    --history_source gt \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode future_overlap \
    --epi_overlap_k 0 \
    --overlap_zero_action \
    || { echo "ERROR: replay_libero_wm_traj.py failed"; exit 1; }

# ===== PER-QUADRANT AGGREGATION + PLOT =====
run_manual_check_aggregation

echo "SLURM job finished at $(date)"
