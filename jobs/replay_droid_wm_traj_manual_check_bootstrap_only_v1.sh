#!/bin/bash
#
#SBATCH --job-name=replay-droid-manual-check-bootstrap-only-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
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

# Uses the bootstrap-only-v1 checkpoint (see jobs/replay_droid_wm_traj_manual_check.sh
# for the v3 variant, _v1.sh/_v2.sh for the v1/v2 variants) -- this job also
# differs in WHICH episodes get replayed: hand-collected trajectories, imported
# via scripts/import_manual_droid_episode.py, instead of droid_ctrl_world's
# held-out val split.
#
# bootstrap_only_v1 is warm-started from v3's checkpoint-220000.pt (bare
# state_dict load, not a full resume -- see configs/training/droid_wm_uq_bootstrap_only_v1.py's
# module docstring) and trains with the SAME zero_overlap_action=True,
# mask_history_from_peek=True settings as v3, so this eval needs the same two
# flags below as the v3/un-suffixed job.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_bootstrap_only_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

DATA_ROOT=/scratch/gpfs/AM43/yy4041/data
SUITE=droid_manual_uq_check
OUT_DIR="${CKPT_DIR}/replay_manual_check"
QUADRANT_CELLS="low_aleatoric/low_epistemic,low_aleatoric/high_epistemic,high_aleatoric/low_epistemic,high_aleatoric/high_epistemic"

# replay_libero_wm_traj.py opens chunk_metrics.jsonl in APPEND mode, so
# re-submitting this job against the same OUT_DIR would duplicate every prior
# chunk's rows and skew the per-quadrant aggregation below. Move any existing
# file aside (never delete) so each submission starts a clean file.
if [ -f "${OUT_DIR}/chunk_metrics.jsonl" ]; then
    mv "${OUT_DIR}/chunk_metrics.jsonl" "${OUT_DIR}/chunk_metrics.$(date +%Y%m%d_%H%M%S).jsonl"
fi

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
    --svd_model_path external/stable-video-diffusion-img2vid \
    --height 192 --width 320

# ===== REPLAY (epistemic UQ: future_overlap / same-range self-consistency) =====
# Same args as jobs/replay_droid_wm_traj_manual_check.sh (v3 variant) EXCEPT
# the checkpoint -- see CKPT_DIR above.
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root "${DATA_ROOT}" \
    --suites "${SUITE}" \
    --split val \
    --stat_root "${DATA_ROOT}/dataset_meta_info" \
    --output_dir "${OUT_DIR}" \
    --num_cams 3 --height 192 --width 320 --down_sample 1 \
    --native_fps_default 15 \
    --history_source gt \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode future_overlap \
    --epi_overlap_k 0 \
    --overlap_zero_action \
    --mask_history_from_peek

# ===== PER-QUADRANT AGGREGATION + PLOT =====
# Groups chunk_metrics.jsonl by the uncertainty_cell each episode's own
# annotation carries (the hand-picked aleatoric/epistemic quadrant --
# see replay_libero_wm_traj.py's no-manifest episode-list branch), reusing
# the same aggregate/plot scripts the LIBERO 2x2 variance/data-scale UQ study
# uses, via their --cells override.
uv run scripts/aggregate_uq_by_cell.py \
    --chunk_jsonl "${OUT_DIR}/chunk_metrics.jsonl" \
    --cells "${QUADRANT_CELLS}" \
    --output_dir "${OUT_DIR}/uq_by_quadrant"

uv run scripts/plot_uq_by_cell.py \
    --csv "${OUT_DIR}/uq_by_quadrant/uq_per_chunk.csv" \
    --cells "${QUADRANT_CELLS}" \
    --output_dir "${OUT_DIR}/uq_by_quadrant/plots"

echo "SLURM job finished at $(date)"
