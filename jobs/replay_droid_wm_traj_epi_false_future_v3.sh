#!/bin/bash
#
#SBATCH --job-name=replay-droid-epi-ff-v3
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

# ===== CHECKPOINT =====
# Trained by jobs/train_wm_uq_flow_matching_4_gpu_false_future_v3_droid.sh
# (configs/training/droid_wm_uq_false_future_v3.py). Identical augmentation
# mix to v2 (p_history_future_overlap=0.5, p_false_future=0.25,
# p_shifted_future=0.25, zero_overlap_action=True) plus
# mask_history_from_peek=True -- see that training config's module docstring
# for the fixed-length/masking rationale, and
# openworld/world_models/ctrl_world/attention_masks.py's module docstring
# for the exact masking rules (true-history and peek each sealed to their
# own block; target attends to everything).
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_false_future_v3"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

# ===== REPLAY (epistemic UQ: future_overlap / same-range self-consistency) =====
# Pass 1: true rolled history (+ v3's fixed-length zero-padded, fully-masked
#         peek slots -- see --mask_history_from_peek below) -> predicts
#         frames [t, t+num_frames-1].
# Pass 2: splices pass-1's OWN predicted future frames into the same
#         fixed-length peek slots (current frame unchanged) and re-predicts
#         the SAME target range, using a re-encoded, extended action
#         embedding for the added slots.
# --overlap_zero_action REQUIRED (matches zero_overlap_action=True training
# setting -- same as v1/v2).
# --mask_history_from_peek REQUIRED for v3 specifically (NOT set for v1/v2 --
# they were trained with a variable-length, unmasked sequence; setting this
# against a v1/v2 checkpoint would evaluate it out-of-distribution). This is
# the one flag that distinguishes this job from
# jobs/replay_droid_wm_traj_epi_false_future.sh (v1's replay job).
# --epi_overlap_k 0 = max overlap; v3 forces the max regardless (see
# replay_libero_wm_traj.py's mask_history_from_peek branch), so this is
# effectively a no-op here, kept for consistency with the v1/v2 invocation.
#
# No dedicated DROID "collected" eval set exists (unlike data/libero_collected),
# so this replays against droid_ctrl_world's held-out val split directly.
# --down_sample 3 corrects for droid_ctrl_world's action/state arrays being at
# 3x the pre-encoded latents' rate (see droid_wm_uq.py's module docstring).
# --native_fps_default 5 (== --target_hz default) makes the native-rate
# restride a no-op, since these latents are already at 5 Hz WM rate.
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root /scratch/gpfs/AM43/yy4041/data \
    --suites droid_ctrl_world \
    --split val \
    --stat_root /scratch/gpfs/AM43/yy4041/data/dataset_meta_info \
    --output_dir "${CKPT_DIR}/replay_epi_false_future" \
    --num_cams 3 --height 192 --width 320 --down_sample 3 \
    --native_fps_default 5 \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode future_overlap \
    --epi_overlap_k 0 \
    --overlap_zero_action \
    --mask_history_from_peek

echo "SLURM job finished at $(date)"
echo "Cross-reference against jobs/replay_droid_wm_traj_epi_false_future_diagnostic.sh's"
echo "pass1_pred vs repeat_last_hist comparison (v1 checkpoint) when interpreting these numbers."
