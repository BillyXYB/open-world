#!/bin/bash
#
#SBATCH --job-name=replay-droid-epi-ff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
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
# Trained by jobs/train_wm_uq_flow_matching_4_gpu_false_future_droid.sh
# (configs/training/droid_wm_uq_false_future_v1.py, p_history_future_overlap=0.5,
# p_false_future=0.5, zero_overlap_action=True). Builds on
# droid_flow_matching_uq_future_overlap_v1, which showed the logvar head
# collapsing to near-uniform overconfidence on overlap slots regardless of
# whether the peeked future was actually plausible -- see
# droid_wm_uq_false_future_v1.py's module docstring. v1's checkpoints/replay
# remain on disk under droid_flow_matching_uq_future_overlap_v1/ for comparison.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_false_future_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

# ===== REPLAY (epistemic UQ: future_overlap / same-range self-consistency) =====
# Pass 1: true rolled history -> predicts frames [t, t+num_frames-1].
# Pass 2: splices pass-1's OWN predicted future frames into history (current
#         frame unchanged) and re-predicts the SAME target range [t, t+num_frames-1],
#         using a re-encoded, extended action embedding for the added slots.
# This checkpoint was trained with p_history_future_overlap=0.5, so pass 2's
# input is in-distribution -- and, unlike v1, it was ALSO trained with
# p_false_future=0.5 (peek sometimes mismatched) and zero_overlap_action=True
# (overlap slot's action conditioning zeroed). --overlap_zero_action below is
# REQUIRED to match that training setting; omitting it would silently
# evaluate this checkpoint under v1's (real-action) regime instead. See
# config.py's zero_overlap_action docstring.
# --epi_overlap_k 0 = max overlap (uses all num_frames-1 of pass-1's predicted
# future frames as pass-2 context); pass e.g. 1 or 2 to ablate a smaller k.
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
    --overlap_zero_action

echo "SLURM job finished at $(date)"
