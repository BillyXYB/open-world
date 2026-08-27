#!/bin/bash
#
#SBATCH --job-name=replay-droid-epi-zero
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

# ===== CHECKPOINT =====
# droid_wm_uq.py (base, v1: down_sample=3 fix) hasn't been trained yet as of
# writing this job; run jobs/train_wm_uq_flow_matching_4_gpu_droid.sh first.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

# ===== REPLAY (epistemic UQ: zero_history, on DROID) =====
# Pass 1: true rolled history.
# Pass 2: his_cond_zero=True (history positions zeroed in the latent sequence).
# Epistemic columns: EpiLTV (always) + EpiVar per t-target (if UQ head present).
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
    --output_dir "${CKPT_DIR}/replay_epi_zero_history" \
    --num_cams 3 --height 192 --width 320 --down_sample 3 \
    --native_fps_default 5 \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode zero_history

echo "SLURM job finished at $(date)"
