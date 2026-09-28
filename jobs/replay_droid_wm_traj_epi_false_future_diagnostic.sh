#!/bin/bash
#
#SBATCH --job-name=replay-droid-epi-ff-diag
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

# ===== Phase 0 diagnostic (fixed-length masked-peek plan) =====
# Runs the SAME replay eval twice against droid_flow_matching_uq_false_future_v1's
# checkpoint, differing only in --epi_overlap_content:
#   1. pass1_pred (default/normal): pass-2's peek is pass-1's own predicted future.
#   2. repeat_last_hist (diagnostic): pass-2's peek is a content-free copy of
#      true history's own last frame -- carries no new visual information, so
#      any pass-1/pass-2 divergence here is entirely the attention-
#      renormalization / positional-embedding artifact the masking plan's
#      Phase 1-3 code fixes, not genuine epistemic signal.
# Compare the two output dirs' chunk_metrics.jsonl (mean_pdf_diff, mean_kl,
# mean_epi_var, mean_epi_ltv, etc.): if repeat_last_hist's divergence is a
# large fraction of pass1_pred's, the confound was real and material.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_false_future_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

for CONTENT in pass1_pred repeat_last_hist; do
    echo "===== Running with --epi_overlap_content ${CONTENT} ====="
    uv run scripts/replay_libero_wm_traj.py \
        --checkpoint "${CKPT}" \
        --data_root /scratch/gpfs/AM43/yy4041/data \
        --suites droid_ctrl_world \
        --split val \
        --stat_root /scratch/gpfs/AM43/yy4041/data/dataset_meta_info \
        --output_dir "${CKPT_DIR}/replay_epi_false_future_diagnostic_${CONTENT}" \
        --num_cams 3 --height 192 --width 320 --down_sample 3 \
        --native_fps_default 5 \
        --predict_uncertainty \
        --uq_vis_t_targets 0.9 0.5 0.1 \
        --uq_epi_mode future_overlap \
        --epi_overlap_k 0 \
        --overlap_zero_action \
        --epi_overlap_content "${CONTENT}"
done

echo "SLURM job finished at $(date)"
echo "Compare: ${CKPT_DIR}/replay_epi_false_future_diagnostic_pass1_pred/chunk_metrics.jsonl"
echo "     vs: ${CKPT_DIR}/replay_epi_false_future_diagnostic_repeat_last_hist/chunk_metrics.jsonl"
