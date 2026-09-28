#!/bin/bash
#
#SBATCH --job-name=replay-droid-epi-bootstrap-v1
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
# Trained by jobs/train_wm_uq_flow_matching_4_gpu_bootstrap_v1_droid.sh
# (configs/training/droid_wm_uq_bootstrap_v1.py). Same augmentation
# architecture as v3 (p_history_future_overlap=0.5, zero_overlap_action=True,
# mask_history_from_peek=True) plus a fourth mix component,
# p_bootstrap_future=0.40, reallocated from v3's p_false_future/
# p_shifted_future=0.25/0.25 down to 0.15/0.15. Eval itself is UNCHANGED from
# v3's replay job -- bootstrap only changes what the checkpoint saw as
# context DURING TRAINING, never at eval time.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_bootstrap_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

# ===== REPLAY (epistemic UQ: future_overlap / same-range self-consistency) =====
# Identical invocation to jobs/replay_droid_wm_traj_epi_false_future_v3.sh
# (same --mask_history_from_peek / --overlap_zero_action / --epi_overlap_k 0
# flags -- required since this checkpoint inherits v3's masked-peek
# architecture via warm start) so chunk_metrics.jsonl from this run is
# directly comparable, chunk-for-chunk-metric-for-metric, against v3's own
# replay_epi_false_future/chunk_metrics.jsonl. The before/after comparison
# that matters: mean_epi_var_signed moving toward/past zero (no longer
# collapsing to "peek present => more confident") while epi_full_latent_mse
# stays non-trivial (still making real errors -- the fix is calibration, not
# error elimination). See scripts/aggregate_uq_by_cell.py --chunk_jsonl.
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root /scratch/gpfs/AM43/yy4041/data \
    --suites droid_ctrl_world \
    --split val \
    --stat_root /scratch/gpfs/AM43/yy4041/data/dataset_meta_info \
    --output_dir "${CKPT_DIR}/replay_epi_bootstrap" \
    --num_cams 3 --height 192 --width 320 --down_sample 3 \
    --native_fps_default 5 \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode future_overlap \
    --epi_overlap_k 0 \
    --overlap_zero_action \
    --mask_history_from_peek

echo "SLURM job finished at $(date)"
echo "Compare against checkpoints/wm_droid/droid_flow_matching_uq_false_future_v3/replay_epi_false_future/chunk_metrics.jsonl"
