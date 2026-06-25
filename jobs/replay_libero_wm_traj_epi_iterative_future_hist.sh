#!/bin/bash
#
#SBATCH --job-name=replay-libero-epi-iter-fh
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

# ===== CHECKPOINT (auto-select latest from future-hist run) =====
CKPT_DIR="checkpoints/wm_libero/libero_flow_matching_uq_future_hist_v0"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

# ===== REPLAY (epistemic UQ: iterative / self-consistency) =====
# Pass 1: true rolled history (closed-loop from GT seed).
# Pass 2: a separate rolled2 buffer seeded from GT but updated with pass-2 predictions,
#         so each chunk conditions on the model's own prior predictions (self-consistency).
# Large EpiLTV = model predictions diverge when conditioned on its own history vs GT history.
# Note: requires ~2x inference per episode; walltime set to 3h accordingly.
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root data/libero_collected \
    --output_dir "${CKPT_DIR}/replay_epi_iterative_1" \
    --stat_root /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_processed_5hz/ \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode iterative

echo "SLURM job finished at $(date)"
