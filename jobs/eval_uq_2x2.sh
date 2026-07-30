#!/bin/bash
#
#SBATCH --job-name=eval-uq-2x2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
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
export WANDB_MODE=offline

# ===== CHECKPOINT (auto-select latest from UQ 2x2 training run) =====
CKPT_DIR="checkpoints/wm_libero/libero_wm_uq_2x2_future_hist_v0"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

REPLAY_DIR="${CKPT_DIR}/replay_uq_2x2_val"
RESULTS_DIR="results/uq_2x2_$(basename ${CKPT_DIR})"

# ===== REPLAY on val split (iterative epistemic UQ) =====
# Pass 1: closed-loop rolled history from GT seed.
# Pass 2 (iterative): each chunk conditions on pass-2's own prior predictions,
#   so epistemic uncertainty reflects self-consistency gap.
# --uq_vis_t_targets 0.9 0.5 0.1 → per-t metrics saved in replay_summary.json
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root data/libero_uq_2x2 \
    --stat_root data/libero_uq_2x2 \
    --split val \
    --output_dir "${REPLAY_DIR}" \
    --manifest data/libero_uq_2x2/test_manifest_by_cell.json \
    --max_episodes_per_cell 20 \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode iterative

echo "Replay finished at $(date)"

# ===== AGGREGATE by 2x2 cell =====
uv run python scripts/aggregate_uq_by_cell.py \
    --replay_summary "${REPLAY_DIR}/replay_summary.json" \
    --data_root data/libero_uq_2x2 \
    --output_dir "${RESULTS_DIR}"

echo "Aggregation finished at $(date)"
echo ""
echo "Results written to: ${RESULTS_DIR}/"
echo "  uq_per_episode.csv  — per-episode metrics with cell label (for plotting)"
echo "  uq_by_cell.json     — per-cell mean/std/CVaR95 (primary comparison)"

echo "SLURM job finished at $(date)"
