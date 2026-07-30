#!/bin/bash
#
#SBATCH --job-name=wm-uq-2x2-pp-future-hist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
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

# ===== STAT.JSON (generate if missing) =====
STAT_FILE="data/libero_uq_2x2_policy_probe/stat.json"
if [ ! -f "$STAT_FILE" ]; then
    echo "stat.json not found — generating from training annotations..."
    uv run python scripts/build_uq_test_manifest.py \
        --data_root data/libero_uq_2x2_policy_probe \
        --gen_stat
    echo "stat.json generated."
fi

# ===== TEST MANIFEST (build if missing) =====
MANIFEST_FILE="data/libero_uq_2x2_policy_probe/test_manifest_by_cell.json"
if [ ! -f "$MANIFEST_FILE" ]; then
    echo "test_manifest_by_cell.json not found — building..."
    uv run python scripts/build_uq_test_manifest.py \
        --data_root data/libero_uq_2x2_policy_probe
fi

# ===== TRAIN =====
uv run accelerate launch --num_processes 4 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/libero_wm_uq_2x2_policy_probe_future_hist.py

echo "SLURM job finished at $(date)"
