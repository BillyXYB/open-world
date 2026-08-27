#!/bin/bash
#
#SBATCH --job-name=collect-active-uq-viz-smoke-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
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
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0

# ===== SMOKE TEST: candidate-comparison visualization (active_uq.viz) =====
# 1 task, 1 trajectory, 4 candidates, capped at 3 decisions -- checks:
#   - candidate_viz/<episode_id>/decision_NNN.mp4 files are written and playable
#   - each has 8 rows (header + pred1/pred2/aleatoric/epi_ltv/epi_var/pdf_diff/kl)
#     and num_candidates columns
#   - the first column's header strip is green, the rest gray
OUT_DIR="data/active_uq_viz_smoke_test"
rm -rf "${OUT_DIR}"

uv run python scripts/run_data_collection_active_uq.py \
    --config configs/collection/libero_pi05_active_uq_viz_smoke_test.yaml \
    --output_root "${OUT_DIR}"

echo "--- candidate_viz output ---"
find "${OUT_DIR}/libero_spatial/candidate_viz" -type f 2>&1

echo "SLURM job finished at $(date)"
