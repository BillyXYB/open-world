#!/bin/bash
#
#SBATCH --job-name=collect-active-uq-smoke-test
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

# ===== SMOKE TEST: run_data_collection_active_uq.py end to end =====
# num_candidates=2, one task, one trajectory, low num_inference_steps -> a
# handful of active-phase decisions should complete in a few minutes.
# Checks to make manually against the printed output / decision_metrics.jsonl:
#   - no shape errors in the batched Pass 1 / Pass 2 pipeline calls
#   - the 2 candidates' compute_uq_metrics values differ meaningfully at
#     each decision (near-identical values across candidates would indicate
#     the batch dim got aliased via a non-contiguous .expand())
#   - decision_metrics.jsonl row count matches the number of active-phase
#     decisions logged to stdout
OUT_DIR="data/active_uq_smoke_test"
rm -rf "${OUT_DIR}"

uv run python scripts/run_data_collection_active_uq.py \
    --config configs/collection/libero_pi05_active_uq.yaml \
    --output_root "${OUT_DIR}"

echo "--- decision_metrics.jsonl ---"
cat "${OUT_DIR}/libero_spatial/decision_metrics.jsonl"
echo "--- annotation (extra_annotation fields) ---"
python3 - "${OUT_DIR}" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1]) / "libero_spatial" / "annotation" / "train"
for p in sorted(root.glob("*.json")):
    ann = json.loads(p.read_text())
    print(p.name, {k: ann[k] for k in (
        "source", "num_candidates", "uq_metric", "num_decisions",
        "mean_chosen_uq_score", "mean_uq_score_spread", "is_success") if k in ann})
EOF

echo "SLURM job finished at $(date)"
