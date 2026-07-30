#!/bin/bash
#
#SBATCH --job-name=collect-uq-2x2-policy-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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

# MuJoCo offscreen rendering via EGL (required on headless compute nodes)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0

# ===== COLLECT (with policy-based variance probe) =====
# --policy_probe: runs the actual pi0.5 policy at each probe friction value
# instead of kinematically replaying a demo. This measures the *effective*
# variance the policy experiences, accounting for any friction compensation.
# Results cached to data/libero_uq_2x2_policy_probe/variance_probe_policy.json.
#
# To inspect probe results before full collection:
#   MUJOCO_GL=egl ... uv run python scripts/run_data_collection_uq.py \
#     --config configs/collection/libero_pi05_uq_2x2.yaml \
#     --policy_probe --probe_only \
#     --output_root data/libero_uq_2x2_policy_probe
uv run python scripts/run_data_collection_uq.py \
    --config configs/collection/libero_pi05_uq_2x2.yaml \
    --policy_probe \
    --output_root data/libero_uq_2x2_policy_probe

echo "SLURM job finished at $(date)"
