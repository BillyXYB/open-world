#!/bin/bash
#
#SBATCH --job-name=verify-sim-rewind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --output=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%j.err

echo "SLURM job started at $(date)"

# ===== WORKDIR =====
mkdir -p /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs
cd /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world

# ===== ENVIRONMENT =====
source scripts/setup.bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0

# ===== VERIFY: sim snapshot/restore determinism (no policy or WM needed) =====
uv run python scripts/verify_sim_rewind.py --task_suite libero_spatial --task_id 0

echo "SLURM job finished at $(date)"
