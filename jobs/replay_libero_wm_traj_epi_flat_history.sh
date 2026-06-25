#!/bin/bash
#
#SBATCH --job-name=replay-libero-epi-flat
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

# ===== REPLAY (epistemic UQ: flat_history) =====
# Pass 1: true rolled history.
# Pass 2: all history slots replaced with the current frame repeated num_history times.
# Epistemic columns: EpiLTV (always) + EpiVar per t-target (if UQ head present).
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/checkpoints/wm_libero/libero_flow_matching_uq_v0/checkpoint-76000.pt \
    --data_root data/libero_collected \
    --output_dir checkpoints/wm_libero/libero_wm_uq/replay_epi_flat_history \
    --stat_root /scratch/gpfs/AM43/yy4041/open-world/data/wm_training/libero_processed_5hz/ \
    --predict_uncertainty \
    --uq_vis_t_targets 0.9 0.5 0.1 \
    --uq_epi_mode flat_history

echo "SLURM job finished at $(date)"
