#!/bin/bash
#
#SBATCH --job-name=gen-bootstrap-droid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%A_%a.err

echo "SLURM job started at $(date)"
echo "Node list: $SLURM_NODELIST"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0} / ${SLURM_ARRAY_TASK_COUNT:-1}"

# ===== WORKDIR =====
mkdir -p /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs
cd /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world
# ===== ENVIRONMENT =====
source scripts/setup.bash

# ===== GENERATE BOOTSTRAP FUTURES =====
# Samples self-generated ("bootstrap") future latents from the frozen
# droid_flow_matching_uq_false_future_v3 checkpoint, conditioned on real
# DROID history+actions -- see scripts/generate_bootstrap_futures_droid.py's
# module docstring for the full design (including why EACH covered anchor
# generates up to 2 cache files, one per possible `skip` a training sample
# might later draw -- guarantees a timestep-exact match at train time
# instead of a ~50% chance of falling back to the other-episode distractor)
# and configs/training/droid_wm_uq_bootstrap_v1.py for how the cache this
# writes is consumed during training.
#
# This is a plain SLURM ARRAY job (no submitit dependency in this repo,
# unlike wm_uq/) -- submit with e.g., for BREADTH-FIRST coverage (every
# episode gets ANCHORS_PER_EP anchors instead of a random episode subset at
# full density -- see --anchors_per_episode's help):
#   sbatch --array=0-15 --export=ALL,SPLIT=train,EVERY_NTH=1,ANCHORS_PER_EP=2 \
#       jobs/generate_bootstrap_futures_droid_array.sh
# or, for the smoke test / full val-split coverage (no array needed):
#   sbatch --export=ALL,SPLIT=val,EVERY_NTH=1,MAX_ANCHORS=200 \
#       jobs/generate_bootstrap_futures_droid_array.sh
#
# Params are picked at submission time (--export=ALL,...); see the plan's
# staged rollout: (1) a small --max_anchors smoke test first to measure
# per-anchor cost and sanity-check output quality, (2) full val-split
# coverage (EVERY_NTH=1 -- val is small enough, and train_wm.py's periodic
# validation loop exercises this same augmentation path), (3) a train-split
# coverage chosen from the smoke test's measured throughput -- ANCHORS_PER_EP
# (breadth-first, every episode touched) is the recommended axis to size
# this by, rather than EVERY_NTH (a random episode subset at full density).
CKPT="checkpoints/wm_droid/droid_flow_matching_uq_false_future_v3/checkpoint-220000.pt"
SPLIT="${SPLIT:-train}"
EVERY_NTH="${EVERY_NTH:-1}"
ANCHORS_PER_EP="${ANCHORS_PER_EP:-0}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"
N_STEPS="${N_STEPS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"

uv run scripts/generate_bootstrap_futures_droid.py \
    --checkpoint "${CKPT}" \
    --data_root /scratch/gpfs/AM43/yy4041/data \
    --stat_root /scratch/gpfs/AM43/yy4041/data/dataset_meta_info \
    --dataset_name droid_ctrl_world \
    --output_root outputs/bootstrap_futures_cache_droid \
    --split "${SPLIT}" \
    --every_nth_episode "${EVERY_NTH}" \
    --anchors_per_episode "${ANCHORS_PER_EP}" \
    --shard_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --max_anchors "${MAX_ANCHORS}" \
    --n_steps "${N_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_cams 3 --height 192 --width 320 --down_sample 3

echo "SLURM job finished at $(date)"
