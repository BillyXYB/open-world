#!/bin/bash
#
# Bootstrap-future generation for DROID's TRAIN split, breadth-first
# coverage (K=10 anchors/episode, every episode touched). Run this yourself
# with:
#
#   sbatch jobs/submit_bootstrap_train_k10.sh
#
# 128 shards, throttled to 16 concurrent (--array=0-127%16 -- your stated
# H200 cap, all of it since this run has no val job competing for slots) --
# many small (~6.3h nominal) jobs backfill much better than a few huge ones
# on a saturated shared partition, and a crash only costs one shard's worth
# of work (skip-if-exists makes even that resumable for free -- just
# resubmit the same array; anchors with EITHER skip variant already cached
# are skipped). See scripts/generate_bootstrap_futures_droid.py's module
# docstring for the full design -- each covered anchor generates exactly
# ONE randomly-chosen skip variant (not both); dataset.py falls back to
# whichever variant IS cached at train time if the exact match isn't
# available, before giving up to the other-episode distractor.
#
# Total for this config (see generate_bootstrap_futures_droid.py --help /
# module docstring for the math): ~876,867 generation calls, ~809 GPU-hours
# total at ~3.32s/anchor on H200, batch_size=32 (measured after the
# mask_history_from_peek padding fix -- correctly running the model at its
# TRAINED sequence length is ~36% slower than the earlier, architecturally
# wrong unpadded call) -- ~51h wall-clock (~2.1 days) if all 16 shards ran
# continuously, longer under partition contention (checked at submission
# time: ailab was ~fully saturated, 143/144 H200s busy).
#
# Check progress any time with:
#   squeue -u $USER
#   find outputs/bootstrap_futures_cache_droid/droid_ctrl_world -name '*.pt' | wc -l
#
#SBATCH --job-name=gen-bootstrap-train-k10
#SBATCH --array=0-127%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs/%x-%A_%a.err

echo "SLURM job started at $(date)"
echo "Node list: $SLURM_NODELIST"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0} / ${SLURM_ARRAY_TASK_COUNT:-1}"

mkdir -p /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/logs
cd /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world
source scripts/setup.bash
export WANDB_MODE=offline

uv run scripts/generate_bootstrap_futures_droid.py \
    --checkpoint checkpoints/wm_droid/droid_flow_matching_uq_false_future_v3/checkpoint-220000.pt \
    --data_root /scratch/gpfs/AM43/yy4041/data \
    --stat_root /scratch/gpfs/AM43/yy4041/data/dataset_meta_info \
    --dataset_name droid_ctrl_world \
    --output_root outputs/bootstrap_futures_cache_droid \
    --split train \
    --every_nth_episode 1 \
    --anchors_per_episode 10 \
    --shard_id "${SLURM_ARRAY_TASK_ID:-0}" \
    --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --n_steps 25 \
    --batch_size 32 \
    --num_cams 3 --height 192 --width 320 --down_sample 3

echo "SLURM job finished at $(date)"
