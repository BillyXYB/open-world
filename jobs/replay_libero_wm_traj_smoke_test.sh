#!/bin/bash
#
#SBATCH --job-name=replay-libero-smoke-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
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

# ===== SMOKE TEST: verify the load_crtl_world() extraction refactor in =====
# ===== replay_libero_wm_traj.py didn't change behavior. 1 episode,      =====
# ===== 1 autoregressive chunk, few inference steps -> should finish in  =====
# ===== well under the time limit.                                      =====
CKPT_DIR="checkpoints/wm_libero/libero_wm_uq_2x2_policy_probe_future_hist_v0"
CKPT="${CKPT_DIR}/checkpoint-200000.pt"
OUT_DIR="${CKPT_DIR}/replay_smoke_test/mean_pdf_diff"

rm -rf "${OUT_DIR}"
# NOTE: data/libero_uq_2x2_policy_probe is stored at NATIVE 20Hz with
# down_sample=1 (confirmed via its annotation JSON's "fps"/"down_sample"
# fields and LiberoLatentDataset._build_frame_ids), unlike the 5Hz-strided
# convention replay_libero_wm_traj.py's --target_hz defaults to (5). Pass
# --target_hz 20 so it doesn't downsample this data by another 4x (which
# shrinks episodes below the minimum length a single chunk needs).
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint "${CKPT}" \
    --data_root data/libero_uq_2x2_policy_probe \
    --stat_root data/libero_uq_2x2_policy_probe \
    --output_dir "${OUT_DIR}" \
    --suites libero_spatial \
    --num_episodes 1 \
    --max_chunks 1 \
    --target_hz 20 \
    --num_inference_steps 4 \
    --predict_uncertainty \
    --uq_epi_mode iterative

echo "--- replay_summary.json ---"
cat "${OUT_DIR}/replay_summary.json"
echo "--- chunk_metrics.jsonl ---"
cat "${OUT_DIR}/chunk_metrics.jsonl"

echo "SLURM job finished at $(date)"
