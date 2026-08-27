#!/bin/bash
#
#SBATCH --job-name=replay-droid-epi-fo-ksweep
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

# Disable online W&B
export WANDB_MODE=offline

# ===== DIAGNOSTIC SWEEP (one-off, NOT a permanent uq_epi_mode-family job): =====
# Tests whether mean_pdf_diff (0.5*|logvar1-logvar2|, pass1 vs pass2) scales
# with epi_overlap_k regardless of scene content. Pass 2 always has
# num_history + k history tokens vs pass 1's num_history (strictly more,
# every call) -- if the logvar head is sensitive to sequence LENGTH itself
# rather than content, mean_pdf_diff should grow ~monotonically with k.
# See jobs/replay_droid_wm_traj_epi_future_overlap.sh for background: a prior
# replay of this checkpoint showed mean_pdf_diff rendering as an almost
# uniformly-red heatmap (CV ~3.7% across 51 chunks) -- large AND nearly
# content-invariant, consistent with this hypothesis.
#
# Deviates from the repo's one-job-per-parameter-value convention on purpose:
# all k's below MUST share exactly ONE checkpoint snapshot for a valid
# controlled comparison (resolved ONCE below, not per-iteration -- training
# is still running, so re-resolving per-k risks a checkpoint save landing
# mid-sweep and confounding "effect of k" with "effect of more training").
# k=4 (== num_frames-1, this checkpoint's num_frames=5) duplicates the
# existing default job's --epi_overlap_k 0, but is re-run here anyway so it
# shares the same checkpoint snapshot as k=1,2,3 rather than reusing that
# job's (possibly staler or newer) output.
CKPT_DIR="checkpoints/wm_droid/droid_flow_matching_uq_future_overlap_v1"
CKPT=$(ls -t "${CKPT_DIR}"/checkpoint-*.pt 2>/dev/null | head -1)
if [ -z "${CKPT}" ]; then
    echo "ERROR: no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint (fixed for the entire sweep): ${CKPT}"

SWEEP_ROOT="${CKPT_DIR}/replay_epi_overlap_k_sweep"
mkdir -p "${SWEEP_ROOT}"

K_VALUES=(1 2 3 4)

for K in "${K_VALUES[@]}"; do
    OUT_DIR="${SWEEP_ROOT}/k${K}"
    echo "=== [$(date)] starting epi_overlap_k=${K} -> ${OUT_DIR} ==="
    uv run scripts/replay_libero_wm_traj.py \
        --checkpoint "${CKPT}" \
        --data_root /scratch/gpfs/AM43/yy4041/data \
        --suites droid_ctrl_world \
        --split val \
        --stat_root /scratch/gpfs/AM43/yy4041/data/dataset_meta_info \
        --output_dir "${OUT_DIR}" \
        --num_cams 3 --height 192 --width 320 --down_sample 3 \
        --native_fps_default 5 \
        --predict_uncertainty \
        --uq_vis_t_targets 0.9 0.5 0.1 \
        --uq_epi_mode future_overlap \
        --epi_overlap_k "${K}"
    status=$?
    if [ $status -ne 0 ]; then
        echo "=== [$(date)] epi_overlap_k=${K} FAILED (exit ${status}) -- continuing to next k ==="
    else
        echo "=== [$(date)] epi_overlap_k=${K} finished OK ==="
    fi
done

echo "--- sweep summary: mean_pdf_diff / mean_epi_var / mean_kl / epi_overlap_latent_mse vs k ---"
for K in "${K_VALUES[@]}"; do
    SUMMARY="${SWEEP_ROOT}/k${K}/replay_summary.json"
    JSONL="${SWEEP_ROOT}/k${K}/chunk_metrics.jsonl"
    if [ -f "${SUMMARY}" ]; then
        python3 -c "
import json
d = json.load(open('${SUMMARY}'))
if not d:
    print('k=${K}: no episodes completed')
else:
    def avg(key):
        vals = [e[key] for e in d if key in e]
        return sum(vals) / len(vals) if vals else float('nan')
    line = (f'k=${K}  n_episodes={len(d)}  '
            f'mean_pdf_diff={avg(\"mean_pdf_diff\"):.4f}  '
            f'mean_epi_var={avg(\"mean_epi_var\"):.4f}  '
            f'mean_kl={avg(\"mean_kl\"):.4f}')
    try:
        chunk_rows = [json.loads(l) for l in open('${JSONL}')]
        ov = [r['epi_overlap_latent_mse'] for r in chunk_rows if 'epi_overlap_latent_mse' in r]
        if ov:
            line += f'  epi_overlap_latent_mse={sum(ov)/len(ov):.6f}'
    except FileNotFoundError:
        pass
    print(line)
"
    else
        echo "k=${K}: ${SUMMARY} not found (job may have failed)"
    fi
done

echo "SLURM job finished at $(date)"
