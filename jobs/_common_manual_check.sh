# Shared configuration for the jobs/replay_droid_wm_traj_manual_check*.sh family.
#
# Sourced, not executed. Source it AFTER `source scripts/setup.bash` and after the `cd` into
# the repo root, because the external-weights detection below resolves `external/` relative
# to the working directory.
#
# Each caller still sets its own CKPT_DIR (which checkpoint) and its own replay flags.
# Everything here is the stuff that must not drift between the five variants.

# ===== DATA ROOTS =====
# Writes go to yx2653's scratch. The suite used to be written into yy4041's tree, which
# yx2653 cannot write.
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/AM43/yx2653/data}"

# ensure_stat_json() bootstraps this suite's stat.json by copying droid_ctrl_world's, and
# that reference dataset lives in yy4041's tree (readable, not writable). The importer's
# --stat_reference_root lets it READ from there while WRITING to DATA_ROOT above.
STAT_REFERENCE_ROOT="${STAT_REFERENCE_ROOT:-/scratch/gpfs/AM43/yy4041/data}"

SUITE="${SUITE:-droid_manual_uq_check}"
SPLIT="${SPLIT:-val}"

# Where the bundles rsynced from the collection laptop land.
# droid/scripts/convert/export_and_sync_uq_eval.py writes here by default.
BUNDLE_DIR="${BUNDLE_DIR:-/scratch/gpfs/AM43/yx2653/data/wm_uq_export}"

QUADRANT_CELLS="${QUADRANT_CELLS:-low_aleatoric/low_epistemic,low_aleatoric/high_epistemic,high_aleatoric/low_epistemic,high_aleatoric/high_epistemic}"

if [ ! -d "${BUNDLE_DIR}" ]; then
    echo "ERROR: BUNDLE_DIR=${BUNDLE_DIR} does not exist."
    echo "       Sync bundles from the collection laptop first:"
    echo "         python3 droid/scripts/convert/export_and_sync_uq_eval.py"
    echo "       or point BUNDLE_DIR at an existing wm_uq_export/ tree."
    exit 1
fi

# ===== EXTERNAL WEIGHTS =====
# The two branches that produced replay_droid_wm_traj_manual_check.sh disagreed on where the
# SVD weights live (external/svd_weights vs external/stable-video-diffusion-img2vid), so
# detect rather than hardcode. LatentEncoder (scripts/preprocess_libero_for_wm.py) calls
# AutoencoderKLTemporalDecoder.from_pretrained(path, subfolder="vae"), so the real test is
# whether <path>/vae/config.json exists. stable-video-diffusion-img2vid is tried first: it is
# what the sibling jobs use, what external/download_models.sh produces, and what
# replay_libero_wm_traj.py defaults to.
SVD_PATH="${SVD_MODEL_PATH:-}"
if [ -z "${SVD_PATH}" ]; then
    for cand in external/stable-video-diffusion-img2vid external/svd_weights; do
        if [ -f "${cand}/vae/config.json" ]; then SVD_PATH="${cand}"; break; fi
    done
fi
if [ -z "${SVD_PATH}" ]; then
    for cand in external/stable-video-diffusion-img2vid external/svd_weights; do
        if [ -d "${cand}" ]; then
            SVD_PATH="${cand}"
            echo "WARNING: ${cand} exists but has no vae/config.json; trying it anyway."
            break
        fi
    done
fi
if [ -z "${SVD_PATH}" ]; then
    echo "ERROR: no SVD weights found. Looked for (relative to $(pwd)):"
    echo "         external/stable-video-diffusion-img2vid/vae/config.json"
    echo "         external/svd_weights/vae/config.json"
    echo "       Run external/download_models.sh, or set SVD_MODEL_PATH=<dir>."
    exit 1
fi
echo "Using SVD weights: ${SVD_PATH}"

CLIP_PATH="${CLIP_MODEL_PATH:-external/clip-vit-base-patch32}"
if [ ! -d "${CLIP_PATH}" ]; then
    echo "ERROR: no CLIP weights at ${CLIP_PATH} (run external/download_models.sh,"
    echo "       or set CLIP_MODEL_PATH=<dir>)."
    exit 1
fi
echo "Using CLIP weights: ${CLIP_PATH}"

# ===== HELPERS =====

# Import the rsynced bundles into <DATA_ROOT>/<SUITE>/. Safe to re-run: the importer skips
# bundles whose source_relpath is already in the suite's annotations.
run_manual_check_import() {
    uv run scripts/import_manual_droid_episode.py \
        --bundle_dir "${BUNDLE_DIR}" \
        --data_root "${DATA_ROOT}" \
        --stat_reference_root "${STAT_REFERENCE_ROOT}" \
        --suite "${SUITE}" \
        --split "${SPLIT}" \
        --svd_model_path "${SVD_PATH}" \
        --height 192 --width 320 \
        || { echo "ERROR: import_manual_droid_episode.py failed"; exit 1; }

    # Nothing downstream checks this, and a replay over an empty or stale suite still
    # produces a plausible-looking chunk_metrics.jsonl -- so fail loudly here instead.
    local n_ep
    n_ep=$(ls -1 "${DATA_ROOT}/${SUITE}/annotation/${SPLIT}"/*.json 2>/dev/null | wc -l)
    echo "Suite ${SUITE}/${SPLIT} now has ${n_ep} episode(s)"
    if [ "${n_ep}" -eq 0 ]; then
        echo "ERROR: import produced no episodes in ${DATA_ROOT}/${SUITE}/annotation/${SPLIT}"
        exit 1
    fi
}

# replay_libero_wm_traj.py opens chunk_metrics.jsonl in APPEND mode, so re-submitting against
# the same OUT_DIR would duplicate every prior chunk's rows and skew the per-quadrant
# aggregation. Move any existing file aside (never delete) so each submission starts clean.
rotate_chunk_metrics() {
    mkdir -p "${OUT_DIR}"
    if [ -f "${OUT_DIR}/chunk_metrics.jsonl" ]; then
        mv "${OUT_DIR}/chunk_metrics.jsonl" \
           "${OUT_DIR}/chunk_metrics.$(date +%Y%m%d_%H%M%S).jsonl"
    fi
}

# Groups chunk_metrics.jsonl by the uncertainty_cell each episode's own annotation carries
# (the hand-picked aleatoric/epistemic quadrant -- see replay_libero_wm_traj.py's
# no-manifest episode-list branch), reusing the same aggregate/plot scripts the LIBERO 2x2
# variance/data-scale UQ study uses, via their --cells override.
run_manual_check_aggregation() {
    uv run scripts/aggregate_uq_by_cell.py \
        --chunk_jsonl "${OUT_DIR}/chunk_metrics.jsonl" \
        --cells "${QUADRANT_CELLS}" \
        --output_dir "${OUT_DIR}/uq_by_quadrant" \
        || { echo "ERROR: aggregate_uq_by_cell.py failed"; exit 1; }

    uv run scripts/plot_uq_by_cell.py \
        --csv "${OUT_DIR}/uq_by_quadrant/uq_per_chunk.csv" \
        --cells "${QUADRANT_CELLS}" \
        --output_dir "${OUT_DIR}/uq_by_quadrant/plots" \
        || { echo "ERROR: plot_uq_by_cell.py failed"; exit 1; }
}
