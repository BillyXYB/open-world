#!/usr/bin/env bash
# Install script for open-world on della (Princeton HPC)
# Run from the open-world project root:
#   cd /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world
#   bash /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/yanbo_utils/env_management/della/local_install.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
ENV_NAME=open-world
BASE_UTILS_DIR=/scratch/gpfs/AM43/yx2653/utilities

export HF_HOME=$BASE_UTILS_DIR/huggingface
export TORCH_HOME=$BASE_UTILS_DIR/torch_hub
export TORCH_HUB_ROOT=$TORCH_HOME
export PIP_CACHE_DIR=$BASE_UTILS_DIR/cache/pip
export CUDA_HOME=${CONDA_PREFIX:-""}
export UV_CACHE_DIR=$BASE_UTILS_DIR/cache/uv
export UV_ENV_DIR=$BASE_UTILS_DIR/envs
export XDG_CACHE_HOME=$BASE_UTILS_DIR/cache

# ---------------------------------------------------------------------------
# Source UV (install if missing)
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "UV not found — installing into ${BASE_UTILS_DIR}/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${BASE_UTILS_DIR}/bin" sh
fi
source "${BASE_UTILS_DIR}/bin/env"

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
uv venv "$UV_ENV_DIR/$ENV_NAME" --python 3.11

# ---------------------------------------------------------------------------
# Clone openpi (required for OpenPI policy)
# ---------------------------------------------------------------------------
if [ ! -d "external/openpi/.git" ]; then
    echo "Cloning openpi ..."
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git external/openpi
else
    echo "external/openpi already present, skipping clone."
fi

# ---------------------------------------------------------------------------
# Initialize LIBERO submodule (editable install required before uv sync)
# ---------------------------------------------------------------------------
git -C external/openpi submodule update --init third_party/libero

# ---------------------------------------------------------------------------
# Install open-world with policy-openpi + libero extras into the named venv
# ---------------------------------------------------------------------------
echo "Running: uv sync --extra policy-openpi --extra libero"
UV_PROJECT_ENVIRONMENT="$UV_ENV_DIR/$ENV_NAME" uv sync --extra policy-openpi --extra libero

# ---------------------------------------------------------------------------
# Download model assets (CLIP, SVD backbone, checkpoints, benchmark)
# ---------------------------------------------------------------------------
# echo "Downloading model assets ..."
# bash external/download_models.sh

# ---------------------------------------------------------------------------
# Start wandb offline sync daemon
# ---------------------------------------------------------------------------
mkdir -p logs
nohup wandb-osh > logs/wandb_osh.log 2>&1 &
echo "wandb-osh started (logs/wandb_osh.log)"

echo ""
echo "Done. Activate the environment with:"
echo "  source \$UV_ENV_DIR/$ENV_NAME/bin/activate"
