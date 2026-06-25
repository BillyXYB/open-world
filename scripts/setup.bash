# pre-run

## Please update these export configs.

# environment name
ENV_NAME=open-world

# base utilities directory
BASE_UTILS_DIR=/scratch/gpfs/AM43/yx2653/utilities

# export huggingface_hub home directory
export HF_HOME=$BASE_UTILS_DIR/huggingface

# export Torch hub home directory
export TORCH_HOME=$BASE_UTILS_DIR/torch_hub
export TORCH_HUB_ROOT=$TORCH_HOME

# export PIP cache dir
export PIP_CACHE_DIR=$BASE_UTILS_DIR/cache/pip

# export the CUDA HOME PATH
export CUDA_HOME=$CONDA_PREFIX

# export cache directory for UV package manager
export UV_CACHE_DIR=$BASE_UTILS_DIR/cache/uv
# alternatively
export XDG_CACHE_HOME=$BASE_UTILS_DIR/cache

# activate virtual environment
export UV_PROJECT_ENVIRONMENT=${BASE_UTILS_DIR}/envs/${ENV_NAME}
source ${BASE_UTILS_DIR}/envs/${ENV_NAME}/bin/activate
