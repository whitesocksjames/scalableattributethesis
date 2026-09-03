#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --job-name=base_rescue_eval

set -euo pipefail
: "${RESCUE_EVAL_COMMAND:?RESCUE_EVAL_COMMAND is required}"

source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2
module load gcc/11.2
module load cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scz0ade/Tanzeyu/envs/unicorn-me-py38
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=/data/run01/scz0ade/Tanzeyu/.cache/torch_extensions
export LD_LIBRARY_PATH=/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=6

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
exec bash -lc "${RESCUE_EVAL_COMMAND}"
