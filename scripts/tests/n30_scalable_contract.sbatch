#!/usr/bin/env bash
#SBATCH --job-name=scalable_contract
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail
: "${SCALABLE_TEST_SOURCE:?set SCALABLE_TEST_SOURCE}"
: "${SCALABLE_TEST_ENV:?set SCALABLE_TEST_ENV}"
source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2 gcc/11.2 cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scz0ade/Tanzeyu/envs/unicorn-me-py38
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR=/data/run01/scz0ade/Tanzeyu/.cache/torch_extensions
source "$SCALABLE_TEST_ENV"
cd "$SCALABLE_TEST_SOURCE"
bash scripts/tests/run_scalable_contract_gpu.sh
