#!/bin/bash -l
#SBATCH --job-name=scalable_contract
#SBATCH --partition=work,v100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail
: "${SCALABLE_TEST_SOURCE:?set SCALABLE_TEST_SOURCE}"
: "${SCALABLE_TEST_ENV:?set SCALABLE_TEST_ENV}"
module load cuda/11.8.0
export PATH="$WORK/software/private/conda/envs/unicorn-me-py38/bin:$PATH"
export PYTHONNOUSERSITE=1
source "$SCALABLE_TEST_ENV"
cd "$SCALABLE_TEST_SOURCE"
bash scripts/tests/run_scalable_contract_gpu.sh
