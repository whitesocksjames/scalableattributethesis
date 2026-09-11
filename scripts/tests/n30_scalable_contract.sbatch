#!/usr/bin/env bash
#SBATCH --job-name=scalable_contract
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail
job_script_started="$(date +%s)"
: "${SCALABLE_TEST_SOURCE:?set SCALABLE_TEST_SOURCE}"
: "${SCALABLE_TEST_ENV:?set SCALABLE_TEST_ENV}"
environment_started="$(date +%s)"
source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2 gcc/11.2 cuda/11.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /REDACTED/N30_PROJECT_ROOT/envs/unicorn-me-py38
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR=/REDACTED/N30_PROJECT_ROOT/.cache/torch_extensions
source "$SCALABLE_TEST_ENV"
cd "$SCALABLE_TEST_SOURCE"
environment_finished="$(date +%s)"
echo "SCALABLE_CONTRACT_JOB_TIMING {\"environment_setup_seconds\":$((environment_finished - environment_started))}"
set +e
bash scripts/tests/run_scalable_contract_gpu.sh
status=$?
set -e
job_script_finished="$(date +%s)"
echo "SCALABLE_CONTRACT_JOB_TIMING {\"total_job_script_walltime_seconds\":$((job_script_finished - job_script_started)),\"exit_code\":${status}}"
exit "$status"
