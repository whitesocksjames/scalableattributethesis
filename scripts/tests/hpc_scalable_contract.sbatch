#!/bin/bash -l
#SBATCH --job-name=scalable_contract
#SBATCH --partition=work,v100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail
job_script_started="$(date +%s)"
: "${SCALABLE_TEST_SOURCE:?set SCALABLE_TEST_SOURCE}"
: "${SCALABLE_TEST_ENV:?set SCALABLE_TEST_ENV}"
environment_started="$(date +%s)"
module load cuda/11.8.0
export PATH="$WORK/software/private/conda/envs/unicorn-me-py38/bin:$PATH"
export PYTHONNOUSERSITE=1
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
