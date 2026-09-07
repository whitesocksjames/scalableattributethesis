#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
export SCALABLE_RUN_GPU_TESTS=1
export SCALABLE_STRICT_FIXTURES=1
test_target="${SCALABLE_TEST_TARGET:-tests.scalable_contract.test_gpu_integration.GPUIntegrationTests}"
runner_started="$(date +%s)"
set +e
python -m unittest -v "$test_target"
status=$?
set -e
runner_finished="$(date +%s)"
echo "SCALABLE_CONTRACT_RUNNER_TIMING {\"total_unittest_walltime_seconds\":$((runner_finished - runner_started)),\"exit_code\":${status}}"
exit "$status"
