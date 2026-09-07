#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
export SCALABLE_RUN_GPU_TESTS=1
export SCALABLE_STRICT_FIXTURES=1
python -m unittest -v \
  tests.scalable_contract.test_gpu_integration.GPUIntegrationTests
