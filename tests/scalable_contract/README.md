# Scalable contract regression tests

This suite validates the thesis-added Base/Full contract: four-stage Unicorn
prefix use, Base independence from ground truth and Enhancement payloads,
strict checkpoint restoration, Base/Full reconstruction consistency, physical
rate accounting, and fail-closed model identity checks.

## Run

```bash
bash scripts/tests/run_scalable_contract_cpu.sh
bash scripts/tests/run_scalable_contract_gpu.sh
```

`run_scalable_contract_cpu.sh` runs regression discovery without requesting
physical GPU coding. `run_scalable_contract_gpu.sh` enables strict single-GPU
integration and accepts an optional fully qualified unittest name in
`SCALABLE_TEST_TARGET`.

## Runtime assumptions

Use a Python environment containing the repository's PyTorch,
MinkowskiEngine, PyTorch3D, NumPy, and codec dependencies. GPU tests require a
CUDA-capable device, a writable codec working directory, and existing local
fixtures supplied through these environment variables:

```text
SCALABLE_TEST_H5
SCALABLE_TEST_GPCC
SCALABLE_RELEASED_2K_R05_8K256_L2048
SCALABLE_2K_RESCUE_U_PATH_STEP500
SCALABLE_2K_ENHANCEMENT_D111_STEP1500
SCALABLE_RELEASED_8K_R03_32K8K_L8192
SCALABLE_8K_BASE_D111_STEP3525
SCALABLE_8K_ENHANCEMENT_D111_STEP1763
SCALABLE_4K_JOINT_FROM_8K_STEP3000
```

The GPU runner sets `SCALABLE_RUN_GPU_TESTS=1` and
`SCALABLE_STRICT_FIXTURES=1`. Fixture paths are machine configuration and are
not stored in the public repository.
