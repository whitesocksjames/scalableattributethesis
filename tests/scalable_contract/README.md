# Scalable contract regression tests

These tests cover only thesis-added scalable behavior. They deliberately follow
the upstream Unicorn coding/rate convention: no file container, fresh-process
decode, physical G-PCC decode closure, min/max header accounting, or packaged
byte identity is required.

## Post-cleanup dependency closure

Audited before implementation at Phase 1 commit `72999db4`. Current runtime and
config paths (`scalable_attribute`, `lossy_attribute`, `basic_models`,
`data_utils`, `cfg`, `configs`, and active `scripts`) contain zero imports or
launcher references to the archived HPC sweep wrappers or fixed-model trace
bundle. Two strings in `lossy_attribute/README.md` are upstream historical
commands for archived lossless/dynamic modes; they are neither runtime nor
configuration dependencies and were intentionally left unchanged.

## Split

```bash
bash scripts/tests/run_scalable_contract_cpu.sh
bash scripts/tests/run_scalable_contract_gpu.sh
```

The suite uses Python's standard-library `unittest`; no test package is installed
on either cluster. The strict GPU command requires the variables documented in
`configs/test_environments/n30.env.example` or `fau_hpc.env.example`. Paths are
machine configuration, not model provenance. Hashes are optional supporting
evidence; the gates primarily check checkpoint metadata, architecture, lambda,
step, strict state restoration and actual reconstruction/rate contracts.

The 4K fixture names encode its real provenance: released R03 `32k8k` at lambda
8192 and selected 8K Base bootstrap, followed by target-lambda 4096 joint
checkpoint step3000. There is intentionally no `SCALABLE_RELEASED_4K` fixture.

The real-checkpoint gates require 2K `canonical_base_rescue_v1`, lambda2048,
step500 and complete rescued Prefix+BaseSynthesis restoration. The 4K gate
requires complete joint state, profile `32k8k`, source lambda8192, target
lambda4096 and step3000. All state loads are strict: missing or unexpected keys
fail. Optional externally recorded SHA256 values are provenance aids, not fixture
availability gates.

Each GPU test uses a new codec working directory. Cross-GPU floating-point or
bitstream byte equality is not required; encode/decode equality is checked
within each run after coordinate alignment.
