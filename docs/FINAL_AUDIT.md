# Final self-audit — PASS WITH LIMITATIONS

2026-11-11. Audited base: `09ad0cc`, branch `handover/minimal-supervisor-20261111`. The fixes preserve all model tensors and frozen thesis measurements. This is a pretrained inference/evaluation handover, not a training-history release.

| Area | Executed evidence |
| --- | --- |
| Code/dependencies | All retained Python parses; internal imports and local documentation links resolve; no embedded private HPC paths found. |
| Checkpoints | All seven model file sizes and SHA-256 values rechecked; 318 tensors per file; all seven strictly load in the declared CPU runtime. |
| CPU tests | Local suite: 28 passed, 1 skipped because local torch/torchac are absent. The complete declared runtime additionally runs all 5 codec tests successfully, including a fresh entropy decoder. |
| GPU | RTX 3080, PyTorch 1.13.1+cu117; 8k, 2k, 4k each strictly load, match the original-stage model exactly on the same GPU, retain physical bits/EL bytes, and pass encoder/decoder consistency. Other four points were not GPU-tested in this audit. |
| Independent decoding | A separate CPU process at 8k and a separate GPU process at all three selected points decode Base without original RGB or EL. Full reuses that decoded Base state. Reconstructed coordinate/RGB tuples match saved output exactly. |
| Clean source quick test | A new extraction, including a path containing spaces, runs the README command with a locally supplied verified checkpoint and no inherited research-source PYTHONPATH. Both reconstructions, rates and `pc_error` metrics are produced. Existing declared dependencies are used; a completely fresh dependency installation was not performed. |
| Scientific results | All 460 endpoint rows / 20 samples remain; all 7 CSVs and 3 SVGs retain frozen hashes. Pareto/PCHIP reproduces 80/80 available comparisons; RD plots regenerate from the complete data. Redraw layout differs from the preserved frozen SVGs. |

The test input is the bundled 512-point synthetic example. GPU job `1871865` completed successfully. No training or full formal evaluation was submitted. [Machine-readable evidence](validation/final_audit.json) records the actual checks, device observations, model statuses and tested source hashes.

**Fixed findings**

- `decode_base` now rejects a geometry stride different from the payload and an incomplete four-stage residual payload.
- Entropy CDF construction now derives channel dimensions from its prior tensor. Previously it read a temporary field initialized only by `compress`, so a fresh decoder process failed. The new implementation preserves the same CDF dimensions for valid existing streams and validates the explicit decoder channel count. New-process decoding of payloads generated before this fix succeeds.
- Added omitted MERL/dmetric and TMC13 dependency notices; documented the specific final 2k/4k restoration behavior without restoring research history.

**Cross-device limitation**

| Point | CPU Base / Full bits | GPU Base / Full bits | Max absolute Base / Full feature difference |
| --- | ---: | ---: | ---: |
| 8k | 208 / 680 | 208 / 688 | 0.004515 / 0.019249 |
| 2k | 144 / 264 | 144 / 256 | 0.021504 / 0.059339 |
| 4k | 224 / 504 | 224 / 496 | 0.012325 / 0.031865 |

These are measured backend differences, not certified negligible round-off. Their detailed numerical/quantization cause was not isolated. The same-backend original/export and encoder/decoder checks remain exact, so these observations do not identify a cleanup/export codec regression. CPU/GPU bit-exact equivalence and cross-backend payload interchange are **not certified**. Keep encoding and decoding on the same backend, and use the original GPU protocol when comparing with thesis measurements. No tolerance was relaxed and no frozen value was changed.

**Distribution and handover decision**

Technical supervisor review in the existing controlled research setting is recommended for the tested runtime and scope. Fresh environment installation, untested GPU/device combinations and cross-backend bit-exact reproduction remain limitations. Further distribution of copied Unicorn source and derived weights remains blocked by an unverified redistribution grant; private sharing does not itself establish permission. [THIRD_PARTY.md](../THIRD_PARTY.md) lists the verified notices and actionable alternatives: author-owned results-only delivery, authorized controlled execution, or independent authorized upstream acquisition and local export. No public Release, main merge, force push or weight upload was performed.

Current delivery: **84 files, 6.36 MB** of source/tools/results, excluding Git history, caches and the separately stored seven models (952,005,347 bytes). The only added files are necessary dependency notices and this audit summary/evidence.
