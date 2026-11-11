# Handover validation

Validation date: 2026-11-11. Original research workspace, original checkpoints, and frozen thesis measurements are preserved. All inference checks below use the bundled 512-point synthetic RGB example; they establish export/runtime equivalence on that input, rather than replacing the 20-sample formal evaluation.

| Check | Result |
| --- | --- |
| Seven source checkpoint sets | All required source file sizes and SHA-256 identities match the frozen manifest |
| Exported tensor identity | 7/7 files; all 318 tensors per file retain exact values, dtype, and shape |
| Self-contained strict loading | 7/7 models load on CPU with no original weight-file arguments |
| Original versus exported inference | 7/7 models produce identical Base and Full features, coordinates, physical bits, and EL bytes on the example |
| Base independence | Geometry-only input and Base payload decode exactly without original RGB or EL payload |
| Full reuse | Enhancement decoding uses the independently decoded Base state and matches the original Full result |
| README quick test | 8k reconstruction, payload reload, physical rates, and `pc_error` metrics pass on CPU |
| Chunk workflow | Bundled pinned partition source, chunk preparation, chunk inference, union validation, and once-after-merge metrics pass; demo bits match ordinary inference |
| Local regression suite | 26 tests pass; preprocessing, aggregation, and metric parsing |
| Frozen result preservation | All 7 retained CSVs and 3 SVGs are byte-identical to their frozen sources |
| Result recomputation | Per-sequence BD-BR, dataset BD-BR, and frontier-audit CSVs regenerate byte-identically |
| RD redraw | All 460 measured endpoint rows are accepted and redrawn; the drawing script uses a different layout from the preserved frozen SVGs |
| Runtime import closure and documentation | Required internal imports resolve; syntax and local documentation links checked |

Machine-readable evidence is in [validation/](validation/): strict loading, physical parity, quick-test metrics, chunk-demo metrics, and frozen-file identities. The CPU environment uses Python 3.8.20, PyTorch 1.13.1+cu117, MinkowskiEngine 0.5.4, torchac 0.9.3, and NumPy 1.24.4. The local preprocessing/aggregation tests used Python 3.13.13, NumPy 2.5.4, and pytest 8.4.2; pinned runtime installation instructions target Python 3.8.

GPU equivalence, a clean installation on a new machine, and the full 20-sample matrix were not run during cleanup. The original frozen GPU results remain the thesis measurements. Model assets are prepared privately and are not uploaded to a public Release. Exact included tool hashes and third-party notices are in [THIRD_PARTY.md](../THIRD_PARTY.md); an explicit redistribution license for the upstream Unicorn code/weights remains unverified.
