# Multi-rate Candidate Data Inventory — 2026-09-04

> **HISTORICAL / SUPERSEDED CANDIDATE INVENTORY.** This captures the 2026-09-04
> selection boundary. Use `docs/repository/CURRENT_OPERATING_POINTS.md` and
> `configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json` for the
> frozen seven-point mapping.

## Current working candidate mapping

| Point | Base candidate | Full candidate | Status |
|---|---|---|---|
| 32K | D111 Base `step5525` | sequential D111→D611 `step3525` | existing mainline |
| 16K | canonical Base `step3525` | D111 Enhancement `step1763` | working candidate; D611 Stage-2 is diagnostic, not selected |
| 8K | canonical Base `step3525` | D111 Enhancement `step1763` | working candidate |
| 4K | 8K→4K TAFA joint `step3000 Base` | same joint checkpoint `step3000 Full` | manager-provisional |
| 2K | rescued `2K-U-PATH step500` | frozen rescued Base + D111 Enhancement `step1500` | manager-frozen |
| 1K | 2k128 Base `step3525` | D111 Enhancement `step1763` | diagnostic candidate |
| 512 | 2k128 Base `step3525` | D111 Enhancement `step1763` | diagnostic candidate |
| 256 | 2k128 Base `step3525` | D111 Enhancement `step1763` | diagnostic candidate |

The repository operating-point JSON is stale for 4K, 2K, 1K, 512, and 256. It must not be treated as the authority for these manager selections until separately updated and reviewed.

## Evaluation coverage

`PASS` means that matching physical hard Base and Full evidence exists for the current working candidate. `MISSING` means a new evaluation is required; historical evidence from an obsolete candidate is not counted.

| Point | RWTT-28Lite Base+Full | 8i fixed-4 Base+Full | Owlii fixed-4 Base+Full | Note |
|---|---:|---:|---:|---|
| 32K | PASS | PASS | PASS | D611 Full evidence retained |
| 16K | PASS | PASS | PASS | Uses D111 Stage-1 Full; D611 Stage-2 external evidence is not required unless promoted |
| 8K | PASS | PASS | PASS | D111 Stage-1 Full |
| 4K | **MISSING** | PASS | PASS | Existing RWTT result belongs to the old Base, not joint step3000 |
| 2K | PASS (on N30) | PASS | PASS | Current rescued Base + Enhancement step1500 |
| 1K | PASS | PASS | PASS | D111 Stage-1 step1763 |
| 512 | PASS | PASS | PASS | D111 Stage-1 step1763 |
| 256 | PASS | PASS | PASS | D111 Stage-1 step1763 |

## Minimum missing evaluation

Exactly one new hard evaluation is needed for a complete three-contract summary:

```text
Dataset: RWTT-28Lite frozen authoritative manifest
Checkpoint: 8K→4K TAFA joint step3000
Endpoints: Base + Full
Profile: 32k8k
Conditioning lambda: 4096
Rate: physical hard x_low+r1-r4 + Enhancement
Correctness: --require-exact, native r5=0, Full_bits=Base_bits+Enhancement_bits
```

No training, 8i rerun, Owlii rerun, Full28, or other checkpoint evaluation is needed to make the requested 256–32K comparison.

## Plot readiness

- 8i: ready now for eight per-sequence/aggregate curves against the retained official R01–R09 reference.
- Owlii: ready now for eight per-sequence/aggregate curves against the retained author-provided R01–R09 reference.
- RWTT-28Lite: blocked only by current 4K joint step3000. All other selected points are available.

Official interpolation is diagnostic only and must use adjacent points without extrapolation. Cross-sequence mean bpp/PSNR points should not replace the per-sequence figures.
