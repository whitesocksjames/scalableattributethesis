# Formal static RGB finalization report — 2026-09-11

## Matrix closure

The final matrix is **320/320 FORMAL_REUSABLE**: 180/180 Official Unicorn and
140/140 Ours logical points across 20/20 samples. Every Ours logical point has a
valid Base and Full endpoint, yielding 460 endpoint rows in total. The 8i
Original rows use the current-formal runtime-completion artifacts; their rerun
does not add logical points to the 320-point matrix.

## House and Shiva consistency

The complete final House and Shiva sets each contain Official R01-R09 and Ours
512/1k/2k/4k/8k/16k/32k on one A100-SXM4-40GB class. All comparisons with
retained runs match checkpoint identity and chunk/source identity.

For House Ours, the six historically shared operating points (12 Base/Full
endpoints) are exact across the old and memory-optimized evaluators: physical
bits, reconstruction SHA-256, and Y/U/V/YUV611 all match. This is the strongest
same-hardware semantic-equivalence check of the memory optimization.

Cross-GPU comparisons are numerically stable but not byte-identical. House
Official V100-to-A100 differences are at most 24 bits (0.001738%) and 0.0001 dB.
Shiva RTX3090-to-A100 differences are at most 56 bits (0.002172%) and 0.0004 dB;
V100-to-A100 differences are at most 16 bits (0.000748%) and 0.0002 dB. Many
reconstruction file hashes therefore differ across GPU classes. These are
bounded hardware-numerical differences, not provenance mismatches; the final
within-sample comparisons use only the complete A100 sets.

## RD competitiveness

Using raw points without pruning, the descriptive review classifies 19/20
samples as `competitive`, Facade as `mixed`, and none as `clearly weaker`.
8i is 4/4 competitive, Owlii 4/4, and CTC 11/12 competitive plus 1 mixed.

The frozen BD-BR calculation gives Ours Full versus Official equal-sequence
means of:

| Dataset | Y BD-BR | YUV611 BD-BR | Availability |
|---|---:|---:|---|
| 8i | +1.452% | +1.390% | 4/4 sequences |
| Owlii | +1.855% | +2.077% | 4/4 sequences |
| CTC | +0.106% | -0.299% | 9/12 sequences |

Negative values mean bitrate savings. The result supports a **competitive but
not uniformly superior** conclusion. The CTC means are subset means and must not
be presented as complete 12-sequence BD-BR.

Of 80 requested per-sequence values (`20 samples × Base/Full × Y/YUV611`), 68
are available and 12 are unavailable. Boxer contributes four unavailable values
because the Ours rate-quality curve is nonmonotonic. CTC loot and redandblack
contribute four each because the Official curve is nonmonotonic. No point was
deleted and no extrapolation was performed.

The Full-minus-Base table contains **13/140 Full endpoints** with a negative Y
or YUV611 Enhancement delta. Eight have negative Y and all 13 have negative
YUV611. The largest observed decrease is -0.2418 dB Y and -0.18065 dB YUV611 at
CTC redandblack 1k; these real performance regressions remain in all figures
and calculations.

## Runtime

All 460 endpoints have positive encode and decode fields under the current
CUDA-synchronized codec-only timing semantics. Official Enc is the physical
`model.forward(..., encode=True)` interval and Official Dec is `model.decode`.
Ours Base Enc is `prefix_encode`; Base Dec is `prefix_decode +
base_synthesis`; Full Enc is `prefix_encode + enhancement_encode`; Full Dec is
`prefix_decode + base_synthesis + enhancement_decode`. Input reading,
checkpoint loading, PLY writing, `pc_error`, and CPU validation copies are
outside these timers. Runtime summaries are separated by dataset, endpoint,
GPU model, and VRAM.

For the complete A10040 sets, median total codec times are:

| Sample | Official | Ours Base | Ours Full |
|---|---:|---:|---:|
| House | 51.435 s | 33.464 s | 59.307 s |
| Shiva | 18.624 s | 15.887 s | 22.749 s |

These medians are sample-internal summaries only. The hardware-labeled table is
the authoritative source for broader reporting; RTX3090 and A100 timing must
not be averaged without an explicit hardware stratification.

## Remaining thesis limitations

- Twelve frozen-contract BD-BR entries are unavailable due to nonmonotonic raw
  curves; reporting them would require changing the declared contract.
- The descriptive competitiveness label is threshold-based and not a formal
  significance test.
- CTC BD-BR dataset means cover 9/12 valid sequences only.
- Cross-GPU bitstreams/reconstruction files can differ at a tiny numerical
  scale even with exact source, chunk, and checkpoint provenance. Hardware must
  remain attached to runtime and reproducibility claims.
- Full can slightly degrade quality at some points; the largest retained cases
  are reported rather than removed.

## Artifact index

- `tables/completeness.csv`: 20 samples plus total closure.
- `tables/sample_competitiveness.csv`, `tables/dataset_competitiveness.csv`:
  descriptive raw-curve verdicts.
- `tables/bdbr_per_sequence.csv`, `tables/bdbr_dataset_summary.csv`: frozen
  PCHIP BD-BR and explicit unavailable reasons.
- `tables/runtime_per_point.csv`, `tables/runtime_summary_by_hardware.csv`:
  complete hardware-labeled runtime data.
- `tables/house_shiva_consistency.csv`: endpoint-level historical comparison.
- `tables/full_minus_base_all_points.csv`, `tables/full_below_base_cases.csv`:
  enhancement deltas and all negative cases.
- `figures/*_raw_rd_curves.svg`: 20-sample Official/Base/Full physical RD plots.
- `evidence/final_raw_rd_points.csv`: normalized final RD evidence without
  machine-private paths.
- `evidence/house_shiva_consistency_summary.json`: grouped consistency bounds.
