# Formal RD competitiveness review — 2026-09-10

> **HISTORICAL / SUPERSEDED AT 318/320.** House and Shiva were incomplete at
> this audit boundary. Do not quote this package as the final thesis result;
> use [`formal_static_rgb_final_20260911`](../formal_static_rgb_final_20260911/README.md).

This directory is a read-only performance review derived from the then-current
`318/320` `FORMAL_REUSABLE` logical results. It does not modify the frozen
scientific contract, remove any raw operating point, or replace formal result
artifacts. House and Shiva are explicitly treated as pending because each was
still missing its Ours 32K logical point at this audit boundary.

Pipeline integrity is an input assumption from the preceding provenance,
checkpoint, physical-bit, endpoint, metric, and CTC aggregation audit. This
directory evaluates performance rather than repeating that audit.

## Classification

- `report/`: human-readable conclusions, declared diagnostic thresholds, and
  the complete Full-below-Base list.
- `figures/`: per-sample raw physical-bpp RD plots. Every panel includes
  Official Unicorn, Ours Base, and Ours Full; no dominated point is deleted.
- `tables/`: review summaries and complete Full-minus-Base statistics.
- `evidence/`: normalized raw RD rows and the point-level diagnostic comparison
  used for the descriptive labels. Operational source roots are represented by
  non-sensitive provenance aliases rather than machine paths.

## Main entry points

- [`report/REVIEW.md`](report/REVIEW.md): primary interpretation.
- [`figures/8i_raw_rd_curves.svg`](figures/8i_raw_rd_curves.svg)
- [`figures/owlii_raw_rd_curves.svg`](figures/owlii_raw_rd_curves.svg)
- [`figures/ctc_raw_rd_curves.svg`](figures/ctc_raw_rd_curves.svg)
- [`tables/sample_competitiveness.csv`](tables/sample_competitiveness.csv)
- [`tables/dataset_competitiveness.csv`](tables/dataset_competitiveness.csv)
- [`tables/full_minus_base_all_points.csv`](tables/full_minus_base_all_points.csv)
- [`tables/full_below_base_cases.csv`](tables/full_below_base_cases.csv)
- [`tables/operating_point_enhancement_summary.csv`](tables/operating_point_enhancement_summary.csv)
- [`evidence/raw_rd_points.csv`](evidence/raw_rd_points.csv)
- [`evidence/diagnostic_full_vs_official.csv`](evidence/diagnostic_full_vs_official.csv)

## Interpretation boundary

The competitiveness labels use log-bpp linear interpolation through every
Official raw point inside the shared bpp range, with no extrapolation. This is
only a descriptive same-rate comparison and is not the frozen BD-BR procedure.
The observed 4K/8K ordering reversal remains visible in the raw figures but is
not treated as a competitiveness blocker. BD-BR remains governed by the formal
execution plan and is not recomputed here by deleting points or changing the
interpolation contract.
