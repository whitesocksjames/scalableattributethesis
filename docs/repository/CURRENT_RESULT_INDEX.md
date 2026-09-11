# Current results — final thesis evidence

The examiner-facing static RGB result authority is
[`formal_static_rgb_final_20260911`](../../results/comparisons/formal_static_rgb_final_20260911/README.md).
It freezes 20 samples, Original `180/180`, Ours `140/140`, and
`320/320 FORMAL_REUSABLE` logical evaluations. Every Ours point has Base and
Full, so the package contains 460 decoded endpoint rows.

| Need | Final artifact |
|---|---|
| Final interpretation and limitations | [FINALIZATION_REPORT.md](../../results/comparisons/formal_static_rgb_final_20260911/report/FINALIZATION_REPORT.md) |
| 20-sample completeness | [completeness.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/completeness.csv) |
| Raw selected RD evidence | [final_raw_rd_points.csv](../../results/comparisons/formal_static_rgb_final_20260911/evidence/final_raw_rd_points.csv) |
| 8i / Owlii / CTC figures | [figures/](../../results/comparisons/formal_static_rgb_final_20260911/figures/) |
| Per-sequence frozen-contract BD-BR | [bdbr_per_sequence.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/bdbr_per_sequence.csv) |
| Dataset BD-BR summaries | [bdbr_dataset_summary.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/bdbr_dataset_summary.csv) |
| Full-minus-Base and negative Enhancement cases | [full_minus_base_all_points.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/full_minus_base_all_points.csv) · [full_below_base_cases.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/full_below_base_cases.csv) |
| Hardware-labeled runtime | [runtime_per_point.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/runtime_per_point.csv) · [runtime_summary_by_hardware.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/runtime_summary_by_hardware.csv) |
| House/Shiva consistency | [house_shiva_consistency.csv](../../results/comparisons/formal_static_rgb_final_20260911/tables/house_shiva_consistency.csv) |

Final numeric invariants are 68/80 available per-sequence BD-BR values, with
CTC dataset summaries explicitly limited to a 9/12 valid-sequence subset;
13/140 Full endpoints have a negative Y or YUV611 Enhancement delta. The worst
retained degradation is `-0.2418 dB Y / -0.18065 dB YUV611`. House and Shiva
use their respective complete A100-SXM4-40GB 16-point sets for final runtime.

The older `formal_rd_competitiveness_20260910` package is a superseded
318-point review. Earlier RWTT/8i/Owlii screening, rescue, and joint-checkpoint
reports are historical selection evidence, not final thesis result authority.
See [HISTORICAL_DOCUMENT_STATUS.md](HISTORICAL_DOCUMENT_STATUS.md).
