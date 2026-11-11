# Final evaluation artifacts

This directory contains the frozen final evidence for the 20-sample static-RGB evaluation. The 460 raw rows comprise 180 Original Unicorn points and 280 scalable endpoint rows (140 operating points, each with Base and Full endpoints).

## Key results

Base and Full BD-BR versus Original Unicorn:

| Dataset | Base Y | Base YUV611 | Full Y | Full YUV611 |
|---|---:|---:|---:|---:|
| 8iVFB | +5.205% | -0.723% | +1.452% | +1.390% |
| Owlii | +10.110% | +4.992% | +1.855% | +2.077% |
| CTC | +1.848% | -0.326% | +0.155% | -0.696% |

Negative values indicate bitrate savings. Values are equal-sequence means of per-sample BD-BR computed with the Pareto-frontier PCHIP protocol described in [Evaluation](../../docs/EVALUATION.md).

## Files

- `evidence/raw_rd_points.csv`: all measured physical-rate and quality values, reconstruction hashes, and result-set identifiers.
- `evidence/pareto_frontier_points.csv`: point-level retained/dominated frontier audit.
- `tables/bdbr_per_sequence.csv`: Base and Full BD-BR per sample and metric, with availability status and shared quality intervals.
- `tables/bdbr_dataset_summary.csv`: equal-sequence dataset means.
- `tables/curve_frontier_audit.csv`: raw and retained point counts and dominated-point records for every curve.
- `tables/runtime_per_point.csv`: encode/decode timing with GPU model, VRAM, and timing semantics on each row.
- `tables/runtime_summary_by_hardware.csv`: runtime summaries grouped by GPU model and VRAM.
- `figures/8i_raw_rd_curves.svg`, `figures/owlii_raw_rd_curves.svg`, `figures/ctc_raw_rd_curves.svg`: raw RD plots showing every measured point.

All CSV and SVG artifacts are byte-preserved copies of the frozen source bundles.

The frozen numeric and figure files have [SHA-256 identities](SHA256SUMS). Run `sha256sum -c SHA256SUMS` from this directory to verify them.
