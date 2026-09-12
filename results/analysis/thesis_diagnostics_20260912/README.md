# Thesis Diagnostic Analysis

This lightweight, post-freeze package contains the final analysis of two thesis
questions: non-monotonic Base-to-Full refinement and the diagnostic 256
low-rate boundary. It does not alter the publication-frozen seven-point formal
curve, checkpoints, metrics, or BD-BR results on `main`.

## Conclusions

- Q3 analyzes 29/29 selected negative/control cases. Twenty-eight reproduce the
  frozen endpoint exactly; redandblack 8K is explicitly labeled bounded
  runtime/hardware sensitivity evidence.
- The historical sequential 256 Enhancement reaches the observed one-byte per
  independently coded input floor. Its refinement is unstable, while the 512
  control carries non-trivial rate and improves YUV611 on 19/20 samples.

Read [THESIS_ANALYSIS_SUMMARY.md](THESIS_ANALYSIS_SUMMARY.md) first. Detailed
interpretation is in [Q3_FINAL_ANALYSIS.md](Q3_FINAL_ANALYSIS.md) and
[CTC256_ANALYSIS.md](CTC256_ANALYSIS.md).

`tables/` contains the compact-ready summaries and complete lightweight audit
rows. `figures/` contains the final SVG figures. `scripts/` contains only
analysis and verification code; GPU runners and infrastructure scripts are
intentionally excluded.

## Reproduction

From the repository root, using an environment with Python, pandas, matplotlib,
and pytest:

```bash
python results/analysis/thesis_diagnostics_20260912/scripts/build_metric_analysis.py
python results/analysis/thesis_diagnostics_20260912/scripts/verify_package.py
```

The first command rebuilds the 140-point Base/Full metric summaries from the
frozen final evidence on `main`. The second validates the Q3 and 256 package
invariants and refreshes `SHA256SUMS`.
