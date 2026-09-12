# Non-monotonic Base-to-Full refinement — final private analysis

This is post-freeze diagnostic evidence. It does not modify the formal results,
the seven-point curve, checkpoint selection, or BD-BR analysis.

## Evidence closure

- 29/29 selected negative/control cases have pre-quantization D111 and
  residual/correction diagnostics.
- 28/29 reproduce frozen physical bits, reconstruction SHA, and Y/U/V/YUV611
  exactly.
- `redandblack_viewdep_vox12` 8K is retained only as bounded sensitivity
  evidence: Base and Full are each 16 bits below frozen, Enhancement bits are
  exact, and all metric deviations are at most 0.0003 dB.
- The important `redandblack_viewdep_vox12` 2K negative case is exact. Its
  YUV611 change is -0.14156 dB, D111 worsens, residual/correction cosine is
  -0.04687, and only 40.94% of points improve under weighted squared error.

## What the 13 negative cases show

| Diagnostic observation | Negative cases | Matched controls |
| --- | ---: | ---: |
| D111 itself worsens after Enhancement | 11/13 | 0/16 |
| D111 improves but YUV611 worsens | 2/13 | 0/16 |
| Residual/correction cosine <= 0.1 | 13/13 | 9/16 |
| Non-positive cosine | 5/13 | 0/16 |
| Fewer than half of points improve | 12/13 | 8/16 |
| Channel trade-off | 12/13 | 11/16 |
| Tiny Enhancement rate (<=0.001 bpp) | 6/13 | 4/16 |

The two cases where D111 improves while YUV611 worsens are boxer 4K and
redandblack 512. They are metric/channel trade-offs rather than failures under
the checkpoint's D111 training-domain objective.

## Supported interpretation

Negative Base-to-Full refinement is not explained by one universal failure
mode. For 11/13 negatives, Enhancement also increases the actual D111
distortion, and these cases show weak or adverse alignment with the oracle Base
residual. For the other 2/13, D111 improves but channel weighting causes
YUV611 to decrease. Low Enhancement rate is associated with some cases but is
neither necessary nor sufficient by itself.

The matched controls are important: none worsens D111, although weak global
cosine, pointwise heterogeneity, and channel trade-offs can still occur in
positive cases. Therefore no single scalar diagnostic alone establishes the
cause. The evidence supports content- and operating-point-dependent correction
quality under an objective with no explicit per-sample monotonic constraint.

## Thesis-safe wording

> The Enhancement layer is not guaranteed to improve every individual sample.
> Among the 13 measured negative YUV611 cases, 11 also increased the
> checkpoint's D111 distortion and exhibited weak or adverse alignment with the
> remaining Base residual. The other two improved D111 but lost YUV611 through
> channel trade-offs. These observations establish non-monotonic refinement as
> a content- and rate-dependent limitation; they do not identify a universal
> training failure mechanism.

## Claims to avoid

- Do not claim that the absence of a monotonic loss is the proven cause.
- Do not attribute every negative case to low Enhancement bitrate.
- Do not claim residual/correction cosine alone predicts endpoint quality.
- Do not describe the 29-case diagnostic subset as a new formal benchmark.
- Do not call redandblack 8K an exact frozen reproduction.

## Artifacts

- `tables/q3_case_diagnostics.csv`
- `tables/negative_case_matched_controls.csv`
