# Formal static RGB finalization — 2026-09-11

This directory freezes the read-only final audit of the complete 20-sample
`8iVFB + Owlii + CTC` matrix. It contains 320 logical points: 180 pinned
Official Unicorn points and 140 Ours points, with both Base and Full endpoints
for every Ours point.

Ours is described as a **full-resolution quality-scalable extension of Unicorn
Part II**: `native truncated Unicorn prefix → learned BaseSynthesis →
full-resolution Base → conditional Enhancement → Full`. The package does not
claim that Unicorn lacks progressive decoding, that layered coding is newly
invented, or that Ours uniformly outperforms Unicorn.

The selected final set uses current-formal RTX3090 results except for Facade,
House, and Shiva on A100-SXM4-40GB. House and Shiva use their complete 16-point
A100 runs so their within-sample Original/Ours runtime comparisons do not mix
GPU classes. The retained older runs remain comparison evidence and are not
silently overwritten.

## Contract

- Physical bpp and Y/U/V/YUV611 are evaluated under the frozen formal metric
  contract.
- CTC uses the frozen canonical source and shared at-most-800k KD-tree chunks,
  global coordinates, and no recentering, requantization, or deduplication.
- BD-BR is computed per sequence as PCHIP of `ln(physical bpp)` against quality
  over the shared quality interval. No extrapolation or result-dependent point
  removal is allowed. A nonmonotonic raw rate-quality curve is reported as
  `UNAVAILABLE`.
- Dataset BD-BR is an equal-sequence mean over explicitly available sequences;
  a subset mean is labeled `AVAILABLE_SUBSET`.
- Runtime is CUDA-synchronized codec time. Tables keep GPU model and VRAM as
  grouping keys; they do not average unlike hardware together.

See [FINALIZATION_REPORT.md](report/FINALIZATION_REPORT.md) for conclusions.
