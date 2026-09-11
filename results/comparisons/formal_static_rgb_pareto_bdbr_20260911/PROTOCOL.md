# Uniform Pareto-frontier BD-BR protocol

This sensitivity analysis uses the complete measured physical-rate RD evidence
from `formal_static_rgb_final_20260911`. It does not modify or replace the raw
evidence or the earlier strict `68/80` analysis.

For every `(dataset, sample, method endpoint, metric)` curve independently:

1. Start from every measured `(physical_bpp, quality)` point. Y and YUV611 are
   processed independently.
2. Mark a point as dominated when another measured point has no higher bitrate
   and no lower quality, with at least one strict improvement.
3. Retain all and only Pareto-efficient points for BD-BR interpolation. Apply
   this identical rule to Original Unicorn, Ours Base, and Ours Full.
4. Sort the retained frontier by increasing quality. The resulting rates must
   be strictly increasing; otherwise that curve fails closed.
5. Interpolate `ln(physical_bpp)` as a function of quality with shape-preserving
   piecewise cubic Hermite interpolation (PCHIP).
6. Integrate only over the non-empty shared quality interval of the compared
   frontier pair. Do not extrapolate.
7. Compute BD-BR as `100 * (exp(mean_log_rate_difference) - 1)`, where Ours is
   the test curve and Original Unicorn is the reference curve. Negative values
   indicate bitrate savings.
8. Compute dataset summaries as equal-sequence arithmetic means. No manual
   point removal or result-dependent exception is allowed.

All raw points remain present in the frozen evidence and RD figures. The
frontier audit records every removed operating point and every measured point
that dominates it.
