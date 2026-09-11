# Formal static-RGB Pareto-frontier BD-BR sensitivity analysis

This package recomputes BD-BR from the frozen 460-row raw physical-RD evidence
using the uniform protocol in [PROTOCOL.md](PROTOCOL.md). It introduces no new
experiment and does not replace the historical strict `68/80` analysis.

The Pareto protocol makes all **80/80** requested comparisons available. The 68
comparisons that were already valid under the strict protocol are reproduced to
within `8.89e-14` percentage points. The 12 newly available comparisons belong
to three CTC samples: `boxer_viewdep_vox12`, `loot_viewdep_vox12`, and
`redandblack_viewdep_vox12`.

## Dataset-level equal-sequence BD-BR

| Dataset | Endpoint | Y | YUV611 | Sequences |
|---|---|---:|---:|---:|
| 8iVFB | Ours Base | +5.204595% | -0.723098% | 4/4 |
| 8iVFB | Ours Full | +1.452161% | +1.390496% | 4/4 |
| Owlii | Ours Base | +10.109716% | +4.992131% | 4/4 |
| Owlii | Ours Full | +1.854754% | +2.077132% | 4/4 |
| CTC | Ours Base | +1.848206% | -0.326300% | 12/12 |
| CTC | Ours Full | +0.155013% | -0.696226% | 12/12 |

Negative values indicate bitrate savings. The 8iVFB and Owlii values are
numerically unchanged from the strict protocol. The CTC values now average all
12 sequences; comparison with the old 9-sequence subset is therefore a change
of population as well as protocol.

| CTC endpoint | Metric | Old strict 9/12 | Pareto 12/12 | Difference (pp) |
|---|---|---:|---:|---:|
| Ours Base | Y | +3.199828% | +1.848206% | -1.351622 |
| Ours Base | YUV611 | +0.924902% | -0.326300% | -1.251202 |
| Ours Full | Y | +0.105696% | +0.155013% | +0.049317 |
| Ours Full | YUV611 | -0.298847% | -0.696226% | -0.397380 |

On the same nine CTC sequences used by the strict subset, the new computation
reproduces every old value. The differences above arise from adding the three
newly resolved sequences to the equal-sequence mean.

## Pareto removals

Twelve of 120 metric-specific curves contain a dominated point. Fourteen
metric-specific point instances are removed for interpolation, corresponding to
seven endpoint operating-point instances because Y and YUV611 are processed
independently:

| Sample and endpoint | Removed | Dominating measured point |
|---|---|---|
| CTC boxer, Ours Base | 4k | 2k |
| CTC boxer, Ours Full | 4k | 2k |
| CTC loot, Original Unicorn | R04 | R02 |
| CTC loot, Original Unicorn | R05 | R03 |
| CTC redandblack, Original Unicorn | R04 | R03 |
| CTC redandblack, Ours Base | 2k | 8k |
| CTC redandblack, Ours Full | 2k | 8k |

No raw point is removed from the source evidence or RD figures.

## Artifacts

- `tables/curve_frontier_audit.csv`: one row for each of the 120 curves, with
  raw/retained counts, removed points, and all measured dominators.
- `evidence/pareto_frontier_points.csv`: point-level retained/dominated audit.
- `tables/bdbr_per_sequence.csv`: all 80 Pareto-frontier comparisons.
- `tables/bdbr_dataset_summary.csv`: equal-sequence dataset means.
- `tables/strict_vs_pareto.csv`: row-wise sensitivity comparison with the
  historical strict analysis.
- `summary.json`: machine-readable availability and removal counts.
