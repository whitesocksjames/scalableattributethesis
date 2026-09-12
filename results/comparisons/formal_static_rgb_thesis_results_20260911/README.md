# Final Static-RGB Thesis Results

This page presents the final evaluation of a **full-resolution
quality-scalable extension of Unicorn Part II**. The evaluation covers 20
samples: four from 8iVFB, four from Owlii, and twelve from CTC. Each sample has
nine Original Unicorn operating points and seven scalable operating points;
each scalable point yields a Base endpoint and a Full endpoint.

## Dataset-level BD-BR versus Original Unicorn

| Dataset | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| 8iVFB | +5.205% | -0.723% | +1.452% | +1.390% |
| Owlii | +10.110% | +4.992% | +1.855% | +2.077% |
| CTC | +1.848% | -0.326% | +0.155% | -0.696% |

## Per-sequence BD-BR versus Original Unicorn

All values are percentages. Negative BD-BR represents bitrate saving relative
to Original Unicorn at equivalent quality. Average rows are read directly from
the frozen equal-sequence dataset summary; displayed values use three decimal
places, while the [compact table source](bdbr_per_sequence_compact.csv) retains
the complete stored precision.

### 8iVFB

| Sample | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| `longdress` | +21.969% | +11.449% | +0.573% | +1.795% |
| `loot` | +0.169% | -5.259% | +2.511% | +1.052% |
| `redandblack` | -4.879% | -9.229% | -0.575% | -0.640% |
| `soldier` | +3.559% | +0.147% | +3.300% | +3.356% |
| **Average** | **+5.205%** | **-0.723%** | **+1.452%** | **+1.390%** |

### Owlii

| Sample | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| `basketball_player` | +6.224% | +2.506% | +1.925% | +1.676% |
| `dancer` | +14.898% | +9.356% | +3.500% | +3.970% |
| `exercise` | +6.435% | +1.495% | -0.569% | -0.999% |
| `model` | +12.882% | +6.611% | +2.563% | +3.662% |
| **Average** | **+10.110%** | **+4.992%** | **+1.855%** | **+2.077%** |

### CTC

| Sample | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| `basketball_player_vox11_00000200` | +1.928% | -0.738% | -3.560% | -3.798% |
| `dancer_vox11_00000001` | +0.760% | -2.591% | -3.385% | -3.880% |
| `Thaidancer_viewdep_vox12` | +2.269% | -2.656% | +0.256% | -0.348% |
| `longdress_viewdep_vox12` | -0.096% | -0.511% | -0.622% | -0.499% |
| `loot_viewdep_vox12` | -1.769% | -3.707% | +1.259% | -1.218% |
| `redandblack_viewdep_vox12` | -4.938% | -6.355% | -2.934% | -4.331% |
| `soldier_viewdep_vox12` | -2.007% | -3.134% | -0.425% | -1.372% |
| `boxer_viewdep_vox12` | +0.087% | -2.178% | +2.584% | -0.116% |
| `Facade_00009_vox12` | +4.753% | +3.558% | +3.942% | +3.636% |
| `House_without_roof_00057_vox12` | +12.076% | +5.521% | +1.726% | +0.476% |
| `Shiva_00035_vox12` | +5.999% | +5.772% | +1.420% | +1.513% |
| `Staue_Klimt_vox12` | +3.116% | +3.102% | +1.598% | +1.581% |
| **Average** | **+1.848%** | **-0.326%** | **+0.155%** | **-0.696%** |

BD-BR is computed on the
Pareto-efficient RD envelope using PCHIP over the shared quality interval. The
same deterministic rule is applied to Original Unicorn, Ours Base, and Ours
Full; there is no extrapolation or manual point deletion.

The Full endpoint remains close to Original Unicorn RD performance while the
scalable bitstream provides an additional independently decodable Base
endpoint. The result supports a competitive, but not uniformly superior,
conclusion.

## Raw RD Figures

Every measured operating point remains visible in these figures, including
dominated or non-monotonic points:

- [8iVFB: 4 samples, Y and YUV611](../formal_static_rgb_final_20260911/figures/8i_raw_rd_curves.svg)
- [Owlii: 4 samples, Y and YUV611](../formal_static_rgb_final_20260911/figures/owlii_raw_rd_curves.svg)
- [CTC: 12 samples, Y and YUV611](../formal_static_rgb_final_20260911/figures/ctc_raw_rd_curves.svg)

The [figure audit](figure_audit.csv) verifies all 40 sample-metric panels and
the presence of Original Unicorn, Ours Base, and Ours Full.

## Results and Evidence

- [Final raw physical-RD points](../formal_static_rgb_final_20260911/evidence/final_raw_rd_points.csv)
- [Per-sequence Pareto BD-BR](../formal_static_rgb_pareto_bdbr_20260911/tables/bdbr_per_sequence.csv)
- [Dataset-level Pareto BD-BR](../formal_static_rgb_pareto_bdbr_20260911/tables/bdbr_dataset_summary.csv)
- [Compact per-sequence thesis table](bdbr_per_sequence_compact.csv)
- [Pareto-frontier curve audit](../formal_static_rgb_pareto_bdbr_20260911/tables/curve_frontier_audit.csv)
- [Detailed Pareto protocol](../formal_static_rgb_pareto_bdbr_20260911/PROTOCOL.md)
- [Hardware-labeled runtime summary](../formal_static_rgb_final_20260911/tables/runtime_summary_by_hardware.csv)

The frozen evidence package remains the authority for raw measurements,
figures, reconstruction consistency, and runtime data. This page is a concise
presentation layer and does not duplicate or alter that evidence.
