# Final Static-RGB Thesis Results

This page presents the final evaluation of a **full-resolution
quality-scalable extension of Unicorn Part II**. The evaluation covers 20
samples: four from 8iVFB, four from Owlii, and twelve from CTC. Each sample has
nine Original Unicorn operating points and seven scalable operating points;
each scalable point yields a Base endpoint and a Full endpoint.

## Per-sequence BD-BR versus Original Unicorn

All values are percentages. Negative BD-BR represents bitrate saving relative
to Original Unicorn at equivalent quality. Average rows are read directly from
the frozen equal-sequence dataset summary; displayed values use six decimal
places, while the [compact table source](bdbr_per_sequence_compact.csv) retains
the complete stored precision.

### 8iVFB

| Sample | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| `longdress` | +21.968931% | +11.448877% | +0.572631% | +1.794887% |
| `loot` | +0.169039% | -5.259241% | +2.510655% | +1.051747% |
| `redandblack` | -4.878505% | -9.229253% | -0.574927% | -0.640324% |
| `soldier` | +3.558913% | +0.147225% | +3.300283% | +3.355673% |
| **Average** | **+5.204595%** | **-0.723098%** | **+1.452161%** | **+1.390496%** |

### Owlii

| Sample | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| `basketball_player` | +6.223647% | +2.506407% | +1.924549% | +1.676353% |
| `dancer` | +14.897868% | +9.356277% | +3.500272% | +3.969554% |
| `exercise` | +6.435334% | +1.494883% | -0.568958% | -0.999246% |
| `model` | +12.882017% | +6.610955% | +2.563152% | +3.661869% |
| **Average** | **+10.109716%** | **+4.992131%** | **+1.854754%** | **+2.077132%** |

### CTC

| Sample | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| `basketball_player_vox11_00000200` | +1.927752% | -0.737883% | -3.559648% | -3.797633% |
| `dancer_vox11_00000001` | +0.759691% | -2.590792% | -3.384578% | -3.880264% |
| `Thaidancer_viewdep_vox12` | +2.269212% | -2.655915% | +0.256157% | -0.348037% |
| `longdress_viewdep_vox12` | -0.096357% | -0.511371% | -0.622159% | -0.498555% |
| `loot_viewdep_vox12` | -1.769466% | -3.706831% | +1.258922% | -1.218368% |
| `redandblack_viewdep_vox12` | -4.937647% | -6.354937% | -2.933618% | -4.330641% |
| `soldier_viewdep_vox12` | -2.006819% | -3.133605% | -0.425384% | -1.372154% |
| `boxer_viewdep_vox12` | +0.087137% | -2.177951% | +2.583585% | -0.116084% |
| `Facade_00009_vox12` | +4.753462% | +3.558223% | +3.942293% | +3.636108% |
| `House_without_roof_00057_vox12` | +12.076455% | +5.521354% | +1.726482% | +0.476300% |
| `Shiva_00035_vox12` | +5.999013% | +5.772001% | +1.419658% | +1.513442% |
| `Staue_Klimt_vox12` | +3.116040% | +3.102106% | +1.598443% | +1.581173% |
| **Average** | **+1.848206%** | **-0.326300%** | **+0.155013%** | **-0.696226%** |

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
