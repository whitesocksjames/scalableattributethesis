# Final Static-RGB Thesis Results

This page presents the final evaluation of a **full-resolution
quality-scalable extension of Unicorn Part II**. The evaluation covers 20
samples: four from 8iVFB, four from Owlii, and twelve from CTC. Each sample has
nine Original Unicorn operating points and seven scalable operating points;
each scalable point yields a Base endpoint and a Full endpoint.

## Full versus Original Unicorn

| Dataset | Y BD-BR | YUV611 BD-BR | Sequences |
| --- | ---: | ---: | ---: |
| 8iVFB | +1.452% | +1.390% | 4/4 |
| Owlii | +1.855% | +2.077% | 4/4 |
| CTC | +0.155% | -0.696% | 12/12 |

## Base versus Original Unicorn

| Dataset | Y BD-BR | YUV611 BD-BR | Sequences |
| --- | ---: | ---: | ---: |
| 8iVFB | +5.205% | -0.723% | 4/4 |
| Owlii | +10.110% | +4.992% | 4/4 |
| CTC | +1.848% | -0.326% | 12/12 |

Negative values indicate bitrate savings. BD-BR is computed on the
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
- [Pareto-frontier curve audit](../formal_static_rgb_pareto_bdbr_20260911/tables/curve_frontier_audit.csv)
- [Detailed Pareto protocol](../formal_static_rgb_pareto_bdbr_20260911/PROTOCOL.md)
- [Hardware-labeled runtime summary](../formal_static_rgb_final_20260911/tables/runtime_summary_by_hardware.csv)

The frozen evidence package remains the authority for raw measurements,
figures, reconstruction consistency, and runtime data. This page is a concise
presentation layer and does not duplicate or alter that evidence.
