# Full28 Enhancement Checkpoint Audit

## Result provenance

- Dataset: RWTT Validation Full28, 28 original models / 792 H5.
- Rate: actual physical hard bits; Base = x_low+r1-r4, Full = Base+Enhancement; min_v/max_v and geometry excluded by the established project contract.
- Quality: author `pc_error`, normalized YUV PSNR 6:1:1.
- Aggregation: point-weighted channel MSE within each original model, PSNR after aggregation, then equal average across 28 models.
- Official R01 U/V: corrected retained-per-H5 aggregation; provenance saved beside this report.

## Full28 aggregate

| Endpoint | Physical bpp | Y PSNR | U PSNR | V PSNR | YUV611 |
|---|---:|---:|---:|---:|---:|
| Official R01 | 1.577542 | 41.043177 | 46.704811 | 47.850168 | 42.601755 |
| Full step1763 | 1.656177 | 41.554582 | 46.641686 | 47.632971 | 42.950268 |
| Full step3525 | 1.586111 | 40.956589 | 46.758877 | 47.779150 | 42.534695 |

## Step3525 minus step1763

- Physical rate: -0.070066 bpp.
- Y / U / V / YUV611: -0.597993 / +0.117192 / +0.146179 / -0.415573 dB.
- Per model: quality improves 0/28, degrades 28/28; rate decreases 27/28, increases 1/28.
- Mean / median per-model ΔYUV611: -0.415573 / -0.409178 dB.

### Per-model Pareto quadrants

- `better_quality_lower_rate`: 0/28
- `better_quality_higher_rate`: 0/28
- `worse_quality_lower_rate`: 27/28
- `worse_quality_higher_rate`: 1/28
- `ties`: 0/28

## Combined manager table

| Dataset | Full1763 vs R01 Δrate% | Full1763 vs R01 ΔYUV611 | Full3525 vs R01 Δrate% | Full3525 vs R01 ΔYUV611 |
|---|---:|---:|---:|---:|
| RWTT Full28 | +4.985% | +0.3485 dB | +0.543% | -0.0671 dB |
| longdress | +4.453% | -0.0118 dB | +3.404% | -0.5927 dB |
| loot | +9.094% | +0.1273 dB | +10.030% | -0.1532 dB |
| redandblack | +8.154% | +0.2315 dB | +9.570% | -0.1467 dB |
| soldier | +3.338% | -0.1712 dB | +4.496% | -0.5281 dB |

## Classification

**MIXED RD TRAJECTORY**

Classification uses aggregate physical RD dominance/trade-off on RWTT Full28; external results are supporting evidence, not mixed into the Full28 aggregation.

Step3525 saves `0.070066 bpp` relative to step1763 but loses `0.415573 dB` YUV611. It is therefore neither a Pareto improvement nor a Pareto deterioration at the aggregate level. The 28-model evidence is systematic in quality direction (28/28 degrade), while almost all models move toward lower rate (27/28).

## Luminance diagnostic

- Step1763 versus R01 Y PSNR: `+0.511405 dB`.
- Step3525 versus R01 Y PSNR: `-0.086588 dB`.
- Step3525 minus step1763 Y PSNR: `-0.597993 dB`.

On RWTT Full28, the luminance deficit emerges during later training; it is not an inherent failure already present at step1763. This is diagnostic evidence only and does not change the loss or architecture.
