# Rescued 2K Enhancement External Analysis

## Provenance

- Model: rescued `2K-U-PATH step500` Base with independent D111 Enhancement Stage-1.
- Evaluation: physical hard coding; Base uses `x_low+r1-r4`, native r5 is absent; Full bits equal Base plus Enhancement bits.
- 8i reference: author operating points reproduced with local physical coding and direct `pc_error` YUV 6:1:1.
- Owlii reference: retained author-provided per-sequence R01-R09 CSV.
- Matched deltas use adjacent-point linear interpolation only; no extrapolation.

## Per-sequence RD results

| Dataset | Sequence | Endpoint | bpp | YUV611 | Official@bpp | Delta to curve |
|---|---|---:|---:|---:|---:|---:|
| 8iVFB | Longdress 1300 | Canonical_Base | 0.260225 | 30.1081 | 30.9169 | -0.8087 |
| 8iVFB | Longdress 1300 | step1500 | 0.443514 | 32.9171 | 33.0112 | -0.0941 |
| 8iVFB | Longdress 1300 | step1763 | 0.439290 | 32.7973 | 32.9678 | -0.1706 |
| 8iVFB | Loot 1200 | Canonical_Base | 0.102304 | 39.6769 | 39.4794 | +0.1974 |
| 8iVFB | Loot 1200 | step1500 | 0.122829 | 40.2466 | 40.1627 | +0.0840 |
| 8iVFB | Loot 1200 | step1763 | 0.124070 | 40.1873 | 40.1945 | -0.0072 |
| 8iVFB | Redandblack 1550 | Canonical_Base | 0.165007 | 35.8020 | 35.5263 | +0.2757 |
| 8iVFB | Redandblack 1550 | step1500 | 0.210925 | 36.7176 | 36.3200 | +0.3976 |
| 8iVFB | Redandblack 1550 | step1763 | 0.215814 | 36.6337 | 36.4045 | +0.2292 |
| 8iVFB | Soldier 0690 | Canonical_Base | 0.170322 | 38.0970 | 38.0479 | +0.0491 |
| 8iVFB | Soldier 0690 | step1500 | 0.205867 | 38.9445 | 38.8947 | +0.0499 |
| 8iVFB | Soldier 0690 | step1763 | 0.206411 | 38.8676 | 38.9048 | -0.0372 |
| Owlii | Basketball 0200 | Canonical_Base | 0.092497 | 39.0924 | 39.2701 | -0.1777 |
| Owlii | Basketball 0200 | step1500 | 0.109116 | 39.7815 | 39.9096 | -0.1281 |
| Owlii | Basketball 0200 | step1763 | 0.109438 | 39.7307 | 39.9191 | -0.1884 |
| Owlii | Dancer 0001 | Canonical_Base | 0.111926 | 37.6064 | 38.3498 | -0.7434 |
| Owlii | Dancer 0001 | step1500 | 0.140779 | 38.9577 | 39.2932 | -0.3355 |
| Owlii | Dancer 0001 | step1763 | 0.140972 | 38.9065 | 39.2978 | -0.3913 |
| Owlii | Exercise 0001 | Canonical_Base | 0.069468 | 38.8620 | 38.8925 | -0.0305 |
| Owlii | Exercise 0001 | step1500 | 0.082612 | 39.2892 | 39.3182 | -0.0290 |
| Owlii | Exercise 0001 | step1763 | 0.081236 | 39.2562 | 39.2819 | -0.0256 |
| Owlii | Model 0001 | Canonical_Base | 0.141159 | 35.5085 | 35.8602 | -0.3517 |
| Owlii | Model 0001 | step1500 | 0.207238 | 37.2134 | 37.5548 | -0.3415 |
| Owlii | Model 0001 | step1763 | 0.206472 | 37.1338 | 37.5420 | -0.4082 |

## Dataset means

| Dataset | Endpoint | Mean bpp | Mean YUV611 | Mean matched delta |
|---|---:|---:|---:|---:|
| 8iVFB | Canonical_Base | 0.174464 | 35.9210 | -0.0716 |
| 8iVFB | step1500 | 0.245784 | 37.2065 | +0.1093 |
| 8iVFB | step1763 | 0.246396 | 37.1215 | +0.0035 |
| Owlii | Canonical_Base | 0.103762 | 37.7673 | -0.3258 |
| Owlii | step1500 | 0.134936 | 38.8104 | -0.2085 |
| Owlii | step1763 | 0.134529 | 38.7568 | -0.2534 |

## Interpretation

- `step1500` strictly RD-dominates `step1763` on **5/8** sequences: Loot 1200, Redandblack 1550, Soldier 0690, Basketball 0200, Dancer 0001.
- The remaining sequences are rate-quality trade-offs: Longdress 1300, Exercise 0001, Model 0001.
- Therefore `step1500` is the stronger external checkpoint overall, but it is not a universal strict winner. Selection should use the per-sequence curve positions, not PSNR alone.
- All Full checkpoints add substantial Enhancement rate over Base; the payload is actively used and has not collapsed.

## Correctness

- Rows checked: 32; all use four Base residual streams and zero native-r5 streams.
- Full hard round-trip maximum absolute difference is zero for every Full row.
