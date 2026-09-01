# 8iVFB External Generalization Audit

## Result provenance

- Data: exact 8iVFB frames Longdress 1300, Loot 1200, Redandblack 1550, Soldier 0690.
- Rate: actual G-PCC `x_low` plus actual torchac bytes; `min_v/max_v` excluded, matching the existing Unicorn project convention.
- Quality: direct `pc_error`, `resolution=1`, `color=1`, YUV-PSNR 6:1:1.
- Canonical Base: step5525, R01/32k8k, conditioning lambda 32768.
- Canonical Full: Enhancement LR `5e-5`, lambda 32768, step1763 and step3525.
- Canonical endpoints within each sequence share exactly one hard-decoded `x_low+r1-r4` prefix state.
- Training performed in this audit: **NO**.

## Primary endpoint results

| Sequence | Official R01 bpp / dB | Canonical Base bpp / dB | Canonical Full3525 bpp / dB |
|---|---:|---:|---:|
| Longdress | 1.898844 / 41.5353 | 0.767107 / 34.8154 | 1.963481 / 40.9426 |
| Loot | 0.450077 / 45.6167 | 0.243223 / 43.0397 | 0.495218 / 45.4634 |
| Redandblack | 0.873823 / 42.7704 | 0.410088 / 39.2362 | 0.957446 / 42.6237 |
| Soldier | 0.728916 / 44.8988 | 0.393370 / 41.7633 | 0.761685 / 44.3707 |

## Full3525 versus Official R01

| Sequence | Delta bpp | Delta rate | Delta YUV611 |
|---|---:|---:|---:|
| Longdress | +0.064637 | +3.404% | -0.5927 dB |
| Loot | +0.045142 | +10.030% | -0.1532 dB |
| Redandblack | +0.083622 | +9.570% | -0.1467 dB |
| Soldier | +0.032769 | +4.496% | -0.5281 dB |

The step3525 endpoint is slightly or clearly dominated by R01 on all four external sequences. This is not Longdress-specific.

## BaseSynthesis external gain

`B_native` and Canonical Base use identical physical Base bits.

| Sequence | Shared Base bpp | Base minus B_native YUV611 |
|---|---:|---:|
| Longdress | 0.767107 | +1.7644 dB |
| Loot | 0.243223 | +0.7256 dB |
| Redandblack | 0.410088 | +0.7639 dB |
| Soldier | 0.393370 | +1.3643 dB |

BaseSynthesis retains a positive external gain on every sequence. The evidence does not support a Base generalization failure.

## Enhancement checkpoint trajectory

| Sequence | step1763 bpp | step3525 bpp | Delta bpp | Delta YUV611 |
|---|---:|---:|---:|---:|
| Longdress | 1.983407 | 1.963481 | -0.019926 | -0.5809 dB |
| Loot | 0.491006 | 0.495218 | +0.004212 | -0.2805 dB |
| Redandblack | 0.945071 | 0.957446 | +0.012374 | -0.3781 dB |
| Soldier | 0.753245 | 0.761685 | +0.008440 | -0.3569 dB |

Step1763 relative to Official R01 is respectively:

- Longdress: +4.453% rate / -0.0118 dB.
- Loot: +9.094% rate / +0.1273 dB.
- Redandblack: +8.154% rate / +0.2315 dB.
- Soldier: +3.338% rate / -0.1712 dB.

All four external sequences lose quality from step1763 to step3525, and three also spend more rate. However, the subsequently completed RWTT Full28 audit shows that step3525 also loses 0.4156 dB while saving 0.0701 bpp on the training-domain validation contract. The combined evidence is therefore a **MIXED RD TRAJECTORY with an external generalization warning**, not sufficient evidence for pure late-training over-specialization.

## Channel evidence

Full3525 minus Official R01 PSNR:

| Sequence | Delta Y | Delta U | Delta V |
|---|---:|---:|---:|
| Longdress | -0.7692 | +0.3945 | -0.5206 |
| Loot | -0.2073 | +0.0089 | +0.0091 |
| Redandblack | -0.2263 | +0.2106 | -0.0260 |
| Soldier | -0.7047 | +0.0073 | -0.0037 |

Y is the worst channel on all four sequences. This supports a **SYSTEMATIC LUMINANCE DEFICIT** diagnostic, but this audit does not change the loss or architecture.

## Author CSV cross-check

Matched-contract author/local quality agrees closely: maximum absolute YUV611 difference is 0.0092 dB (Longdress), 0.0124 dB (Loot), 0.0116 dB (Redandblack), and 0.0040 dB (Soldier).

The retained author CSVs contain one known overlapping-lambda provenance inconsistency:

- Longdress and Redandblack lambda8192 rows point to `R3` (32k8k lambda8192).
- Loot and Soldier lambda8192 rows point to `R4` (the original traversal's 8k256 lambda8192 overlap).

Those two `R4` rows differ from the thesis official R03 mapping by about 0.43 dB and are explicitly marked `AUTHOR_OVERLAP_USES_8k256_L8192`; they are not treated as same-contract mismatches.

## Classification

- `LONGDRESS-SPECIFIC WEAKNESS`: **NO**.
- `BASE GENERALIZATION WARNING`: **NO** based on four positive BaseSynthesis gains.
- `ENHANCEMENT GENERALIZATION WARNING`: **YES for step3525**, qualified by much stronger step1763 results.
- `LATE-TRAINING OVER-SPECIALIZATION`: **NOT CONFIRMED** after the RWTT Full28 step1763 audit.
- `LATE-TRAINING RD TRAJECTORY`: **MIXED**; RWTT trades lower rate for lower quality, while all external sequences lose quality.
- `SYSTEMATIC LUMINANCE DEFICIT`: **YES, diagnostic evidence**.

No architecture decision or new training is made in this audit.
