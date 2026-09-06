# Owlii operating-point diagnosis and joint-256 review

## Result provenance

- Owlii frames: author Part II static selections, author-compatible vox11→vox10 preprocessing.
- Point-count fingerprint: Basketball 796217; Dancer 702038; Exercise 645135; Model 657755.
- Official curves: repository-retained author-provided per-sequence R01–R09 CSVs.
- Candidate rate: actual physical hard attribute bits; metric: `pc_error` Y/U/V and YUV611.
- Matched delta: linear interpolation in each sequence's own bpp–YUV611 curve; no extrapolation.
- 32K results are reused; 16K–256 are new N30 evaluations.
- 32K Full in this bundle is the retained D111 step3525 result, not the later D611 recipe.
- Hard/bit correctness: PASS for all 32 sequence-point outputs checked.

## Four-sequence equal-mean endpoints

| Point | Endpoint | bpp | YUV611 | Official@bpp | Δmatched |
|---|---|---:|---:|---:|---:|
| 32K | Base | 0.277480 | 40.9603 | 41.2576 | -0.2972 |
| 32K | Full | 0.613282 | 43.8715 | 44.3176 | -0.4460 |
| 16K | Base | 0.210352 | 40.2698 | 40.4661 | -0.1963 |
| 16K | Full | 0.421973 | 42.7564 | 42.7218 | +0.0346 |
| 8K | Base | 0.157413 | 39.3115 | 39.5171 | -0.2056 |
| 8K | Full | 0.280885 | 41.1200 | 41.2904 | -0.1705 |
| 4K | Base | 0.110649 | 37.6384 | 38.2254 | -0.5870 |
| 2K | Base | 0.088488 | 37.1706 | 37.4127 | -0.2421 |
| 1K | Base | 0.068755 | 36.5124 | 36.5247 | -0.0124 |
| 1K | Full | 0.083246 | 37.1660 | 37.1885 | -0.0225 |
| 512 | Base | 0.054547 | 35.7910 | 35.7384 | +0.0526 |
| 512 | Full | 0.058255 | 36.0295 | 36.0048 | +0.0247 |
| 256 | Base | 0.042125 | 34.8923 | N/A | N/A |
| 256 | Full | 0.042137 | 34.8812 | N/A | N/A |

## 8K→4K→2K→1K per-sequence Base pattern

| Sequence | Point | bpp | YUV611 | Δmatched |
|---|---|---:|---:|---:|
| basketball_player | 8K | 0.131713 | 40.2528 | -0.3250 |
| basketball_player | 4K | 0.100103 | 39.0479 | -0.5951 |
| basketball_player | 2K | 0.078863 | 38.4434 | -0.1545 |
| basketball_player | 1K | 0.060988 | 37.6744 | +0.0676 |
| dancer | 8K | 0.163000 | 39.6802 | -0.1434 |
| dancer | 4K | 0.120746 | 37.3832 | -1.3087 |
| dancer | 2K | 0.095642 | 36.9457 | -0.7725 |
| dancer | 1K | 0.072600 | 36.2513 | -0.4209 |
| exercise | 8K | 0.107450 | 39.6865 | -0.2869 |
| exercise | 4K | 0.082984 | 38.9983 | -0.3297 |
| exercise | 2K | 0.061432 | 38.4669 | -0.0737 |
| exercise | 1K | 0.044059 | 37.7409 | +0.0550 |
| model | 8K | 0.227489 | 37.6266 | -0.2666 |
| model | 4K | 0.138763 | 35.1244 | -0.6612 |
| model | 2K | 0.118014 | 34.8264 | -0.3062 |
| model | 1K | 0.097374 | 34.3829 | -0.1003 |

## Joint-256 RWTT-28Lite

| Recipe | Step | Endpoint | bpp | YUV611 | Δmatched | EL Δbpp | Full−Base dB |
|---|---:|---|---:|---:|---:|---:|---:|
| TAFA_joint | 500 | Base | 0.077402 | 31.5302 | +0.1812 | 0.006943 | +0.1268 |
| TAFA_joint | 500 | Full | 0.084345 | 31.6570 | +0.1154 | 0.006943 | +0.1268 |
| TAFA_joint | 1500 | Base | 0.076714 | 31.6342 | +0.3043 | 0.005609 | +0.0713 |
| TAFA_joint | 1500 | Full | 0.082323 | 31.7054 | +0.2200 | 0.005609 | +0.0713 |
| Sequential_Enhancement | 1763 | Base | 0.076068 | 31.6954 | +0.3835 | 0.000109 | -0.0096 |
| Sequential_Enhancement | 1763 | Full | 0.076177 | 31.6858 | +0.3708 | 0.000109 | -0.0096 |

## Evidence synthesis

- **Owlii Base:** 4K is below the matched Official envelope on all four sequences and is the clearest weak point (mean −0.5870 dB). 2K improves over 4K on every sequence but remains negative (mean −0.2421 dB). 1K recovers to near-envelope mean performance (−0.0124 dB), with Dancer still weak.
- **Family pattern:** the cross-sequence 8K→4K degradation and 2K→1K recovery reproduce the earlier 8i concern. This supports an `8k256`-family/native-truncation contribution, modulated by sequence content; it is not merely a Longdress-only effect.
- **Owlii Full:** 16K Stage-1 is near/slightly above the matched author curve (+0.0346 dB); 8K is moderately below (−0.1705 dB); 1K and 512 are near the envelope. The 256 endpoints lie below the author R09 rate and therefore have no interpolated delta.
- **32K provenance:** the reused 32K Full is D111 step3525 and is −0.4460 dB matched on Owlii; this result must not be labeled as the later D611 recipe.
- **Joint-256:** step1500 improves Full over its own Base by +0.0713 dB using 0.005609 bpp Enhancement, but its matched efficiency is lower than sequential step1763. Sequential uses only 0.000109 bpp Enhancement and has the larger positive matched delta; the joint probe does not currently justify replacing the sequential recipe.
- **Joint training status:** scientific conclusion is provisional because BS4 OOM stopped training at step1609; nevertheless both retained hard checkpoints are valid evaluation points.

## Interpretation guardrails

- Owlii interpolation is diagnostic, not BD-rate.
- Joint step500/1500 are incomplete-training checkpoints after a BS4 OOM at step1609.
- Joint and sequential rows use the same RWTT-28Lite physical hard contract.
- No new training or automatic recipe selection is implied.
