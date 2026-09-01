# Y-aware Equilibrium and Human-data Audit

## Result provenance

- Training jobs: D111 `5435847`, D411 `5435848`, D611 `5435849` (all `COMPLETED`, exit code 0).
- Hard-evaluation job: `5435851` (`COMPLETED`, exit code 0).
- Data: fixed 28-H5 subset, one H5 from each of 28 RWTT models.
- Training continuation: each arm resumed its own step 2063 checkpoint and ran to step 3263.
- Evaluation: actual hard Enhancement coding; the hard Prefix was shared once per H5 across checkpoints.
- Metric: direct `pc_error` Y/U/V and YUV-PSNR 6:1:1.
- Scope: screening evidence only; this is not Full28 formal RD.

## Fixed-28 trajectory

| Arm | Relative updates | Global step | EL bpp | Full bpp | Y PSNR | U PSNR | V PSNR | YUV 6:1:1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D111 | +300 | 2063 | 1.195251 | 1.727740 | 41.8011 | 47.1278 | 48.3792 | 43.2892 |
| D111 | +500 | 2263 | 1.205337 | 1.737825 | 41.8867 | 47.1246 | 48.3719 | 43.3521 |
| D111 | +750 | 2513 | 1.194712 | 1.727201 | 41.8420 | 47.0225 | 48.3228 | 43.2997 |
| D111 | +1000 | 2763 | 1.203179 | 1.735668 | 41.8940 | 47.1150 | 48.3587 | 43.3547 |
| D111 | +1500 | 3263 | 1.199841 | 1.732330 | 41.8544 | 47.1452 | 48.4155 | 43.3359 |
| D411 | +300 | 2063 | 1.562370 | 2.094859 | 44.2795 | 46.5623 | 47.9672 | 45.0258 |
| D411 | +500 | 2263 | 1.582704 | 2.115193 | 44.4385 | 46.5022 | 47.9266 | 45.1325 |
| D411 | +750 | 2513 | 1.566287 | 2.098775 | 44.3687 | 46.4366 | 47.9156 | 45.0705 |
| D411 | +1000 | 2763 | 1.585424 | 2.117913 | 44.5138 | 46.4376 | 47.9001 | 45.1776 |
| D411 | +1500 | 3263 | 1.578590 | 2.111079 | 44.4844 | 46.4744 | 47.9044 | 45.1606 |
| D611 | +300 | 2063 | 1.623752 | 2.156240 | 44.6941 | 46.3869 | 47.8381 | 45.2987 |
| D611 | +500 | 2263 | 1.644698 | 2.177187 | 44.8563 | 46.3172 | 47.7856 | 45.4051 |
| D611 | +750 | 2513 | 1.625749 | 2.158238 | 44.7826 | 46.2174 | 47.7461 | 45.3324 |
| D611 | +1000 | 2763 | 1.649066 | 2.181554 | 44.9629 | 46.2027 | 47.7290 | 45.4636 |
| D611 | +1500 | 3263 | 1.644471 | 2.176959 | 44.9342 | 46.2851 | 47.7683 | 45.4573 |

Base rate is constant at `0.532489 bpp`. All 420 hard reconstructions have `max_abs_difference = 0`.

## Interpretation

1. **D411/D611 high rate is an equilibrium, not a transient spike.** From +300 to +1500, D411 EL rate remains `1.562–1.585 bpp`; D611 remains `1.624–1.649 bpp`. Continued training does not return them toward D111 rate.
2. **Quality largely plateaus near +1000.** D411 peaks at `45.1776 dB` and D611 at `45.4636 dB`; +1500 is slightly worse (`-0.0169` and `-0.0063 dB`) despite nearly unchanged rate.
3. **Y-aware weighting does protect Y, but reallocates quality away from chroma.** At +1500 versus D111, D411 gains `+2.6300 dB` Y while losing `-0.6708/-0.5111 dB` U/V; D611 gains `+3.0798 dB` Y while losing `-0.8601/-0.6472 dB` U/V.
4. **D411 is the more economical Y-aware point.** D611 spends about `0.0659 bpp` more than D411 at +1500 for only `+0.2967 dB` YUV 6:1:1. This remains a screening observation, not a formal selection.
5. A later formal study would need **rate calibration** if D411/D611 are to be compared at matched rate. Extending the same run alone does not resolve their high-rate behavior.

## Human-data inventory and feasibility

Local read-only inventory found one vox9 frame each for MVUB Andrew, David, Phil and Sarah; Ricardo and Volucap Thomas were not found. Queen/Boxer remain test content, and only Thaidancer configs/results were found. See `human_data_inventory.csv` in the earlier audit directory.

The no-training Andrew probe (`5435854`) passed:

- 279,664 points partitioned into four H5 blocks;
- DataLoader, YUV conversion, MinkowskiEngine SparseTensor, frozen Base Prefix and Enhancement forward all finite;
- zero optimizer updates.

This proves pipeline feasibility only. Four isolated vox9 frames are not an approved or sufficient human-data training set.

## Round conclusion

- D411/D611 high-rate behavior: **persistent equilibrium**.
- Y protection: **effective, with measurable chroma trade-off**.
- More training at the same setup: **low expected value after about +1000 updates**.
- Human-data path: **runtime feasible, dataset availability insufficient for training**.
- No new architecture, Full28 evaluation, lambda sweep or data mixing was started.

## Longdress same-contract RD comparison

The three +1500 checkpoints were additionally evaluated on the exact Longdress
frame 1300 used by the retained official R01--R09 reference. Rate is physical
codec bpp and quality is direct `pc_error` YUV-PSNR 6:1:1.

| Endpoint | Physical bpp | Y PSNR | YUV 6:1:1 |
|---|---:|---:|---:|
| Official R04 | 0.653807 | 33.8028 | 34.5541 |
| Canonical Base | 0.767107 | 33.8162 | 34.8154 |
| Official R03 | 1.126301 | 37.4815 | 38.2224 |
| Official R02 | 1.483294 | 39.4788 | 39.9848 |
| Official R01 | 1.898844 | 41.3234 | 41.5353 |
| D111 +1500 | 2.017059 | 41.5604 | 41.6922 |
| D411 +1500 | 2.346172 | 44.2815 | 43.5454 |
| D611 +1500 | 2.391032 | 44.7461 | 43.8108 |

D111 is only marginally above R01 in both rate and quality. D411 and D611 are
clear high-rate extensions beyond R01: they gain `+2.0101 dB` and `+2.2755 dB`
YUV respectively over R01 while spending `+0.4473` and `+0.4922 bpp`.
Because the official curve has no released point beyond R01, this establishes
their same-content physical RD position but does not by itself prove Pareto
efficiency against a denser native high-rate curve.
