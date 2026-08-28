# B1 Universality Screen v1

> Fixed 4-H5 physical screening. No training, model modification, quantizer search or Final Architecture Decision.

## Contract

- Same deterministic B1 nested mapping as B1-v0.
- Fixed `step=4/8` only.
- Each derived point compared only with official neighbors evaluated on the exact same four H5.
- Same-family：R01/R02/R03 (`32k8k`)；cross-family：R04 (`8k256`)。

## Results

Submitted：

- R01-derived：Job `1786644`；same-H5 neighbors R01/R02/R03。
- R03-derived：Job `1786645`；same-H5 neighbors R02/R03/R04/R05。
- R04-derived：Job `1786646`；same-H5 neighbors R03/R04/R05/R06。

All three jobs completed on RTX3080 in `1:47–2:04`:

- R01-derived：Job `1786644`。
- R03-derived：Job `1786645`。
- R04-derived：Job `1786646`。

### R01-derived (`32k8k`, lambda=32768)

| Endpoint | Physical bpp | direct YUV-PSNR 6:1:1 | Full−Base | layered/original | max diff |
|---|---:|---:|---:|---:|---:|
| step4 Base | 0.444795 | 39.703138 dB | — | — | — |
| step4 Full | 1.010110 | 42.902019 dB | +0.565315 / +3.198881 dB | 1.000195 | 0 |
| step8 Base | 0.343784 | 39.158103 dB | — | — | — |
| step8 Full | 1.008430 | 42.902019 dB | +0.664646 / +3.743916 dB | 0.999839 | 0 |

Same-H5 official：R01 `1.009102 / 42.902019`；R02 `0.758304 / 41.615453`；R03 `0.550941 / 40.197600`。

Assessment：Base 位于同 family local RD progression的合理延伸位置；Full exact；ratio≈1。**PASS（fixed-4）**。

### R03-derived (`32k8k`, lambda=8192)

| Endpoint | Physical bpp | direct YUV-PSNR 6:1:1 | Full−Base | layered/original | max diff |
|---|---:|---:|---:|---:|---:|
| step4 Base | 0.199613 | 37.842809 dB | — | — | — |
| step4 Full | 0.551012 | 40.197600 dB | +0.351399 / +2.354791 dB | 0.999905 | 0 |
| step8 Base | 0.171427 | 37.694278 dB | — | — | — |
| step8 Full | 0.551112 | 40.197600 dB | +0.379686 / +2.503322 dB | 1.002424 | 0 |

Same-H5 official：R02 `0.758304 / 41.615453`；R03 `0.550941 / 40.197600`；R04 `0.216411 / 38.144622`；R05 `0.142772 / 37.242894`。

Assessment：两个 Base 均落在 R04/R05 local region附近；Full exact；ratio误差≤0.25%。**PASS（fixed-4）**。

### R04-derived (`8k256`, lambda=4096; cross-family)

| Endpoint | Physical bpp | direct YUV-PSNR 6:1:1 | Full−Base | layered/original | max diff |
|---|---:|---:|---:|---:|---:|
| step4 Base | 0.125590 | 37.246209 dB | — | — | — |
| step4 Full | 0.216238 | 38.144622 dB | +0.090648 / +0.898413 dB | 1.001422 | 0 |
| step8 Base | 0.110275 | 37.167194 dB | — | — | — |
| step8 Full | 0.216762 | 38.144622 dB | +0.106487 / +0.977428 dB | 1.006888 | 0 |

Same-H5 official：R03 `0.550941 / 40.197600`；R04 `0.216411 / 38.144622`；R05 `0.142772 / 37.242894`；R06 `0.081701 / 36.192975`。

Assessment：step4 Base在该 fixed-4 上以更低 rate获得与 R05近似的 quality；step8也位于 R05/R06 local progression中；Full exact；最大ratio偏差约0.69%。**PASS（fixed-4 cross-family）**。

## Gate

R01/R03/R04 均通过三个 gate：

1. Base 接近 same-H5 official local RD progression；
2. Full exact recover original released point (`max_abs_difference=0`)；
3. layered/original physical ratio接近1（本轮范围约 `0.99984–1.00689`）。

结论：B1 fixed nested mapping 获得初步 universality evidence，覆盖 same-family R01/R03与 cross-family R04。该结论仍是 fixed-4 screen，不等价于 Full28 formal generalization，也不在本轮自动授权或实现 B1-H。
