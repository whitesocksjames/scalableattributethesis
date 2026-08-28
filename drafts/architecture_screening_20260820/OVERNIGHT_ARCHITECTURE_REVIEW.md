# Overnight Architecture Review

> Evidence-only review draft. No Final Architecture Decision.

## Execution contract

- B1：existing Full28 fixed-mapping `step=4/8` shards；single raw hard-code pass，derive Full28 and Dev14。
- C2：official R08 frozen Base (`2k128`, `lambda_Base=256`)；`lambda_E=650/2000`；each exactly 3,525 updates。
- C2 training：BS4，Adam，constant `lr=5e-5`，seed0，objective `R_E + lambda_E D_F`，architecture/native initialization unchanged。
- Trajectory checkpoints：705 / 1410 / 2115 / 2820 / 3525。
- No C1/C3/B2, no B1 tuning/training, no new loss or architecture.

## Slurm execution

- B1 Full28 shards（保持原样）：`1785647 / 1785648 / 1785649 / 1785650`。
- B1 afterok aggregation：`1785675`，dependency = all four shards successful。
- C2 lambda_E=650 training：`1785676`；fixed-4 trajectory：`1785678`；Val28 final：`1785679`。
- C2 lambda_E=2000 training：`1785677`；fixed-4 trajectory：`1785680`；Val28 final：`1785681`。
- C2 evaluation jobs分别使用对应 training job 的 `afterok` dependency；两档 training彼此独立。

## B1 Full28 / Dev14 evidence

### Full28（28 models / 792 H5）

| Endpoint | Physical bpp | direct pc_error YUV-PSNR 6:1:1 | Full−Base |
|---|---:|---:|---:|
| B1 step4 Base | 0.567024 | 36.484142 dB | — |
| B1 step4 Full | 1.239581 | 41.045287 dB | +0.672557 bpp / +4.561144 dB |
| B1 step8 Base | 0.452689 | 35.490102 dB | — |
| B1 step8 Full | 1.239110 | 41.045287 dB | +0.786421 bpp / +5.555185 dB |

Official same-contract neighbors：

| R point | Physical bpp | YUV-PSNR 6:1:1 |
|---|---:|---:|
| R02 | 1.240016 | 41.045284 dB |
| R03 | 0.952912 | 39.330371 dB |
| R04 | 0.426228 | 35.228702 dB |

- step4 layered/original physical ratio：`0.999558`。
- step8 layered/original physical ratio：`0.999383`。
- 两个 step 的 exact Full `max_abs_difference=0`。
- step8 Base 相对 R04：`+0.026461 bpp / +0.261400 dB`。这是直接 endpoint comparison，不使用 interpolation 作 formal conclusion。

### Dev14（14 models / 312 H5，从同一 Full28 per-H5 raw data派生）

| Endpoint | Physical bpp | direct pc_error YUV-PSNR 6:1:1 | Full−Base |
|---|---:|---:|---:|
| B1 step4 Base | 0.578251 | 36.532649 dB | — |
| B1 step4 Full | 1.292194 | 40.971315 dB | +0.713943 bpp / +4.438666 dB |
| B1 step8 Base | 0.450342 | 35.636466 dB | — |
| B1 step8 Full | 1.291685 | 40.971315 dB | +0.841343 bpp / +5.334849 dB |

Official Dev14：R02 `1.292161 / 40.971315 dB`；R03 `0.994678 / 39.263161 dB`；R04 `0.422514 / 35.293846 dB`。

Full28 与 Dev14 方向一致。B1 fixed step4/8 generalization PASS；本轮没有 quantizer search、B1 training或 Stage5 specialization。

## C2 fixed-4 trajectory

Base fixed：`0.028598 bpp / 34.81388 dB`。

### lambda_E=650

| Step | EL bpp | Full bpp | Full PSNR | Gain | nonzero | active channels | range |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 705 | 0.022264 | 0.050862 | 35.113906 | +0.300015 dB | 0.000327 | 1/32 | [-2,2] |
| 1410 | 0.016717 | 0.045315 | 35.084895 | +0.271004 dB | 0.000239 | 1/32 | [-2,2] |
| 2115 | 0.027162 | 0.055759 | 35.122504 | +0.308613 dB | 0.000370 | 1/32 | [-2,2] |
| 2820 | 0.019891 | 0.048488 | 35.032848 | +0.218957 dB | 0.000284 | 1/32 | [-2,2] |
| 3525 | 0.014887 | 0.043485 | 35.026789 | +0.212898 dB | 0.000228 | 1/32 | [-2,2] |

### lambda_E=2000

| Step | EL bpp | Full bpp | Full PSNR | Gain | nonzero | active channels | range |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 705 | 0.098485 | 0.127082 | 36.029047 | +1.215168 dB | 0.002095 | 1/32 | [-4,4] |
| 1410 | 0.096080 | 0.124678 | 36.018749 | +1.204869 dB | 0.002110 | 1/32 | [-5,5] |
| 2115 | 0.103978 | 0.132576 | 36.075131 | +1.261252 dB | 0.002239 | 1/32 | [-5,4] |
| 2820 | 0.106014 | 0.134612 | 36.156835 | +1.342956 dB | 0.002302 | 1/32 | [-4,4] |
| 3525 | 0.093776 | 0.122373 | 35.895122 | +1.081242 dB | 0.002019 | 1/32 | [-5,4] |

两档均有 GT-dependent hard symbols和真实 quality gain，但 trajectory 非单调；`1/32` active-channel pattern经过完整 epoch没有改变。Final checkpoint并非 fixed-4 上的最佳 intermediate checkpoint。

## C2 Val28 generalization

| Endpoint | Base bpp/PSNR | EL bpp | Full bpp/PSNR | Full−Base |
|---|---|---:|---|---:|
| lambda_E=650, step3525 | 0.099592 / 31.537844 dB | 0.040517 | 0.140109 / 32.019667 dB | +0.481823 dB |
| lambda_E=2000, step3525 | 0.099590 / 31.537847 dB | 0.191079 | 0.290669 / 33.446791 dB | +1.908944 dB |

Official Full28 direct comparisons：

- R07：`0.152390 / 32.523601 dB`。lambda650 比 R07 少 `0.012281 bpp`，但低 `0.503934 dB`。
- R06：`0.218998 / 33.399184 dB`。lambda2000 多 `0.071671 bpp`，仅高 `0.047607 dB`。
- R05：`0.324253 / 34.477281 dB`。lambda2000 少 `0.033584 bpp`，但低 `1.030490 dB`。

因此 C2 one-epoch 证明了 generalization，而不是 micro-overfit；但两个 endpoint 均没有显示接近 official native RD efficiency。lambda650 没有在完整训练后追上 rate-matched official enhancement；lambda2000 的较大 gain伴随明显 rate cost。

Training protocol/runtime：两档均 exactly 3,525 updates、BS4、constant `lr=5e-5`、seed0；V100 runtime分别约 `2:09:04 / 2:10:23`，peak model-reported memory约 `11.185 GiB`。训练与 hard evaluation无 NaN/Inf failure。

## Architecture Review evidence summary

| Candidate | Evidence from this round | Review status（非 Final Decision） |
|---|---|---|
| B1 fixed nested native latent | Full28/Dev14均保持 near-envelope Base；exact R02 Full；layered physical ratio约1 | **strengthened / formal prototype candidate evidence** |
| C2 appended native ResidualVAE | 一完整 epoch可 generalize，lambda response有效；但1/32 channel持续且RD明显弱于official native curve | **generalization PASS, efficiency WARNING** |

本轮不包含 B1 quantizer tuning/training，也不包含 C1/C3/B2。以上仅为下一次 Architecture Review 的 evidence，不宣布 Final Architecture Decision。
