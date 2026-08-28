# C2 / B1 Parallel De-risking

## 2026-08-21 local-correction / generalization round

- C2 same-4 official R07/R08 local reference：Job `1785553`；输出根目录：
  `$WORK/scalable_attribute_thesis/experiments/c2_calibration_local_reference_r07_r08_v1/`。
- B1 fixed mapping (`step=4/8`) Full28 shards：Jobs `1785554–1785557`；输出根目录：
  `$WORK/scalable_attribute_thesis/experiments/b1_fixed_mapping_generalization_v1/`。
- B1 GPU evaluation 只生成 Full28 raw `per_h5.csv`；完成后由同一 raw data 派生 Full28/Dev14 `per_model.csv` 与 `endpoint_summary.csv`，不重复 hard coding。
- 本轮不做 quantizer search、不训练 Stage5、不提交 C2 one-epoch、不恢复 C3。
- 首次 submission `1785546–1785550` 因 `sbatch --wrap` 的 `/bin/sh` 不支持 `set -o pipefail`，在 evaluation 前退出/取消；修正只移除了 Bash-only shell option，实验参数未改变。

> 日期：2026-08-21  
> 状态：C2 calibration + B1 physical RD completed。  
> 本文件不是 Final Architecture Decision。

## Track A — C2 calibration

固定 contract：official R08 Base、150 updates、BS4、lr `5e-5`、seed 0、相同 RWTT shuffled sequence、相同 4-H5 RWTT Val hard subset。

### Training final

| lambda_E | step-150 R_E | step-150 D_F | step-150 total loss | EMA R_E | EMA D_F | EMA loss |
|---:|---:|---:|---:|---:|---:|---:|
| 650 | 0.063179 | 0.000307996 | 0.263377 | 0.087538 | 0.000535202 | 0.435419 |
| 2000 | 0.179466 | 0.000222895 | 0.625255 | 0.237841 | 0.000389617 | 1.017075 |
| 6500 | 0.336438 | 0.000188005 | 1.558473 | 0.413414 | 0.000337992 | 2.610365 |

Raw step-150 是同一 shuffled-sequence 的最后一个 batch；EMA 用于避免单 batch complexity 波动掩盖整体趋势。

### Fixed 4-H5 actual hard validation

| lambda_E | Base bpp | EL bpp | Full bpp | Base YUV-PSNR | Full YUV-PSNR | Gain | nonzero fraction | active channels | range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 650 | 0.028598 | 0.017403 | 0.046001 | 34.813879 | 34.876496 | +0.062616 dB | 0.000215 | 1/32 (`29`) | [-2,2] |
| 2000 | 0.028598 | 0.102837 | 0.131434 | 34.813879 | 35.924187 | +1.110308 dB | 0.002062 | 1/32 (`29`) | [-4,4] |
| 6500 | 0.028598 | 0.241666 | 0.270264 | 34.813879 | 37.249177 | +2.435297 dB | 0.005664 | 1/32 (`29`) | [-7,7] |

所有 bpp 均为 actual Base/EL hard-coded bytes；所有 PSNR 均为 direct/per-model-corrected author `pc_error` YUV 6:1:1。固定 subset 中每个 original model 只有一个 H5。

### Runtime and memory

| lambda_E | Slurm elapsed | median s/update | mean s/update | 4-H5 hard seconds | peak GPU memory |
|---:|---:|---:|---:|---:|---:|
| 650 | 6m46s | 2.127 | 2.125 | 27.28 | 10.107 GiB |
| 2000 | 6m51s | 2.178 | 2.178 | 26.77 | 10.107 GiB |
| 6500 | 6m40s | 2.131 | 2.131 | 27.26 | 10.107 GiB |

### C2 provenance

```text
RESULT PROVENANCE

Result:
Three-point lambda calibration with monotonic hard rate/activity/quality response.

Source:
Training estimated likelihood rate; actual torchac hard EL bytes;
author pc_error direct YUV-PSNR 6:1:1.

Data:
Training: normal RWTT Train manifest, first 150 updates of identical seeded order.
Hard validation: fixed 4 H5 / 4 original RWTT Val models.

Training:
150 updates, BS4, lr=5e-5, constant LR, official R08 frozen Base.

Metric:
Training normalized-YUV 1:1:1 MSE and N0-normalized rate;
hard physical bpp and author YUV-PSNR 6:1:1.

Can be used for formal RD?
NO — calibration subset and 150 updates only.
```

### C2 calibration decision input

- `lambda_E` 增大时，hard EL rate、nonzero fraction、symbol range、Full quality 均明显增大；distortion 下降。
- 因此 C2 不属于 `FAIL-TO-CALIBRATE`，也不是三档全部 collapse。
- 三档始终只使用 channel 29；按 contract 不修改 architecture、不加 auxiliary loss。
- Official R08→R07 target 约 `+0.053 bpp / +0.95–0.99 dB`。λ650 明显 under-spend/under-correct；λ6500 明显 overspend；λ2000 quality gain 最接近，但 fixed-4 EL rate约为 target 的 1.9 倍。
- Recommended lambda for one-epoch：**2000**。它是三个已测设置中唯一处于 sensible middle regime 的点；λ650 under-correct，λ6500 overspend。λ2000 在 fixed-4 上的 EL rate 仍约为 global R08→R07 target increment 的 1.9 倍，这是后续 one-epoch 审核必须保留的 warning，而不是 formal RD prediction。
- C2 calibration status：**HEALTHY**。存在清晰、方向正确的 λ response；150 updates 不能证明 formal RD/generalization。
- 不自动提交 one-epoch，等待 Project Lead 明确放行。

## Track B — B1 physical RD

Completed Job：`1785501`，2m17s，RTX3080。前两个 attempts 仅为 execution failure（missing GPCC symlink；CPU/CUDA decoded-symbol placement），未产生 RD evidence；quantizer、samples、steps 和 native traversal 未改变。

### Physical Base/Full RD

四个 H5 分属四个 RWTT Validation models；表中是 model-equal mean。Base bits 包含 actual native `x_low+r1...r4` prefix 和 qB torchac；Full bits 再加 conditional qE torchac。

| Step | Base bpp | Full bpp | Base direct YUV-PSNR | Full direct YUV-PSNR | Full−Base | mean layered/original ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.526691 | 0.764491 | 39.309713 | 41.615453 | +2.305741 dB | 1.008552 |
| 4 | 0.300154 | 0.758767 | 38.760156 | 41.615453 | +2.855297 dB | 1.000541 |
| 8 | 0.242482 | 0.758268 | 38.452803 | 41.615453 | +3.162650 dB | 1.001126 |

所有 step：

- qB physical round-trip exact；
- qB + conditional qE exact 恢复 qfine；
- layered Full 与 official R02 hard reconstruction `max_abs_difference=0`；
- Full direct `pc_error` 与 official R02 完全相同。

### Same-H5 official physical baselines

| Official point | Physical bpp | direct pc_error YUV-PSNR 6:1:1 |
|---|---:|---:|
| R02 | 0.758304 | 41.615453 dB |
| R03 | 0.550941 | 40.197600 dB |
| R04 | 0.216411 | 38.144622 dB |

Apples-to-apples interpretation：

- step2 Base rate 接近 R03，但低约 `0.888 dB`，该 mapping 较弱。
- step4 Base 位于 R04/R03 之间，接近两点的 local interpolation；Full overhead 约 `0.054%`。
- step8 Base 位于 R04 稍上方，quality 也相应提高；Full overhead 约 `0.113%`。
- 本轮未使用 estimated B1 rate 与 physical official rate 混合判断，也未按 evaluation H5 单独选择 step。

### B1 provenance

```text
RESULT PROVENANCE

Result:
Physical Base/Full RD for deterministic nested steps 2/4/8;
exact Full preservation for all 12 sample-step cases.

Source:
Native prefix GPCC + r1-r4 actual bytes;
qB and conditional qE actual torchac bytes;
direct author pc_error YUV-PSNR 6:1:1.

Data:
Fixed 4 H5 / 4 original RWTT Validation models.

Training:
None. Released official R02 model; no Stage5 training/unsharing.

Metric:
Physical Base/Full bpp and direct pc_error on exactly the same H5;
official R02/R03/R04 baselines recomputed on those H5.

Can be used for formal RD?
NO — four-model candidate screening evidence, not Full28/Dev14.
```

### B1 status

- B1 physical RD：**PROMISING**。
- Main remaining bottleneck：**quantizer mapping**。
- Syntax overhead、decoder-known mapping 和 exact Full preservation 已不再是当前主要风险。
- 若继续 B1，应只在 calibration/train data 上选择一个 global deterministic mapping，然后 freeze 后评价 held-out data；不能 per-evaluation-H5 oracle selection。

## Round-level review input

```text
C2 recommended lambda for one-epoch: 2000
C2 calibration status: HEALTHY

B1 physical RD: PROMISING
B1 main remaining bottleneck: quantizer mapping
```

本轮 evidence 支持同时保留：

- A：C2 R08/λE=2000 one-epoch generalization，需 Project Lead 明确放行；
- B：B1 升级为 main-candidate-level investigation，下一步仍应是 frozen global quantizer evidence，而非训练/unsharing。

没有恢复 C3，没有新增 architecture，也没有提交 C2 one-epoch。
