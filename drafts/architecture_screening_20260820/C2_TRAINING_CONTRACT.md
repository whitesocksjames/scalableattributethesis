# C2 Training Calibration Contract

> 日期：2026-08-21  
> 状态：Calibration Gate contract；不是 one-epoch 放行。  
> Architecture frozen：本轮不修改 C2 topology、不加 auxiliary loss。

## Dataset contract

Split：`model_95_5_seed0`，按 original RWTT model 划分。

| Split | Original models | H5 blocks | Shuffle | drop_last |
|---|---:|---:|---|---|
| Train | 523 | 14,098 | True | False |
| Validation | 28 | 792 | False | False |

- Train/Validation model overlap：`0`。
- Proposed batch size：`4`。
- Train steps per full epoch：`ceil(14,098 / 4) = 3,525`。
- 最后一个 train batch：`2 H5`，不会丢弃。
- Full validation loader steps：`792 / 4 = 198`。
- Calibration training 每档只跑 `150 updates`，约 `0.0426 epoch`。
- 三个 λ 使用相同 `seed=0`、相同 DataLoader policy 和相同 shuffled sequence。

Calibration hard validation 固定为 4 个 RWTT Validation H5，不重新抽样：

```text
RWT115/model_mesh_P0.h5
RWT182/572883_P15.h5
RWT380/ujety_svah_ske_P15.h5
RWT541/marco_cat_mesh_P9.h5
```

四个 H5 分属四个 original models。该 subset 仅用于 λ calibration，不作为 formal RD dataset。

## Model and parameter contract

Base operating point：official R08：

```text
checkpoint = 2k128/epoch_last.pth
lambda_Base = 256
```

| Component | Parameters | State |
|---|---:|---|
| Frozen Unicorn Base | 17,085,155 | `eval()`, `requires_grad=False`, `no_grad()` |
| C2 appended native ResidualVAE | 12,871,139 | trainable |

- Base reconstruction/state：`B, F_U, D_U`，由同一个 frozen R08 Base deterministic symbol path 得到。
- C2 输入/输出仍为既定 native append：`B, A, F_U, D_U, lambda embedding → Full`。
- C2 initialization：从 official R08 checkpoint 中 released shared `ResidualVAE` 权重完整复制到 appended VAE；不是 random initialization。
- Optimizer 只接收 `model.enhancement.parameters()`；Base parameters 不进入 optimizer。
- 每次 forward/backward 后检查 Base 没有 gradients。

## Objective contract

```text
L = R_E + lambda_E * D_F
```

### Rate

```text
R_E = -sum(log2(likelihood_E)) / N0
```

- `N0 = len(A)`：当前 batch 的 full-resolution active point count。
- 分母不是 latent point count，也不是 batch size。
- sum 覆盖全部 C2 latent sites 和 channels。
- Training 使用原 author-style Uniform noise differentiable quantization。
- Hard calibration 使用 actual torchac compress/decompress bytes。

### Distortion

```text
D_F = mean((A.F - Full.F)^2)
```

- domain：normalized YUV `[0,1]` tensor domain。
- channels：Y/U/V 等权 `1:1:1`。
- mean 覆盖 full-resolution points × 3 Attribute channels。
- 该 loss-domain distortion 不等于 hard evaluation 的 author `pc_error YUV-PSNR 6:1:1`；后者只用于 validation endpoint reporting。

## Optimization contract

```text
optimizer   = Adam
lr          = 5e-5
betas       = (0.9, 0.999)
weight_decay= 0
scheduler   = None
warmup      = None
grad_clip   = None
batch_size  = 4
seed        = 0
```

- Constant LR。
- Non-finite loss/rate/distortion/gradient：fail fast。
- 不 skip batch、不自动改 batch size/λ/lr、不自动 retry semantic failure。

### Protocol provenance

以下 training protocol 来自原作者邮件说明，并在 thesis calibration 中保持原语义：

- loss-domain Attribute distortion：normalized YUV `1:1:1` MSE；
- rate normalization：使用 full-resolution point count `N0`；
- initial learning rate：`5e-5`；
- learning rate policy：constant LR，不加 scheduler/warmup。

Hard evaluation 仍使用作者 `pc_error` YUV-PSNR `6:1:1`；它与 training loss-domain `1:1:1` 是两个不同但固定的 convention。

## Lambda calibration bracket

固定 architecture、optimizer 和 data sequence，只比较：

```text
lambda_E = [650, 2000, 6500]
```

选择依据是现有 R08 C2 smoke step 2：

```text
R_E = 0.098318
D_F = 0.000113562

lambda=650  -> lambda*D ~= 0.0738
lambda=2000 -> lambda*D ~= 0.2271
lambda=6500 -> lambda*D ~= 0.7382
```

因此 bracket 从 roughly rate-balanced 覆盖到 distortion-dominant，且保留 `6500` 作为已有 upper/reference setting。它不是 formal multi-λ sweep。

每档：

```text
150 normal-RWTT Train updates
final checkpoint only
same fixed 4-H5 RWTT Val hard evaluation
```

记录：actual hard EL bpp、Base/Full author `pc_error YUV-PSNR 6:1:1`、quality gain、nonzero fraction、symbol range、active channels。

## Calibration target

Corrected official neighboring increment：

| Dataset | R08→R07 Δbpp | R08→R07 ΔYUV-PSNR 6:1:1 |
|---|---:|---:|
| Full28 | +0.052798 | +0.985754 dB |
| Dev14 | +0.053226 | +0.953599 dB |

Calibration 不要求命中 formal curve；只选择一个既不明显 collapse、也不明显 overspend rate 的 λ，供后续单点 one-epoch generalization 审核。

## Measured resource evidence and estimate

直接 C2 evidence：

- Previous R08 C2 200-update BS1 probe：`0.834 s/update`，peak约 `2.02 GiB`。
- Normal-RWTT C2 BS1 smoke：首步含初始化 `3.32 s`，第二步 `0.69 s`，peak `2.02 GiB`；train + 1-H5 hard evaluation 整个 Slurm job `34 s`。

BS4 resource envelope（已有 external EL V100 run，不冒充 C2 direct measurement）：

- 3,525-step epoch median `3.015 s/update`，training compute累计约 `2.96 h`。
- observed peak `14.12 GiB`；worst-4 one-step smoke `9.89 GiB`。

因此 calibration 保守估计：

```text
150 updates training: about 7–12 min
4-H5 hard validation: about 0.5–1 min
requested walltime: 30 min
```

由于已有 BS4 envelope 可超过 RTX3080 10 GiB，三个 calibration jobs 使用同型 V100，避免不同 λ 落到不同 GPU 或在 RTX3080 上发生硬件容量混杂。Calibration 完成后以其实际 peak memory / step time 更新 one-epoch estimate。

## Gate

- Calibration 结束后只提交结果，不能自动提交 one-epoch。
- `1/32 active channels` 不触发 architecture change 或 auxiliary loss。
- Project Lead 明确选择一个 λ 并放行后，才允许提交 official R08 one-epoch generalization run。

## Calibration execution

| lambda_E | Job ID | Resource | Output root |
|---:|---:|---|---|
| 650 | `1785475` | V100, 30 min | `.../c2_lambda_calibration_r08_v1/lambda_650/` |
| 2000 | `1785476` | V100, 30 min | `.../c2_lambda_calibration_r08_v1/lambda_2000/` |
| 6500 | `1785477` | V100, 30 min | `.../c2_lambda_calibration_r08_v1/lambda_6500/` |

三个 jobs 独立，无相互 dependency，也没有连接 one-epoch job。提交时均为正常 `PENDING`；结果完成后先统一回到 Project Lead 审核。

### Calibration measured update

三个 jobs 均 `COMPLETED`，直接 C2 BS4 measurement：

| lambda_E | Slurm elapsed | median s/update | mean s/update | peak memory | 4-H5 hard time |
|---:|---:|---:|---:|---:|---:|
| 650 | 6m46s | 2.127 | 2.125 | 10.107 GiB | 27.28 s |
| 2000 | 6m51s | 2.178 | 2.178 | 10.107 GiB | 26.77 s |
| 6500 | 6m40s | 2.131 | 2.131 | 10.107 GiB | 27.26 s |

按纯 training step time 外推，3,525 steps 约 `2.08–2.13 h`；正式 job 还必须另加 validation/checkpoint 开销并留 walltime margin。该估算不构成 one-epoch submission authorization。
