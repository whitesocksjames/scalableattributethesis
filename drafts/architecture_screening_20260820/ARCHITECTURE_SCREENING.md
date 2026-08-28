# Architecture Screening — Round 1

日期：2026-08-20

本轮目标不是选择最终 architecture，而是用最低成本证据判断三条 family
是否存在 RD headroom 和 thesis contribution potential。正式 Unicorn source、
geometry、entropy codec 均未修改；没有启动长训练、sweep、Dev14 或 full eval。

## 固定测试条件

- Family A/B：official R02，`32k8k + lambda=16384`。
- Family C：official R09，`2k128 + lambda=128`。
- 固定四个 H5：
  - `RWT115/model_mesh_P0.h5`
  - `RWT182/572883_P15.h5`
  - `RWT380/ujety_svah_ske_P15.h5`
  - `RWT541/marco_cat_mesh_P9.h5`
- A 的前两个 H5 是 `micro_train`，后两个是 `holdout`。
- A 的主要 metric 是 author `pc_error` YUV-PSNR 6:1:1；B/C 是
  screening-only tensor PSNR。
- B/C 的 `ideal bpp` 是 per-channel empirical marginal entropy，不是
  `torchac` physical bits，不能直接作为正式 RD 数据。

## Family A — Native no-bit continuation

测试 boundary 是 r5，仅作为最低成本 test bed。Base payload 仍是
`x_low+r1+r2+r3+r4`；Base rate 沿用此前同一四 H5 的 physical probe：
R02 平均约 `0.23851 bpp`，released Full 约 `0.78226 bpp`。

比较：

- `symbols=0`；
- `symbols=round(loc)`；
- continuous `loc` diagnostic；
- learned no-bit predictor：复制 native `loc_net`，冻结整个 Base，只训练该
  独立 predictor 100 updates，MSE-only，`lr=1e-3`；predictor 输出 continuous
  EQ-domain value，经 frozen DQ/decoder/fusion/out path 得到 Base。

四个模式均不增加 per-sample Base bits。结果是两样本算术平均：

| Split | Continuation | Base author YUV-PSNR | Released Full | Remaining gap |
|---|---|---:|---:|---:|
| micro-train | zero | 41.7181 | 43.0286 | 1.3106 |
| micro-train | round(loc) | 41.7509 | 43.0286 | 1.2778 |
| micro-train | continuous loc | 41.7686 | 43.0286 | 1.2601 |
| micro-train | learned | 41.8110 | 43.0286 | 1.2177 |
| holdout | zero | 35.0107 | 40.2023 | 5.1916 |
| holdout | round(loc) | 35.0991 | 40.2023 | 5.1032 |
| holdout | continuous loc | 35.1326 | 40.2023 | 5.0696 |
| holdout | learned | 35.0567 | 40.2023 | 5.1455 |

事实：learned 相对 zero 在 micro-train 上约 `+0.093 dB`，但只比
continuous loc 高约 `+0.042 dB`；holdout 上 learned 比 zero 高约
`+0.046 dB`，却比 round(loc) 低约 `0.042 dB`、比 continuous loc 低约
`0.076 dB`。

结论边界：当前最小 predictor 没有显示出足以优先投入正式 architecture 的
no-bit continuation capacity。它没有证明所有 learned Base continuation 都不可行，
但足以把这条具体 r5/loc-net formulation 暂时降级。

## Family B — Within-stream successive refinement oracle

对捕获的 native r5 integer symbols 做 offline step-size split：

```text
coarse = round(symbol / step) * step
refinement = symbol - coarse
```

`coarse+refinement` 可无损恢复原 symbol。没有改 `torchac`，rate 只比较
empirical marginal entropy：

| Step | Original bpp | Coarse bpp | Refinement bpp | Layered bpp | Rate ratio | Coarse PSNR | Full PSNR | Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.7152 | 0.1617 | 0.5663 | 0.7279 | 1.0079 | 40.6705 | 43.1620 | 2.4915 |
| 4 | 0.7152 | 0.0484 | 0.6773 | 0.7258 | 1.0066 | 40.0258 | 43.1620 | 3.1362 |
| 8 | 0.7152 | 0.0076 | 0.7112 | 0.7188 | 1.0018 | 39.6646 | 43.1620 | 3.4975 |

事实：在这个四-H5/R02/r5 oracle 中，coarse symbols 能形成有用的 lower-rate
reconstruction，且把 coarse/refinement 分开编码的 empirical entropy overhead
只有约 `0.18%–0.79%`。

结论边界：这是目前最强 positive signal，但尚未证明 conditional entropy 下仍有
同样结果，也未证明真正 nested arithmetic payload、其他 stages/rate points 或训练
可行性。

## Family C — Hybrid residual representation oracle

目标 residual 是 `A-B_R09`。比较 stride-1 residual 与按 geometry-derived
stride-2/4 parent support 求均值的 spatial representation。每个 parent value 只计
一次 symbol，rate 以 full-resolution `N0` 归一化；decode 时再 broadcast 回原 support。

下表为四 H5 平均：

| Spatial factor | Quant step | Parent count | Ideal bpp | Oracle gain | Distortion removed |
|---:|---:|---:|---:|---:|---:|
| 1 | 4/255 | 82152 | 5.0998 | +11.8075 dB | 91.49% |
| 1 | 8/255 | 82152 | 2.5880 | +7.3007 dB | 76.22% |
| 2 | none | 22845 | — | +6.8725 dB | 75.70% |
| 2 | 4/255 | 22845 | 1.3354 | +5.0198 dB | 67.22% |
| 2 | 8/255 | 22845 | 0.6470 | +3.2866 dB | 52.74% |
| 4 | none | 5974 | — | +3.8859 dB | 55.26% |
| 4 | 4/255 | 5974 | 0.3207 | +2.8990 dB | 46.86% |
| 4 | 8/255 | 5974 | 0.1453 | +1.8153 dB | 33.57% |

四个 sample 的差异明显：stride-2 unquantized gain 约 `3.31–9.22 dB`，
stride-4 unquantized gain 约 `1.42–5.75 dB`。这说明 residual 含有可压缩的
lower-resolution component，但 signal 的 spatial character strongly
content-dependent。

结论边界：EL V1 collapse 不能解释为 residual 本身没有 headroom。相反，结果支持
研究 geometry-derived spatial bottleneck 和 native conditional representation；
它不支持直接编码 raw RGB/YUV residual，因为 full-resolution empirical rate 很高。

## Round-1 status

| Family | 当前状态 | 原因 |
|---|---|---|
| A: learned native no-bit continuation | 暂时降级 | 仅有很小 train gain，holdout 不超过 deterministic loc baselines |
| B: within-stream successive refinement | 优先保留 | coarse endpoint 有用，ideal split overhead 极小 |
| C: hybrid/native-spatial residual | 优先保留 | stride-4 在约 0.15–0.32 empirical bpp 下仍保留约 1.8–2.9 dB oracle gain |

这张表是 screening priority，不是最终 architecture selection。
