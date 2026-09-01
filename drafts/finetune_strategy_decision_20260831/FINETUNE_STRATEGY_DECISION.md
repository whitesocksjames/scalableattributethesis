# 32K Fine-tuning Strategy Evidence Review

## 结论先行

当前最有证据支持的 32K recipe 是：

- 保留现有 **D111 Canonical Base step5525**，并在 Enhancement training 中完全冻结；
- Enhancement 使用 released R01 `ResidualVAE` 的 independent clone 初始化；
- 使用 RWTT、`LR=5e-5`、`BS=4`、constant LR、`rd_lambda=conditioning_lambda=32768`；
- 当前 provisional best 是 `D111 1763 updates -> D611 1762 updates`，总共 3525 optimizer updates；
- 不采用 Base D611、full-unfreeze、TAFA-style random Base/Full 或 MVUB mixing 作为当前正式 recipe。

封版前只剩 **一个必要实验**：将 clean Direct-D611 M0 从 step1762 resume 到总 step3525，并与 two-stage D611 step3525 做 matched-budget hard RD comparison。在这项证据回来前，`D111 warm-up required?` 必须保持 **UNRESOLVED**。

本轮是 read-only synthesis，没有启动 training/evaluation job，也没有修改 model、loss 或 architecture。

## 1. Evidence contract 与来源

主要结果根目录：

```text
/home/liltan/projects/Scalable-Attribute-Thesis-results/n30_20260831/
```

Formal RWTT contract：28 original models / 792 H5；physical hard rate；每个 H5 先保留 channel MSE，point-weighted 聚合为 per-model MSE，再转换 PSNR，最后 model-equal average；quality 为 author `pc_error` Y/U/V 与 YUV 6:1:1。

External contract：8iVFB fixed-4，即 Longdress 1300、Loot 1200、Redandblack 1550、Soldier 0690；physical hard rate；四 sequence equal average。

本报告直接核对的主要 raw artifacts：

- Base D611：`base_d611_r01/train_3525/resolved_args.json`、`eval_full28/per_model.csv` 和 `eval_8ivfb/*/physical_rd.csv`；
- D411/D611：`D411_step3263_full28/per_model.csv`、`D611_step3525_full28/per_model.csv`；
- D111 formal：`drafts/canonical_scalable_formal_r01/full28_step1763/`、`full28_checkpoint_audit/`、`8ivfb_external/*/physical_rd.csv`；
- MVUB A/B/C：`overnight_pass1_5c71343/*/train/resolved_args.json`、`MVUB_OVERNIGHT_ALL_RESULTS.csv`、`supplementary_eval_v1/`；
- Direct-D611 mixing：`direct_d611_mixing_v1/train/*/resolved_args.json`、`8ivfb/*/physical_rd.csv`、`rwtt_full28/*/per_model.csv`。

为补齐本地结果包未复制的 provenance，本轮还只读核对了 N30 `Tanzeyu` namespace 内 D111、D411、D611 和 Base 的 `resolved_args.json`。没有读取其他用户目录。

## 2. Model lineage：global step 不等于单段训练预算

| Model/checkpoint | Initialization | Trainable modules | Loss | Dataset | LR / BS | Actual optimizer updates | Optimizer |
|---|---|---|---|---|---|---:|---|
| Canonical Base D111 step5525 | clean zero-init BaseSynthesis + released R01 prefix | BaseSynthesis only；prefix/native modules frozen | D111 Base MSE | RWTT | 3e-4 / 4 | 5525 total：0→500、500→2000、2000→5525 | fresh at step0；weights + Adam resumed across segments |
| Base D611 continuous step3525 | clean zero-init BaseSynthesis + released R01 prefix | BaseSynthesis only | D611 Base MSE | RWTT | 3e-4 / 4 | 3525 continuous | fresh Adam |
| Enhancement D111 step1763 | independent exact clone of released R01 ResidualVAE；fixed D111 Base step5525 | EnhancementVAE only；Base/prefix frozen | `R_E + 32768*D111_Full` | RWTT | 5e-5 / 4 | 1763 from clean Enhancement initialization | fresh Adam |
| D411 candidate step3263 | D111 step1763 | EnhancementVAE only | `R_E + 32768*D411_Full` | RWTT | 5e-5 / 4 | +1500；total 3263 | Enhancement weights + Adam resumed；LR unchanged |
| D611 pre-MVUB step3525 | D111 step1763 | EnhancementVAE only | `R_E + 32768*D611_Full` | RWTT | 5e-5 / 4 | +1762；total 3525 | Enhancement weights + Adam resumed；LR unchanged |
| A1 step7944 | pre-MVUB D611 step3525 | EnhancementVAE only | Full D611 | MVUB David/Phil/Ricardo/Sarah | 1e-5 / 2 | 7944 = one MVUB pass | fresh Adam |
| B1 step4000 selected | pre-MVUB D611 step3525 | prefix + BaseSynthesis + EnhancementVAE (`full`) | Full-only D611 with `R_B_est+R_E_est` | same MVUB | 1e-5 / 2 | 4000 selected from a 7944-update run | fresh Adam |
| C2 step4000 selected | pre-MVUB D611 step3525 | prefix + BaseSynthesis + EnhancementVAE (`full`) | 50/50 random Base/Full endpoint | same MVUB | 2e-5 / 2 | 4000 selected from a 7944-update run | fresh Adam |
| Direct M0 step1762 | independent exact clone of released R01 ResidualVAE；fixed D111 Base step5525 | EnhancementVAE only | Full D611 | RWTT only | 5e-5 / physical=effective 4 | 1762 | fresh Adam；没有加载旧 Enhancement checkpoint |
| Direct M10 step1762 | same clean initialization as M0 | EnhancementVAE only | Full D611 | 1586 RWTT + 176 MVUB updates | 5e-5 / 4 | 1762 | fresh Adam |
| Direct M25 step1762 | same clean initialization as M0 | EnhancementVAE only | Full D611 | 1321 RWTT + 441 MVUB updates | 5e-5 / 4 | 1762 | fresh Adam |

重要解释：

- Base `step5525` 确实是 5525 次累计 optimizer updates，但最后一段 `2000→5525` 才是一次完整 RWTT loader pass；它不是“一个 5525-step epoch”。
- D411 `step3263` 是 D111 1763 后再训练 1500 次；D611 `step3525` 是 D111 1763 后再训练 1762 次。
- Direct M0 `step1762` 只有 1762 次总训练；不能与总预算 3525 的 two-stage D611 强行归因比较。

## 3. Hypothesis evidence matrix

### H1 — Base 是否需要 D611？

同一 hard Base bits 下，D611 continuous step3525 没有改善 quality：

| Dataset | Base | bpp | Y | U | V | YUV611 |
|---|---|---:|---:|---:|---:|---:|
| RWTT Full28 | D111 step5525 | 0.523916 | 32.3849 | 45.2608 | 46.6829 | 35.7816 |
| RWTT Full28 | D611 step3525 | 0.523916 | 32.3663 | 45.0282 | 46.4766 | 35.7129 |
| 8i fixed-4 | D111 step5525 | 0.453447 | 38.0281 | 45.3850 | 44.1556 | 39.7136 |
| 8i fixed-4 | D611 step3525 | 0.453447 | 38.0042 | 45.1881 | 44.0533 | 39.6583 |

RWTT `ΔYUV611=-0.06877 dB`，8i `ΔYUV611=-0.05534 dB`；physical Base bpp 不变。RWTT 28/28 models、8i 4/4 sequences 都向负方向移动。

训练预算并不严格 matched（D111 5525 vs D611 3525），所以这不是“D611 objective 本质更差”的纯因果证明；但作为是否替换当前 Base 的工程决策，证据明确是 **NO-GO**。按此前 gate，不再为这个负结果追加 clean D111 control。

### H2 — Base/Prefix 是否需要 full-unfreeze？

pre-MVUB 与选中的 A1/B1/C2 Full28：

| Candidate | Base bpp / YUV611 | Full bpp / YUV611 | 主要现象 |
|---|---|---|---|
| pre-MVUB D611 | 0.523916 / 35.7816 | 2.133239 / 45.2926 | reference |
| A1 step7944 | 0.523916 / 35.7816 | 1.933296 / 43.7296 | Base exact preserved；Full `-1.5631 dB` |
| B1 step4000 | 0.674082 / 35.5469 | 2.038450 / 43.6087 | Base `+0.1502 bpp/-0.2347 dB`；Full `-1.6840 dB` |
| C2 step4000 | 0.914242 / 35.8133 | 2.145622 / 43.6109 | Base `+0.3903 bpp/+0.0317 dB`；Full `-1.6818 dB` |

8i 上 B/C 虽提升 Base Y，却显著损失 U/V；Full 也都比 pre-MVUB 左下移动。C 的 50/50 endpoint sampling 确实作用到了 Base，但没有形成合理的 general-purpose RD improvement。

结论：**NO-GO（High confidence）**。正式 32K recipe 保持 Prefix、native modules 和 BaseSynthesis frozen，只训练 Enhancement。

### H3 — MVUB 是否应该进入正式 recipe？

Pure MVUB A1 已经表明 target-domain fine-tuning 会损害 generalization：8i Full 从 `1.383888/45.9603` 移到 `1.209140/44.6818`，RWTT Full 从 `2.133239/45.2926` 移到 `1.933296/43.7296`。

更干净的 causal comparison 是同初始化、同预算的 M0/M10/M25：

| Dataset | Arm | Full bpp | YUV611 | 相对 M0 |
|---|---|---:|---:|---|
| 8i fixed-4 | M0 | 1.381671 | 45.8483 | reference |
| 8i fixed-4 | M10 | 1.362507 | 45.8242 | `-0.01916 bpp/-0.02412 dB` |
| 8i fixed-4 | M25 | 1.354207 | 45.6547 | `-0.02746 bpp/-0.19364 dB` |
| RWTT Full28 | M0 | 2.120552 | 45.1747 | reference |
| RWTT Full28 | M10 | 2.140772 | 45.1468 | `+0.02022 bpp/-0.02789 dB`，strictly dominated |
| RWTT Full28 | M25 | 2.102163 | 45.0373 | `-0.01839 bpp/-0.13736 dB` |

M10 在 8i 只是极小的 rate-quality trade-off，在 RWTT 则被 M0 严格支配；M25 的 quality loss 更明显。当前没有证据支持把 MVUB 混入正式 general-purpose recipe。

结论：**NO-GO（Medium/High confidence）**。MVUB 结果保留为 domain-adaptation evidence，不进入封版 recipe。

### H4 — D611 Enhancement 是否应该保留？

| RWTT Full28 | Full bpp | Y | U | V | YUV611 |
|---|---:|---:|---:|---:|---:|
| D111 step1763 | 1.656177 | 41.5546 | 46.6417 | 47.6330 | 42.9503 |
| D411 step3263 | 2.069583 | 44.4558 | 46.0573 | 47.2945 | 45.0108 |
| D611 step3525 | 2.133239 | 44.8843 | 45.8979 | 47.1376 | 45.2926 |

D611 相比 D411 是 `+0.06366 bpp/+0.28181 dB YUV611`，并进一步把 allocation 移向 Y；相比 D111 step1763 则是显著更高 rate 与更高 quality。Longdress 上 D611 相比 D411 同样为约 `+0.02142 bpp/+0.24291 dB`。

这些不是 matched-rate curve，因此不能声称 D611 在 BD-rate 上优于 D411；但在当前单点、目标 metric 即 YUV611 的 recipe 中，D611 是已测候选里最高的 formal Full quality，且是 MVUB/direct studies 使用的稳定起点。

结论：**GO，保留 D611（Medium confidence）**。这表示保留当前 metric-aligned candidate，不表示已经证明 D611 rate-efficiency 最优。

### H5 — D111 warm-up 是否仍有必要？

当前比较不匹配：

```text
two-stage: released init -> D111 1763 -> D611 1762 = 3525 total updates
direct M0: released init -> D611 1762 = 1762 total updates
```

两者都使用 frozen D111 Base、RWTT、LR 5e-5、BS4 和 Enhancement-only，但总 optimizer budget 差 1763 updates。现有 M0 不能回答“direct D611 训到同样总预算后，是否仍弱于 two-stage”。

结论：**UNRESOLVED**。

## 4. Final decision table

| Question | Decision | Confidence | 核心理由 |
|---|---|---|---|
| Base 改用 D611？ | NO-GO | High for current replacement；Medium causal | same rate 下 RWTT/8i 均下降，且方向系统一致 |
| Prefix/Base full-unfreeze？ | NO-GO | High | Base rate inflation、chroma damage、Full RWTT/8i degradation |
| MVUB mixing 进入正式 recipe？ | NO-GO | Medium/High | M10 对 M0 在 RWTT dominated；M25 quality loss；A1 generalization degradation |
| Enhancement 保留 D611？ | GO | Medium | 当前目标 metric 下 formal Full quality最高；但尚无 matched-rate curve |
| D111 warm-up 必要？ | UNRESOLVED | High that evidence is insufficient | direct M0 与 two-stage 总 training budget 不匹配 |

## 5. 推荐的 final 32K recipe

### 已由现有数据支持

```text
Operating point: released R01 / 32k8k / lambda 32768

Base:
  checkpoint: existing Canonical Base D111 step5525
  train provenance: RWTT, BaseSynthesis-only, LR 3e-4, BS4,
                    5525 cumulative optimizer updates
  during Enhancement training: frozen

Enhancement:
  module: independent ResidualVAE
  initialization: exact released R01 ResidualVAE clone
  trainable scope: EnhancementVAE only
  Base/Prefix: frozen
  objective: R_E + 32768 * D611_Full
  rate normalization: -sum(log2(likelihood_E)) / N_full
  training data: RWTT only
  optimizer: Adam, betas=(0.9,0.999), weight_decay=0
  LR: 5e-5 constant
  BS: 4
  current checkpoint: D111 1763 -> D611 1762, total step3525
```

Checkpoint selection不能只看 final step、training loss 或 estimated rate。必须同时看：

- RWTT Full28 physical hard RD；
- 8i fixed-4 physical hard RD；
- author `pc_error` Y/U/V/YUV611；
- exact hard round-trip 与 `Full_bits = Base_bits + Enhancement_bits`。

### 尚未确定

- D111 warm-up 是否必要；
- 因此最终 Enhancement schedule 是 `D111 1763 + D611 1762`，还是更简洁的 `Direct-D611 3525`。

除此之外，现有 evidence 不支持再扫 Base loss、full-unfreeze、MVUB ratio、LR 或 architecture。

## 6. 唯一必要实验

**Matched-budget Direct-D611 control：**

```text
resume: Direct M0 step1762
target: total step3525（再训练 1763 updates）
data: RWTT only
Base/Prefix: frozen
trainable: EnhancementVAE only
loss: D611
conditioning_lambda = rd_lambda = 32768
LR = 5e-5 constant
BS = 4
optimizer: restore M0 Adam state
```

只需在 final step3525 使用与 two-stage D611 完全相同的 RWTT Full28 和 8i fixed-4 hard contract。它唯一要消除的 ambiguity 是：

> two-stage 的优势来自 D111 warm-up，还是仅来自多训练了 1763 updates？

判断：

- Direct-D611 3525 与 two-stage 相当或更好：删除 D111 warm-up，采用更简单的 Direct-D611 recipe；
- Direct-D611 3525 明显更差：保留 D111 warm-up；
- 两者在不同数据集有 trade-off：按 RWTT+8i Pareto 明确记录 MIXED，不再泛化 sweep。

除这一个 matched-budget experiment 外：**NO MORE 32K TRAINING IS CURRENTLY JUSTIFIED**。

