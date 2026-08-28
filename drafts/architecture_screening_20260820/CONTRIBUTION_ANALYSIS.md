# Contribution Analysis — Round 1

本文件把 fact、evidence、inference 和 recommendation 分开。当前没有最终
architecture commitment。

## Family A — Learned native no-bit continuation

**Problem**

Native prefix truncation 虽然满足 `Base bits subset Full bits`，但 omitted suffix 后仍需
一个 full-resolution Base reconstruction；zero/entropy-location continuation 不是为 Base
distortion 优化的。

**Possible contribution**

利用 decoder-known prefix state 学习一个零 per-sample bits 的 Base-only continuation，
同时保持 released Full branch 不变。

**Evidence**

- r5 prefix 本身已通过 structural/physical falsification；
- 当前 loc-net-sized predictor 在 micro-train 上只比 zero 高约 `0.093 dB`；
- holdout 上不超过 `round(loc)` 或 continuous `loc`。

**Inference**

当前 r5 predictor formulation 的收益太小，尚不足以证明需要新增 learned branch。
较早 boundary 或更强 continuation 可能不同，但会增加 missing-transition 难度与分支成本。

**Natural ablation**

`zero -> round(loc) -> continuous loc -> learned continuation`，并报告相同 prefix rate、
Base PSNR 和 Full preservation。

**Thesis strength**

若未来能在多个 boundary/rate point 上以零 bits 稳定提高 Base RD，它可成为清晰的
decoder-side contribution；以当前证据单独作为 thesis 主线偏弱，更像 baseline
improvement。

**Status**

暂时降级，不建议下一步优先做正式训练。若 B/C 后续失败，最低成本复核应是扩大
micro-train capacity/steps，而不是立即实现完整 branch。

## Family B — Within-stream successive refinement

**Problem**

Unicorn 的每个 native arithmetic stream 是 monolithic payload；只能按整条 spatial
stage 截断，不能在同一 stage 内给出 finer quality endpoint。

**Proposed contribution candidate**

把 native integer latent 表示成 nested coarse symbols 与 refinement symbols，使 Base
先获得一个 full-resolution coarse reconstruction，Full 再 exact/refined recover latent。

**Why necessary**

Raw stream truncation 的 rate granularity 由 spatial stages 固定。R02 的 r5 占比很大，
而 oracle 表明 r5 内部仍可形成有用的 coarse endpoint。

**Evidence**

- step-2 coarse 在四个 R02 H5 上平均约 `0.162 empirical bpp`；
- coarse PSNR 为 `40.67 dB`，Full 为 `43.16 dB`；
- separate coarse+refinement marginal entropy 只比 original marginal entropy 高约
  `0.8%`；step-4/8 overhead 同样小。

**Inference**

Native symbols 至少在 signal representation 层面具有 successive-refinement headroom。
这条路线同时具备 RD 动机和明确 mechanism，当前 contribution potential 最高。

**Training contribution**

正式方案很可能需要 endpoint-aware objective，让 coarse endpoint 与 Full endpoint
共同优化；但当前没有证据要求立刻改变 AQL 或全部 stages。

**Natural ablation**

- native whole-stream truncation；
- offline coarse-only oracle；
- proposed nested representation；
- coarse step / endpoint loss weight；
- physical Base/Full rate 与 Full preservation。

**Main missing evidence**

1. `loc/scale` conditional domain 中 coarse/refinement 的真实 rate，而不是 marginal entropy；
2. 一个最小 nested entropy syntax 是否能保持 `Base bits subset Full bits`；
3. coarse reconstruction 是否在 R01/R05/R09 和其他 stages 仍成立；
4. Full exactness 或 retrained Full RD 代价。

**Thesis strength**

如果 physical nested coding 与 endpoint-aware optimization 成功，这不是简单删 stream，
而是为 Unicorn native latent 引入 successive refinement，足以形成完整硕士 contribution。

## Family C — Hybrid/native-spatial residual enhancement

**Problem**

EL V1 的 hard symbols collapse；它直接把 full-resolution residual 经固定 bottleneck
编码，没有证明 residual 的哪一部分适合 transmitted Enhancement。

**Proposed contribution candidate**

把 `A-B` 分解到 geometry-derived multiscale/native support，在较低 spatial support 上
编码可压缩 correction，再用 decoder-known Base/native state 条件恢复 full-resolution
Enhancement。

**Why necessary**

Raw full-resolution residual empirical rate 很高；同时 stride-2/4 oracle 表明相当比例的
distortion 能由 coarse spatial component 解释。EL V1 的单一路径没有显式利用这种
representation structure。

**Evidence**

- stride-4、step `8/255`：约 `0.145 empirical bpp`，平均 `+1.82 dB` oracle gain；
- stride-4、step `4/255`：约 `0.321 empirical bpp`，平均 `+2.90 dB`；
- stride-2、step `8/255`：约 `0.647 empirical bpp`，平均 `+3.29 dB`；
- no-quant stride-2/4 分别移除约 `75.7%/55.3%` residual distortion；
- sample variation 很大，说明 conditional/adaptive representation 可能必要。

**Inference**

V1 collapse 更像 formulation/optimization 与 representation mismatch，不是 Enhancement
signal 不存在。最有希望的 C-family 不是原样 external EL，而是 native-spatial residual
representation。

**Training contribution**

可能需要 multiscale residual target、decoder-known conditioning 与 endpoint-aware RD
loss。当前 evidence 不足以决定具体 network、AQL 或 entropy model。

**Natural ablation**

- EL V1；
- full-resolution residual bottleneck；
- stride-2/stride-4 representation；
- quantization level；
- conditioning state；
- Base/Full hard rate and author metric。

**Main missing evidence**

1. 使用 decoder-known native features 的 conditional entropy estimate；
2. learned synthesis 能否从 parent symbols 恢复 oracle 中未表示的 high-frequency part；
3. short hard-symbol gate 是否避免 V1 collapse；
4. parameter/compute cost 与 Base independence。

**Thesis strength**

若能证明 native-spatial representation 解决 V1 collapse 并改善 hard RD，它可形成完整
architecture contribution；若只传 quantized averaged residual，则只是 engineering
baseline，不足以作为最终贡献。

## Cross-family judgement

| Family | RD signal | Mechanism novelty | Implementation risk | Round-1 priority |
|---|---|---|---|---|
| A no-bit continuation | weak/ambiguous | medium | low–medium | 3 |
| B successive refinement | strong oracle signal | high | high | 1 |
| C native-spatial hybrid | strong oracle signal | medium–high | medium | 2 |

## Recommendation — next 1–2 directions

### 1. 优先：Family B conditional-rate legality probe

下一步只做一个更接近 codec semantics、仍不训练的 discriminating experiment：在固定
R02/r5 symbols 上，基于 decoder-known native `loc/scale` 计算 coarse/refinement 的
conditional estimated rate，并明确最小 nested decode dependency。只有它仍显示合理
Base rate和低 layered overhead，才进入 physical prototype / endpoint-aware training。

### 2. 并行备选：Family C native-spatial conditional oracle

保留同一四 H5，把 stride-4 parent correction 与 decoder-known native/Base state 做一个
小型 conditional entropy/capacity probe。Gate 是：低于 raw marginal rate、非零 hard
symbols、并在 holdout 保留稳定 correction。PASS 后再讨论 EL V2 topology。

### 暂不投入

Family A 不做正式 architecture implementation。当前 evidence 只支持把它保留为
no-bit baseline/control；后续只有在 B/C 被 falsify 时才复核其 capacity。

以上 recommendation 只把 design space 收窄到 B/C，不宣布最终 thesis architecture。
