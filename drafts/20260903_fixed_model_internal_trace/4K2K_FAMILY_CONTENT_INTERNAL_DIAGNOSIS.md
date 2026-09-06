# 4K/2K Family × Content Internal Diagnosis

## 结论

本轮固定模型诊断全部通过 correctness/provenance gate，没有训练或修改模型。综合证据更支持：

> **checkpoint-family representation 与 content complexity 的 interaction。**

`8k256` 的弱点在 `B_native` 之前已经出现；`BaseSynthesis` 在所有 external sequence 上都提供正增益，因此不是 degradation 的制造者，但它对困难内容的补偿明显不足。相同 λ=8192 时，`8k256` 从 r1 stage 就与 `32k8k` 拉开差距，说明问题不是仅在最终 Base synthesis 才产生。

## 1. Execution 与数据完整性

| Job | 内容 | 状态 | Runtime |
|---|---|---|---:|
| 5439979 | External 8 sequences × core 5 configs | PASS | 1:17:41 |
| 5439980 | External alternative-family probes | PASS | 0:51:20 |
| 5439981 | RWTT Train60 + Val28 × core 5 configs | PASS | 1:18:00 |

- External：8i 56 blocks + Owlii 32 blocks，共 88 blocks/config。
- RWTT：Train low/typical/high 各20 blocks + fixed Val28，共88 H5/config。
- 每个配置均有 `x_low,r1…r5` stage-local trace、`B_unpool/B_native/Official_Full`；存在 selected Base checkpoint 的正式点还有 `Canonical_Base`。
- `(configuration, source, block, stage/endpoint)` 无重复、无缺失。
- 两次独立执行重叠的 official R04/R05 数值逐字段 exact 一致（只忽略是否加载 Canonical Base 所产生的 metadata 差异）。
- `run_info.json` 中所有 released/Base SHA-256 与 config 声明一致。完整 hash/path 保留在 `analysis_provenance.json`。

Aggregation：block MSE 按 point count 聚合后计算 PSNR；external group 对四个 sequence 等权。Intermediate native quality 是 stage-local reconstruction 对 exact pooled GT，禁止跨 resolution 解读成 RD endpoint。`pc_error` 只用于 full-resolution endpoints。

## 2. Full-resolution decomposition

External 四序列等权均值：

| Point | Dataset | B_unpool | B_native | Canonical Base | Official Full | BaseSynthesis gain | Native r5 gap | r5 Δbpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8i | 36.188 | 36.976 | 37.695 | 40.673 | +0.719 | 3.697 | 0.307 |
| 4K | 8i | 31.541 | 32.956 | 35.145 | 38.639 | +2.189 | 5.683 | 0.188 |
| 2K | 8i | 30.093 | 31.965 | 34.887 | 37.305 | +2.922 | 5.340 | 0.119 |
| 1K | 8i | 28.727 | 31.266 | 34.330 | 35.528 | +3.064 | 4.262 | 0.056 |
| 8K | Owlii | 37.512 | 38.255 | 39.288 | 41.956 | +1.033 | 3.700 | 0.193 |
| 4K | Owlii | 32.242 | 34.293 | 37.582 | 39.945 | +3.289 | 5.652 | 0.070 |
| 2K | Owlii | 30.447 | 33.458 | 37.131 | 38.776 | +3.672 | 5.318 | 0.041 |
| 1K | Owlii | 29.042 | 32.872 | 36.430 | 37.277 | +3.558 | 4.405 | 0.020 |

结论：4K/2K 在进入 learned BaseSynthesis 前已经具有更大的 native-to-Full gap；BaseSynthesis 对所有 aggregate 和所有单 sequence 都是正增益。它在补救，而不是制造 degradation。

但补救具有 content sensitivity。例如 4K 的 synthesis gain 在 Longdress 只有 `+0.767 dB`，Loot 为 `+3.531 dB`；2K 分别为 `+1.002/+4.433 dB`。这与 Longdress 弱、Loot 健康的方向一致。

## 3. Native r5 dependence

4K 的 native r5 quality gap 在 Longdress/Dancer/Model 为 `6.725/6.168/5.940 dB`，高于 Exercise 的 `4.736 dB`；2K 对应为 `5.980/5.799/5.348/4.685 dB`。r5 physical increment 也在 Longdress 最大（4K `0.397 bpp`，2K `0.285 bpp`）。

不过关系并不统一：Loot 的 native r5 gap 仍为 4K `5.463 dB`、2K `5.406 dB`，但其 matched-rate Base 表现接近健康。因此：

- bad content **往往**更依赖 native r5，尤其 Longdress；
- r5 dependence 不是充分解释，必须和 prefix/BaseSynthesis 的 content response 一起看。

Across 8 sequences，`native_r5_gap` 与 matched-rate delta 的 Spearman ρ 为 4K `-0.643`、2K `-0.500`；仅作 descriptive association。

## 4. Clean same-λ family control：λ8192

以下为 `8k256@8192 - 32k8k@8192`，两个 checkpoint 使用相同 λ，但前者是 diagnostic、不是 official operating point：

| Dataset | Endpoint | Δbpp | ΔYUV611 |
|---|---|---:|---:|
| 8i | B_native | -0.061 | **-3.434 dB** |
| 8i | Official Full | -0.107 | -0.959 dB |
| Owlii | B_native | -0.022 | **-3.438 dB** |
| Owlii | Official Full | -0.109 | -1.112 dB |
| RWTT Train strata mean | B_native | -0.059 | -6.734 dB |
| RWTT Val28 | B_native | -0.105 | -2.427 dB |

`x_low` exact相同；stage-local quality 从 r1 已分离。External 上 `8k256-32k8k` 的 r1 差约 `-9.4` 至 `-11.7 dB`，到 r4 仍约 `-4.7` 至 `-8.3 dB`。这强烈支持 family-dependent native multiscale representation，而不是仅由 r4→full synthesis 引起。

重要限制：两个 family 虽 λ 相同，但 physical rate 不相同；因此它证明 representation/behavior 不同，不证明作者 training lineage，也不是公平的 matched-rate winner comparison。

## 5. Alternative-family probes

### 2k128@2048 vs official 8k256@2048

`2k128 - 8k256`：

| Dataset | Endpoint | Δbpp | ΔYUV611 |
|---|---|---:|---:|
| 8i | B_native | -0.009 | **+0.529 dB** |
| Owlii | B_native | -0.004 | **+0.527 dB** |
| 8i | Full | -0.028 | -0.586 dB |
| Owlii | Full | -0.008 | -0.414 dB |

这说明 2k128 family 在 λ2048 的 truncated native Base behavior 确实改善，而且 rate 还略低；但其 Full 反而更低。它支持“family 会改变 Base/Enhancement information allocation”，不支持把该 diagnostic config 直接替代 official R05。其 released training λ range 无 metadata 可恢复，故继续标记 `RANGE_UNKNOWN`。

### 32k8k@4096 vs official 8k256@4096

`32k8k - 8k256`：B_native 在 8i/Owlii 分别为 `+3.789/+3.780 dB`，同时多 `+0.083/+0.043 bpp`；Full 为 `+1.692/+1.728 dB`，同时多 `+0.193/+0.159 bpp`。这是明显的 high-rate shift，而不是相同 rate 下的免费改善。由于 λ4096 是否位于 32k8k released training range无法从 checkpoint metadata确认，只能作为 exploratory evidence。

## 6. RWTT energy-stratified evidence

RWTT 的 20 low / 20 typical / 20 high blocks按既有 `r2_E_D111` 固定选择；另有独立 Val28。

同 λ8192 下，`8k256-32k8k` 的 B_native 差异为：

| Group | Δbpp | ΔYUV611 |
|---|---:|---:|
| Train high-energy | -0.097 | **-3.141 dB** |
| Train typical | -0.085 | -2.285 dB |
| Val28 | -0.105 | -2.427 dB |

因此 high-energy RWTT blocks 也呈现与 external bad content 同方向的 family weakness。它并非 external-only 现象。

low-energy strata 是 strongest counterexample：它出现极高/近无残差 PSNR、几乎零 r5 rate，family PSNR差被数值放大，不适合作为一般内容效应的线性对照。它提示“energy越高、family gap必然越大”并非单调定律。

## 7. 与既有 content metrics 对齐

| Metric vs matched-rate degradation | 4K ρ | 2K ρ |
|---|---:|---:|
| GT r2 residual energy | -0.905 | -0.905 |
| TV6 | -0.833 | -0.833 |
| V variance | -0.524 | -0.524 |
| native r5 quality gap | -0.643 | -0.500 |
| r5 Δbpp | -0.476 | -0.619 |
| BaseSynthesis gain | +0.595 | +0.833 |

`BaseSynthesis gain` 的正相关意味着：matched-rate 表现越健康的 sequence，learned synthesis 往往补得越多。它进一步支持 BaseSynthesis 是不充分的补救模块，而非 degradation 的起点。所有相关均为 `n=8` exploratory/descriptive，不作 significance 或 causation claim。

## 8. Required questions

| Question | Answer | Classification | Strongest counterexample |
|---|---|---|---|
| A. bad content 更依赖 native r5？ | Longdress及部分 Owlii是；总体只有中等关联 | **WEAKLY SUPPORTED** | Loot r5 gap仍大但 Base healthy |
| B. weakness 在 B_native 前已出现？ | 是；same-λ trace 从 r1 已明显分离 | **SUPPORTED** | 8k256并非所有 content/endpoint 都弱 |
| C. BaseSynthesis 补救还是制造？ | 所有 sequence 均正增益，属于补救但不足 | **SUPPORTED（补救）**；制造假说 **WEAKENED** | 困难 content 的 gain 明显较小 |
| D. 8192 same-λ control？ | 8k256 B_native 在两 external set 均低约3.44 dB，且早期 stage 已分离 | **SUPPORTED** | rate并不相同，不能作公平RD结论 |
| E. 2k128@2048 改善 native behavior？ | B_native约+0.53 dB且略低rate，但Full更差 | **SUPPORTED（仅native）** | Full低0.41–0.59 dB；range unknown |
| F. 32k8k@4096？ | quality大幅上升但显著增加rate | **INCONCLUSIVE as replacement** | 不同rate且可能out-of-range |
| G. RWTT high-energy复现？ | 同λ family gap在high group为-3.14 dB | **SUPPORTED** | low-energy组非单调且近零残差 |
| H. content、family或interaction？ | family决定结构差异，content调制严重度 | **SUPPORTED：interaction** | n=8 external aggregates，非因果证据 |

## 9. Final assessment

最强证据链是：

1. `x_low` 一致，但 same-λ family trace 从 r1 就分离；
2. 4K/2K 的 `B_native→Full` gap 大于8K/1K；
3. BaseSynthesis 始终提供正增益，却在 Longdress 等困难内容上补偿较少；
4. high-energy RWTT blocks 也能复现 family weakness；
5. 2k128@2048 改善 B_native、却牺牲 Full，直接表明 family 改变 Base/Full information allocation。

因此当前 evidence 最支持 **checkpoint-family representation × content complexity interaction**。它不支持把问题简化为“r5 bits太多”，也不支持把 BaseSynthesis认定为根因。

本轮到此停止；没有启动 training、fine-tuning 或 architecture change。

## 10. Artifacts

- `analysis/internal_trace_blocks_merged.csv`
- `analysis/full_resolution_blocks_merged.csv`
- `analysis/internal_trace_sequences.csv`
- `analysis/full_resolution_sequences.csv`
- `analysis/reconstruction_decomposition_sequences.csv`
- `analysis/family_controls_sequences.csv`
- `analysis/content_internal_alignment.csv`
- `analysis/internal_degradation_correlations.csv`
- `analysis/validation_checks.csv`
- `analysis/analysis_provenance.json`
- `analysis/figure1_native_stage_local_trace.png`
- `analysis/figure2_native_r5_quality_gap.png`
- `analysis/figure3_full_resolution_decomposition.png`
- `analysis/figure4_same_lambda_and_family_controls.png`
- `analysis/figure5_content_internal_vs_degradation.png`
