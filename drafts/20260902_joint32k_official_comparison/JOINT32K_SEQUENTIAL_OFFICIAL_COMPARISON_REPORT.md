# 32K TAFA Joint 与 Sequential Canonical 对比报告

## 1. 目的

本报告比较 32K operating point 下两种 scalable training protocol：

- **Sequential**：先训练并冻结 selected Canonical Base，再训练独立 EnhancementVAE；
- **TAFA-style Joint**：从同一个 trained selected Base 初始化，以 50/50 概率随机选择 Base 或 Full endpoint，联合更新 Prefix、BaseSynthesis 和 EnhancementVAE。

本轮不用于替代正式 sequential recipe，而是回答：Joint optimization 能否稳定使用 Enhancement、是否改善 Full endpoint，以及是否破坏原 selected Base 的 RD efficiency。

## 2. 比较对象

图中包含五个 thesis endpoints：

1. **Selected Base**：Canonical Base D111 step5525；
2. **Sequential D111 Full**：frozen Base + Enhancement D111 stage-1 step1763；
3. **Sequential D611 Full**：同一 frozen Base + D611 continuation step3525；
4. **Joint32K Base**：TAFA-style Joint step3525 的 Base endpoint；
5. **Joint32K Full**：TAFA-style Joint step3525 的 Full endpoint。

Sequential D111 与 D611 使用完全相同的 selected Base，因此 Base 只画一次。所有图同时叠加对应数据集、对应 sequence 的 Official Unicorn R01–R09 physical RD curve。

## 3. Evaluation contract

共同 contract：

```text
physical attribute bpp
x_low + native r1-r4 Base payload
Full bits = Base bits + Enhancement payload
author pc_error Y/U/V
YUV-PSNR 6:1:1
hard arithmetic encode/decode
hard Full round-trip exact
```

数据集：

- **RWTT-28Lite**：固定 28 H5、28 original models，model-equal aggregation；
- **8iVFB fixed4**：Longdress1300、Loot1200、Redandblack1550、Soldier0690，分别报告，不做跨 sequence 掩盖。

注意：RWTT 图使用轻量固定 RWTT-28Lite，而不是 Full28。所有比较均在同一数据集内部进行；不得把 RWTT-28Lite 数值与 Full28 数值直接混合。

## 4. 结果总表

### 4.1 RWTT-28Lite

| Endpoint | Physical bpp | YUV611 (dB) |
|---|---:|---:|
| Official R01 | 1.517274 | 42.7278 |
| Selected Base | 0.511988 | 36.3600 |
| Sequential D111 Full | 1.599862 | 43.0448 |
| Sequential D611 Full | 2.072941 | 45.3378 |
| Joint32K Base | 0.763606 | 37.1285 |
| Joint32K Full | 1.615403 | 43.1908 |

Joint Full 相对 Sequential D111 Full：

```text
Δbpp = +0.015541
ΔYUV611 = +0.145980 dB
```

两者非常接近，Joint Full 以约 0.0155 bpp 换取约 0.146 dB。Sequential D611 达到更高质量，但同时使用明显更高码率；它与另外两点形成不同的 rate–quality operating regime，不能只按最高 PSNR 判断胜负。

### 4.2 8iVFB per-sequence

| Sequence | Endpoint | Physical bpp | YUV611 (dB) |
|---|---|---:|---:|
| Longdress | Selected Base | 0.767107 | 34.8154 |
| Longdress | Sequential D111 Full | 1.983407 | 41.5235 |
| Longdress | Sequential D611 Full | 1.963481 | 40.9426 |
| Longdress | Joint32K Base | 1.050214 | 35.3403 |
| Longdress | Joint32K Full | 2.046347 | 41.3207 |
| Loot | Selected Base | 0.243223 | 43.0397 |
| Loot | Sequential D111 Full | 0.491006 | 45.7439 |
| Loot | Sequential D611 Full | 0.495218 | 45.4634 |
| Loot | Joint32K Base | 0.437530 | 45.2037 |
| Loot | Joint32K Full | 0.521922 | 46.1425 |
| Redandblack | Selected Base | 0.410088 | 39.2362 |
| Redandblack | Sequential D111 Full | 0.945071 | 43.0018 |
| Redandblack | Sequential D611 Full | 0.957446 | 42.6237 |
| Redandblack | Joint32K Base | 0.630262 | 40.3516 |
| Redandblack | Joint32K Full | 1.020384 | 43.0212 |
| Soldier | Selected Base | 0.393370 | 41.7633 |
| Soldier | Sequential D111 Full | 0.753245 | 44.7276 |
| Soldier | Sequential D611 Full | 0.761685 | 44.3707 |
| Soldier | Joint32K Base | 0.640666 | 43.7608 |
| Soldier | Joint32K Full | 0.801637 | 45.1866 |

Joint Full 相对 Sequential D111 Full：

| Sequence | Δbpp | ΔYUV611 |
|---|---:|---:|
| Longdress | +0.062940 | -0.202850 dB |
| Loot | +0.030916 | +0.398588 dB |
| Redandblack | +0.075313 | +0.019412 dB |
| Soldier | +0.048393 | +0.458988 dB |

Joint 在 Loot 和 Soldier 上取得明显质量提升，Redandblack 基本持平，但 Longdress 出现更高码率且更低质量。由此可见 Joint Full 的 external behavior 存在明显 sequence dependence，不能仅以四序列平均值宣称普遍改善。

## 5. Official curve 中的位置

### 5.1 Base endpoints

Selected Base 在不同数据集上的 same-bpp Official interpolation 差值：

| Dataset | Selected Base Δmatched | Joint Base Δmatched |
|---|---:|---:|
| RWTT-28Lite | -0.1312 dB | -1.2827 dB |
| Longdress | -0.6183 dB | -2.2915 dB |
| Loot | +0.5299 dB | -0.2569 dB |
| Redandblack | +0.1653 dB | -0.9763 dB |
| Soldier | +0.3746 dB | -0.3424 dB |

Joint training 在所有五个 contract 上都使 Base 的 matched-rate position 相对 selected Base 恶化。虽然 Joint Base 的 absolute PSNR 通常更高，但它也显著增加 Base physical rate；增加的质量不足以补偿增加的码率。

以 RWTT-28Lite 为例：

```text
Selected Base : 0.511988 bpp / 36.3600 dB
Joint Base    : 0.763606 bpp / 37.1285 dB

Base rate drift    = +0.251618 bpp（约 +49%）
Base quality gain  = +0.768452 dB
Δmatched           = -0.1312 → -1.2827 dB
```

因此，Joint32K 最明确的问题不是 Enhancement collapse，而是 **Base endpoint RD efficiency drift**。

### 5.2 Full endpoints

多数 Full endpoints 的 bpp 已高于各自 Official R01，因此超出 Official R01–R09 interpolation range。本报告遵守 no-extrapolation 规则，不为这些点计算正式 `Δmatched`。

只能将它们与 R01 端点做直观位置比较：

- RWTT：Joint Full 比 R01 多 0.0981 bpp、高 0.4630 dB；
- Longdress：Joint Full 比 R01 多 0.1475 bpp、低 0.2146 dB；
- Loot：Joint Full 比 R01 多 0.0718 bpp、高 0.5259 dB；
- Redandblack：Joint Full 比 R01 多 0.1466 bpp、高 0.2509 dB；
- Soldier：Joint Full 比 R01 多 0.0727 bpp、高 0.2878 dB。

这些是 endpoint differences，不是 matched-rate、BD-rate 或 extrapolated conclusions。

## 6. 对五个研究问题的回答

### 6.1 EL 是否被稳定使用？

**是。** Joint step3525 在 RWTT-28Lite 上：

```text
Base bpp = 0.763606
Full bpp = 1.615403
EL bpp   = 0.851797
```

Enhancement payload 显著非零，且 Full 相对 Base 提升约 6.06 dB。

### 6.2 EL 是否随训练 collapse？

**没有。** step500–3525 始终存在大量 Enhancement rate，hard encode/decode 均 PASS。step3525 的 Enhancement rate低于 step2500，但仍为约 0.852 bpp，属于 rate reallocation，而不是 collapse。

### 6.3 Base rate/quality 是否发生不合理 drift？

**是，这是当前最主要问题。** Joint Base 比 selected Base 使用更多物理码率，但在 Official matched-rate curve上的相对位置明显变差，并且该现象在 RWTT、Longdress、Loot、Redandblack、Soldier五个 contract上一致出现。

### 6.4 Full 是否形成有效 scalable refinement？

**是。** Full hard reconstruction exact，`Full_bits = Base_bits + Enhancement_bits`，且质量显著高于 Joint Base。但“refinement有效”不等同于“两端点整体RD最优”；Base drift削弱了 scalable pair 的整体价值。

### 6.5 与 Joint256 convergence pattern 是否不同？

**明显不同。** Joint256 后期 Enhancement 接近 zero-rate/collapse；Joint32K 始终大量使用 Enhancement。说明 joint protocol 在 high-rate 32K 下能够维持 transmitted Enhancement information，但仍不能自动保证 Base/Full 两端同时高效。

## 7. 综合判断

### Diagnostic conclusion

```text
Enhancement utilization: PASS
Hard scalable correctness: PASS
Full refinement capability: PASS
Base RD preservation: FAIL/WARNING
External consistency: MIXED
```

Joint32K 证明了 TAFA-style joint optimization 在 high-rate setting 下不会必然导致 EL collapse，并能得到有效 Full refinement。然而，它同时显著改变 native Prefix/Base rate，并使 Base endpoint 的 matched-rate efficiency系统性下降。

因此当前证据支持：

> **Joint32K 是有价值的 diagnostic control，但不应替代当前 selected Base frozen + sequential Enhancement 的 canonical recipe。**

若论文强调两个 independently useful scalable endpoints，现有 sequential protocol仍更稳健：它保留了更好的 Base RD position，并把 Enhancement优化限制在 Full refinement。Joint protocol的价值主要是揭示 high-rate 下 Base/Enhancement可以共同学习，但需要额外机制才能保护 Base RD；本阶段不应据此自动增加新 loss、重新训练或扩展 sweep。

## 8. Artifacts

- `official_and_32k_endpoints.csv`：所有绘图数值；
- `rwtt28lite_official_and_32k_endpoints.png/.svg`；
- `8ivfb_longdress_official_and_32k_endpoints.png/.svg`；
- `8ivfb_loot_official_and_32k_endpoints.png/.svg`；
- `8ivfb_redandblack_official_and_32k_endpoints.png/.svg`；
- `8ivfb_soldier_official_and_32k_endpoints.png/.svg`。

N30补测 Jobs：

```text
5439852_[0-3]  Joint32K 8i fixed4
5439853        Sequential D111 RWTT-28Lite
```

全部成功完成。FAU→N30 checkpoint 使用 SSH agent forwarding直接服务器传输，SHA256 两端一致。
