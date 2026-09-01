# Direct-D611 matched-budget result

比较目标：

- Two-stage：D111 1763 updates → D611 1762 updates，总计 3525；
- Direct：从第一步开始使用 D611；本轮从 Direct step1762 恢复 optimizer/weights，并继续到 step3525；
- Base/Prefix frozen，Enhancement-only，RWTT，LR `5e-5`，BS4，R01/λ32768。

8i fixed-4 trajectory 的 shortlist 是 Direct step3000。相对 Two-stage step3525：

- Direct step3000：`1.401847 bpp / 45.920788 dB`；
- Two-stage step3525：`1.383888 bpp / 45.960266 dB`；
- Direct 为 `+0.017959 bpp / -0.039478 dB`，在该 external contract 上被 Two-stage 严格支配。

RWTT Full28：

- Direct step3000：`2.129593 bpp / 45.225938 dB`；
- Two-stage step3525：`2.133239 bpp / 45.292648 dB`；
- Direct 为 `-0.003646 bpp / -0.066710 dB`，是很小的 rate-quality trade-off，不构成胜出。

Per-model Full28 中 Direct 的 YUV611 仅 5/28 models 改善，median difference `-0.032316 dB`。主要损失来自 chroma：mean ΔU `-0.349615 dB`，mean ΔV `-0.182507 dB`；Y 基本不变（mean ΔY `-0.000260 dB`）。RWT449 是最大负向 outlier（`-0.769411 dB`），但去掉单个 outlier 也不会改变总体方向。

当前 evidence 更支持保留 D111 warm-up。该判断是单个 32K operating point 的 matched-budget evidence，不是 BD-rate conclusion。

