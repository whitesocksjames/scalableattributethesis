# 2K/4K Longdress training-step diagnosis

目的：判断 low-rate Canonical Base 在 Longdress1300 上的 weakness 是随着 fine-tuning 增加而恶化，还是在早期已经存在。

数据：

- RWTT-28Lite：固定 28 models / 28 H5 trajectory；
- Longdress：8iVFB `longdress_vox10_1300`；
- checkpoints：step500/1000/2000/3000/3525；
- 每个 operating point 的所有 checkpoints 共用相同 physical Base bits；
- Longdress hard prefix 在同一 point 内只执行一次；所有 rows 的 `soft_hard_max_abs_difference=0`。

| Point | Step | RWTT-28Lite YUV611 | Longdress YUV611 |
|---|---:|---:|---:|
| 2K | 500 | 33.345313 | 28.576400 |
| 2K | 1000 | 33.405715 | 28.640225 |
| 2K | 2000 | 33.423482 | 28.671963 |
| 2K | 3000 | 33.430148 | 28.697013 |
| 2K | 3525 | 33.445679 | 28.704200 |
| 4K | 500 | 33.650233 | 28.517663 |
| 4K | 1000 | 33.688265 | 28.557288 |
| 4K | 2000 | 33.727919 | 28.586913 |
| 4K | 3000 | 33.727097 | 28.644000 |
| 4K | 3525 | 33.727699 | 28.645313 |

结论：

- 2K Longdress 从 step500 到 step3525 单调改善 `+0.1278 dB`；RWTT-28Lite 同期改善 `+0.1004 dB`。
- 4K Longdress 同期改善 `+0.12765 dB`；RWTT-28Lite 改善 `+0.07747 dB`，step2000 后进入几乎完全的平台期。
- 因此 Longdress weakness 不是随 fine-tuning step 增加而恶化。它在最早保存的 step500 已经存在，后续训练只带来小幅改善。
- 没有 step0 external metric，所以不能严格断言 weakness 在 optimizer 第一步前就存在；可以确定的是它不属于 step500→3525 的 overfitting degradation。
- 4K 的 RWTT winner step2000 仅比 step3525 高约 `0.00022 dB`；Longdress 则 step3525 比 step2000 高约 `0.0584 dB`。本诊断不改变已冻结 checkpoint selection。

