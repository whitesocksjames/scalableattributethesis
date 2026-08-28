# Cheap Experiment Log

日期：2026-08-20

## Scope discipline

- 只使用 experiment-only scripts。
- 未修改 `lossy_attribute/`、geometry、AQL、entropy model 或正式 model topology。
- 未运行 long training、parameter sweep、Dev14/full evaluation。
- 调查文档只写入本机 `drafts/architecture_screening_20260820/`。

## Local scripts

- `screen_native_and_successive.py`
  - Family A deterministic/learned no-bit continuation；
  - Family B offline successive-refinement oracle。
- `screen_hybrid_residual_oracle.py`
  - Family C spatial/quantization oracle。

HPC 上的脚本副本仅位于本轮 experiment directory：

```text
$WORK/scalable_attribute_thesis/experiments/
  architecture_screening_20260820/scripts/
```

## Jobs

| Job ID | Purpose | Resource | State | Runtime | Note |
|---:|---|---|---|---:|---|
| 1785080 | Family A+B first attempt | V100 | FAILED | startup | copied predictor inherited `requires_grad=False`; no scientific result |
| 1785081 | Family C initial oracle | all-GPU, allocated V100 | COMPLETED | 23 s | reconstruction valid; stride-2/4 bpp used broadcast symbols and is superseded |
| 1785082 | Family A+B corrected | V100 | COMPLETED | 2 m 42 s | `predictor.requires_grad_(True)` only |
| 1785085 | Family C parent-symbol correction | all-GPU, allocated V100 | COMPLETED | 24 s | authoritative Family C CSV |

`1785080` 是普通 experiment-script bug；修复没有改变 architecture、training
objective 或 probe configuration。

## Authoritative raw artifacts

Family A/B：

```text
$WORK/scalable_attribute_thesis/experiments/architecture_screening_20260820/
  family_a_b_r02/
    family_a_continuations.csv
    family_a_training.json
    family_a_capacity_predictor.pth
    family_b_successive_oracle.csv
```

Family C：

```text
$WORK/scalable_attribute_thesis/experiments/architecture_screening_20260820/
  family_c_r09_parent_symbols/
    family_c_residual_oracle.csv
```

旧目录 `family_c_r09/` 保留为实验历史，但其中 stride-2/4 的
`ideal_entropy_bpp` 不可用于结论。其 PSNR/reconstruction 与 corrected run 一致。

## Probe implementation notes

### Family A

- native stages r1-r4 使用 deterministic rounded symbols 重建 decoder state；
- 在读取 r5 payload 前捕获 `x_in/f_in/prior_dec/prior/loc/scale`；
- learned predictor 是 native `loc_net` 的独立 copy，输入 decoder-known `prior`；
- Base 和 released Full modules 保持 frozen；
- predictor 使用 continuous EQ-domain output，经 native DQ/synthesis；
- 100 updates 交替使用两个固定 H5，Adam `lr=1e-3`，MSE-only。

`family_a_training.json` 的 cheap logger 保存的是各记录窗口内最后一次 loss，不能当作
严格的 step-0/20/50/100 learning curve。最终 checkpoint/evaluation 是有效的，本轮
不根据这四个 log 点推断 convergence。

### Family B

- 捕获 r5 native rounded integer symbols；
- step `2/4/8` 的 coarse/refinement pair 能 exact recover original symbols；
- reconstruction 只把 coarse symbols 送入原 DQ/decoder；
- rate 是 channel-wise empirical marginal entropy；
- 没有实现或声称 physical progressive bitstream。

### Family C

- R09 Base 由现有 deterministic BaseAdapter 产生并保持 frozen；
- support 由 known geometry 按 `floor(coords/factor)` 聚合；
- stride-2/4 只对 unique parent values 计 entropy bits；
- parent symbols decode 后按 inverse mapping broadcast 到 stride-1 support；
- denominator 始终是 full-resolution `N0`。

## Data-quality gates

- 三个成功 jobs exit code 均为 `0:0`。
- A/B/C 使用同一组四个 H5，便于观察 content variation。
- A 的 released Full 路径使用原 native r5 symbols；此前 r5-only falsification 已证明
  同一 reconstruction procedure 与 released hard decode `max_abs_diff=0`。
- B/C 结果严格标记为 oracle/diagnostic；不能进入正式 reference RD curve。
