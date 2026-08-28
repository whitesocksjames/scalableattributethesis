# PSNR Aggregation Peak=1 Correction

> 日期：2026-08-21  
> 状态：patch 和独立 re-derivation regression PASS  
> 原 frozen reference 未覆盖。

## Bug and patch

`pc_error` 输出的 Attribute channel MSE 已归一化到 `[0,1]`。旧 helper 在从 aggregated MSE 重算 PSNR 时错误使用 `peak=255`：

```python
10 * log10(255**2 / mse)
```

现已在 `scalable_attribute/evaluation.py` 修正为：

```python
10 * log10(1**2 / mse)
```

这只影响由 MSE 派生的 absolute PSNR，不影响 bits、bpp、MSE 或 codec output。

## Aggregation contract

当前实现不是直接 average per-H5 PSNR。顺序为：

1. H5 是 execution unit，每个 H5 保存 physical bits、point count 和 author `pc_error` per-channel MSE。
2. 在每个 original RWTT model 内，按 point count 累积每通道 SSE：`sum(MSE_h5 * N_h5)`。
3. 除以该 model 的 `sum(N_h5)`，得到 model-level Y/U/V MSE。
4. 使用 normalized peak `1` 分别计算 Y/U/V PSNR。
5. 计算 author Attribute `YUV-PSNR 6:1:1`。
6. 最后在 original models 之间等权平均。

这符合现有 contract：H5 execution、original-model aggregation、model equal average；直接平均 per-H5 PSNR 会让 partition 数量影响权重，因此没有采用。

## Corrected artifacts

Retained `per_h5.csv` 被只读复用；没有重新运行 codec：

```text
$WORK/scalable_attribute_thesis/experiments/
  rwtt_unicorn_reference_v1_corrected_peak1_v1/
    reference_full/
      R01...R09/per_model.csv
      reference_curve.csv
    reference_dev/
      R01...R09/per_model.csv
      reference_dev_curve.csv
    c2_r08_smoke/
      per_model.csv
      endpoint_summary.csv
    regression_checks.json
    provenance.json
```

Source frozen artifacts remain at：

```text
$WORK/scalable_attribute_thesis/experiments/rwtt_unicorn_reference_v1/
```

## Regression checks

所有检查 PASS：

- R01–R09 Full28 和 Dev14 的 bpp 修复前后完全相同。
- 每个 per-model Y/U/V/YUV PSNR 和最终 curve PSNR 都统一下降：

```text
20 * log10(255) = 48.1308036086791 dB
```

- C2 smoke Base/Full bpp 完全不变。
- C2 smoke endpoint `Full−Base` PSNR difference 修复前后不变。
- C2 single-H5 aggregated Base 与 direct `pc_error` 差约 `3.4e-5 dB`；Full 差约 `2.1e-5 dB`，来自 retained MSE/PSNR 文本精度，低于 `0.001 dB` regression tolerance。
- 没有发现第二个 aggregation bug。

## Corrected official neighboring values

### Full RWTT Validation — 28 models / 792 H5

| Point | Checkpoint | lambda | mean bpp | corrected author YUV-PSNR 6:1:1 |
|---|---|---:|---:|---:|
| R07 | 2k128 | 512 | 0.1523901476 | 32.5236010 dB |
| R08 | 2k128 | 256 | 0.0995921908 | 31.5378472 dB |
| R09 | 2k128 | 128 | 0.0706404850 | 30.7022151 dB |

### Fixed Dev14 — 14 models / 312 H5

| Point | Checkpoint | lambda | mean bpp | corrected author YUV-PSNR 6:1:1 |
|---|---|---:|---:|---:|
| R07 | 2k128 | 512 | 0.1357621149 | 32.5270453 dB |
| R08 | 2k128 | 256 | 0.0825363202 | 31.5734456 dB |
| R09 | 2k128 | 128 | 0.0545764470 | 30.7615103 dB |

## Result provenance

```text
RESULT PROVENANCE

Result:
Corrected R01-R09 Full28 and Dev14 model-level/curve PSNR.

Source:
Retained author-reference per_h5.csv; no codec rerun.

Data:
Full: 28 original models / 792 H5 / 57,904,570 points.
Dev: fixed 14 original models / 312 H5 / 25,002,819 points.

Training:
None. Official released Unicorn checkpoints and fixed official mapping.

Metric:
Point-weighted per-channel normalized MSE within original model
→ peak=1 PSNR → YUV 6:1:1 → equal average across models.

Can be used for formal RD?
YES, subject to project review of this correction.
```

## C2 smoke corrected endpoint

```text
Base = 0.0199567796 bpp / 37.7823718 dB
Full = 0.0255261135 bpp / 37.4139086 dB
EL   = 0.0055693339 bpp
```

该结果仍只是 2-update / 1-H5 path smoke，不能用于 formal RD；修复只纠正其 derived absolute PSNR。
