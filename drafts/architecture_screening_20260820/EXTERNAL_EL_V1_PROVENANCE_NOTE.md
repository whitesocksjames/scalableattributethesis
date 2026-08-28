# External-EL V1 provenance note

本页只记录当前 repository 与 retained experiment artifacts 能直接确认的事实，不重新训练，也不扩展为 autopsy。

## Architecture / objective facts

- analysis target：`E = A - B`。
- analysis conditioning：`E + B + F_U`（先编码 residual，再与 `B/F_U` fusion）。
- entropy conditioning：`B + F_U`。
- synthesis conditioning：`y + B + F_U`（`synthesis_condition=b_fu`）。
- `analysis_scale=2`，即 stride-4 latent。
- `latent_channels=64`。
- training quantization：fixed unit-width uniform noise；deterministic/hard path：unit-step rounding。
- 未复用 native AQL `EQ/DQ`。
- Enhancement parameters 从随机初始化开始，除非显式 `--resume`。
- objective：`R_E + lambda_E * D_F`；`R_E` 以 full-resolution point count `N0` 归一化。

## Retained collapse-run provenance

Legacy one-epoch run：

- resolved args：`$WORK/scalable_attribute_thesis/experiments/el_coarse_rd_lr_seed0_bs4_1ep_20260819/b2048_rd350_lr0p0001_cz64/train/resolved_args.json`
- Base：legacy `2k128 + lambda_Base=2048`
- `lambda_E=350`，`lr=1e-4`，`batch_size=4`，`seed=0`
- planned/completed checkpoint endpoint：`step_3525.pth`（另有 periodic checkpoints）
- 该旧 artifact 的 resolved args 尚未显式保存 `synthesis_condition`；按 retained implementation/default，它对应 `b_fu`，但此字段不能仅从旧 JSON 独立恢复。

Explicit short control：

- resolved args：`$WORK/scalable_attribute_thesis/experiments/el_synthesis_condition_r09_short_v1/r09_syn_b_fu_rd6500_lr5e-5/train/resolved_args.json`
- checkpoint：`step_200.pth`
- Base：official R09，`2k128 + lambda_Base=128`
- `lambda_E=6500`，`lr=5e-5`，`batch_size=4`，`synthesis_condition=b_fu`

另有 zero-centered causal control 位于：

`$WORK/scalable_attribute_thesis/experiments/el_zero_centered_r09_causal_v1/r09_bfu_zc_rd6500_lr5e-5/train/`

## Scope conclusion

> External-EL V1 Hybrid failed; this is not evidence that the C1 layered-enhancement family fails.

因此 C1-A/native B-only 与 ScalablePCAC-inspired C1-B 仍保留在 design space；本轮不实现。
