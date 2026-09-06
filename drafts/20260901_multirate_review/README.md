# Multi-rate Review（2026-09-01，source `d707ce7`）

本目录集中保存本轮 A/B/C 的 data-only review artifacts。没有复制 checkpoints，
没有重新运行 evaluation，也没有混用 RWTT-28Lite、Full28 与 8i contracts。

## 目录

- `01_figures/`：5 张供直接审阅的图（PNG + SVG）
- `02_summary_tables/`：编号后的汇总 CSV
- `03_raw_evidence/`：保留的原始 CSV/JSON/Markdown，不含 checkpoint
- `04_reproduction/`：从 raw evidence 重新生成图表的脚本
- `review_info.json`：本 bundle 的 provenance、contract 和缺失项

## 先看这些图

1. `01_figures/Fig01_16K_Enhancement_RD_RWTT28Lite.png`
2. `01_figures/Fig02_8K_Enhancement_RD_RWTT28Lite.png`
3. `01_figures/Fig03_LowRate_Base_Training_Trajectory_RWTT28Lite.png`
4. `01_figures/Fig04_Selected_Base_RD_RWTT28Lite.png`
5. `01_figures/Fig05_Selected_Base_RD_8iVFB.png`
6. `01_figures/Fig06_Selected_Base_RD_8iVFB_PerSequence.png`
7. `01_figures/Fig07_Fixed_Full_RD_RWTT28Lite.png`
8. `01_figures/Fig08_Fixed_Full_RD_8iVFB_Mean.png`
9. `01_figures/Fig09_Fixed_Full_RD_8iVFB_PerSequence.png`

`Fig05` 是四个 8iVFB sequence 的算术平均；`Fig06` 将 Longdress、Loot、
Redandblack、Soldier 分开，避免平均值掩盖 sequence-specific behavior。

所有 RD 图均叠加对应 contract 下的 Official Unicorn R01–R09 Full curve。
Canonical Base 使用独立 marker；Base 与 Official Full 不是同一种 endpoint，不能把两条线解释成
同一 codec 的普通 single-layer RD curve。

## 汇总表

- `Table01`：16K/8K Enhancement trajectory
- `Table02–03`：1K/512/256 Base trajectory 与 8iVFB
- `Table04–06`：selected Base 与 Official R01–R09
- `Table07–10`：8192 overlap 与 8K/4K/2K decomposition
- `Table11`：R01–R09 residual-stream share
- `Table12–13`：selected Base 与 Official R01–R09 的 8iVFB per-sequence 数据
- `Table14`：6 个 Base points 相对同 sequence Official 曲线的 matched-bpp 插值差值
- `Table15–17`：固定六个 Full points 的 RWTT/8i mean/per-sequence数据
- `Table18–19`：固定 Full points 相对 Official曲线的 matched-bpp诊断
- `Table20`：每个固定 point 的 Base/Full matched-bpp对照

完整文字汇总见 `BASE_FULL_MATCHED_OFFICIAL_REPORT.md`。

`03_raw_evidence/` 保留本轮 N30 CSV/JSON/Markdown/command data，以及既有 selected Base 和 official
RWTT-28Lite reference。没有 `.pth`、PLY、GPCC temporary files 或 Slurm logs。

## 重画

```bash
MPLCONFIGDIR=/tmp/matplotlib-multirate \
conda run -n unicorn-attr-cu121 \
python drafts/20260901_multirate_review/04_reproduction/build_review.py
```

更完整 provenance 与明确缺失项见 `review_info.json`。
