# B1 Universality Generalization — R01 + R04

> Evidence-only draft. No B1-H, training, mapping search or architecture modification.

## Frozen contract

- R01: official `32k8k`, lambda=32768.
- R04: official `8k256`, lambda=4096.
- Fixed deterministic `step=4/8`; qB + conditional qE exactly recovers original q.
- Full28 hard evaluation once per source; Dev14 derived from the same raw per-H5 rows.

## Results

Submitted:

- R01 Full28 shards: `1786673 / 1786674 / 1786675 / 1786676`.
- R01 afterok aggregation: `1786677`.
- R04 Full28 shards: `1786678 / 1786679 / 1786680 / 1786681`.
- R04 afterok aggregation: `1786682`.

All shards and both dependency aggregations completed successfully.

| Source | Dataset | Step | Base bpp | Base PSNR | Full bpp | Full PSNR | ratio | exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R01 | Full28 | 4 | 0.777586 | 37.798036 | 1.576960 | 42.601751 | 0.999561 | yes |
| R01 | Full28 | 8 | 0.606325 | 36.448086 | 1.575718 | 42.601751 | 0.998768 | yes |
| R01 | Dev14 | 4 | 0.796937 | 37.829586 | 1.638234 | 42.524176 | 0.999833 | yes |
| R01 | Dev14 | 8 | 0.612112 | 36.583579 | 1.637029 | 42.524176 | 0.999255 | yes |
| R04 | Full28 | 4 | 0.239900 | 33.050201 | 0.425166 | 35.228701 | 0.997793 | yes |
| R04 | Full28 | 8 | 0.193945 | 32.466012 | 0.425576 | 35.228701 | 0.999536 | yes |
| R04 | Dev14 | 4 | 0.232016 | 33.259470 | 0.421595 | 35.293845 | 0.998387 | yes |
| R04 | Dev14 | 8 | 0.186172 | 32.871960 | 0.421996 | 35.293845 | 1.000182 | yes |

For every row: qB physical round-trip exact = yes; qB+qE exact original q = yes; Full `max_abs_difference=0`.

### R01 — same-family high-rate generalization

Full28 official neighbors：R01 `1.577542 / 42.601755`；R02 `1.240016 / 41.045284`；R03 `0.952912 / 39.330371`。

- step4 Base `0.777586 / 37.798036`；step8 Base `0.606325 / 36.448086`。
- 两个Base均是从R03继续向低rate方向的合理local progression；linear interpolation仅作visual diagnostic时约低于R03-R04 segment `0.17–0.18 dB`，不作为formal claim。
- Dev14与Full28方向一致；Full exact恢复R01；physical ratio与1偏差≤0.13%。

Assessment：**PASS**。

### R04 — cross-family generalization

Full28 official neighbors：R03 `0.952912 / 39.330371`；R04 `0.426228 / 35.228702`；R05 `0.324253 / 34.477281`；R06 `0.218998 / 33.399184`；补充观察R07 `0.152390 / 32.523601`。

- step4 Base `0.239900 / 33.050201`：比R06多 `0.020902 bpp`且低 `0.348983 dB`，被R06严格支配。
- step8 Base `0.193945 / 32.466012`：比R07多 `0.041555 bpp`且低 `0.057589 dB`，被R07严格支配。
- Dev14 step4同样被R06支配；step8虽不被单个neighbor严格支配，但仍位于local progression弱侧。
- Full28和Dev14方向一致，因此不是Dev14 subset reversal；但fixed-4中观察到的near-envelope优势未generalize。
- Full preservation和physical legality全部PASS，ratio与1最大偏差约0.22%。

Assessment：**WARNING**。

## Generalization gate

- R01 same-family high-rate：PASS。
- R04 cross-family：WARNING（correctness PASS，Base RD generalization弱于fixed-4）。
- R02此前Full28：PASS。

因此当前证据不满足“R02 + R01 + R04均Full28 PASS”。可以正式陈述B1 fixed mapping具有R01/R02跨operating-point evidence，但尚不能把cross-checkpoint-family R04写成PASS。

本轮不授权B1-H；不自动触发任何新实验。
