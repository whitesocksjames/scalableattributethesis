# Longdress multi-rate Canonical Base V1

数据：8iVFB `longdress_vox10_1300`，857,966 points。

评价 contract：physical hard attribute rate；author `pc_error` Y/U/V；YUV-PSNR 6:1:1。

蓝线是作者 released Unicorn R01–R09 checkpoints 在相同 frame 上的本地 physical reproduction。其质量与作者数据相符；physical bpp 包含本地实际 codec bytes，因此与作者表中的 rounded/estimated bpp 可能有极小差异。

橙线是 Canonical Base 五个 operating points，只包含 `x_low+r1+r2+r3+r4`，不包含 native r5 或 Enhancement bits。Lower-rate checkpoint 由 frozen RWTT-28Lite selection 和 8i fixed-4 sanity 决定。

输入数据见 `longdress_official_r01_r09_and_base_5pt.csv`。

