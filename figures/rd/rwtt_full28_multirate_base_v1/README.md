# RWTT Full28 multi-rate Canonical Base V1

数据：固定 RWTT Validation split，28 original models、792 H5、57,904,570 points。

评价 contract：physical hard attribute rate；author `pc_error` Y/U/V；YUV-PSNR 6:1:1；H5 distortion point-weighted 聚合到 original model，再对 28 models equal-average。

蓝线是 released Unicorn R01–R09 在相同 Full28 contract 上的本地正式 reference；橙线是 Canonical Base 2K/4K/8K/16K/32K formal endpoints。

`CANONICAL_BASE_FORMAL_FULL28_5PT.csv/json` 是主要机器可读 artifact；PNG 为快速查看图。

