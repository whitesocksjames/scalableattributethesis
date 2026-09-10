# Formal RD competitiveness review — 2026-09-10

Read-only review of 318/320 FORMAL_REUSABLE logical points. House and Shiva are pending; no point was removed and no extrapolation was used.

## Classification rule

For descriptive classification only, each Ours Full raw point inside the Official bpp range is compared with a log-bpp linear interpolation through all Official raw points sorted by physical bpp. This is not BD-BR. `competitive`: median delta >= -0.25 dB and at least 60% within -0.5 dB; `clearly weaker`: median delta <= -1.0 dB and at least 70% below -0.5 dB; otherwise `mixed`. Overall requires agreement from Y and YUV611.

## Dataset summary

| Dataset | Competitive | Mixed | Clearly weaker | Pending | Median ΔY | Median ΔYUV611 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 8i | 4 | 0 | 0 | 0 | -0.132 | -0.088 | competitive |
| Owlii | 4 | 0 | 0 | 0 | -0.095 | -0.082 | competitive |
| CTC | 9 | 1 | 0 | 2 | 0.074 | 0.087 | competitive |

Overall, Ours Full is competitive with Official Unicorn on the currently
closed evidence: 17 samples are classified competitive, Facade is mixed, and
none is clearly weaker. House and Shiva remain pending and are not used for a
final sample verdict. The repeated frozen 4K/8K point-order reversal is shown
in the raw plots but is not treated as a competitiveness blocker.

## Full minus Base summary

Across all 138 currently reusable Ours points (including the six available
points each for pending House and Shiva), the mean deltas are +1.468 dB Y,
+1.148 dB YUV611, and +0.136463 bpp; medians are +0.526 dB Y, +0.396 dB
YUV611, and +0.023349 bpp. There are 8 negative-Y points and 13
negative-YUV611 points, whose union is 13 logical points.

For the 126 points excluding pending House/Shiva, the dataset summaries are:

| Dataset | Points | Mean ΔY | Median ΔY | Mean ΔYUV611 | Median ΔYUV611 | Y positive | YUV611 positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8i | 28 | 2.569 | 1.871 | 2.095 | 1.507 | 28/28 | 27/28 |
| Owlii | 28 | 2.188 | 1.518 | 1.680 | 1.164 | 28/28 | 28/28 |
| CTC | 70 | 0.784 | 0.158 | 0.597 | 0.135 | 62/70 | 58/70 |

The enhancement therefore has substantial practical effect at medium/high
rates. Negative cases are concentrated in CTC low/mid-rate points: four at
1K, two at 2K, four at 4K, two at 8K, and one at 512. There are no negative
16K or 32K cases. Only three logical points exceed a 0.10 dB degradation:
CTC loot 1K and CTC redandblack 1K/2K.

## Per-sample verdict

| Dataset | Sample | Y | YUV611 | Overall | Median ΔY | Median ΔYUV611 |
|---|---|---|---|---|---:|---:|
| 8i | longdress | competitive | competitive | competitive | -0.160 | -0.164 |
| 8i | loot | competitive | competitive | competitive | -0.107 | -0.039 |
| 8i | redandblack | competitive | competitive | competitive | -0.096 | -0.025 |
| 8i | soldier | competitive | competitive | competitive | -0.156 | -0.136 |
| CTC | Facade_00009_vox12 | mixed | competitive | mixed | -0.261 | -0.234 |
| CTC | House_without_roof_00057_vox12 | pending | pending | pending | -0.119 | -0.064 |
| CTC | Shiva_00035_vox12 | pending | pending | pending | -0.180 | -0.172 |
| CTC | Staue_Klimt_vox12 | competitive | competitive | competitive | -0.168 | -0.137 |
| CTC | Thaidancer_viewdep_vox12 | competitive | competitive | competitive | -0.024 | 0.017 |
| CTC | basketball_player_vox11_00000200 | competitive | competitive | competitive | 0.103 | 0.112 |
| CTC | boxer_viewdep_vox12 | competitive | competitive | competitive | -0.017 | 0.074 |
| CTC | dancer_vox11_00000001 | competitive | competitive | competitive | 0.092 | 0.118 |
| CTC | longdress_viewdep_vox12 | competitive | competitive | competitive | 0.076 | 0.122 |
| CTC | loot_viewdep_vox12 | competitive | competitive | competitive | 0.080 | 0.022 |
| CTC | redandblack_viewdep_vox12 | competitive | competitive | competitive | 0.110 | 0.158 |
| CTC | soldier_viewdep_vox12 | competitive | competitive | competitive | 0.071 | 0.100 |
| Owlii | basketball_player | competitive | competitive | competitive | -0.035 | -0.014 |
| Owlii | dancer | competitive | competitive | competitive | -0.164 | -0.149 |
| Owlii | exercise | competitive | competitive | competitive | 0.023 | 0.044 |
| Owlii | model | competitive | competitive | competitive | -0.155 | -0.188 |

## Enhancement negatives

Negative logical points: 13. Magnitude classes: {'tiny (≤0.05 dB)': 5, 'material (>0.10 dB)': 3, 'small (0.05–0.10 dB)': 5}.

| Dataset | Sample | Point | Δbpp | ΔY | ΔYUV611 | Magnitude |
|---|---|---|---:|---:|---:|---|
| CTC | boxer_viewdep_vox12 | 4k | 0.001067 | 0.0290 | -0.0082 | tiny (≤0.05 dB) |
| CTC | boxer_viewdep_vox12 | 8k | 0.003161 | -0.0041 | -0.0030 | tiny (≤0.05 dB) |
| CTC | longdress_viewdep_vox12 | 1k | 0.000827 | -0.0406 | -0.0254 | tiny (≤0.05 dB) |
| CTC | loot_viewdep_vox12 | 1k | 0.000011 | -0.1279 | -0.0561 | material (>0.10 dB) |
| CTC | loot_viewdep_vox12 | 2k | 0.000358 | 0.0006 | -0.0818 | small (0.05–0.10 dB) |
| CTC | loot_viewdep_vox12 | 4k | 0.000491 | 0.0236 | -0.0550 | small (0.05–0.10 dB) |
| CTC | loot_viewdep_vox12 | 8k | 0.001851 | -0.0671 | -0.0586 | small (0.05–0.10 dB) |
| CTC | redandblack_viewdep_vox12 | 1k | 0.000552 | -0.2418 | -0.1806 | material (>0.10 dB) |
| CTC | redandblack_viewdep_vox12 | 2k | 0.001799 | -0.1669 | -0.1416 | material (>0.10 dB) |
| CTC | redandblack_viewdep_vox12 | 512 | 0.000012 | -0.0763 | -0.0357 | small (0.05–0.10 dB) |
| CTC | soldier_viewdep_vox12 | 4k | 0.001515 | 0.0171 | -0.0505 | small (0.05–0.10 dB) |
| 8i | loot | 4k | 0.001868 | 0.0162 | -0.0147 | tiny (≤0.05 dB) |
| CTC | Facade_00009_vox12 | 1k | 0.003910 | -0.0090 | -0.0315 | tiny (≤0.05 dB) |

## Clearly weak samples

None under the declared descriptive threshold.

Facade is the only watch-list sample (`mixed`, not `clearly weaker`). Across
its six Full points within Official's bpp range (0.571–1.660 bpp), Y is
0.161–0.455 dB lower and YUV611 is 0.148–0.284 dB lower. This is a modest,
broad deficit rather than a severe collapse at one bitrate range.
