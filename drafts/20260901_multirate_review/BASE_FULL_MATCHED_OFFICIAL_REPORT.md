# Base/Full matched-rate comparison against Official R01–R09

Metric: physical attribute bpp and author `pc_error` YUV-PSNR 6:1:1.
Interpolation: linear in the bpp–YUV611 plane between adjacent Official points.
A positive delta means the endpoint lies above the same-contract Official interpolation.
No extrapolation is performed outside R01–R09.

## RWTT-28Lite

| Point | Base bpp | Base dB | Base Δmatched | Full bpp | Full dB | Full Δmatched | Δbpp Full−Base | ΔdB Full−Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 0.511988 | 36.3600 | -0.131 | 2.072941 | 45.3378 | N/A | 1.560953 | +8.9778 |
| 16K | 0.404211 | 35.8402 | +0.172 | 1.162033 | 41.0472 | +0.012 | 0.757822 | +5.2071 |
| 8K | 0.312247 | 35.1701 | +0.231 | 0.835552 | 39.0808 | +0.121 | 0.523304 | +3.9108 |
| 1K | 0.118964 | 32.8772 | +0.534 | 0.204794 | 33.8466 | +0.124 | 0.085830 | +0.9694 |
| 512 | 0.097287 | 32.3731 | +0.473 | 0.130883 | 32.8456 | +0.274 | 0.033596 | +0.4725 |
| 256 | 0.076068 | 31.6954 | +0.383 | 0.076177 | 31.6858 | +0.371 | 0.000109 | -0.0096 |

## 8iVFB-fixed4-mean

| Point | Base bpp | Base dB | Base Δmatched | Full bpp | Full dB | Full Δmatched | Δbpp Full−Base | ΔdB Full−Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 0.453447 | 39.7136 | +0.181 | 1.044457 | 43.3501 | N/A | 0.591010 | +3.6365 |
| 16K | 0.353982 | 38.8299 | +0.301 | 0.747377 | 41.8704 | -0.309 | 0.393394 | +3.0405 |
| 8K | 0.269079 | 37.7085 | +0.339 | 0.517961 | 40.0326 | -0.116 | 0.248882 | +2.3241 |
| 1K | 0.117917 | 34.4291 | +0.106 | 0.160664 | 35.3826 | +0.058 | 0.042747 | +0.9535 |
| 512 | 0.092824 | 33.7449 | +0.263 | 0.105673 | 34.1216 | +0.209 | 0.012849 | +0.3768 |
| 256 | 0.068070 | 32.7891 | +0.241 | 0.068079 | 32.7824 | +0.234 | 0.000009 | -0.0068 |

## Notes

- The 32K Full endpoint is above the maximum Official R01 bpp on both datasets, so its matched delta is N/A.
- Interpolation is a diagnostic, not a formal BD-rate result.
- All listed endpoints use physical hard rate and passed their recorded hard correctness checks.
