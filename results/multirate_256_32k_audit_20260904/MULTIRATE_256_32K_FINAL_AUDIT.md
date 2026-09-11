# Unified 256–32K Multirate Scalable RD Audit

> **HISTORICAL SCREENING EVIDENCE (2026-09-04).** The formal curve later froze
> at `512/1K/2K/4K/8K/16K/32K`; 256 is not a final thesis point. Recommendation
> labels below do not override the frozen registry or final 320-point package.

This report uses physical hard-rate evidence. Official-curve and neighbor interpolation are diagnostic local linear interpolation only; no extrapolation is used.

## Manager-facing decision table

| Point | Base RD | Full RD | Base ladder | Full ladder | Enh usefulness | Correctness | RWTT | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 8i +0.089 (4/4); Owlii +0.020 (2/4) | 8i +0.081 (4/4); Owlii +0.006 (2/4) | 8/8 | 8/8 | raw 0/8; RD 0/8 | PASS | PASS | KEEP_WITH_WARNING |
| 512 | 8i -0.020 (4/4); Owlii -0.009 (4/4) | 8i +0.076 (4/4); Owlii -0.006 (4/4) | 8/8 | 8/8 | raw 8/8; RD 3/8 | PASS | PASS | FREEZE |
| 1K | 8i -0.136 (4/4); Owlii -0.100 (4/4) | 8i +0.029 (4/4); Owlii -0.058 (4/4) | 8/8 | 8/8 | raw 8/8; RD 3/8 | PASS | PASS | FREEZE |
| 2K | 8i -0.072 (4/4); Owlii -0.326 (4/4) | 8i +0.109 (4/4); Owlii -0.208 (4/4) | 8/8 | 8/8 | raw 8/8; RD 7/8 | PASS | PASS | FREEZE |
| 4K | 8i +0.059 (4/4); Owlii -0.365 (4/4) | 8i +0.066 (4/4); Owlii -0.238 (4/4) | 0/8 | 8/8 | raw 7/8; RD 5/8 | PASS | PASS | KEEP_WITH_WARNING |
| 8K | 8i +0.234 (4/4); Owlii -0.255 (4/4) | 8i -0.109 (4/4); Owlii -0.179 (4/4) | 0/8 | 8/8 | raw 8/8; RD 3/8 | PASS | PASS | FREEZE |
| 16K | 8i +0.223 (4/4); Owlii -0.223 (4/4) | 8i -0.316 (4/4); Owlii +0.035 (4/4) | 8/8 | 8/8 | raw 8/8; RD 4/8 | PASS | PASS | KEEP_WITH_WARNING |
| 32K | 8i +0.113 (4/4); Owlii -0.288 (4/4) | 8i N/A (0/4); Owlii -0.446 (4/4) | 8/8 | 8/8 | raw 8/8; RD 0/8 | PASS | PASS | KEEP_WITH_WARNING |

## Candidate definitions

- **32K:** Base `D111 Base step5525`; Full `sequential D111-to-D611 step3525`.
- **16K:** Base `canonical Base step3525`; Full `D111 Enhancement step1763`.
- **8K:** Base `canonical Base step3525`; Full `D111 Enhancement step1763`.
- **4K:** Base `8K-to-4K joint step3000 Base`; Full `8K-to-4K joint step3000 Full`.
- **2K:** Base `2K-U-PATH step500`; Full `D111 Enhancement step1500`.
- **1K:** Base `2k128 Base step3525`; Full `D111 Enhancement step1763`.
- **512:** Base `2k128 Base step3525`; Full `D111 Enhancement step1763`.
- **256:** Base `2k128 Base step3525`; Full `D111 Enhancement step1763`.

## Main findings

1. **Correctness is closed for all eight selected points.** Every retained result comes from the physical hard path; Base uses exactly `x_low+r1-r4`, native r5 is absent, and Full hard round-trip/bit identity passed.
2. **The 4K joint step3000 Full is a usable scalable refinement, but its Base is rate-misplaced.** Full lies between the frozen 2K/8K Full rates on all eight external sequences and improves Base quality on 7/8; its Base rate exceeds the selected 8K Base on all eight external sequences. This is a ladder warning, not a codec-legality failure.
3. **The weakest Enhancement endpoint is 256.** Its Enhancement payload is nearly zero and quality is slightly below Base on most external sequences. The 256 Base remains a meaningful low-rate endpoint, but the selected Full does not add useful refinement.
4. **512, 1K and rescued 2K remain distinct low-rate points.** They provide rate separation; 512/1K Enhancements are useful, and the rescued 2K step1500 is consistently stronger than its discarded step1763 checkpoint.
5. **16K has a dataset-dependent warning.** Its RWTT-28Lite Full is close to the official curve, while its 8i mean is below the matched official interpolation. That is evidence for an external-generalization warning, not a correctness issue.
6. **The 8K Base `0/8` ladder count is induced by 4K Base overshoot.** It does not mean that 8K itself is RD-invalid: the selected 4K Base lies to the right of 8K Base on all external sequences.

Numbers in parentheses in the decision table are the count of sequences bracketed by the official R01-R09 curve. `N/A` means no extrapolation was performed.

## RWTT-28Lite selected endpoints

| Point | Base bpp | Base YUV611 | Base Δofficial | Full bpp | Full YUV611 | Full Δofficial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0761 | 31.695 | +0.383 | 0.0762 | 31.686 | +0.371 |
| 512 | 0.0973 | 32.373 | +0.473 | 0.1309 | 32.846 | +0.274 |
| 1K | 0.1190 | 32.877 | +0.534 | 0.2048 | 33.847 | +0.124 |
| 2K | 0.1734 | 33.820 | +0.535 | 0.3225 | 35.090 | +0.056 |
| 4K | 0.3714 | 35.454 | +0.040 | 0.6065 | 37.641 | +0.428 |
| 8K | 0.3122 | 35.170 | +0.231 | 0.8356 | 39.081 | +0.121 |
| 16K | 0.4042 | 35.840 | +0.172 | 1.1620 | 41.047 | +0.012 |
| 32K | 0.5120 | 36.360 | -0.131 | 2.0729 | 45.338 | N/A |

## 4K step1500 versus step3000

- **step1500:** Base mean Δofficial `-0.275 dB`; Full mean Δofficial `-0.116 dB`; Base rate-in-range `0/8`; Full rate-in-range `8/8`; Enhancement RD-positive `7/8`.
- **step3000:** Base mean Δofficial `-0.153 dB`; Full mean Δofficial `-0.086 dB`; Base rate-in-range `0/8`; Full rate-in-range `8/8`; Enhancement RD-positive `5/8`.

step3000 remains the manager-provisional point: it gives the stronger Full endpoint overall, while neither checkpoint cures the cross-lambda Base-rate overshoot.

## Answers to the review questions

- **Freeze now:** 512, 1K, rescued 2K, and 8K have sufficient evidence under the current contracts.
- **Keep with warning:** 4K step3000 (Base rate placement), 16K (8i Full efficiency), and 32K (legacy exception/out-of-range interpolation).
- **True pathology:** the selected 256 Full is effectively redundant; this does not invalidate the 256 Base. No selected point has a hard correctness failure.
- **Low-rate value:** 256/512/1K Bases are distinct. The 256 Full is not; 512 and 1K Full remain useful.
- **2K:** yes, the selected U-PATH step500 Base + D111 step1500 Full is stable enough to freeze.
- **4K checkpoint trade-off:** step1500 is earlier and marginally less rate-drifted on some content; step3000 has the stronger overall Full quality/RD evidence. Both retain the Base-placement warning.
- **8K/16K:** 8K needs no further tuning evidence. 16K only warrants an external checkpoint/Stage-2 selection review if one more experiment is allowed.
- **If only 1–2 GPU experiments remain:** first address 256 Enhancement usefulness; second evaluate/select a 16K Stage-2 candidate externally. Do not retrain 4K before deciding whether its Base-rate placement is acceptable in the thesis ladder.

## Figures

- `RWTT_28LITE_MULTIRATE_RD.png`: Official Unicorn, selected Base ladder, selected Full ladder.
- `8IVFB_MULTIRATE_RD_FOUR_PANEL.png`: four 8i sequences, no dataset averaging.
- `OWLII_MULTIRATE_RD_FOUR_PANEL.png`: four Owlii sequences, no dataset averaging.

## Data completeness

All three requested contracts are complete for the selected 256–32K set. No additional GPU evaluation is required for this audit.
