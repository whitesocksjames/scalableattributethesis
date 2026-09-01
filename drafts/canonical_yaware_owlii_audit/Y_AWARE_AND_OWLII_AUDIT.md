# Y-aware and Owlii Audit

## 1. Owlii preprocessing

The four read-only Owlii vox11 frames were mapped with the author-compatible operation `floor(xyz / 2)`, followed by duplicate-voxel RGB mean and rounding. Point-count fingerprinting gives an exact 4/4 match:

| Sequence | Source | Derived | Author target | Exact |
|---|---:|---:|---:|---|
| Basketball | 2,925,514 | 796,217 | 796,217 | YES |
| Dancer | 2,592,758 | 702,038 | 702,038 | YES |
| Exercise | 2,391,718 | 645,135 | 645,135 | YES |
| Model | 2,458,429 | 657,755 | 657,755 | YES |

The `round(xyz / 2)` alternative fails all four fingerprints. Local Official R01 versus the retained author CSV differs by at most 0.0028 dB in YUV611, so preprocessing and metric provenance pass.

## 2. Owlii generalization

| Sequence | R01 bpp/dB | Full1763 bpp/dB | Full3525 bpp/dB | 3525-1763 bpp/dB |
|---|---:|---:|---:|---:|
| Basketball | 0.531031 / 45.2183 | 0.487234 / 44.7433 | 0.483275 / 44.4493 | -0.003959 / -0.2940 |
| Dancer | 0.642301 / 45.1166 | 0.593062 / 44.5605 | 0.593221 / 44.2306 | +0.000160 / -0.3298 |
| Exercise | 0.569667 / 43.8875 | 0.548400 / 43.8161 | 0.511322 / 43.3655 | -0.037078 / -0.4506 |
| Model | 0.914347 / 44.4966 | 0.875317 / 43.9444 | 0.865307 / 43.4408 | -0.010010 / -0.5037 |

From step1763 to step3525, Y decreases by 0.3823/0.4395/0.5932/0.6501 dB and YUV611 decreases on all four sequences. Rate saving is inconsistent: three save rate, one increases slightly. Every canonical hard round-trip is exact, and each sequence shares one hard prefix state.

**Owlii confirms the late-stage external generalization issue already observed on 8iVFB.**

## 3. Y-aware training and Full28

All arms resume the same step1763 model and Adam state, use LR `1e-5`, BS4, 300 additional updates, and keep Base/prefix frozen. D111 equivalence and gradient gates pass. The fixed 28-H5 trajectory is hard-coded with 28 shared prefix invocations and all nine checkpoint round-trips exact.

| Endpoint | bpp | Y | U | V | YUV611 |
|---|---:|---:|---:|---:|---:|
| Official R01 | 1.577542 | 41.0432 | 46.7048 | 47.8502 | 42.6018 |
| Original step1763 | 1.656177 | 41.5546 | 46.6417 | 47.6330 | 42.9503 |
| Original step3525 | 1.586111 | 40.9566 | 46.7589 | 47.7792 | 42.5347 |
| D111 +300 | 1.664691 | 41.6455 | 46.7323 | 47.8066 | 43.0515 |
| D411 +300 | 2.041408 | 44.1312 | 46.1634 | 47.3734 | 44.7905 |
| D611 +300 | 2.102934 | 44.5448 | 45.9800 | 47.2320 | 45.0601 |

D111 moves only +0.00851 bpp / +0.10121 dB from step1763. D411 and D611 strongly preserve/improve Y, but cost +0.38523 and +0.44676 bpp respectively. They therefore demonstrate a quality-rate tradeoff, not a controlled RD-efficiency improvement under the specified same-or-lower-rate criterion. Increasing Y weight also sacrifices U/V quality relative to step1763.

## 4. Manager decision

- **Does Owlii confirm external late-stage degradation? YES.**
- **Does Y-aware loss improve Full28 RD efficiency? NO under the approved criterion.** It raises quality by spending substantially more rate.
- **Best next candidate: Original step1763** for the current Canonical Full V1 evidence set. D111 +300 is close but does not dominate it.
- **Recommended next action:** retain step1763 and review whether the next experiment should explicitly target rate matching before considering further Y-aware training.

No new external evaluation of the Y-aware arms and no additional training were launched.
