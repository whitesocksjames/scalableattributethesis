# Thesis Diagnostic Analysis Summary

## Verified facts

- The diagnostic 256 lineage is the pre-CTC sequential candidate; it is not part of the frozen seven-point formal curve.
- Diagnostic 256 completeness: 20/20 samples; incomplete: 0.
- Every diagnostic 256 Enhancement payload is at the observed coder floor of one byte per independently coded input (one byte per CTC chunk).
- Diagnostic 256 mutually exclusive Base-to-Full outcomes: positive above 0.01 dB 8, near-zero (`|Delta YUV611| <= 0.01 dB`) 6, negative below -0.01 dB 6.
- Q3 reconstruction diagnostics: 29/29 analyzed; exact frozen reproduction 28/29; bounded runtime-sensitivity evidence 1/29; incomplete 0; failed 0.

## Supported interpretation

- Low-rate boundary decision under the predeclared diagnostic rule: **NO claim**.
- The 256 result supports a near-zero-rate/coder-floor statement, but not universal Base/Full equality or universal ineffectiveness: CTC contains measurable positive and negative content-dependent corrections.
- Q3 classifications are evidence labels, not universal causal claims: {"globally_aligned_content_dependent_outcome": 9, "minority_pointwise_improvement": 15, "nonpositive_residual_alignment": 5}.

## Remaining uncertainty

- A checkpoint objective permits, but does not by itself explain, a negative per-sample Base-to-Full change.
- Content-specific behavior and quantized physical reconstruction effects cannot be promoted to causal claims without additional controlled training studies.
- One Q3 case (redandblack 8K) remains non-exact: Base and Full are each 16 bits below frozen, the Enhancement payload is exact, and metric differences are at most 0.0003 dB. It is labeled sensitivity evidence rather than exact reproduction.
- This package is complete; missing/failed cases must not be silently omitted.

## Safe thesis wording

- Report the 256 candidate as a post-freeze diagnostic using the historically established sequential lineage.
- State that 256 reaches a one-byte-per-input coder floor and gives unreliable, dataset-sensitive Base-to-Full separation; do not imply that 256 was excluded after inspecting CTC.
- Describe negative refinements as measured non-monotonic cases, then report whether D111 improved, channel trade-offs occurred, and residual/correction alignment was weak.

## Claims to avoid

- Do not claim that the training objective is the proven cause of every negative case.
- Do not claim universal Enhancement improvement.
- Do not call 256 a universal latent collapse without persisted latent-symbol evidence.
- Do not add 256 to the frozen formal curve or recompute final BD-BR with it.

## Thesis and appendix material

- Main thesis: `tables/diagnostic256_dataset_summary.csv`, the 256-vs-512 summary, and `figures/diagnostic256_full_minus_base.svg`.
- Detailed low-rate interpretation: `CTC256_ANALYSIS.md`.
- Enhancement Effectiveness Analysis: retain the frozen 140-point aggregate statistics.
- Non-monotonic Base-to-Full Refinement: summarize the 13 negatives by evidence category.
- Appendix: `tables/q3_case_diagnostics.csv`, full matched-control mapping, provenance gates, and per-case reconstruction statistics.
- Final Q3 interpretation: `Q3_FINAL_ANALYSIS.md`.
