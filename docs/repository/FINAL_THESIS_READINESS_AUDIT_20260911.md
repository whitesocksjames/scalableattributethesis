# Final repository consistency and thesis-readiness audit — 2026-09-11

Scope: repository documentation, indexes, frozen registry, and final artifact
package. No scheduler query, GPU experiment, model/checkpoint change, result
mutation, metric change, or BD-BR contract change was performed.

## Initial findings

| Class | Finding before repair | Severity |
|---|---|---|
| Final artifacts | `formal_static_rgb_final_20260911` existed locally but was untracked and absent from remote HEAD `d86bf98`. | Hard repository inconsistency |
| CURRENT navigation | Root README and CURRENT result/operating-point pages still described working/provisional candidates and included 256. | Hard documentation inconsistency |
| Frozen registry | `current_candidates.json` still had eight working candidates and unverified identities instead of the frozen seven-point manifest. | Hard provenance/index inconsistency |
| Evaluation entrypoint | The CURRENT page predated the official pinned runner and final CTC/formal harness and did not fully define final runtime components. | Hard documentation inconsistency |
| Historical results | The 318/320 competitiveness package and 256–32K screening reports could be mistaken for final evidence when opened directly. | Stale-document risk |
| Terminology | CURRENT pages lacked one canonical contribution phrase and an explicit claim boundary. | Terminology risk |

The final package itself had no numeric mismatch: it consistently recorded 20
samples, 180/180 Original, 140/140 Ours, 320/320 logical evaluations, 460
endpoints, 68/80 available BD-BR values, CTC 9/12 valid-sequence subset,
13/140 negative Full endpoints, and the worst `-0.2418 dB Y / -0.18065 dB
YUV611` degradation.

## Repairs

- Promoted the complete final package to the CURRENT result index.
- Replaced working/provisional navigation with the seven-point frozen mapping.
- Rebuilt `current_candidates.json` as a final registry and checked all seven
  points against `formal_static_rgb_v1_checkpoints.json`: point membership,
  lambda, profile, Base/Full loader, and checkpoint SHA-256 pass 7/7.
- Marked 256 as historical screening only; retained the legacy CLI recipe with
  an explicit non-authority notice.
- Standardized the contribution as **full-resolution quality-scalable extension
  of Unicorn Part II** and the codec flow as `native truncated Unicorn prefix →
  learned BaseSynthesis → full-resolution Base → conditional Enhancement →
  Full`.
- Added explicit boundaries against claiming that Unicorn lacks progressive
  decoding, that layered coding is first introduced here, or that Ours uniformly
  outperforms Unicorn.
- Documented Official and Ours Enc/Dec component boundaries, CUDA
  synchronization, and hardware-separated runtime aggregation.
- Added direct superseded banners to the highest-risk dated reports and a
  repository-wide historical-document status index.

## Post-repair authority

- Project/current navigation: `README.md`, `docs/repository/CURRENT_*.md`.
- Scientific/checkpoint contract:
  `docs/reproduction/FORMAL_8I_OWLII_CTC_EXECUTION_PLAN_20260909.md` and
  `configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json`.
- Final results: `results/comparisons/formal_static_rgb_final_20260911/`.
- Historical routing: `docs/repository/HISTORICAL_DOCUMENT_STATUS.md`.

## Remaining non-blocking limitations

- Numerous user-owned untracked historical raw/draft artifacts remain outside
  this commit. They were neither staged nor altered. A clean clone is not a
  guarantee for reconstructing every historical screening plot.
- Dated handoffs retain old counts and scheduler observations as historical
  evidence. Their superseding headers must remain intact.
- The 12 unavailable BD-BR entries, CTC 9/12 subset limitation, negative
  Enhancement cases, and cross-GPU numerical variance are scientific reporting
  limitations, not repository inconsistencies.

Verdict after tracked-package validation and successful push: **THESIS-READY
WITH EXPLICIT LIMITATIONS**.
