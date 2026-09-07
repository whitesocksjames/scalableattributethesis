# Current operating points

This is the human-readable companion to
[`configs/scalable_attribute/current_candidates.json`](../../configs/scalable_attribute/current_candidates.json).
The entries below are current working candidates, not a final freeze. Every
point has a Base and Full identity; `effective` is the state used for that
endpoint, while `bootstrap` is the initial loader/source state.

| Point | λ / released profile | Base effective checkpoint | Full effective checkpoint | Loader / status |
|---|---:|---|---|---|
| 32K | 32768 / `32k8k` | D111 Base `step5525` | sequential D111→D611 `step3525` | base + sequential enhancement; working |
| 16K | 16384 / `32k8k` | canonical Base `step3525` | D111 Enhancement `step1763` | independent enhancement; working |
| 8K | 8192 / `32k8k` | canonical Base `step3525` | D111 Enhancement `step1763` | independent enhancement; working |
| 4K | 4096 / joint source `32k8k` | 8K→4K joint `step3000` | 8K→4K joint `step3000` | source λ 8192 → target λ 4096; working/provisional |
| 2K | 2048 / `8k256` | rescued `2K-U-PATH step500` | D111 Enhancement `step1500` | rescued Base + independent enhancement; working |
| 1K | 1024 / `2k128` | 2k128 Base `step3525` | D111 Enhancement `step1763` | independent enhancement; working |
| 512 | 512 / `2k128` | 2k128 Base `step3525` | D111 Enhancement `step1763` | independent enhancement; working |
| 256 | 256 / `2k128` | 2k128 Base `step3525` | D111 Enhancement `step1763` | independent enhancement; working; remains evaluable |

## Provenance rules

Checkpoint paths in the machine registry are portable paths relative to a
recorded `origin_root`. Repository evidence paths point to retained
`resolved_args.json`, raw endpoint summaries, commands, or review tables. A
missing checkpoint hash is represented as `sha256: null` and
`verification: unverified`; no hash is inferred from a filename or location.

The registry distinguishes `initial_loader_bootstrap` from
`effective_checkpoint` / `effective_full_state_checkpoint`. This matters for
the 4K joint candidate: its effective joint state is `step3000`, but it was
bootstrapped from the 8K Base (`32k8k`, source λ 8192) and evaluated at target
λ 4096. The 2K entry similarly records the rescue initialization separately
from the selected U-PATH `step500` state.

The runtime mapping in
[`canonical_operating_points.json`](../../configs/scalable_attribute/canonical_operating_points.json)
is intentionally unchanged. Its official point schema cannot express the 4K
joint source-profile/target-lambda contract, so this registry is descriptive
and is not consumed as a runtime replacement.

## Evidence anchors

- [Current 256–32K master review](../../results/multirate_256_32k_audit_20260904/MULTIRATE_256_32K_MASTER_REVIEW.csv)
- [4K joint step3000 review](../../results/joint_8k_to_4k_external_20260904/selected_step3000_review/SELECTED_4K_STEP3000_REVIEW.md)
- [2K rescued enhancement analysis](../../results/rescued_2k_enh_external_20260903/RESCUED_2K_ENH_EXTERNAL_ANALYSIS.md)
- [Historical candidate inventory](../../results/multirate_candidate_inventory_20260904/MULTIRATE_CANDIDATE_DATA_INVENTORY.md) — its 4K RWTT MISSING and freeze labels are superseded; use [current result index](CURRENT_RESULT_INDEX.md).
