# Current frozen operating points

The thesis formal curve is frozen at seven points:
`512 / 1K / 2K / 4K / 8K / 16K / 32K`. Selection is predeclared and must not
be changed from test performance. The authoritative content-identity manifest
is [`formal_static_rgb_v1_checkpoints.json`](../../configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json);
the human/machine-readable final registry is
[`current_candidates.json`](../../configs/scalable_attribute/current_candidates.json).

| Point | λ / released profile | Frozen Base checkpoint | Frozen Full checkpoint | Loader |
|---|---:|---|---|---|
| 32K | 32768 / `32k8k` | D111 Base `step5525` | sequential D611 `step3525` | base synthesis / sequential enhancement |
| 16K | 16384 / `32k8k` | canonical Base `step3525` | D111 Enhancement `step1763` | base synthesis / independent enhancement |
| 8K | 8192 / `32k8k` | canonical Base `step3525` | D111 Enhancement `step1763` | base synthesis / independent enhancement |
| 4K | 4096 / joint source `32k8k` | 8K→4K joint `step3000` | same joint `step3000` | joint scalable Base / Full |
| 2K | 2048 / `8k256` | rescued `2K-U-PATH step500` | rescued D111 Enhancement `step1500` | rescued Base / independent enhancement |
| 1K | 1024 / `2k128` | 1K Base `step3525` | D111 Enhancement `step1763` | base synthesis / independent enhancement |
| 512 | 512 / `2k128` | 512 Base `step3525` | D111 Enhancement `step1763` | base synthesis / independent enhancement |

The 4K joint state is conditioned at target λ 4096 but has a frozen 8K Base
bootstrap lineage and `base_checkpoint_lambda=8192`. This is provenance, not a
license to substitute the old standalone 4K checkpoint. The 2K frozen state is
the U-PATH rescue plus its selected Enhancement, not the older canonical 2K
Base. Paths alone are insufficient: runtime validation requires the manifest
SHA-256, size, architecture, lambda/profile, loader branch, and parent lineage.

`256` is retained only in dated screening evidence and the legacy CLI recipe.
It is not part of the final formal curve and must not enter thesis RD/BD-BR
tables.

## Authority boundary

[`canonical_operating_points.json`](../../configs/scalable_attribute/canonical_operating_points.json)
is a legacy CLI compatibility recipe. Its historical 2K/4K selections and 256
entry are not final checkpoint authority. Formal runs and final reporting use
the frozen manifest and seven-point registry above.

## Evidence anchors

- [Formal scientific contract](../reproduction/FORMAL_8I_OWLII_CTC_EXECUTION_PLAN_20260909.md)
- [Final 320-point package](../../results/comparisons/formal_static_rgb_final_20260911/README.md)
- [4K joint selection evidence](../../results/joint_8k_to_4k_external_20260904/selected_step3000_review/SELECTED_4K_STEP3000_REVIEW.md)
- [2K rescue selection evidence](../../results/rescued_2k_enh_external_20260903/RESCUED_2K_ENH_EXTERNAL_ANALYSIS.md)
