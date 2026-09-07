# Phase 1 repository rationalization — 2026-09-07

Status: staged organization, not final publication freeze. No experiments authorized by this document.

## Phase 1 completion

Completed on 2026-09-07 following manager approval. The root README now provides
five current navigation routes. Inventory and candidate provenance are published;
stale runtime selection and broken historical HPC submission instructions are
explicitly marked. Original runtime configuration values, source semantics and
selected results are unchanged. No files were physically moved or deleted.

Validation: navigation links and registry evidence paths resolve locally; JSON
and inventory CSV parse; the existing operating-point resolver accepts the added
descriptive metadata; existing runtime config values match the parent commit;
whitespace checks pass. Checkpoint hashes remain explicitly unverified where
not retained. Phase 2/3/4 are not started by this completion.

Recovery reference: `audit/pre-finalization-20260906` at
`997643570834e0fa3e6b280d37cddda985243539`. Work branch:
`organize/phase1-20260907`. The snapshot does **not** contain every local untracked
raw table/report/SVG; retain these local files, too. Do not assume Git can recover them.

## Before / target

Before: root README presents upstream Unicorn; current candidates are distributed
between reports, evaluator arguments and plotting scripts; old official-mapping
JSON still contains superseded selections; historical HPC sweep commands target
missing scripts.

Target: root README leads to five current navigation pages. Runtime and evidence
paths remain stable. A separate current candidate registry describes effective
checkpoints without forcing joint checkpoints through the old official schema.

| Action | Exact scope | Dependency / replacement | Recovery |
| --- | --- | --- | --- |
| Keep | `scalable_attribute/`, `lossy_attribute/`, `basic_models/`, `data_utils/`, `cfg/`, `third_party/` | Current canonical code depends on native modules; rescue/joint loaders remain needed | Existing Git history |
| Keep | `scripts/`, `configs/`, `data_splits/` | Preserve historical CLI/schema compatibility | Snapshot and local untracked files |
| Keep | all `drafts/` and `results/` | Current aggregate builders read historical tables; names do not imply dead code | Snapshot where tracked; otherwise original local files |
| Update | `README.md`, `HPC.md`, `HPC_GUIDE.md`, `docs/CANONICAL_TRAINING_PROTOCOL.md` | Link current navigation; identify historical submission instructions | Parent commit |
| Annotate | `configs/scalable_attribute/canonical_operating_points.json` | Retain existing resolver semantics; point readers to current registry | Parent commit |
| Create | `docs/repository/`, `configs/scalable_attribute/current_candidates.json` | Current navigation, provenance and inventory | New branch |
| Move/archive physically | None | Logical historical classification only | Not applicable |
| Delete | None | No dependency-safe deletion needed to achieve 30-second navigation | Not applicable |

## Order and checks

1. Inventory major directories and key stale/broken paths.
2. Publish architecture, candidates, training, evaluation and result entry points.
3. Annotate stale JSON and historical documentation without changing selections or recipes.
4. Validate JSON, internal navigation links and Git diff; confirm no source semantics changed.

No new regression suite, upstream workspace, training, evaluation, automatic
Stage-2 or checkpoint promotion in this phase. Duplicate-looking files are not
deleted without content and caller checks. See [inventory](INVENTORY.md).

## Risks remaining

- Checkpoint pointers are not checkpoint binaries; remote existence/hash checks
  are not implied by local evidence inspection.
- Some current evidence inputs are untracked and absent from the snapshot.
- Existing `--point` mode uses the legacy official-mapping configuration; it is
  not a universal current candidate loader, especially for joint 4K.
- Previously written FREEZE/MISSING labels are historical, not new manager decisions.
- Legacy HPC sweep submit/retry paths remain unusable until a separately reviewed
  adapter update; current canonical entry points are documented instead.
