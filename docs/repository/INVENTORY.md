# Repository inventory — Phase 1

> **Historical inventory with current routing addendum.** The file-level
> classifications below retain their 2026-09-07 meaning. For final thesis
> navigation use [`CURRENT_RESULT_INDEX.md`](CURRENT_RESULT_INDEX.md) and
> [`HISTORICAL_DOCUMENT_STATUS.md`](HISTORICAL_DOCUMENT_STATUS.md); the final
> static RGB package is `results/comparisons/formal_static_rgb_final_20260911/`.

Physical cleanup update (2026-09-07): six non-current upstream mode directories
now live under `archive/upstream_out_of_scope/`; two root ZIPs are retained under
`archive/packages/`. See [exact relocation/recovery table](../../archive/README.md).
The CSV remains the accepted pre-cleanup inventory, not a post-move path listing.
Current model and result paths have not moved.

Inventory date: 2026-09-07. This is a read-only navigation inventory for the
Phase 1 rationalization. It does not authorize deletion, moving, renaming,
rewriting, or retraining. The machine-readable companion is
[`inventory.csv`](inventory.csv).

## Classification vocabulary

`ACTIVE_CORE`, `ACTIVE_TRAINING`, `ACTIVE_EVALUATION`, `ACTIVE_CONFIG`, and
`ACTIVE_RESULT` describe the current role. `UPSTREAM_UNICORN_DEPENDENCY`
identifies source, data, metric, or codec components retained because the
current scalable Attribute code imports or wraps them. `LEGACY_EXPERIMENT`,
`LEGACY_DIAGNOSTIC`, and `HISTORICAL_EVIDENCE` identify
old experiments, diagnostics, reports, and source evidence that remain useful
but are not the current recipe. `GENERATED_ARTIFACT` identifies plots, raw
codec output, caches, and packaged binaries. `BROKEN_ENTRYPOINT` identifies a
path whose current caller resolves to a missing target. `DUPLICATE` is used only
where a full-file SHA-256 match was checked. `UNKNOWN` means that the available
local evidence is insufficient for a stronger claim.

Multiple classifications in the CSV are semicolon-separated. A directory row
is intentionally a scope summary; use key-file rows and the linked reports for
the exact role of individual contents.

## Navigation summary

| Scope | Classification | Evidence / current route |
| --- | --- | --- |
| [`scalable_attribute/`](../../scalable_attribute/) | `ACTIVE_CORE` | Canonical model, frozen Unicorn prefix, Base/Full assembly, data and aggregation helpers. See [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md). |
| [`scripts/scalable_attribute/canonical/`](../../scripts/scalable_attribute/canonical/) | `ACTIVE_CORE;ACTIVE_TRAINING;ACTIVE_EVALUATION;ACTIVE_RESULT` | Current Base, Enhancement, joint, rescue, formal evaluation, aggregation, and plotting entrypoints. See [`CURRENT_TRAINING_ENTRYPOINTS.md`](CURRENT_TRAINING_ENTRYPOINTS.md) and [`CURRENT_EVALUATION_ENTRYPOINTS.md`](CURRENT_EVALUATION_ENTRYPOINTS.md). |
| [`scripts/data/`](../../scripts/data/) and [`data_splits/`](../../data_splits/) | `ACTIVE_TRAINING;ACTIVE_CONFIG` | Dataset preparation adapters and model-level manifests. Raw H5/PLY data is outside this bounded source scan. |
| [`configs/`](../../configs/) | `ACTIVE_CONFIG` | Operating-point and rescue configuration. The legacy official mapping remains part of the resolver contract; it is not a complete current-candidate registry. |
| [`scripts/hpc/`](../../scripts/hpc/) | `ACTIVE_CONFIG;BROKEN_ENTRYPOINT` | Reference submission path is present; generic train/eval sweep path is historical and broken because its spec and CLI targets are absent. |
| [`docs/`](../../docs/) | `ACTIVE_CORE;ACTIVE_CONFIG` | Protocol and this navigation/inventory layer. |
| [`drafts/`](../../drafts/) | `LEGACY_EXPERIMENT;LEGACY_DIAGNOSTIC;HISTORICAL_EVIDENCE;GENERATED_ARTIFACT` | Dated reports, raw evidence, tables, plots, and reproduction inputs. Preserve paths; several directories/files are untracked. |
| [`results/`](../../results/) | `ACTIVE_RESULT;HISTORICAL_EVIDENCE;GENERATED_ARTIFACT` | Current candidate tables/plots and older PCAC/PCGC/reference outputs. See [`CURRENT_RESULT_INDEX.md`](CURRENT_RESULT_INDEX.md). |
| [`basic_models/`](../../basic_models/), [`lossy_attribute/`](../../lossy_attribute/), [`data_utils/`](../../data_utils/), [`cfg/`](../../cfg/), [`third_party/`](../../third_party/) | `UPSTREAM_UNICORN_DEPENDENCY` | Current canonical code imports native backbones, released lossy Attribute modules, data readers, CLI configuration, metrics, and G-PCC/PCError tools. |
| [`lossless_attribute/`](../../archive/upstream_out_of_scope/lossless_attribute/), [`dynamic_attribute/`](../../archive/upstream_out_of_scope/dynamic_attribute/), [`lossless_geometry/`](../../archive/upstream_out_of_scope/lossless_geometry/), [`lossy_geometry/`](../../archive/upstream_out_of_scope/lossy_geometry/), [`dynamic_geometry/`](../../archive/upstream_out_of_scope/dynamic_geometry/), [`dynamic_geometry_lidar/`](../../archive/upstream_out_of_scope/dynamic_geometry_lidar/), [`pipelines/`](../../pipelines/) | `UPSTREAM_UNICORN_DEPENDENCY;HISTORICAL_EVIDENCE` | Original Unicorn source families retained for compatibility/recovery. Geometry and non-scalable modes are out of current thesis scope. |
| [`figures/`](../../figures/) | `ACTIVE_RESULT;GENERATED_ARTIFACT;HISTORICAL_EVIDENCE` | Curated figures with source CSV/provenance; not a runtime entrypoint. |
| [`references/`](../../references/) | `ACTIVE_CONFIG;HISTORICAL_EVIDENCE` | Read-only thesis task and Unicorn Part II references. |

## Exact broken-entrypoint evidence

The following are stale submission paths, not deletion candidates:

- [`scripts/hpc/submit_train_sweep.py`](../../scripts/hpc/submit_train_sweep.py:46)
  constructs `scripts/scalable_attribute/train.py`; that file is missing.
- [`scripts/hpc/submit_eval_sweep.py`](../../scripts/hpc/submit_eval_sweep.py:92)
  constructs `scripts/scalable_attribute/evaluate.py`; that file is missing.
- [`scripts/hpc/remote_submit.py`](../../scripts/hpc/remote_submit.py:24) defaults
  to `scripts/hpc/sweep_spec.py`; that file is missing. Its train/eval actions
  dispatch to the two stale wrappers above.
- [`HPC_GUIDE.md`](../../HPC_GUIDE.md:75) still tells a reader to edit that
  absent `sweep_spec.py`. [`HPC.md`](../../HPC.md:4) now labels the block as
  historical/broken, while retaining the scheduler policy as evidence.

The working routes are the explicit canonical scripts linked from the current
training/evaluation pages, and the present reference route
[`scripts/hpc/submit_reference_sweep.py`](../../scripts/hpc/submit_reference_sweep.py)
plus [`scripts/hpc/reference_sweep_spec.py`](../../scripts/hpc/reference_sweep_spec.py).
No replacement generic sweep adapter is inferred in this phase.

## Exact duplicate policy and findings

Names, matching stems, and copied outputs are not enough to call a duplicate.
The following full-file SHA-256 matches were observed:

- `0766c54897e209cd76ffa6d83ef8c9ce6714faf9a5267197a959fda865d64a1a`:
  `dynamic_attribute/README.md`, `lossless_attribute/README.md`,
  `lossy_attribute/README.md`.
- `d4d526d98debd39041c99d3448567287a3866bdc50b0a1538521edac3277cc28`:
  the seven package marker files `basic_models/__init__.py`, `cfg/__init__.py`,
  `data_utils/__init__.py`, `data_utils/attribute/__init__.py`,
  `data_utils/geometry/__init__.py`, `pipelines/__init__.py`, and
  `third_party/__init__.py`.

These are recorded as `DUPLICATE` rows in the CSV for auditability; no action is
proposed. Many generated codec payloads also hash identically across runs, but
that is not by itself evidence that an experiment directory is redundant, so
those rows remain `GENERATED_ARTIFACT;HISTORICAL_EVIDENCE` or `UNKNOWN`.

## Coverage, exclusions, and missing raw evidence

- Directory scope covers the requested major directories and the upstream
  source directories listed above. Key-file scope covers the canonical model,
  current training/evaluation/aggregation paths, configs/manifests, stale HPC
  wrappers, representative reports/results, and exact duplicate groups.
- This bounded inventory did not exhaustively scan local training datasets,
  released checkpoints, HPC experiment roots, Slurm logs, or external-workspace
  state. `.gitignore` explicitly excludes `/data/`, `/checkpoints/`,
  `/experiments/`, `*.h5`, and `*.pth`; those namespaces are outside this
  inventory. Ignored raw PLY, codec binaries, and logs may still exist.
- Present local raw evidence under dated `drafts/*/raw*` and
  `results/*/raw/` is included in directory-level classification, but several
  such paths are untracked in the current worktree. Git history alone is not a
  recovery guarantee for them; do not delete or move them in Phase 1.
- Generated caches (`__pycache__/`), root archives (`results.zip` and
  `scalable-geometry-roi-channel_split_lambda.zip`), `.git/`, `.agents/`,
  `.codex/`, and editor metadata are excluded from detailed file-by-file
  coverage. Caches/archives are classified as generated or unknown where
  represented, not treated as cleanup authorization.
- No raw-data provenance, checkpoint hash, or remote path is asserted unless a
  local file provides it. The CSV uses `UNKNOWN` where ownership, callers, or
  recoverability cannot be established from this checkout.

## No physical action

This inventory only adds navigation evidence. It does not delete, move,
rename, overwrite, or modify any source/model/training artifact.
