# Readability Phase 2 proposal

Status: proposal only. No source file has been moved or modified by this
proposal.

Baseline:

- branch base: `f77a5dca225b64a423ddd4496334807bee8caea9`
- 2K rescued Base plus independent Enhancement slow hard gate: PASS
- 4K joint step3000 lightweight reconstruction gate: PASS
- 8K standard Base plus independent Enhancement lightweight reconstruction
  gate: PASS

The refactor must preserve architecture behavior, state-dict tensor names,
checkpoint metadata, coding/rate conventions, selected weights and experiment
results.

## 1. Proposed directory tree

```text
scalable_attribute/
├── models/
│   ├── __init__.py
│   ├── config.py             # BaseSynthesisConfig only
│   ├── prefix.py             # r1-r4 decoded Prefix state
│   ├── base_synthesis.py     # learned feature compensation c_B
│   ├── base.py               # full-resolution Base reconstruction
│   ├── enhancement.py        # independent EnhancementVAE
│   └── scalable.py           # Base/Full model composition and forward paths
├── training/
│   ├── __init__.py
│   ├── data_schedule.py      # reproducible continuation sampler
│   ├── joint_endpoint.py     # random single-endpoint objective
│   └── base_rescue.py        # rescue sampler/objective support
├── evaluation/
│   ├── __init__.py
│   └── base_validation.py    # tensor Base validation/aggregation
├── runtime/
│   ├── __init__.py
│   ├── checkpoints.py        # strict Base and complete scalable loaders
│   └── operating_points.py   # legacy runtime operating-point resolver
└── canonical/                # compatibility import surface during transition
    ├── __init__.py
    ├── config.py
    ├── prefix.py
    ├── base_synthesis.py
    ├── model.py
    ├── enhancement.py
    ├── scalable_model.py
    ├── data_schedule.py
    ├── joint_endpoint.py
    ├── base_rescue.py
    ├── evaluation.py
    └── operating_points.py

scripts/scalable_attribute/
├── training/                 # current supported training CLIs
├── evaluation/               # current supported evaluation CLIs
├── diagnostics/              # analysis/probe/summarization CLIs
├── historical/               # reproducibility-only experiment CLIs
└── canonical/                # compatibility wrappers during transition
```

The first useful milestone is the library split under `scalable_attribute/`.
Script relocation is deliberately last: recorded commands and Slurm launchers
refer to current script paths, so moving scripts has more compatibility cost
than moving internal modules behind re-export shims.

## 2. Current file to proposed location

### Architecture

| Current file | Proposed implementation file | Responsibility |
| --- | --- | --- |
| `canonical/config.py` | `models/config.py` | Immutable BaseSynthesis architecture configuration. The compatibility CLI helper remains temporarily and moves to `runtime/cli.py` in a later reviewed batch. |
| `canonical/prefix.py` | `models/prefix.py` | `PrefixState`, released Unicorn r1-r4 traversal, decoded native transition to x5p/f5p/d5p, synthesis access and lambda embedding. |
| `canonical/base_synthesis.py` | `models/base_synthesis.py` | x4/f4 to feature compensation c_B. No rate, training loop or metrics. |
| `canonical/model.py` | `models/base.py` | `CanonicalBaseModel`, Base reconstruction and native Base baselines. |
| `canonical/enhancement.py` | `models/enhancement.py` | Independent ResidualVAE wrapper and deterministic/encode/decode interfaces. |
| Architecture part of `canonical/scalable_model.py` | `models/scalable.py` | `CanonicalScalableModel` and Base/Full forward composition. Existing trainable-scope methods remain attached during Batch 1/2 for move-only equivalence. |

The model directory should read in dataflow order:

```text
Prefix -> BaseSynthesis -> Base -> Enhancement -> Scalable Base/Full model
```

Two known boundaries are intentionally deferred rather than mixed into the
move-only batches:

- `add_base_architecture_arguments()` is CLI/runtime support. Its final home is
  `runtime/cli.py`; `models/config.py` ultimately contains architecture config
  only.
- `CanonicalScalableModel.set_trainable_scope()` and related freeze/unfreeze
  behavior are training policy. Batch 1/2 retain their methods and public API;
  a later review may extract their implementation to `training/scopes.py`.

### Checkpoint and runtime support

| Current file/symbol | Proposed location | Responsibility |
| --- | --- | --- |
| `load_frozen_base()` from `canonical/scalable_model.py` | `runtime/checkpoints.py` | Strict loading of standard BaseSynthesis-only and rescued complete-Base formats. |
| `load_finetuned_scalable()` and `FINE_TUNE_ARCHITECTURE` | `runtime/checkpoints.py` | Strict loading and architecture validation of complete joint scalable state. |
| `canonical/operating_points.py` | `runtime/operating_points.py` | Existing legacy runtime resolver only; it does not replace `current_candidates.json`. |
| `configs/scalable_attribute/current_candidates.json` | unchanged | Human/machine-readable current candidate provenance registry. |
| `configs/scalable_attribute/canonical_operating_points.json` | unchanged | Existing CLI compatibility mapping, explicitly marked non-authoritative for current rescued/joint selection. |

### Training support

| Current file | Proposed location | Responsibility |
| --- | --- | --- |
| `canonical/data_schedule.py` | `training/data_schedule.py` | Ordered-manifest fingerprint, continuation sampler and resume compatibility. |
| `canonical/joint_endpoint.py` | `training/joint_endpoint.py` | Endpoint sampling, estimated-rate/distortion calculation and TAFA-style single-endpoint objective. |
| `canonical/base_rescue.py` | `training/base_rescue.py` | Difficulty score loading, deterministic weighted sampler and Base rescue objective. |

Current supported training entry points move only in the final script batch:

| Current CLI | Proposed CLI implementation |
| --- | --- |
| `scripts/.../canonical/train_base.py` | `scripts/.../training/train_base.py` |
| `scripts/.../canonical/train_enhancement.py` | `scripts/.../training/train_enhancement.py` |
| `scripts/.../canonical/train_joint_endpoint.py` | `scripts/.../training/train_joint_endpoint.py` |

Reproduction-only `train_base_rescue.py`, `train_mvub_finetune.py` and
`train_enhancement_mixed.py` belong under `scripts/.../historical/`, not the
current training surface. Their current paths remain wrappers while retained
commands depend on them.

### Evaluation support

| Current file | Proposed location | Responsibility |
| --- | --- | --- |
| `scalable_attribute/evaluation.py` | `evaluation/__init__.py` | Preserve the existing public PSNR, sample-identity and model-aggregation import surface while converting the module into a package. |
| `canonical/evaluation.py` | `evaluation/base_validation.py` | Tensor distortion and Base validation aggregation. |
| `scripts/.../canonical/evaluate_scalable_formal.py` | `scripts/.../evaluation/evaluate_scalable_formal.py` | H5 physical Base/Full evaluation. |
| `scripts/.../canonical/evaluate_8ivfb_sequence.py` | `scripts/.../evaluation/evaluate_external_sequence.py` | Existing 8i/Owlii prepared-input physical evaluation. The rename is deferred until callers are audited. |
| `scripts/.../canonical/evaluate_base_formal.py` | `scripts/.../evaluation/evaluate_base_formal.py` | Canonical Base physical evaluation. |
| `scripts/.../canonical/evaluate_base.py` | `scripts/.../evaluation/evaluate_base_tensor.py` | Tensor-only Base validation. |

`evaluate_base_rescue.py` and `evaluate_base_rescue_arm.py` remain historical
reproduction tools. `analyze_*`, `audit_*`, `build_*`, `plot_*`, `probe_*`,
`screen_*` and `summarize_*` belong under `scripts/.../diagnostics/`; they are
not model architecture.

## 3. Dependency direction

The intended dependency graph is one-way:

```text
upstream Unicorn implementation
        ↓
scalable_attribute.models
        ↓
scalable_attribute.runtime
        ↓
scalable_attribute.training / scalable_attribute.evaluation
        ↓
scripts
```

More precisely:

- `models` may import PyTorch, MinkowskiEngine and unchanged upstream modules.
- `models` must not import `training`, `evaluation`, scripts, manifests or
  experiment paths.
- `runtime` may import `models` and PyTorch for strict state restoration.
- `training` may import `models` and `runtime`; it must not import formal
  evaluation CLIs.
- `evaluation` may import `models`, `runtime` and metric/data utilities; it must
  not import training loops or optimizer code.
- scripts parse CLI/path configuration and call the appropriate package layer.

Duplicated `_same_support()` helpers should not be consolidated in the first
move batch. Consolidation is behavior-bearing cleanup and should occur only
after move-only equivalence is established.

## 4. Checkpoint compatibility

No checkpoint will be rewritten or resaved.

Compatibility requirements:

1. Keep module attribute names (`base`, `prefix`, `base_synthesis`,
   `enhancement`, `vae`) unchanged so state-dict keys remain byte-for-byte
   identical.
2. Keep architecture strings unchanged, including
   `canonical_base_predict_correct`, `canonical_base_rescue_v1`,
   `canonical_independent_enhancement` and
   `canonical_scalable_mvub_finetune_v1`.
3. Move loader functions without changing accepted metadata, real-path checks,
   lambda checks or `strict=True` behavior.
4. Keep compatibility modules at every old `scalable_attribute.canonical.*`
   path. They re-export the new implementation symbols and contain no duplicate
   implementation.
5. Compare pre/post `state_dict().keys()` for standard 8K, rescued 2K and joint
   4K loaders. Missing and unexpected keys remain fatal.
6. Existing dictionary/state-dict checkpoints do not depend on Python class
   pickle locations. Compatibility shims still protect historical imports and
   any unreviewed whole-object artifacts.

## 5. Public API and CLI compatibility

New preferred imports:

```python
from scalable_attribute.models import (
    BaseSynthesis, CanonicalBaseModel, CanonicalScalableModel,
    EnhancementVAE, FrozenUnicornPrefix, PrefixState)
from scalable_attribute.runtime.checkpoints import (
    load_finetuned_scalable, load_frozen_base)
```

Existing imports continue to work:

```python
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_finetuned_scalable, load_frozen_base)
```

Old modules should initially be silent re-export shims, not emit deprecation
warnings into training logs. Removal is outside this phase.

`scalable_attribute/canonical/README.md` and its package documentation make the
compatibility-only role explicit. The preferred reading path is
`models/ -> training/evaluation/runtime`; no new implementation belongs in the
compatibility package.

Recorded CLI paths remain executable. If scripts are relocated in the final
batch, the old files become thin `main()` forwarding wrappers. Argument names,
defaults, output layouts and exit behavior remain unchanged. No scheduler
framework is introduced.

## 6. Upstream Unicorn files kept unchanged

The readability refactor does not move or edit:

```text
lossy_attribute/
basic_models/
data_utils/
cfg/
third_party/
pipelines/
```

In particular, native `MultiscaleVAE`, `ResidualVAE`, entropy coding, pooling,
geometry support, G-PCC wrapper behavior and upstream bitrate convention remain
untouched. Necessary compatibility patches already documented by Phase 1 are
not expanded here.

## 7. Small execution batches and gates

### Batch 1 — model package, move-only

- Move config, Prefix, BaseSynthesis, Base and Enhancement implementations.
- Add old-path re-export shims.
- Do not yet split `scalable_model.py`.

Gates:

- existing fast CPU source/loader suite;
- import and `--help` smoke for current training/evaluation entry points;
- 8K standard lightweight GPU gate.

### Batch 2 — scalable composition and checkpoint loaders

- Move `CanonicalScalableModel` to `models/scalable.py`.
- Move strict loaders/constants to `runtime/checkpoints.py`.
- Leave `canonical/scalable_model.py` as a compatibility re-export.

Gates:

- complete fast CPU suite, including strict standard/rescue/joint loaders;
- 2K rescued and 4K joint loader contracts;
- 4K joint plus 8K standard lightweight GPU gates.

### Batch 3 — training support modules

- Move data schedule, joint objective and rescue support.
- Update internal imports; preserve old-path shims.

Gates:

- fast CPU suite;
- data-schedule/resume compatibility tests already present;
- training entry-point import and `--help` smoke.

No optimizer job or training is submitted.

### Batch 4 — evaluation and operating-point runtime modules

- Move tensor Base evaluation utility.
- Move legacy operating-point resolver to runtime.
- Update internal imports and retain shims.

Gates:

- fast CPU suite;
- evaluation entry-point import and `--help` smoke;
- no formal dataset evaluation.

### Batch 5 — script navigation, last and separately reviewed

- Relocate only confirmed current training/evaluation implementations.
- Convert recorded old CLI paths to forwarding wrappers.
- Classify diagnostic and historical scripts without deleting them.
- Update current entry-point documentation.

Gates:

- fast CPU suite;
- all current CLI import/`--help` smoke;
- 4K/8K lightweight GPU gates if imports changed.

### Final pre-refactor equivalence milestone

Run the complete fast suite and the existing 4K/8K lightweight gates. Run one
2K slow physical-code integration gate only once after all batches, not after
every move. The expected outcome is identical reconstruction/rate contracts;
performance benchmarking is not a refactor acceptance criterion.

### Final readability boundary review

After Batch 4-5 and their existing equivalence gates pass, review the remaining
training-policy coupling in `models/base.py` and `models/scalable.py` symbol by
symbol. In particular, classify `TRAINABLE_SCOPES`, `set_trainable_scope()`,
freeze/unfreeze parameter policy, and the `base_synthesis_only`, `base_path`,
`full`, and `enhancement_only` policies.

The goal is to keep architecture, model state, and necessary differentiable
forward interfaces in `models/`, while moving experiment-specific trainable
scope policy to `training/` where compatibility risk permits. PyTorch
`train()` overrides and model-owned differentiable forwards are not moved merely
for directory purity. This is a reviewed boundary decision after Batch 5, not a
change authorized by the move-only batches.

### Deferred experiment/result organization

The current `results/`, `drafts/`, plotting artifacts and historical evidence
remain in place throughout the code-structure refactor. After the readability
batches, a separately reviewed `Experiment / Result / Figure organization`
phase will distinguish:

- raw experiment outputs;
- aggregated result tables;
- analysis and plotting scripts;
- final thesis figures.

That later phase must support the independent Unicorn baseline, CTC evaluation,
BD-BR analysis and thesis RD figures. Nothing in these artifact trees is moved
or deleted by the current readability batches.

## 8. Risks and stop conditions

- Import cycles can appear if checkpoint loaders remain in the model module;
  this is why runtime extraction follows the initial model moves.
- Script relocation can invalidate retained Slurm commands; it is delayed and
  protected by wrappers.
- `canonical_operating_points.json` is stale for current 2K/4K selection. The
  refactor must not silently promote it over `current_candidates.json`.
- Joint 4K uses source profile/lambda `32k8k@8192` and effective target lambda
  4096. Any simplification to a fictional released 4K loader is a blocker.
- Rescued 2K requires complete Prefix plus BaseSynthesis restoration. Loading
  only BaseSynthesis is a blocker.
- Moving and consolidating logic in the same commit would obscure semantic
  drift. Each batch should be move/re-export first; cleanup follows only after
  equivalence.

Stop a batch if state-dict keys, strict-loader results, support/stride/channels,
Base/Full reconstruction or rate identities differ from the accepted baseline.
