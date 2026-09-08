# Final readability boundary review

This review closes the code-structure phase after Batches 1-5. It does not
change architecture, checkpoint tensor keys, rate conventions, CLI behavior,
or selected experiment semantics.

## Recommended final tree

```text
scalable_attribute/
├── models/
│   ├── config.py          BaseSynthesis architecture values
│   ├── prefix.py          released Unicorn r1-r4 and decoder-known transition
│   ├── base_synthesis.py  learned feature compensation c_B
│   ├── base.py            full-resolution Base composition
│   ├── enhancement.py     independent ResidualVAE wrapper
│   └── scalable.py        Base/Full composition and hard endpoint contract
├── training/
│   ├── data_schedule.py   reproducible continuation order
│   ├── joint_endpoint.py  random single-endpoint objective
│   └── base_rescue.py     retained rescue sampler/objective support
├── evaluation/
│   ├── __init__.py        shared identity and formal aggregation primitives
│   └── base_validation.py tensor-domain Base validation
└── runtime/
    ├── checkpoints.py     strict Base/full-state restoration
    └── operating_points.py released-point/config resolution
```

This is already close to the minimum useful granularity. No additional
`training/scopes.py`, `runtime/cli.py`, metrics package, or one-class-per-file
expansion is recommended.

## Files retained separately

| File | Decision and reason |
| --- | --- |
| `models/prefix.py` | Retain. It owns the most important scalable boundary: exactly r1-r4 plus decoder-known `x5p/f5p/d5p`. |
| `models/base_synthesis.py` | Retain. `c_B` is the thesis-added learned module and should be independently visible to a reader. |
| `models/base.py` | Retain. It explains how Prefix state and `c_B` become a full-resolution Base. Merging it with BaseSynthesis would obscure the distinction between learned module and endpoint composition. |
| `models/enhancement.py` | Retain. Independence from native r5 and released initialization are standalone thesis claims. |
| `models/scalable.py` | Retain. It is the shortest complete reading path for Base/Full composition, hard reconstruction, and bit identities. |
| `models/config.py` | Retain for checkpoint/config and import compatibility. Its CLI helper is a small known boundary imperfection; creating another file solely for that helper would increase navigation cost. |
| `training/*.py` | Retain all three. Continuation order, joint endpoint optimization, and historical Base-rescue sampling are distinct experiment concepts. |
| `evaluation/base_validation.py` | Retain. It is tensor-domain training validation; merging it into formal aggregation primitives in `evaluation/__init__.py` would mix evaluation contracts. |
| `runtime/*.py` | Retain both. Strict state restoration and operating-point resolution have separate failure modes and callers. |

The only safe consolidation performed in this review removes the duplicate
`_complete_state_trainable()` implementation. Both deterministic/hard callers
already execute under `torch.no_grad()`, so the shared `_complete_state()` keeps
the differentiable training path without changing deterministic behavior.

## Training-policy boundary

| Symbol | Decision | Rationale |
| --- | --- | --- |
| `CanonicalBaseModel.train()` | Keep in `models/base.py` | PyTorch lifecycle invariant: a frozen Prefix must remain in eval mode when its parent enters train mode. |
| `FrozenUnicornPrefix.train()` | Keep in `models/prefix.py` | Same lifecycle invariant, including the author model's quantization/eval behavior. |
| `forward_trainable()` and `forward_base_training()` | Keep in `models/base.py` | These define differentiable graph paths and returned Prefix likelihood/state contracts. Reimplementing them in a trainer would require reaching through model internals. |
| `forward_base_train()` and `forward_full_train()` | Keep in `models/scalable.py` | They are endpoint graph definitions. In particular, Base must not invoke Enhancement, while Full must connect the complete graph. |
| `set_trainable_scope()` and scope names | Keep for now | The scope choice is training policy, but applying it safely also coordinates `requires_grad`, nested module mode, frozen Prefix behavior, and embedding gradients. Moving only the names would add indirection; moving the behavior risks semantic drift and breaks recorded callers. Training scripts remain responsible for selecting the scope. |
| `freeze()` / `set_trainable()` | Keep as compatibility lifecycle APIs | Strict loaders and current scripts use them to establish model invariants. They do not add checkpoint state. |
| `TRAINABLE_SCOPES` | Keep beside `CanonicalScalableModel` | It validates that model's two supported lifecycle states. A separate policy module for one tuple is not warranted. |

This is an intentional pragmatic boundary: experiment code chooses a scope;
the model owns the safe state transition and differentiable endpoint graph.

## Compatibility impact

- Module attribute names and state-dict keys are unchanged.
- Architecture strings and strict checkpoint validation are unchanged.
- Existing `canonical/` Python and CLI compatibility layers remain intact.
- Training/evaluation arguments, defaults, output layout, and rate/metric
  conventions are unchanged.
- No result, draft, checkpoint, dataset, or historical evidence is moved.

## Regression result

The isolated consolidation commit `6c3fcbf` was tested on N30 before this
review was finalized:

| Gate | Result |
| --- | --- |
| Fast contract suite | PASS: 12 tests passed; 4 explicitly requested GPU/slow tests skipped |
| Selected 4K joint `step3000` lightweight reconstruction | PASS (`5448309`) |
| Selected 8K Base `step3525` + Enhancement `step1763` lightweight reconstruction | PASS (`5448310`) |
| Local source-contract tests | PASS: 5/5 |
| Python compile and `git diff --check` | PASS |

The cluster outputs are under
`/data/run01/scz0ade/Tanzeyu/experiments/readability_final_6c3fcbf/`.
No training, physical formal evaluation, or slow 2K hard-code integration was
run for this private-helper consolidation.

## Remaining readability debt

1. `add_base_architecture_arguments()` and `BaseSynthesisConfig.from_args()`
   couple the architecture config to argparse-shaped objects. Moving two small
   helpers would require either a new thin module or duplicated CLI code, so it
   is deferred unless future CLI growth gives that module independent value.
2. Scope names encode experiment history (`base_synthesis_only`, `base_path`,
   `enhancement_only`, `full`). They remain public compatibility vocabulary.
   Renaming is not justified before a publication-facing API is designed.
3. `evaluation/__init__.py` contains reusable formal aggregation functions.
   A future larger evaluation package may justify a named `metrics.py`, but the
   current size does not.
4. Experiment/result/figure organization is explicitly outside this review and
   remains the next separately designed repository concern.
