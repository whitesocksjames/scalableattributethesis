# Current training entry points

This page locates reproduction code; the thesis architecture/checkpoint selection
and formal experiment matrix are frozen. It does not authorize a new run or
change a recipe. Use [frozen operating points](CURRENT_OPERATING_POINTS.md) for
initialization evidence.

| Purpose | Entry point | Role |
| --- | --- | --- |
| Canonical BaseSynthesis training | [train_base.py](../../scripts/scalable_attribute/training/train_base.py) | Active |
| Frozen Base + independent Enhancement | [train_enhancement.py](../../scripts/scalable_attribute/training/train_enhancement.py) | Active, including rescued Base compatibility |
| Joint random Base/Full endpoint | [train_joint_endpoint.py](../../scripts/scalable_attribute/training/train_joint_endpoint.py) | Active for joint lineage; inspect exact initialization contract |
| Base rescue scopes/samplers | [train_base_rescue.py](../../scripts/scalable_attribute/historical/train_base_rescue.py) | Historical experiment entry, still needed for reproduction |
| MVUB-only/mixed adaptation | [train_mvub_finetune.py](../../scripts/scalable_attribute/historical/train_mvub_finetune.py), [train_enhancement_mixed.py](../../scripts/scalable_attribute/historical/train_enhancement_mixed.py) | Historical experiments; reusable code is not deletion authority |

`--point` reads the legacy CLI recipe. It is not formal checkpoint authority and
does not safely resolve the frozen rescued/joint set. Reproduction must use the
frozen checkpoint manifest and explicit options recorded in retained commands. In particular,
4K joint target lambda=4096 with source family=32k8k is not official R04 (8k256).
Do not replace a full-state checkpoint by its initial Base checkpoint.

Source definitions: [data schedule](../../scalable_attribute/training/data_schedule.py),
[joint objective](../../scalable_attribute/training/joint_endpoint.py),
[rescue support](../../scalable_attribute/training/base_rescue.py).
Architecture, optimizer recipe and sampler history remain separate concerns.
No additional Stage-2 candidate selection belongs to the frozen formal matrix.

The old `scripts/scalable_attribute/canonical/` command paths remain thin
compatibility wrappers for retained commands. New invocations should use
`training/`; see the [script index](../../scripts/scalable_attribute/README.md).

Cluster wrappers: [N30 guide](../../N30_GUIDE.md), [HPC guide](../../HPC_GUIDE.md).
`scripts/hpc/submit_train_sweep.py` and `submit_eval_sweep.py` target deleted legacy
train/evaluate scripts. Their submission and retry workflows are not current
canonical launchers. No automatic replacement CLI is claimed by this phase.
