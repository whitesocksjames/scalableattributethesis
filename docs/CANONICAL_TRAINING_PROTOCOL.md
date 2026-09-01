# Canonical Training Protocol

Status: thesis working protocol, 2026-09-01. This document separates settings
supported by current evidence from provisional stage-gated choices.

## Scope

The canonical scalable Attribute codec is:

```text
x_low + r1+r2+r3+r4 -> frozen Unicorn prefix -> learned Base synthesis -> Base
Base + independent EnhancementVAE bitstream                         -> Full
```

Base is independently decodable. Base physical bits are exactly `x_low` plus
the four native residual streams. BaseSynthesis transmits no bits. Full adds an
separate entropy-coded Enhancement payload conditioned on decoded Base; native
`r5` is not transmitted. The Enhancement payload is not independently
decodable: its entropy decoding and reconstruction require the decoded Base
state and the matching fixed operating-point model/configuration.

## Official operating-point mapping

Do not infer a checkpoint from its filename alone and do not invent a new
lambda/checkpoint pairing.

| Point | Released checkpoint family | lambda |
|---|---|---:|
| R01 | 32k8k | 32768 |
| R02 | 32k8k | 16384 |
| R03 | 32k8k | 8192 |
| R04 | 8k256 | 4096 |
| R05 | 8k256 | 2048 |
| R06 | 2k128 | 1024 |
| R07 | 2k128 | 512 |
| R08 | 2k128 | 256 |
| R09 | 2k128 | 128 |

The current five-point canonical Base study uses R01-R05 (32K, 16K, 8K,
4K, 2K). Every run must record the explicit checkpoint path and lambda.

R06-R08 (1K/512/256) begin with `role=diagnostic` and no selected Base. Their
five-checkpoint Base trajectories may produce a machine-readable `candidate`
after RWTT-28Lite selection and 8i sanity. Candidate status permits a later
review for Full28 or Enhancement Stage 1; it is not an automatic promotion to
`canonical_selected` and does not add the point to the formal curve.

Current evidence-snapshot selections are:

| Rate label | Point | Selected Base checkpoint |
|---|---|---|
| 32K | R01 | existing D111 `step5525` lineage |
| 16K | R02 | clean continuous `step3525` |
| 8K | R03 | clean continuous `step3525` |
| 4K | R04 | clean continuous `step2000` |
| 2K | R05 | clean continuous `step3525` |

The existing 32K `step5525` identifier is a cumulative global step from the
earlier `0 -> 500 -> 2000` exploration plus the final approximately one-pass
continuation. It is a selected artifact, **not** evidence that a fresh Base
should be trained for 5525 updates. New comparable Base runs use a clean,
continuous 3525-update budget unless a reviewed experiment says otherwise.

## Dataset and evaluation contract

- Training: RWTT `model_95_5_seed0` training split, model-level split.
- Development selection: frozen RWTT-28Lite contract.
- External sanity: fixed 8iVFB sequences/frames under the existing protocol.
- Formal retention/evaluation: RWTT Full28, 28 original models / 792 H5.
- Physical rate: actual codec bytes divided by full-resolution point count.
- Formal quality: author `pc_error` Y/U/V and YUV-PSNR 6:1:1.
- Full28 aggregation: per-H5 distortion, point-weighted per-model distortion,
  per-model PSNR, then model-equal average. Do not average per-H5 PSNR.
- Tensor-domain normalized YUV distortion is a training diagnostic and must not
  be mixed with formal `pc_error` results on one RD curve.

## Base training

The evidence-backed canonical Base recipe is:

```text
released Unicorn prefix       frozen, eval mode
native Upscaler/fuseNet/outNet frozen, with autograd through them
BaseSynthesis                 trainable, clean zero initialization
input                         x4 + f4
objective                     D111 = mean channel MSE
optimizer                     Adam, betas=(0.9, 0.999), weight_decay=0
learning rate                 3e-4, constant
batch size                    4
seed                          0
continuous budget             3525 updates (about one RWTT pass)
```

Recommended checkpoints are `500/1000/2000/3000/3525`. The final checkpoint
is not automatically the winner. Selection order is:

1. RWTT-28Lite trajectory;
2. 8iVFB sanity for the selected candidate;
3. RWTT Full28 only for the selected formal candidate.

The Base D611 ablation is evidence, not the default recipe: it did not justify
replacing D111 across the current evaluation evidence.

## Enhancement training: provisional stage-gated protocol

This is deliberately **not** declared an immutable per-lambda final budget.

### Stage 1

```text
Prefix and selected Base      frozen
EnhancementVAE                independent, initialized from released VAE
training data                 RWTT only
objective                     R_E + lambda_E * D111
conditioning lambda           official operating-point lambda
rd lambda                     same operating-point lambda unless reviewed
optimizer                     fresh Adam, betas=(0.9, 0.999), weight_decay=0
learning rate                 5e-5, constant
batch size                    4
budget                        1763 updates
```

At step 1763, run the frozen RWTT-28Lite stage gate. If its physical RD point
is already reasonable, early stop is allowed and preferred.

### Stage 2, only after review

If Stage 1 is insufficient, resume the **same Enhancement model and Adam
state** and change only the distortion weighting to:

```text
D611 = (6*D_Y + D_U + D_V) / 8
additional budget <= 1762 updates
maximum cumulative step = 3525
```

Stage 2 is not automatic. It must not reload released VAE weights, create a
fresh optimizer, alter architecture, change lambda, or add MVUB/full-unfreeze.
Save intermediate states so selection uses the physical RD trajectory rather
than assuming the last checkpoint is best.

### Data-order continuation contract

The data order is a function of global step, not of process launch or stage:

```text
steps_per_epoch = ceil(num_train_h5 / batch_size)
epoch           = global_step // steps_per_epoch
batch_in_epoch  = global_step %  steps_per_epoch
permutation     = randperm(num_train_h5, seed + epoch)
```

`drop_last=False`; therefore the last batch of an epoch may contain fewer than
four H5 samples. A Stage-2 resume at step 1763 consumes the batch assigned to
global step 1763, rather than generating a new Stage-2 shuffle. With 14098 H5
and batch size 4, Stage 1 consumes 1763 of 3525 batches and Stage 2 consumes the
remaining 1762 batches, completing exactly one scheduled pass.

The checkpoint stores the schedule policy, seed, manifest path, ordered
manifest-content fingerprint, resolved data root, sample count, batch size and
steps per epoch. Resume fails if these fields differ or if the checkpoint
predates this explicit schedule. This protects scientific continuation without
adding sampler-state or RNG-state recovery machinery.
Uniform-noise quantization remains unchanged; this contract governs dataset
order only.

The matched-budget 32K result supports this default two-stage procedure, but
does not prove that every lower-rate point needs or optimally uses all 1762
Stage-2 updates.

### Operating-point CLI

Canonical Enhancement runs select the official mapping and selected Base via:

```text
--point {32k,16k,8k,4k,2k}
--released-root <released-checkpoint-root>
--canonical-experiment-root <canonical-experiments-root>
```

The machine-readable mapping and provisional stage defaults live only in
`configs/scalable_attribute/canonical_operating_points.json`. Stage 1 is the
default. Stage 2 requires both `--enhancement-stage 2` and an explicit
`--resume-checkpoint`; no evaluation or training script launches it
automatically.

## Checkpoint and correctness requirements

Every run records resolved arguments, command, Git commit, dataset split,
initialization checkpoint, optimizer settings and global step. Resume must
restore model weights, Adam state and global step; it must never silently
restart.

Required hard gates:

- exactly four native residual streams for Base and no native `r5` stream;
- `Base_bits = bits_xlow + bits_r1 + bits_r2 + bits_r3 + bits_r4`;
- `Full_bits = Base_bits + Enhancement_bits`;
- hard prefix decode completes and reconstructs Base from decoder-known state;
- Enhancement hard encode/decode is exact under the existing hard contract;
- Prefix/Base remain frozen during Enhancement-only training;
- finite loss, rate, distortion and gradients.

Soft-vs-hard Base difference is a diagnostic, not an encode/decode round-trip
exactness claim.

## Explicitly unresolved

- The optimal lower-rate Stage-1/Stage-2 stopping step is not yet established.
- A single universal lower-rate Enhancement budget has not been demonstrated.
- MVUB mixing, full-unfreeze and random Base/Full training are not part of the
  current canonical recipe.
- Final multi-rate Enhancement packaging is deferred until the lower-rate
  stage gates are reviewed.
