# Evidence Map for Scalable Attribute Design

Evidence is organized after defining the candidate space. A PASS here supports
only the stated claim, not an architecture commitment.

## 1. Source-confirmed evidence

| Evidence | Status | Supports | Does not prove |
|---|---|---|---|
| Five ordered arithmetic streams follow `x_low` | confirmed | Native prefix/suffix boundaries exist | Any particular cut has good RD |
| Each entropy prior uses decoder-known `x_in/f_in/prior_dec` | confirmed | No hyperprior bits are needed; suffix is sequentially conditioned | Later streams can decode without their prefix |
| All five RWTT calls share one ResidualVAE and one Upscaler | confirmed | Checkpoint parameters can seed stage-specialized copies | Unsharing improves compression |
| `stage=3` has three VAE/Upscaler modules over 15 anisotropic transitions | confirmed | Multiple learned transition modules are an implementation precedent | Partial specialization of the released five isotropic stages |
| Backbone supports scale 0, positive and negative transforms | confirmed | Native-like continuation/refinement modules are reusable | Any scale is appropriate for EL residuals |
| AQL EQ/DQ conditions all stages on lambda | confirmed | Variable-rate state transformation is native | AQL is necessary for a new EL |
| Conditional entropy stream is monolithic torchac bytes | confirmed | Physical native stream bits are real arithmetic bytes | The stream is progressively decodable internally |
| File-level Attribute encode/decode APIs are TODO | confirmed | In-memory experiments match repository reality | Metadata-free standalone codec exists |
| Public static Attribute training script is absent | confirmed gap | Evaluation/architecture semantics remain inspectable | Exact released training loss can be reconstructed |
| Released RWTT checkpoints have only top-level `model` | confirmed for all three profiles | Parameter tensors are recoverable | Args, optimizer, epoch, exact loss/`last_weight`, lambda sampling |
| Official `model.test()` passes in-memory `x_low` into decode | confirmed | G-PCC rate is counted | A physical G-PCC `x_low` decode path exists |

## DECODER_KNOWN_NO_BIT_OPTIONS

Before reading the current arithmetic payload, `loc/scale` depend only on the
decoded prefix, geometry support, fixed model/configuration and lambda. `loc`
is the conditional center in post-EQ, pre/at-quantization symbol space, not in
DQ feature space.

| Continuation | Encoder-only data | Extra bits | Domain | Codec status | Synthesis |
|---|---:|---:|---|---|---|
| `symbols=0` | none | none | integer EQ symbol | deterministic missing-stream rule, not an official public API | `DQ(0)=bias`, then native decoder/fusion/output |
| `symbols=round(loc)` | none | none | integer EQ symbol | decoder-known integer rule, not an official public API | DQ then native decoder/fusion/output |
| `latent=loc` | none | none | continuous EQ value | bypasses released integer arithmetic-symbol semantics | can be experimentally passed through DQ/synthesis |

These options make a contiguous omitted suffix deterministic. They do not make
an arbitrary subset valid: after an omitted interior stage, unchanged released
later bytes have a different conditional CDF and cannot generally be decoded as
the released Full suffix.

## 2. Native integrity evidence

| Artifact/result | Observation | Implication |
|---|---|---|
| Base interface probe | Normal deterministic and hard decode B/F_U agree; coordinates stride 1 | Current BaseAdapter is a trustworthy observation boundary |
| Hard decode | Decoder obtains reconstruction from Base payload without encoder-only state | Native hard-path comparisons are valid |
| Frozen Base checks | Released Base remains eval/frozen and receives no gradients | External or branched experiments can preserve Base |

## 3. Native state trace evidence

Artifact:

```text
$WORK/scalable_attribute_thesis/experiments/native_state_trace_20260820/results/
```

Observed on fixed RWT115/RWT541 R09 samples:

- Final `B`, `F_U`, `D_U` equal final `x_out`, `f_out`, `dec` exactly.
- All are full-resolution, coordinate-aligned decoder-known states.
- Upscaler preserves aggregate RMS approximately but redistributes channel
  magnitudes strongly (channel-RMS cosine roughly 0.68–0.75).
- `dec -> Unpool(dec)` preserves channel distributions almost exactly.
- Native E5 RMS was roughly three times `A-B`, while local variation magnitude
  was similar; `A-B` is relatively higher-frequency/localized.
- R09 Stage 4/5 symbols were all zero on these two samples, while DQ produced
  nonzero decoded states.

Implications:

- Decoder state reuse is structurally possible.
- Feature identity and residual-state identity do not have equal semantics.
- Fine residual enhancement should not assume aggressive spatial reduction is
  harmless.
- Two R09 samples cannot establish high-rate stream allocation.

## 4. R5-only falsification evidence

Artifact:

```text
$WORK/scalable_attribute_thesis/experiments/
  scalable_architecture_survey_20260820/r5_only/results/
```

Scope: R01/R02/R03, four fixed H5, released `32k8k` checkpoint.

| Point | Base-r5 bpp | Full bpp | r5 bpp | r5 share | Base PSNR | Full PSNR |
|---|---:|---:|---:|---:|---:|---:|
| R01 | 0.32360 | 1.03883 | 0.71524 | 68.85% | 38.9082 | 42.9020 |
| R02 | 0.23851 | 0.78226 | 0.54375 | 69.51% | 38.3644 | 41.6154 |
| R03 | 0.17353 | 0.56963 | 0.39610 | 69.54% | 37.6695 | 40.1976 |

All 12 Base reconstructions were full-resolution and read no r5 payload. Full
hard reconstruction matched the released hard path with maximum absolute
difference zero. This is evidence that one native suffix cut is structurally
and locally RD-plausible. It is not an architecture selection, a dataset-wide
result, or evidence that r5 is globally the best cut.

## 5. External EL evidence

| Experiment | Observation | Implication |
|---|---|---|
| EL V1 coarse sweep | Hard latent symbols 100% zero; endpoints nearly overlap | Estimated-rate learning did not produce transmitted enhancement information |
| Synthesis-condition variants | Changing `y+B+F_U`, `y+B`, `y-only` did not establish a healthy hard stream | Collapse is not explained solely by one conditioning choice |
| Zero-centered causal test | Zero-symbol correction removed exactly, but hard symbols remained 100% zero | Zero-input synthesis shortcut was not the sole cause |
| Gradient diagnostic | Rate, distortion and synthesis gradients were finite/nonzero | Collapse was not a simple disconnected-loss bug |

This evidence falsifies the tested EL V1 formulation, not all post-native or
hybrid enhancement families.

## 6. Official reference evidence

Artifacts:

```text
$WORK/scalable_attribute_thesis/experiments/rwtt_unicorn_reference_v1/
```

- R01–R09 full hard reference covers all 792 Validation H5 / 28 models.
- It provides reliable total physical bits and author pc_error YUV-PSNR 6:1:1.
- Existing reference CSV does not contain `x_low/r1...r5` physical breakdown.
- The requested `released_longdress_streambits.csv` was not found under the
  currently inspected `$HOME/$WORK` paths.

The full reference can judge endpoint RD position but cannot by itself select a
native cut.

## 7. Remaining evidence gaps by family

| Family | Highest-value missing evidence | Cheapest source of evidence |
|---|---|---|
| Native prefix/suffix | Dataset-wide cut robustness and operating-point dependence | Fixed-H5 physical cut probes, then Dev14 only if justified |
| Separate Base continuation | No-bit continuation capacity/generalization | 20–50-step frozen-prefix micro-overfit |
| Stage specialization | Clone equivalence, parameter cost, isolated trainability | Read-only clone-equivalence/parameter probe |
| Within-stream refinement | Coarse/refinement entropy benefit | Offline captured-symbol analysis |
| Post-native/hybrid | Practical oracle RD and viable hard symbols | No-training residual oracle, then one short hard-gated run |

## 8. Evidence discipline

- Physical arithmetic bits and estimated entropy remain separate columns.
- DQ outputs are not transmitted symbols.
- A fixed-sample PASS is a falsification result, not a formal RD conclusion.
- No earlier result commits the thesis to r5, native split, external EL, AQL,
  unsharing or joint training.
