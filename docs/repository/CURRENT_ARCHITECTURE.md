# Current architecture

The frozen thesis method is a **full-resolution quality-scalable extension of
Unicorn Part II**. Its canonical flow is:

```text
native truncated Unicorn prefix → learned BaseSynthesis
→ full-resolution Base → conditional Enhancement → Full
```

Each of the seven formal operating points has Base and Full; the seven lambdas
are not seven layers of one universal stream. The contribution is this
full-resolution extension and its verified Base/Full contract, not the first
proposal of layered coding. Unicorn Part II already has progressive residual
refinement, so the repository must not describe it as lacking progressive
decoding. Source and frozen manifests are authoritative; selected weights have
different training histories.

## Added BaseSynthesis

```text
x4 (3 channels) + f4 (128 channels), tensor_stride=2
  -> concatenate (131)
  -> Linear 131 -> 128
  -> 2 ResNet units on tensor_stride=2 support
  -> ONE SparseConvTranspose, kernel 3, spatial upsampling x2
     tensor_stride 2 -> 1; channels 128 -> 128
  -> 2 ResNet units on tensor_stride=1 support
  -> Linear 128 -> 128 (zero-initialized weight and bias at clean init)
  -> c_B (feature compensation, not RGB)
```

Each unit is Conv3 -> ReLU -> Conv3 + identity. Both internal convolution strides
are 1. There are four units in total, no BatchNorm, no post-addition ReLU. A trained
checkpoint is not zero-valued just because its config records `zero_init=true`.

```text
c_B + native f5p -> native fuseNet -> F_B -> native outNet -> delta_B
Base = x5p + delta_B; x5p = UnP(x4)
```

| Component | Actual source |
| --- | --- |
| BaseSynthesis | [base_synthesis.py](../../scalable_attribute/models/base_synthesis.py) |
| Exact Backbone / ResNet definitions | [backbone.py](../../basic_models/backbone.py), [resnet.py](../../basic_models/resnet.py) |
| Prefix and native transition | [prefix.py](../../scalable_attribute/models/prefix.py) |
| Base assembly / scopes | [base.py](../../scalable_attribute/models/base.py) |
| Independent EnhancementVAE | [enhancement.py](../../scalable_attribute/models/enhancement.py) |
| Scalable assembly | [scalable.py](../../scalable_attribute/models/scalable.py) |
| Strict checkpoint restoration | [checkpoints.py](../../scalable_attribute/runtime/checkpoints.py) |
| Native residual implementation | [model_resvae.py](../../lossy_attribute/model_resvae.py) |

## Information flow and rate

Base uses x_low+r1-r4, no native r5. BaseSynthesis uses decoded state and adds no
payload. Full uses the same Base and a separate entropy-coded Enhancement payload
conditioned on decoded Base (`Base`, `F_B`, `d5p`, embedding). Enhancement decode
does not take GT attributes. Geometry support/model identity follow upstream assumptions.

Base bits = x_low attribute bits + r1-r4 string bits.
Full bits = Base bits + Enhancement string bits.
Follow upstream `len(strings)*8` and x_low G-PCC attribute accounting, including
upstream treatment of min/max side information and encoder-side x_low handoff.
Container/fresh-process closure is outside this phase, not a thesis blocker.

Training r1-r4 estimated rate omits fixed x_low rate; x_low does not depend on
learned parameters, so this omission does not remove a parameter gradient.
Do not confuse this estimated residual rate with final physical Base rate.

Initial sequential Base training freezes native modules. Selected rescued 2K and
joint 4K have updated native weights: calling all current native modules 'frozen
released weights' is incorrect. Formal evaluation freezes all parameters. 4K
Base rate need not be below 8K Base rate for scalable correctness. The final
curve is `512/1K/2K/4K/8K/16K/32K`; 256 is historical screening evidence only.
