# Unicorn V1 Attribute Repository Capability Map

Scope: static, read-only survey of the released Attribute codec. This document
describes capabilities and dependency boundaries; it does not select a scalable
architecture.

## 1. Released operating structure

For the released RWTT checkpoints the effective configuration is `scale=5`,
`stage=1`, `Vmode=1`, RGB converted to normalized YUV, 128 state channels and a
shared `ResidualVAE`. The public test mapping uses three variable-rate
checkpoints (`32k8k`, `8k256`, `2k128`) and lambda embeddings.

```text
full-resolution A and geometry support
          |
          | encoder-only average-pooling pyramid x_set[1,2,4,8,16,32]
          v
quantized x_low at stride 32 -------------------------------+
          |                                                  |
          | GPCC physical payload                            | decoder input
          v                                                  v
      transition -> r1 -> transition -> r2 -> ... -> r5 -> reconstruction B
                    ^                    ^              ^
                    | arithmetic streams, coarse-to-fine |
                    +-------------------------------------+
```

`r1...r5` below mean the five arithmetic payloads in encode/decode traversal
order. `r5` is the final refinement call producing stride-1 output.

## 2. Per-stage dependency graph

At the start of refinement stage `k`, the decoder has the prior stage states:

```text
x_out_(k-1), f_out_(k-1), dec_(k-1)
        |              |          |
        |              |          +-> Unpool -> prior_dec_k
        |              +------------> cat with x_out -> learned Upscaler -> f_in_k
        +---------------------------> Unpool -> x_in_k
```

The encoder additionally has `x_gt_k` from the input-only pooling pyramid:

```text
encoder-only:
    residual_k = x_gt_k - x_in_k
    latent_raw  = Encoder(residual_k)
    latent_EQ   = EQ(latent_raw, lambda_embedding)
    symbols_k   = round(latent_EQ)

shared encoder/decoder probability computation:
    prior_k = BlockPrior(cat(x_in_k, f_in_k, prior_dec_k))
    loc_k   = LocNet(prior_k)
    scale_k = abs(ScaleNet(prior_k)).clamp(min=1e-8)

transmitted:
    arithmetic bytes for symbols_k
    min_v/max_v required by the released in-memory decoder API

decoder:
    symbols_k = arithmetic_decode(bytes, loc_k, scale_k, min_v, max_v)
    q_k       = DQ(symbols_k, lambda_embedding)
    dec_k     = Decoder(q_k)
    f_out_k   = FuseNet(f_in_k + dec_k)
    x_out_k   = OutNet(f_out_k) + x_in_k
```

There is no transmitted hyperprior. `loc/scale`, sparse latent support and all
feature/reconstruction states are reproducible from the decoded prefix,
geometry support, lambda and fixed model parameters.

## 3. Information ownership

| Information/state | Encoder | Decoder | Transmitted? | Dependency |
|---|---:|---:|---:|---|
| Full-resolution input `A` | yes | no | no | source sample |
| Geometry/coordinate support `x0` | yes | yes | outside Attribute streams | Geometry codec/input contract |
| Pooling pyramid `x_gt_k` | yes | no | no | `A` |
| `x_low` quantized attributes | yes | yes | GPCC physical payload | deepest pooled `A` |
| `x_in_k` | yes | yes | no | decoded `x_out_(k-1)` + unpool |
| `f_in_k` | yes | yes | no | decoded `f_out_(k-1),x_out_(k-1)` + Upscaler |
| `prior_dec_k` | yes | yes | no | decoded `dec_(k-1)` + unpool |
| encoder residual/latent | yes | no | no | `x_gt_k-x_in_k` |
| rounded latent symbols | yes | yes after decode | yes, arithmetic payload | EQ output |
| entropy `loc/scale` | yes | yes | no | `x_in_k,f_in_k,prior_dec_k` |
| `dec_k/f_out_k/x_out_k` | yes | yes | no | decoded symbols + prefix state |
| lambda embedding | yes | yes | no in current API | operating-point/profile contract |

The author bit accounting counts arithmetic byte length and GPCC bits. It does
not provide a complete file container or account for metadata such as profile,
lambda and `min_v/max_v` serialization.

## 4. Structural mechanisms

### Spatial hierarchy

- Five encoder average-pooling operations produce the GT pyramid.
- `x_low` is the coarsest quantized anchor.
- Five parameter-free pooling-transpose operations expand reconstruction and
  decoded residual state support.
- `Backbone(scale=-1)` is a learned feature transition, not plain interpolation.
- `Backbone(scale=1)` in `ResidualVAE` maps each stage residual/prior to the
  next-coarser latent support; its decoder uses `scale=-1`.

### Residual compression

- True compression happens inside `ResidualVAE`: residual analysis, AQL EQ,
  conditional entropy coding, DQ and synthesis.
- `BlockPrior` conditions each stream on three decoder-known states.
- The conditional entropy model uses a Laplace distribution and actual
  `torchac` arithmetic coding.
- Noise is the only differentiable quantization mode used by the native forward
  training path; symbols use rounding for deterministic/hard execution.

### Variable rate / AQL

- One lambda embedding conditions EQ and DQ at every stage.
- EQ/DQ are affine transforms `x * scale(lambda) + bias(lambda)`, not
  `x*(1+scale)+bias`.
- DQ means zero transmitted symbols need not imply zero decoded correction.
- Different released checkpoint profiles cover different lambda ranges; lambda
  alone must not be treated as a unique checkpoint selector.

### State reuse

- `stage=1` RWTT models instantiate one learned Upscaler and one
  `ResidualVAE`; both are called at all five spatial stages.
- EQ/DQ, entropy prior, encoder, decoder, FuseNet and OutNet are therefore
  shared across all five calls.
- Pooling/unpooling list entries are distinct modules but parameter-free.
- `stage=3, scale=5` does **not** mean five isotropic stages with three
  specialized VAEs. It constructs 15 anisotropic transitions: the encoder
  repeats `[2,1,1] -> [1,2,1] -> [1,1,2]` five times, while decode traverses
  `[1,1,2] -> [1,2,1] -> [2,1,1]` five times. Three learned
  Upscaler/ResidualVAE modules are cycled with `idx % 3` to match those
  axis-specific transitions. This is only a multiple-module implementation
  precedent, not direct evidence for specialization of the released five
  isotropic RWTT stages.

## 5. Legal branch boundaries

### Between native streams

Any boundary after `x_low` or `r_k` can define a logical prefix. A Base decoder
must still obtain full-resolution attributes by one of:

1. deterministic continuation through omitted stages;
2. a separately trained Base continuation branch;
3. transmitting later information through a different nested enhancement path.

Native later streams cannot be decoded independently of earlier states because
their entropy priors and reconstruction states depend on the decoded prefix.
Suffix grouping is dependency-compatible; arbitrary isolated stream selection
is not automatically compatible.

## DECODER_KNOWN_NO_BIT_OPTIONS

Before reading stage `k` arithmetic bytes, the decoder has already computed:

```text
x_in_k      = Unpool(x_out_(k-1))
f_in_k      = Upscaler(cat(f_out_(k-1), x_out_(k-1)))
prior_dec_k = Unpool(dec_(k-1))
prior_k     = BlockPrior(cat(x_in_k, f_in_k, prior_dec_k))
loc_k       = LocNet(prior_k)
scale_k     = abs(ScaleNet(prior_k)).clamp(min=1e-8)
```

For stage 1, `x_low`, `linear_in(x_low)` and a zero `curr_dec` replace the
previous-stage states. Thus every `loc/scale` is determined by decoded prefix
state, geometry support, fixed model parameters and lambda/profile. No
encoder-only `x_gt` or residual enters the prior path.

`loc` parameterizes the distribution of the **EQ output in unit-step
quantization/symbol space**. The encoder path is
`latent_raw -> EQ -> round -> symbols`; likelihood and arithmetic CDF compare
these EQ-space symbols against `loc/scale`. `loc` is neither raw encoder latent
nor DQ-space decoder feature.

### `symbols = 0`

- Requires no encoder-only information and adds no transmitted bits.
- It is an integer symbol in EQ/quantization space.
- It follows `0 -> DQ(0,lambda) -> Decoder -> FuseNet/OutNet`.
- For Vmode 1, `DQ(0,lambda) = bias_DQ(lambda)`, generally nonzero. The tested
  r5-only Base is therefore a deterministic lambda-conditioned Stage-5
  reconstruction, **not** “zero residual” or “no Stage-5 correction”.
- It is mechanically usable at any stage whose latent support is reconstructed
  from the decoder prior path.

### `symbols = round(loc)`

- Requires no encoder-only information and adds no transmitted bits.
- It is an integer EQ/quantization-space symbol.
- It follows `round(loc) -> DQ -> Decoder -> FuseNet/OutNet`.
- It is a decoder-predicted conditional-center continuation. It is compatible
  with integer symbol semantics, but not exposed as an official missing-stream
  API.
- It is mechanically usable at any native stage.

### `latent = loc`

- Requires no encoder-only information and adds no transmitted bits.
- It is a continuous EQ-space value before rounding, not a legal
  arithmetic-decoded integer symbol in the released codec.
- An experimental decoder may run `loc -> DQ -> Decoder -> FuseNet/OutNet`, but
  this bypasses native quantization/arithmetic semantics and adds floating-point
  reproducibility assumptions.
- It is a possible modified continuation at any stage, not a native omitted
  stream decoded normally.

For every option, unchanged released later bytes cannot generally follow an
omitted interior stream. The continuation changes `x_out/f_out/dec`, hence the
later `loc/scale` CDF. A literal native subset must omit a contiguous suffix.
Later bytes could instead be newly encoded under the same continuation, but
then they are not the unchanged released Full suffix. Full decoding can restart
from the common prefix and use the actual complete suffix.

### Inside a native stream

The released arithmetic stream is monolithic. Bitplane layering, coarse/fine
symbol refinement or successively refinable entropy coding would require a new
entropy representation/API. It is a structural opportunity, not a native
capability.

### At a reconstruction state

Every `x_out_k` is a native reconstruction at its current spatial support.
It may feed:

- native continuation;
- a new deterministic/learned full-resolution Base branch;
- a specialized later native stage.

### At the final state

`B=x_out_5`, `F_U=f_out_5`, and `D_U=dec_5` are all full-resolution and
decoder-known. A post-decoder enhancement can condition on any of them without
transmitting the conditioning state, but its own latent must carry genuine new
information.

### Existing conditional reference hook

`MultiscaleVAE` accepts a per-stride `ref_set`; the dynamic Attribute model
uses it through `inter_compensator`. This proves that decoder-known multiscale
side information can be fused into native feature states. Static scalability
does not automatically have such a reference, so using this hook would require
a decoder-reproducible source and retraining.

## 6. Modification impact map

| Modification location | Base impact | Full impact | Entropy/bit dependency impact |
|---|---|---|---|
| Change `x_low` | direct | all later stages | all stream priors change |
| Change early stage `r1-r3` decode/state | direct | all suffix stages | all later entropy priors change |
| Omit a suffix with deterministic continuation | changes Base only | can preserve released Full | prefix bytes remain unchanged |
| Add separate Base continuation at prefix | changes Base only | can preserve released Full | no change if branch consumes no new bits |
| Clone/specialize later ResidualVAE | depends on split | changes Full if trained | affected stage and suffix priors change |
| Change shared ResidualVAE | likely both endpoints | all five calls | all native streams change |
| Change AQL/embedder | likely both | all stages | symbols and entropy distributions change |
| Split symbols within a stream | both format paths | potentially preservable only by design | requires new entropy syntax/model |
| Add post-final external refinement | Base can remain released | Full adds independent stream | native Base streams unchanged |
| Change Upscaler | downstream state changes | affected suffix changes | later conditional priors change |

## 7. Independently decodable endpoint contract

The official in-memory test path is not a standalone Attribute bitstream. A
minimal independently decodable endpoint requires:

| Required information | Official source status |
|---|---|
| Full-resolution geometry/coordinate support `x0` | supplied in memory; assumed available from geometry coding |
| Physical `x_low` attribute payload | G-PCC encode runs for bit counting, but no `.bin` decode/handoff is implemented |
| Included r-stream arithmetic bytes | real `torchac` bytes are produced |
| Per-stream `min_v/max_v` | passed in-memory and required to rebuild each CDF alphabet |
| Lambda | decode argument; not serialized |
| Checkpoint/profile identity | external configuration; not serialized |
| `scale/stage/Vmode`, channels, color format/normalization | fixed configuration; not serialized |
| Stream count/cut position and ordering | implicit in API/profile; not serialized |

`model.test()` calls G-PCC encode, receives its reported bit count, and passes
the original in-memory `x_low` SparseTensor directly to `decode()`. It never
decodes the generated G-PCC bitstream. The thesis
`BaseAdapter.hard_reconstruct()` retains the same in-memory handoff: r-streams
are actually compressed/decompressed and G-PCC bits are counted, but there is
no physical `x_low` serialization/decode path. That capability must not be
attributed to official source or the current wrapper.

At a suffix boundary, bytes that can literally satisfy `Base subset Full` are
the common physical `x_low` payload (once implemented), exact prefix r-stream
bytes and their metadata. Enhancement is the exact suffix bytes plus suffix
metadata. Lambda/profile/model and cut signaling are common fixed metadata. The
current evidence establishes logical/in-memory nesting, not a packaged codec.

## 8. API and release limitations

- `LossyAttributeCoder.encode/decode` file APIs are TODO; `test()` is the real
  released hard in-memory path.
- `MultiscaleVAE.decode` expects `x_low` and five encoded dictionaries; missing
  streams are not a public native API and require an experimental continuation.
- Public `out_list` reverses traversal order; `out_list[0]` is final output.
- Public `Qvalue_list` is derived from returned `Qlatent`, which has already
  passed DQ in `Vmode=1`; it must not be interpreted as transmitted symbols.
- The repository contains no static `lossy_attribute/train.py`. Exact released
  multiscale training loss/`last_weight` semantics cannot be uniquely recovered
  from the public Attribute source alone.
- CPU inspection of all three released RWTT checkpoints (`32k8k`, `8k256`,
  `2k128`) found exactly one top-level key: `model`. No args, optimizer, epoch,
  loss configuration, `last_weight` or lambda-sampling metadata are stored, and
  no sidecar logs/configs exist beside them. Released training metadata is
  therefore **not recoverable** from the available package.
- The current thesis compatibility flags exposing deterministic reconstruction
  and final feature do not alter the native mathematical chain, but they are not
  part of the original public API.
