# Scalable Attribute Design Space

This is a candidate-family map, not an architecture decision or ranking.

## Family 1 — Native prefix/suffix stream partition

**Base definition**  
`x_low` plus a coarse-to-fine native prefix `r1...rK`.

**Enhancement definition**  
The native suffix `r(K+1)...r5`.

**Full definition**  
Released/native traversal with all streams.

**Base subset mechanism**  
The Base payload is a literal ordered subset of native payloads. Omitted suffix
stages use a deterministic continuation known to both sides.

**Full-resolution Base mechanism**  
Run every remaining spatial transition using zero/default/predicted symbols or
another fixed no-bit continuation.

Source-legality distinguishes three continuations: `symbols=0` and
`symbols=round(loc)` are integer EQ-space choices; `latent=loc` is decoder-known
but bypasses native integer-symbol semantics. All still run DQ and synthesis.
In particular, `DQ(0,lambda)=bias_DQ(lambda)` is generally nonzero.

**Repository components reused**  
`x_low`, native transitions, ResidualVAE decoder, AQL DQ, native suffix streams,
conditional entropy model.

**Components changed**  
Decode control/API for omitted suffix streams; optional endpoint signaling.

**Training requirement**  
None for a diagnostic zero-symbol baseline; likely needed to optimize Base RD.
Raw truncation is a capability/control baseline, not by itself a sufficient
thesis architecture contribution.

**Full preservation potential**  
High: Full may retain all released bytes and released decode path exactly.

**Main technical risk**  
Deterministic suffix continuation may yield a weak Base; the best cut can vary
with operating point. Earlier cuts compound state and transition errors. An
omitted interior stream cannot be followed by unchanged released later bytes,
because the changed reconstruction state changes their entropy CDFs; unchanged
native nesting therefore requires a contiguous suffix cut.

**Evidence already available**  
R5-only R01-R03 four-H5 falsification passed all structural gates and produced
meaningful Base RD points; r5 represented roughly 47–75% per-sample Full rate.

**Missing evidence**  
Dataset-level robustness, other cuts, low-rate behavior and retrained Base
continuation behavior.

**Cheapest discriminating experiment**  
For a proposed cut, run released hard streams on 2–4 fixed H5, omit only the
suffix, and compare physical rate/author PSNR against neighboring official RD
points. Do not train until that endpoint survives.

## Family 2 — Prefix Base with a separate full-resolution Base continuation

**Base definition**  
A native prefix ending at an intermediate reconstruction/state.

**Enhancement definition**  
Native suffix streams or a separately coded suffix branch.

**Full definition**  
Released Full path, or a retrained Full branch beginning at the same prefix.

**Base subset mechanism**  
Both endpoints share the native prefix. The Base-only continuation consumes no
additional payload; Enhancement bytes select the Full continuation.

**Full-resolution Base mechanism**  
A learned deterministic upsampling/refinement branch converts the intermediate
state to stride-1 attributes.

**Repository components reused**  
Native prefix encoder/decoder, `Backbone(scale=-1)`, intermediate
`x_out/f_out/dec`, optionally native suffix for Full.

**Components changed**  
New Base continuation branch and endpoint routing.

**Training requirement**  
Train the Base continuation; native prefix and Full may remain frozen initially.

**Full preservation potential**  
High if the released Full suffix remains untouched and the Base branch is
separate; lower if shared transition parameters are updated.

**Main technical risk**  
Base continuation may hallucinate/smooth attributes, and duplicated decoder
branches may weaken architecture coherence.

**Evidence already available**  
Native intermediate states exist at every stride; learned Upscaler transitions
change feature semantics substantially.

**Missing evidence**  
Whether a no-bit learned continuation improves RD over native zero-symbol
continuation and whether its gains generalize.

**Cheapest discriminating experiment**  
Freeze a chosen prefix and overfit only a very small continuation for 20–50
steps on a fixed micro-set; reject if it cannot beat zero continuation without
altering Full.

## Family 3 — Stage-specialized native residual refinement

**Base definition**  
A native prefix or a native deterministic Base endpoint.

**Enhancement definition**  
One or more explicit late ResidualVAE calls whose parameters are specialized for
the enhancement endpoint.

**Full definition**  
Base state plus transmitted specialized residual stream(s).

**Base subset mechanism**  
Base payload stops before specialized streams; Full appends them.

**Full-resolution Base mechanism**  
Native deterministic continuation or a separate Base continuation.

**Repository components reused**  
Residual encoder/decoder, BlockPrior, loc/scale nets, SymmetricConditional,
FuseNet/OutNet, transitions and checkpoint weights.

**Components changed**  
Shared ResidualVAE may be cloned/unshared or supplemented with endpoint-specific
parameters; encode/decode traversal becomes stage-explicit.

**Training requirement**  
Train only specialized modules initially; freeze native Base path.

**Full preservation potential**  
Initially high if every clone is copied from the shared released weights;
training specialized stages intentionally moves Full away from released Full.

**Main technical risk**  
Breaking shared-weight regularization, propagating changed states into later
entropy priors, and increasing parameters without guaranteed RD benefit.

**Evidence already available**  
RWTT `stage=1` reuses one ResidualVAE five times. Repository `stage=3` is only a
multiple-module precedent: three axis-specific modules cycle over 15
anisotropic transitions, not the released five isotropic stages. State trace
confirms the released shared weights face different residual distributions,
but five-stage specialization remains a proposed modification.

**Missing evidence**  
Exact parameter cost, clone numerical equivalence and stable isolated training.

**Cheapest discriminating experiment**  
Source-level deep-copy equivalence probe only: initialize explicit stage copies
from one released VAE and require identical per-stage outputs and Full before any
training.

## Family 4 — Successive refinement inside native latent streams

**Base definition**  
Coarse quantization/representation of one or more native latent symbol fields.

**Enhancement definition**  
Refinement symbols or bitplanes that recover finer latent values.

**Full definition**  
Decode coarse plus refinement symbols through native DQ/synthesis or a compatible
retrained synthesis path.

**Base subset mechanism**  
Enhancement payload refines already transmitted coarse symbols rather than
adding spatial stages.

**Full-resolution Base mechanism**  
All five decoder stages still execute; Base uses coarse latent values.

**Repository components reused**  
Native state chain, conditional prior networks, DQ, decoder and reconstruction
heads.

**Components changed**  
Quantization representation, entropy syntax/model and encode/decode API.

**Training requirement**  
Endpoint-aware retraining is required.

**Full preservation potential**  
Medium: exact released Full is possible only if refinement reconstructs original
symbols and entropy dependencies remain compatible; otherwise Full changes.

**Main technical risk**  
The released `torchac` stream is monolithic and not prefix-decodable. Designing
true nested entropy payloads is substantially more complex than stream suffix
partitioning.

**Evidence already available**  
Native symbols are integer-valued and conditional priors are decoder-known.

**Missing evidence**  
No released progressive-symbol API, no rate benefit evidence and no training
method in the repository.

**Cheapest discriminating experiment**  
Offline symbol-only analysis: coarsen captured native symbols, measure empirical
coarse/refinement entropy and reconstruction impact without changing the codec.

## Family 5 — Post-native or hybrid residual enhancement

**Base definition**  
A complete native full-resolution reconstruction, or a native prefix with a
full-resolution Base continuation.

**Enhancement definition**  
An additional residual stream conditioned on decoder-known native states such as
reconstruction, feature and decoded residual states.

**Full definition**  
`Base + decoded enhancement correction` at full resolution.

**Base subset mechanism**  
Native Base payload is unchanged; Enhancement payload is appended independently.

**Full-resolution Base mechanism**  
Provided directly by native Full or by the chosen Base continuation.

**Repository components reused**  
Backbone, ResidualVAE concepts, SymmetricConditional and decoder-known native
states; the dynamic `ref_set/inter_compensator` demonstrates conditional feature
fusion as a repository pattern.

**Components changed**  
New analysis/synthesis/entropy path or a hybrid native residual module.

**Training requirement**  
Train enhancement modules with Base frozen; hard nested rate must be validated.

**Full preservation potential**  
Native Base is fully preserved, but Full is new rather than released Full.

**Main technical risk**  
Small high-frequency residuals can collapse under quantization; decoder-known
conditioning can create zero-bit shortcuts, and excessive spatial reduction can
discard the target signal.

**Evidence already available**  
External EL V1 and zero-centered causal runs produced 100% zero hard symbols.
Native state trace showed `A-B` is smaller-amplitude but relatively more local
than E5, and native ResidualVAE uses scale 1 rather than the failed EL's scale 2.

**Missing evidence**  
Whether any full-resolution/native-style formulation carries nonzero hard
information and produces worthwhile incremental RD.

**Cheapest discriminating experiment**  
Before training, run direct residual/spatial-bottleneck oracle analysis. Only if
the oracle is positive, test one 20–100-step architecture with hard-symbol gates.

## Cross-family constraints

- A valid design must define the physical Base payload and Enhancement payload,
  not only two neural outputs.
- Every Base decoder must reach the original stride-1 coordinate support.
- Decoder conditioning may use only decoded payloads, fixed parameters, geometry
  and explicit operating-point information.
- Changing an early decoded state changes every later entropy prior; released
  suffix bytes are preserved only when their preceding decoded state is
  preserved.
- Released checkpoint reuse is initialization evidence, not proof that a new
  endpoint has coherent RD behavior.
- No family is selected or ranked by this survey.
