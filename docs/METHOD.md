# Method

This thesis develops a **full-resolution quality-scalable extension of Unicorn
Part II** for static point-cloud attribute compression:

```text
native truncated Unicorn prefix -> learned BaseSynthesis
-> full-resolution Base -> conditional Enhancement -> Full
```

Unicorn Part II already supports progressive residual refinement. The thesis
contribution is therefore not progressive decoding or layered coding in
general, but a full-resolution Base/Full extension with a verified payload and
reconstruction contract.

## Base endpoint

The Base payload consists of `x_low` and `r1-r4`; it does not include the native
`r5` payload. Decoding this prefix exposes the decoded low-resolution attribute
state and native intermediate features.

[`BaseSynthesis`](../scalable_attribute/models/base_synthesis.py) consumes the
decoded prefix state and generates feature compensation for the
full-resolution Base path without additional payload bits. Its main structure
is:

```text
concatenated decoded prefix features
-> linear projection
-> two stride-2 residual units
-> one sparse transposed convolution to stride 1
-> two stride-1 residual units
-> zero-initialized output projection
```

The compensation is combined with the native full-resolution feature path.
The native fusion and output networks then predict the residual used to form
the Base reconstruction. BaseSynthesis alone does not produce the final Base
reconstruction.

## Full endpoint

The Full payload is the Base payload plus a separately entropy-coded
Enhancement payload. Enhancement coding is conditioned only on decoded Base
state and does not use ground-truth attributes at the decoder. Full decoding
reuses the exact same decoded Base reconstruction produced by the Base
endpoint, then applies the decoded Enhancement residual.

## Training objective

The Base objective combines the estimated `r1-r4` prefix rate with Base
distortion, while the Full objective additionally includes Enhancement rate:

```text
L_Base = R_Base + lambda_Base * D_Base
L_Full = R_Base + R_Enhancement + lambda_Full * D_Full
```

The fixed `x_low` rate is omitted from the differentiable training-rate term
because it does not depend on learned parameters. It is included in every
formal physical-rate measurement. Some selected points use sequential training
and others joint random Base/Full endpoint training; their exact frozen
identities and loader branches are recorded in the
[checkpoint manifest](../configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json).

## Source map

| Component | Source |
| --- | --- |
| Truncated Unicorn prefix | [`prefix.py`](../scalable_attribute/models/prefix.py) |
| BaseSynthesis | [`base_synthesis.py`](../scalable_attribute/models/base_synthesis.py) |
| Base assembly | [`base.py`](../scalable_attribute/models/base.py) |
| Conditional Enhancement codec | [`enhancement.py`](../scalable_attribute/models/enhancement.py) |
| Base/Full composition | [`scalable.py`](../scalable_attribute/models/scalable.py) |
| Checkpoint identity validation | [`checkpoints.py`](../scalable_attribute/runtime/checkpoints.py) |
| Joint endpoint objective | [`joint_endpoint.py`](../scalable_attribute/training/joint_endpoint.py) |
| Current training commands | [`scripts/scalable_attribute/training/`](../scripts/scalable_attribute/training/) |

The released Unicorn implementation remains the upstream authority for native
components. The thesis implementation does not claim ownership of those
components.
