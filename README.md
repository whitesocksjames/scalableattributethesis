# Scalable Point Cloud Attribute Compression Based on Unicorn Part II

This repository contains the implementation and final evaluation of a
**full-resolution quality-scalable extension of Unicorn Part II**. The method
turns a native truncated Unicorn attribute prefix into a full-resolution Base
reconstruction and uses an optional conditional Enhancement payload to produce
the Full endpoint. It therefore provides two full-resolution decoding endpoints
while retaining competitive Full-endpoint rate-distortion performance relative
to Original Unicorn.

## Method

```text
truncated Unicorn prefix
    -> BaseSynthesis
    -> full-resolution Base
    -> conditional Enhancement
    -> Full
```

- The Base payload consists of `x_low` and `r1-r4`.
- BaseSynthesis generates feature compensation for the full-resolution Base
  path without additional payload bits.
- The Full payload consists of the Base payload plus the Enhancement payload.
- Full decoding reuses the same decoded Base reconstruction.
- Base and Full are reconstructed on the original full-resolution geometry
  support.

## Main Results

Full versus Original Unicorn:

| Dataset | Y BD-BR | YUV611 BD-BR |
| --- | ---: | ---: |
| 8iVFB | +1.452% | +1.390% |
| Owlii | +1.855% | +2.077% |
| CTC | +0.155% | -0.696% |

Negative values indicate bitrate savings. The complete evaluation covers 20
samples: four 8iVFB, four Owlii, and twelve CTC samples. BD-BR is computed on
the Pareto-efficient RD envelope using PCHIP over the shared quality range;
all measured operating points remain visible in the raw RD plots.

The results show that the Full endpoint remains close to Original Unicorn RD
performance while the scalable bitstream also provides an independently
decodable Base endpoint. The results are competitive, but not uniformly
superior.

See the [final thesis results](results/comparisons/formal_static_rgb_thesis_results_20260911/README.md)
for Base results, per-sequence tables, raw RD evidence, and figures.

Method and evaluation details are documented in [Method](docs/METHOD.md) and
[Formal Evaluation](docs/EVALUATION.md).

## Repository Structure

```text
scalable_attribute/
|-- models/
|   |-- prefix.py
|   |-- base_synthesis.py
|   |-- base.py
|   |-- enhancement.py
|   `-- scalable.py
|-- training/
|-- evaluation/
`-- runtime/
```

- [`prefix.py`](scalable_attribute/models/prefix.py) implements truncated
  Unicorn prefix coding and exposes the decoded prefix state.
- [`base_synthesis.py`](scalable_attribute/models/base_synthesis.py) implements
  learned feature compensation for the full-resolution Base path.
- [`base.py`](scalable_attribute/models/base.py) assembles the Base
  reconstruction path.
- [`enhancement.py`](scalable_attribute/models/enhancement.py) implements the
  conditional Enhancement codec.
- [`scalable.py`](scalable_attribute/models/scalable.py) composes the Base and
  Full endpoints.

## Key Entry Points

- Model implementation: [`scalable_attribute/models/`](scalable_attribute/models/)
- Training: [`scripts/scalable_attribute/training/`](scripts/scalable_attribute/training/)
- Formal scalable evaluation: [`evaluate_scalable_formal.py`](scripts/scalable_attribute/evaluation/evaluate_scalable_formal.py)
- Original Unicorn evaluation: [`evaluate_unicorn_official_physical.py`](scripts/scalable_attribute/evaluation/evaluate_unicorn_official_physical.py)
- Frozen checkpoint manifest: [`formal_static_rgb_v1_checkpoints.json`](configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json)
- Final results: [`formal_static_rgb_thesis_results_20260911`](results/comparisons/formal_static_rgb_thesis_results_20260911/README.md)

## Reference

This work builds on the official
[NJUVISION/Unicorn](https://github.com/NJUVISION/Unicorn) implementation and the
attribute-compression method described in *A Versatile Point Cloud Compressor
Using Universal Multiscale Conditional Coding -- Part II: Attribute
Compression*.

- [Unicorn Part II paper](https://ieeexplore.ieee.org/document/10682566)
- [Original Unicorn project page](https://njuvision.github.io/Unicorn/)

Please cite the Original Unicorn paper and repository when using the upstream
method or code.
