# Formal Evaluation

The frozen static-RGB evaluation compares **Original Unicorn**, **Ours Base**,
and **Ours Full** under one physical-rate and metric contract. The completed
matrix contains 20 samples: four 8iVFB, four Owlii, and twelve CTC samples.
Each sample has nine Original Unicorn points and seven scalable points; every
scalable point yields both a Base and a Full endpoint.

## Operating points and provenance

Original Unicorn uses the official `NJUVISION/Unicorn` source at commit
`b50d6c1bd033185b9e893b755d5d316cca2d4448` with this fixed mapping:

| Point | Profile | Lambda |
| --- | --- | ---: |
| R01 | `32k8k` | 32768 |
| R02 | `32k8k` | 16384 |
| R03 | `32k8k` | 8192 |
| R04 | `8k256` | 4096 |
| R05 | `8k256` | 2048 |
| R06 | `2k128` | 1024 |
| R07 | `2k128` | 512 |
| R08 | `2k128` | 256 |
| R09 | `2k128` | 128 |

Lambda 8192 always uses `32k8k@8192`; checkpoints are never selected per
sequence. Ours uses the frozen `512/1K/2K/4K/8K/16K/32K` mapping in the
[checkpoint manifest](../configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json).
This manifest is the formal authority for checkpoint paths, SHA-256 identities,
loader branches, architecture, profile, lambda, and parent lineage; mismatches
fail closed.

## Samples

- 8iVFB: `longdress`, `loot`, `redandblack`, and `soldier` at the standard
  vox10 frames.
- Owlii: `basketball_player`, `dancer`, `exercise`, and `model` using the frozen
  author-compatible vox10 inputs.
- CTC: native vox11 `basketball_player` and `dancer`, plus the ten frozen vox12
  objects in the [sample manifest](../configs/evaluation/formal_static_rgb_v1_samples.tsv).

Owlii and CTC inputs sharing a sequence name are distinct canonical artifacts.
Their identities, sizes, point counts, and hashes are frozen in the manifest.

## CTC preprocessing

CTC uses deterministic KD-tree chunks of at most 800,000 points. Coordinates
remain global: there is no local recentering, requantization, or
deduplication. Original Unicorn and Ours consume the same chunk manifest. Bits
and codec runtime are summed across chunks; reconstruction quality is measured
once after merging all chunks on the canonical full sample.

## Rate, quality, and endpoints

- Rate is measured from physical payload bytes and divided by the canonical
  full-input point count.
- Base bits are the physical `x_low` attribute payload plus `r1-r4` entropy
  strings.
- Full bits are the exact Base bits plus Enhancement entropy strings.
- Quality uses MPEG `pc_error`: Y, U, V, and `YUV611=(6Y+U+V)/8`.
- Original Unicorn and Ours use the same input, preprocessing, denominator, and
  metric implementation.

## Runtime semantics

Original encode/decode time covers synchronized CUDA execution of the official
physical encoder/decoder calls. Ours reports:

```text
Base Enc = prefix encode
Base Dec = prefix decode + BaseSynthesis
Full Enc = prefix encode + Enhancement encode
Full Dec = prefix decode + BaseSynthesis + Enhancement decode
```

Input loading, checkpoint loading, PLY writing, metric execution, and CPU
validation copies are outside codec timing. Runtime tables remain labeled and
grouped by GPU model; unlike GPU classes are not combined into an unlabeled
mean.

## BD-BR protocol

All measured raw points remain in the evidence and figures. For BD-BR only, a
deterministic Pareto-efficient frontier is constructed independently for every
sample, method, endpoint, and metric. PCHIP interpolation is applied to
`ln(physical_bpp)` over shared quality overlap only, without extrapolation or
manual point deletion. See
[`recompute_pareto_bdbr.py`](../scripts/scalable_attribute/analysis/recompute_pareto_bdbr.py).

## Reproduction entry points

| Purpose | Entry point |
| --- | --- |
| Original Unicorn physical evaluation | [`evaluate_unicorn_official_physical.py`](../scripts/scalable_attribute/evaluation/evaluate_unicorn_official_physical.py) |
| Combined Base/Full physical evaluation | [`evaluate_scalable_formal.py`](../scripts/scalable_attribute/evaluation/evaluate_scalable_formal.py) |
| CTC chunk preparation | [`prepare_ctc_formal_chunks.py`](../scripts/scalable_attribute/evaluation/prepare_ctc_formal_chunks.py) |
| CTC merged aggregation | [`aggregate_formal_chunk_results.py`](../scripts/scalable_attribute/evaluation/aggregate_formal_chunk_results.py) |
| Frozen contract validation | [`validate_formal_static_rgb_contract.py`](../scripts/scalable_attribute/evaluation/validate_formal_static_rgb_contract.py) |
| Pareto BD-BR analysis | [`recompute_pareto_bdbr.py`](../scripts/scalable_attribute/analysis/recompute_pareto_bdbr.py) |
| Final RD plotting | [`plot_final_static_rgb_rd.py`](../scripts/scalable_attribute/analysis/plot_final_static_rgb_rd.py) |

Final measurements and presentation tables are indexed by the
[thesis-results landing page](../results/comparisons/formal_static_rgb_thesis_results_20260911/README.md).
