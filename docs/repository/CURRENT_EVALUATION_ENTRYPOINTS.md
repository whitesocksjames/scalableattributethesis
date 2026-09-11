# Current formal evaluation contract and entry points

The formal experiment matrix is complete and frozen. These entry points document
the implementation that produced the evidence; they are not authorization to
submit new experiments.

| Role | Entry point |
|---|---|
| Pinned Official Unicorn physical coding | [evaluate_unicorn_official_physical.py](../../scripts/scalable_attribute/evaluation/evaluate_unicorn_official_physical.py) |
| Ours combined Base+Full physical coding | [evaluate_scalable_formal.py](../../scripts/scalable_attribute/evaluation/evaluate_scalable_formal.py) |
| CTC canonical ≤800k chunk preparation | [prepare_ctc_formal_chunks.py](../../scripts/scalable_attribute/evaluation/prepare_ctc_formal_chunks.py) |
| CTC merged aggregation | [aggregate_formal_chunk_results.py](../../scripts/scalable_attribute/evaluation/aggregate_formal_chunk_results.py) |
| Frozen identity/contract gate | [validate_formal_static_rgb_contract.py](../../scripts/scalable_attribute/evaluation/validate_formal_static_rgb_contract.py) |
| Resumable point orchestration | [formal_ctc_pack_resume.py](../../scripts/scalable_attribute/evaluation/formal_ctc_pack_resume.py) |

Original Unicorn comes from pinned upstream commit
`b50d6c1bd033185b9e893b755d5d316cca2d4448`; the thesis implementation is not
used as a substitute. The frozen Ours checkpoint authority is
[`formal_static_rgb_v1_checkpoints.json`](../../configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json).

## Endpoint and runtime definitions

Official encode time covers the synchronized GPU `model.forward(...,
encode=True)` physical coding call, including entropy strings and physical
G-PCC `x_low` accounting. Official decode time covers the synchronized GPU
`model.decode(...)` call. Input reading, checkpoint loading, reconstruction PLY
writing, and `pc_error` are outside both codec timers.

Ours uses one combined `(sample, operating point)` execution so Base and Full
share the exact native prefix and Base reconstruction:

- `Base Enc = prefix_encode`;
- `Base Dec = prefix_decode + base_synthesis`;
- `Full Enc = prefix_encode + enhancement_encode`;
- `Full Dec = prefix_decode + base_synthesis + enhancement_decode`.

Therefore `Base codec = Base Enc + Base Dec`, and `Full codec = Full Enc + Full
Dec`. Full includes the complete cost needed to obtain its Base dependency. CUDA
is synchronized at every timing boundary. Metric computation, CPU validation
copies, PLY I/O, model/checkpoint loading, and task wall time are not mixed into
Enc/Dec.

For CTC, each component is summed over serial chunks and quality is measured
once on the merged full-sample reconstruction. Runtime aggregation is always
hardware-labeled and grouped by GPU model/VRAM. Unlike GPU models must never be
combined into an unlabeled mean. House and Shiva final runtime uses each
sample's complete A100-SXM4-40GB Original9+Ours7 set.

## Scientific boundary

The common contract is canonical input, shared preprocessing/chunks, physical
rate, full-input bpp denominator, MPEG `pc_error` Y/U/V, and
`YUV611=(6Y+U+V)/8`. CTC keeps global coordinates and does not recenter,
requantize, or deduplicate. Base is `x_low+r1+r2+r3+r4`; Full is that exact Base
plus the Enhancement payload.

Legacy evaluators such as `evaluate_unicorn_reference.py`, Base-only screening,
and historical rescue scripts remain diagnostic evidence. They are not the
formal baseline/runtime authority. Final outputs are indexed in
[CURRENT_RESULT_INDEX.md](CURRENT_RESULT_INDEX.md).
