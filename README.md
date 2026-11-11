# Scalable Point Cloud Attribute Compression

Final implementation and evaluation of a full-resolution Base/Full quality-scalable extension of Unicorn Part II for static point-cloud attributes.

## OURS architecture

```text
Base payload: x_low + r1-r4
    -> Prefix decoding + learned BaseSynthesis
    -> full-resolution Base and decoded features
    -> optional conditional Enhancement decoding
    -> full-resolution Full
```

BaseSynthesis completes the truncated prefix without extra payload bits. Full adds an independently entropy-coded Enhancement payload and reuses the decoded Base state. The seven operating points are `512 / 1k / 2k / 4k / 8k / 16k / 32k`. See [Method](docs/METHOD.md) for the graph and training objective.

**Seven pretrained checkpoints are required for the complete handover**, supplied separately from Git objects. Each contains Prefix, BaseSynthesis and Enhancement states for both Base and Full; no Original Unicorn weight file is needed. The [manifest](configs/checkpoints.json) and [SHA-256 list](configs/SHA256SUMS) identify the audited files (952,005,347 bytes total).

**Checkpoint delivery status: not yet published.** GitHub Release assets and a separate GitLab package copy are prepared as the distribution targets. Redistribution permission for inherited Unicorn code and derived weights remains unconfirmed, and the GitLab repository has not been specified. No checkpoint download is currently available on either platform; see [Checkpoint delivery](docs/CHECKPOINTS.md). Datasets are not bundled.

## Final results

The frozen evaluation covers 20 samples: four 8iVFB, four Owlii and twelve CTC. BD-BR relative to Original Unicorn is:

| Dataset | Base Y | Base YUV611 | Full Y | Full YUV611 |
| --- | ---: | ---: | ---: | ---: |
| 8iVFB | +5.205% | -0.723% | +1.452% | +1.390% |
| Owlii | +10.110% | +4.992% | +1.855% | +2.077% |
| CTC | +1.848% | -0.326% | +0.155% | -0.696% |

Negative BD-BR indicates bitrate savings at equivalent quality. [Final results](results/final/README.md) contain all 460 measured endpoint rows, per-sample Base/Full and Original Unicorn comparisons, Pareto/PCHIP tables, raw RD figures and hardware-labeled runtime data. The [evaluation protocol](docs/EVALUATION.md) defines preprocessing, physical rate, metrics and interpolation.

## Installation and environment

[INSTALL.md](docs/INSTALL.md) provides Linux dependency setup and pinned `tmc3` / `pc_error` build commands. [environment.json](configs/environment.json) records the actual validated Python 3.8.20, PyTorch 1.13.1+cu117, MinkowskiEngine 0.5.4 and remaining runtime versions. The instructions require no personal cluster directory or account. A completely fresh dependency installation and MinkowskiEngine rebuild have not been certified.

## Repository structure

| Path | Contents |
| --- | --- |
| `scalable_attribute/models/` | Prefix, BaseSynthesis, Base and Enhancement composition |
| `scalable_attribute/runtime/` | Complete-checkpoint loading and Base/Full encode/decode |
| `scalable_attribute/evaluate.py` | RGB PLY inference, physical rates and optional quality metrics |
| `basic_models/`, `lossy_attribute/`, `data_utils/`, `cfg/` | Required network, entropy, I/O and fixed configuration dependencies |
| `results/final/` | Frozen measurements, BD-BR tables and RD figures |
| `configs/`, `examples/` | Checkpoint manifest/checksums, sample identities, environment record and synthetic example |
| `scripts/scalable_attribute/` | CTC preprocessing/aggregation and result analysis |
| `tests/`, `docs/`, `third_party/` | Focused regression tests, method/protocol, required tools and notices |

## Evaluation instructions

### Reproduce tables and figures without weights

These analysis commands use only the Python standard library:

```bash
python scripts/scalable_attribute/analysis/recompute_pareto_bdbr.py \
  --raw results/final/evidence/raw_rd_points.csv --output outputs/bdbr
python scripts/scalable_attribute/analysis/plot_final_static_rgb_rd.py \
  --raw results/final/evidence/raw_rd_points.csv --output-dir outputs/rd-figures
(cd results/final && sha256sum -c SHA256SUMS)
```

### Verify a checkpoint and run the Quick Test

This step requires a verified checkpoint. The platform download step is blocked as described above. Once authorized assets are available, save `ours_8k.pth` in `checkpoints/` and check its identity before loading it:

```bash
(cd checkpoints && sha256sum --ignore-missing -c ../configs/SHA256SUMS)
```

Check that the output explicitly says `ours_8k.pth: OK`; an absent file is not a successful verification of that point. For all seven models, omit `--ignore-missing`.

Then run:

```bash
python -m scalable_attribute.evaluate \
  --checkpoint checkpoints/ours_8k.pth \
  --input examples/example.ply --output outputs/example \
  --gpcc third_party/tmc3_v21 --pc-error third_party/pc_error_d
```

The synthetic example produces `base.ply`, `full.ply`, payload containers and `rates.json`; it is not a thesis benchmark sample. Use a new output directory. `--endpoint base` exercises Base without an Enhancement payload. CPU and GPU outputs are not bit-exact interchangeable; use one backend for encoding and decoding. The containers are model-specific evaluation artifacts, with geometry/container overhead excluded under the frozen physical-rate convention.

For CTC, prepare lossless global-coordinate chunks, evaluate the chunk manifest and merge before full-sample quality calculation:

```bash
python scripts/scalable_attribute/evaluation/prepare_ctc_formal_chunks.py \
  --input /path/to/canonical_sample.ply --output-root outputs/chunks
python -m scalable_attribute.evaluate \
  --checkpoint checkpoints/ours_8k.pth \
  --input /path/to/canonical_sample.ply --chunk-manifest outputs/chunks/manifest.json \
  --output outputs/chunk-evaluation \
  --gpcc third_party/tmc3_v21 --pc-error third_party/pc_error_d
sh outputs/chunk-evaluation/aggregate_command.txt
```

Run focused checks with `python -m pytest -q` after installing `requirements-test.txt`; codec-dependent tests need the declared runtime. Analysis and preprocessing checks do not require pretrained weights.

## Reference and attribution

This work builds on [NJUVISION/Unicorn](https://github.com/NJUVISION/Unicorn/tree/b50d6c1bd033185b9e893b755d5d316cca2d4448) and *A Versatile Point Cloud Compressor Using Universal Multiscale Conditional Coding — Part II: Attribute Compression* ([paper](https://ieeexplore.ieee.org/document/10682566)). OURS adds the learned full-resolution BaseSynthesis path and conditional Enhancement endpoint. [THIRD_PARTY.md](THIRD_PARTY.md) preserves attribution, tool notices and the unresolved redistribution status of inherited code.
