# Scalable Point Cloud Attribute Compression

This repository implements a full-resolution Base/Full quality-scalable extension of Unicorn Part II for static point-cloud attributes. Base decodes the truncated `x_low` and `r1-r4` prefix and synthesizes a full-resolution reconstruction. Full adds a conditional Enhancement payload and reuses the decoded Base state.

## Final results

The frozen evaluation covers 20 samples: four 8iVFB, four Owlii, and twelve CTC. Full-endpoint BD-BR relative to Original Unicorn is:

| Dataset | Y | YUV611 |
|---|---:|---:|
| 8iVFB | +1.452% | +1.390% |
| Owlii | +1.855% | +2.077% |
| CTC | +0.155% | -0.696% |

Negative BD-BR indicates bitrate savings at equivalent quality. Per-sample Base and Full results, raw measurements, Pareto audit, figures, and hardware-labeled runtimes are in [results/final](results/final/README.md).

## Setup and evaluation

Use the [verified FAU environment setup](docs/INSTALL.md) and [exact version profile](configs/fau_environment.json). With a locally supplied checkpoint and the required dependencies installed, evaluate an ASCII or binary RGB PLY sample:

```bash
python -m scalable_attribute.evaluate \
  --checkpoint checkpoints/ours_8k.pth \
  --input examples/example.ply \
  --output outputs/example \
  --gpcc third_party/tmc3_v21 \
  --pc-error third_party/pc_error_d
```

This writes Base and Full reconstructions, trusted PyTorch payload containers, geometry support, and a rate/quality report. Rate counts G-PCC-reported `x_low` attribute bits, `r1-r4` strings, and Enhancement strings; it excludes PyTorch container overhead and geometry support. The payloads are model-specific evaluation artifacts, not standalone interoperable compressed streams. See [Evaluation](docs/EVALUATION.md) for the decoding and accounting contract.

Seven exported checkpoints are recorded in `configs/checkpoints.json` and `configs/SHA256SUMS` (952,005,347 bytes total). Each has 318 tensors with exact source tensor identity. All seven pass strict independent loading and original/export Base/Full physical equivalence on the bundled 512-point example using CPU inference; representative GPU checks for 8k, 2k, and 4k also pass on RTX 3080, including fresh-process Base/Full decoding. CPU and GPU results are not bit-exact interchangeable. See the [final audit](docs/FINAL_AUDIT.md) and [validation results](docs/VALIDATION.md). Model assets and download URLs remain unpublished pending upstream permission and handover approval. The downloader verifies supplied local files and will download only from a configured HTTPS mirror after a URL is approved:

```bash
python -m scalable_attribute.download --point 8k --output checkpoints
```

With authorized local model supply, place the supplied `ours_8k.pth` in `checkpoints/` and run `python -m scalable_attribute.download --point 8k --output checkpoints` to verify it against the manifest.

To check strict checkpoint loading:

```bash
mkdir -p outputs
python -m scalable_attribute.verify --models checkpoints --point 8k --load-only --output outputs/verify-8k.json
```

## Results analysis

Rebuild raw RD figures:

```bash
python scripts/scalable_attribute/analysis/plot_final_static_rgb_rd.py \
  --raw results/final/evidence/raw_rd_points.csv \
  --output-dir outputs/rd-figures
```

Recompute Pareto/PCHIP BD-BR tables and audits from the raw evidence:

```bash
python scripts/scalable_attribute/analysis/recompute_pareto_bdbr.py \
  --raw results/final/evidence/raw_rd_points.csv \
  --output outputs/pareto-bdbr
```

Prepare CTC chunks from a local canonical source PLY, then evaluate the chunk manifest. Use the pinned upstream `partition.py` file whose SHA-256 is required by the preparation tool. The evaluator writes per-chunk records and an `aggregate_command.txt` for the merger:

```bash
python scripts/scalable_attribute/evaluation/prepare_ctc_formal_chunks.py \
  --input /path/to/ctc_sample.ply \
  --output-root outputs/ctc_chunks \
  --partition-source third_party/unicorn_partition.py.txt
python -m scalable_attribute.evaluate \
  --checkpoint checkpoints/ours_8k.pth \
  --input /path/to/ctc_sample.ply \
  --chunk-manifest outputs/ctc_chunks/manifest.json \
  --output outputs/ctc_ours_8k \
  --gpcc third_party/tmc3_v21 \
  --pc-error third_party/pc_error_d
# Run the generated merger command:
sh outputs/ctc_ours_8k/aggregate_command.txt
```

Run the configured test suite with:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

## Method and attribution

Method and protocol details: [docs/METHOD.md](docs/METHOD.md), [docs/EVALUATION.md](docs/EVALUATION.md). This work builds on the [NJUVISION/Unicorn implementation at commit `b50d6c1`](https://github.com/NJUVISION/Unicorn/tree/b50d6c1bd033185b9e893b755d5d316cca2d4448) and *A Versatile Point Cloud Compressor Using Universal Multiscale Conditional Coding—Part II: Attribute Compression* ([paper](https://ieeexplore.ieee.org/document/10682566)). The thesis contribution is the full-resolution BaseSynthesis path and conditional Enhancement endpoint. See [third-party notices](THIRD_PARTY.md) for attribution and included dependency notices.
