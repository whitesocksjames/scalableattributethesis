# Evaluation

The frozen static-RGB evaluation compares Original Unicorn, Ours Base, and Ours Full on 20 samples: four 8iVFB, four Owlii, and twelve CTC. Each sample has nine Original Unicorn operating points and seven scalable operating points; each scalable point produces Base and Full endpoints. [configs/samples.tsv](../configs/samples.tsv) records sample IDs, dataset-relative paths, point counts, source hashes, and preprocessing.

## Rate and quality

Rate is physical payload bits divided by the canonical full-input point count. Base rate includes G-PCC-reported `x_low` attribute bits and `r1-r4` entropy strings. Full rate adds Enhancement strings. Serialized PyTorch payload containers and geometry support are excluded from reported rate. The containers hold model-specific tensors and require the known geometry support; they are trusted evaluation artifacts, not standalone interoperable compressed streams.

Quality is measured using MPEG `pc_error`: Y, U, V, and `YUV611=(6Y+U+V)/8`. Both methods use the same inputs, preprocessing, denominator, and metric implementation.

CTC uses deterministic KD-tree chunks of at most 800,000 points. Coordinates stay global, without recentering, requantization, or deduplication. Both codecs use the same chunk manifest. Bits and codec runtime are summed across chunks; reconstruction quality is measured after merging chunks on the canonical full sample.

## BD-BR protocol

For each sample, method endpoint, and metric, the analysis retains Pareto-efficient measured RD points. Y and YUV611 are processed independently. PCHIP interpolates `ln(physical_bpp)` against quality over the shared quality interval, with no extrapolation. Dataset summaries are equal-sequence arithmetic means. All measured raw points remain in the raw evidence and figures; the frontier audit records retained and dominated points.

## Runtime

Codec time is CUDA-synchronized. Original Unicorn timing covers its physical encoder and decoder calls. Ours Base encode is prefix encoding; Base decode is prefix decoding plus BaseSynthesis. Ours Full encode adds Enhancement encoding; Full decode adds Enhancement decoding to the Base decode path. Input and checkpoint loading, PLY writing, metrics, and CPU validation copies are outside codec timing. Runtime summaries retain GPU model and VRAM as grouping keys.

For CTC, `prepare_ctc_formal_chunks.py` verifies the upstream KD-tree partition source and produces a source/chunk manifest. `scalable_attribute.evaluate --chunk-manifest ...` independently encodes and decodes each chunk, writes a per-chunk record, and emits `aggregate_command.txt` for `aggregate_formal_chunk_results.py` to merge reconstructions and calculate full-sample metrics.

See [results/final](../results/final/README.md) for raw points, final BD-BR tables, figures, the frontier audit, and runtime evidence.

The runtime is verified within each tested backend. CPU and RTX 3080 results on the synthetic demo are not bit-exact interchangeable: Full payloads differ by one byte and decoded feature differences are measurable. Original-stage and exported models agree exactly within the same backend, and fresh-process decoding succeeds on that backend. Cross-device payload interchange has not been certified; these findings do not modify the frozen GPU measurements.

## Running evaluation

Datasets are not bundled. Seven self-contained pretrained checkpoints are required external delivery assets; they are available on the [GitHub Release](CHECKPOINTS.md), with GitLab delivery pending a target repository. With an authorized, hash-verified checkpoint, run the command in the repository README. Its format is defined by `scalable_attribute/runtime/checkpoints.py`: point/lambda, Prefix configuration, BaseSynthesis configuration, and the complete `model_state` are required. The seven formal state identifiers are in [checkpoints.json](../configs/checkpoints.json).

The shipped 512-point synthetic example exercises Base/Full reconstruction and reports physical bits and optional `pc_error` metrics. The output directory must be new. To evaluate CTC, prepare its deterministic chunks, run `scalable_attribute.evaluate --chunk-manifest`, and execute the emitted `aggregate_command.txt`. Each model loads independently of historical training directories and Original Unicorn weight files.

Result analysis works without checkpoints: the plotting and BD-BR scripts consume `results/final/evidence/raw_rd_points.csv`. The frozen result files can be checked with `(cd results/final && sha256sum -c SHA256SUMS)`.
