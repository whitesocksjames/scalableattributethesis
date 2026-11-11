# Evaluation support

The main inference interface is `python -m scalable_attribute.evaluate`.

- `evaluation/prepare_ctc_formal_chunks.py` preserves the deterministic CTC preprocessing contract.
- `evaluation/aggregate_formal_chunk_results.py` validates and merges chunk reconstructions before full-sample metrics.
- `analysis/recompute_pareto_bdbr.py` recomputes the final BD-BR tables from raw RD points.
- `analysis/plot_final_static_rgb_rd.py` redraws the final RD figures.

See [evaluation documentation](../../docs/EVALUATION.md) for commands and rate/metric definitions.
