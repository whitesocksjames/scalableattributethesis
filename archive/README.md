# Archived material

Physical cleanup, 2026-09-07. This is storage, not an active import namespace.

| Original paths | Destination | Role / dependency finding | Replacement / recovery |
| --- | --- | --- | --- |
| `dynamic_attribute/`, `dynamic_geometry/`, `dynamic_geometry_lidar/`, `lossless_attribute/`, `lossless_geometry/`, `lossy_geometry/` | `upstream_out_of_scope/<original-directory>/` | Non-current upstream modes; imports between these directories exist. Static lossy Attribute/canonical runtime does not import them. | Current method remains in `lossy_attribute/` + `scalable_attribute/`. For historical execution restore original layout from snapshot. |
| `results.zip`, `scalable-geometry-roi-channel_split_lambda.zip` | `packages/<original-filename>` | Packaged reference material, not current result outputs or runtime imports | Retain original bytes locally; no assumption that ignored ZIPs exist in Git history. |
| `scripts/hpc/submit_train_sweep.py`, `scripts/hpc/submit_eval_sweep.py` | `legacy_hpc/` | Retired sweep wrappers target missing `scripts/scalable_attribute/train.py` and `evaluate.py`; their only source caller was `remote_submit.py`. | Current canonical entrypoints are unchanged. Old submit commands now reject before sync/SSH. Restore original layout from snapshot for historical inspection. |
| Eight `scripts/diagnostics/` fixed-model trace files listed below | `diagnostics/fixed_model_internal_trace/` | Completed fixed-model diagnosis; only executable caller is its own historical Slurm template. Current training/evaluation has no imports of these tools. | Retained reports/raw outputs are unchanged. All eight are in the recovery snapshot; restore original layout for historical execution. No relocated launcher is advertised as runnable. |

Batch 2 trace filenames: `fixed_model_internal_trace.py`,
`summarize_fixed_model_internal_trace.py`, `build_internal_trace_rwtt_manifest.py`,
`fixed_model_family_probes_n30.json`, `fixed_model_rwtt_trace_n30.json`,
`fixed_model_internal_trace_n30.json`, `fixed_model_internal_trace_external_n30.tsv`,
`n30_fixed_model_internal_trace.sbatch`. Bytes are unchanged, including historical
paths. `scripts/diagnostics/build_multirate_256_32k_audit.py` remains active;
the two content-statistics tools remain in place for possible reuse.

Recovery snapshot: `audit/pre-finalization-20260906` at
`997643570834e0fa3e6b280d37cddda985243539`. The six tracked source directories
are recoverable there. The archived source is not promised runnable from this
new location: no import or path semantics were rewritten. Package archives may
be ignored by Git and are retained locally at their new paths.

No model, current experiment output, checkpoint, dataset, or historical evidence
table is deleted. Original/current `lossy_attribute`, `basic_models`, `cfg`,
`data_utils`, `third_party`, `pipelines`, canonical runtime and result paths stay
in place. `docs/repository/inventory.csv` records the pre-cleanup inventory;
apply the relocation table above when resolving these archived entries.
