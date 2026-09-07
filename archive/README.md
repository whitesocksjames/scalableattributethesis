# Archived material

Physical cleanup, 2026-09-07. This is storage, not an active import namespace.

| Original paths | Destination | Role / dependency finding | Replacement / recovery |
| --- | --- | --- | --- |
| `dynamic_attribute/`, `dynamic_geometry/`, `dynamic_geometry_lidar/`, `lossless_attribute/`, `lossless_geometry/`, `lossy_geometry/` | `upstream_out_of_scope/<original-directory>/` | Non-current upstream modes; imports between these directories exist. Static lossy Attribute/canonical runtime does not import them. | Current method remains in `lossy_attribute/` + `scalable_attribute/`. For historical execution restore original layout from snapshot. |
| `results.zip`, `scalable-geometry-roi-channel_split_lambda.zip` | `packages/<original-filename>` | Packaged reference material, not current result outputs or runtime imports | Retain original bytes locally; no assumption that ignored ZIPs exist in Git history. |

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
