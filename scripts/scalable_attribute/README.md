# Scalable attribute command-line tools

Use these directories as the reading and invocation entry points:

| Directory | Responsibility |
| --- | --- |
| `training/` | Current Base, Enhancement, and joint endpoint training commands |
| `evaluation/` | Current Base/Full, external-sequence, and Unicorn-reference evaluators |
| `diagnostics/` | Probes, audits, aggregation, screening, and plotting utilities |
| `historical/` | Reproduction entry points for retained rescue and adaptation experiments |
| `canonical/` | Compatibility wrappers for previously recorded command paths |

The wrappers in `canonical/` preserve their existing CLI arguments, defaults,
outputs, and behavior by forwarding to the classified implementation. New
commands and documentation should use the classified paths above.

Model architecture lives in `scalable_attribute/models/`; reusable training,
evaluation, and checkpoint support lives in the corresponding package
directories. This script tree contains executable entry points, not model
definitions.
