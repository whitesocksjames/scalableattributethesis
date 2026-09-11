# Scalable attribute command-line tools

Use these directories as the reading and invocation entry points:

| Directory | Responsibility |
| --- | --- |
| `training/` | Current Base, Enhancement, and joint endpoint training commands |
| `evaluation/` | Current Base/Full, external-sequence, and Unicorn-reference evaluators |
| `analysis/` | Final Pareto-frontier BD-BR analysis and raw-RD plotting |

Model architecture lives in `scalable_attribute/models/`; reusable training,
evaluation, and checkpoint support lives in the corresponding package
directories. This script tree contains executable entry points, not model
definitions.
