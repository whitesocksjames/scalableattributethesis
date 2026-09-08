# Compatibility command paths

This directory is not the preferred reading path. Its Python files preserve
historically recorded `scripts/scalable_attribute/canonical/...` CLI commands
and forward unchanged arguments to implementations in:

- `../training/`
- `../evaluation/`
- `../diagnostics/`
- `../historical/`

Use [the script index](../README.md) to locate current implementations. Keep a
wrapper here while a retained Slurm command, document, or experiment record may
still invoke its old path.
