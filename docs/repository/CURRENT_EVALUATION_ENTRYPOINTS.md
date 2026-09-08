# Current evaluation entry points

| Input / endpoint | Entry point |
| --- | --- |
| H5 Base+Full physical hard (RWTT-28Lite / existing Full28 contract) | [evaluate_scalable_formal.py](../../scripts/scalable_attribute/evaluation/evaluate_scalable_formal.py) |
| External PLY sequences, Base/Full including joint | [evaluate_8ivfb_sequence.py](../../scripts/scalable_attribute/evaluation/evaluate_8ivfb_sequence.py) (also existing Owlii prepared input) |
| Canonical Base-only physical | [evaluate_base_formal.py](../../scripts/scalable_attribute/evaluation/evaluate_base_formal.py) |
| Rescued Base physical | [evaluate_base_rescue.py](../../scripts/scalable_attribute/historical/evaluate_base_rescue.py), [arm wrapper](../../scripts/scalable_attribute/historical/evaluate_base_rescue_arm.py) |
| Tensor distortion / training validation | [evaluate_base.py](../../scripts/scalable_attribute/evaluation/evaluate_base.py) |
| Existing locally reproduced Unicorn reference | [evaluate_unicorn_reference.py](../../scripts/scalable_attribute/evaluation/evaluate_unicorn_reference.py) |

Names do not establish provenance: the existing reference evaluator imports this
repository's native implementation. It is not yet the proposed independent
upstream workspace runner. Author-provided CSV, local reproduction and ours must
remain distinguishable.

Check [candidate registry](CURRENT_OPERATING_POINTS.md) before selecting a loader.
Joint 4K needs the complete scalable state, target conditioning 4096 and actual
32k8k profile; any bootstrap Base lambda8192 is initialization metadata. 2K needs
the rescued Prefix+BaseSynthesis as well as selected Enhancement.

Keep metric/bitrate/input conventions aligned with upstream. Tensor PSNR used in
training is not interchangeable with pc_error Y/U/V and weighted YUV611.
Soft/hard Base differences are diagnostic; residual/Enhancement hard round-trip
and Base/Full accounting are separate checks. No container-overhead addition is
introduced here.

Pass frozen manifests and actual paths explicitly. Each codec process needs its
own working directory for upstream temporary files. N30 persistent files stay
under `/data/run01/scz0ade/Tanzeyu/`; job-local staging follows its guide. Existing
Slurm templates are recipe-specific, not authorization to relaunch experiments.

No evaluation was submitted by Phase 1. Regression tests, independent upstream
workspace and new comparison runs belong to later reviewed phases.

The old root and `canonical/` paths remain CLI compatibility wrappers. New
invocations should use `evaluation/`; aggregation, plotting, and probes are
classified under `diagnostics/`. See the
[script index](../../scripts/scalable_attribute/README.md).
