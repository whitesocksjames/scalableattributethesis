# HPC Workspace

日常 submit、query、logs、cancel、retry 和 result 操作请使用中文手册：[HPC_GUIDE.md](HPC_GUIDE.md)。本文件只记录稳定的 workspace 与 TinyGPU rules。

The official HPC workspace is:

```text
$WORK/scalable_attribute_thesis/
├── datasets/
├── checkpoints/
├── experiments/
└── logs/
```

- The local repository is the source of truth. Training scripts will be uploaded to the HPC later.
- Do not assume that local paths and HPC paths are identical. Paths must be configured explicitly for each environment.
- Do not place datasets, checkpoints, experiment outputs, or logs inside the source tree.

## TinyGPU rules verified on this account

- Submit with `sbatch.tinygpu` (equivalent to `sbatch -M tinygpu`).
- Query the live queue with `squeue.tinygpu`.
- Query accounting with `sacct -M tinygpu`; there is no `sacct.tinygpu` wrapper.
- Cancel with `scancel -M tinygpu`.
- Do not pass `--mem` for these GPU jobs; the cluster rejected it in a previous job.

Default resource profiles are kept only in `scripts/hpc/resource_profiles.py`:

```text
train:  partition=v100          gres=gpu:v100:1       cpus=8  time=1-00:00:00
train6: partition=v100          gres=gpu:v100:1       cpus=8  time=06:00:00
smokeV: partition=v100          gres=gpu:v100:1       cpus=4  time=00:30:00
eval:   partition=work,rtx3080  gres=gpu:rtx3080:1    cpus=8  time=04:00:00
smoke:  partition=work,rtx3080  gres=gpu:rtx3080:1    cpus=8  time=00:30:00
```

There is no automatic GPU fallback. A100 or RTX2080Ti use requires a deliberate,
explicit resource profile after its exact request has been verified.

## Local-first sweep workflow

Edit `scripts/hpc/sweep_spec.py` locally, then use the thin entry point:

```bash
python scripts/hpc/remote_submit.py submit-train --study STUDY
python scripts/hpc/remote_submit.py submit-eval --study STUDY
python scripts/hpc/remote_submit.py queue
python scripts/hpc/remote_submit.py result STUDY/EXPERIMENT
python scripts/hpc/remote_submit.py logs STUDY/EXPERIMENT --follow
python scripts/hpc/remote_submit.py cancel STUDY/EXPERIMENT
python scripts/hpc/remote_submit.py retry STUDY/EXPERIMENT --resume
```

Submission syncs source code through SSH alias `tinyx` unless `--no-sync` is
given. Dataset, checkpoints, experiments, logs, upstream results, and `.git` are
excluded. If sync or SSH fails, no Slurm submission is attempted. Persistent
`.sbatch` files are not generated; jobs use `sbatch.tinygpu --wrap`.

Each run is located deterministically at:

```text
$WORK/scalable_attribute_thesis/experiments/STUDY/EXPERIMENT/
├── manifest.json
├── train/
│   ├── resolved_args.json
│   ├── enhancement_config.json
│   ├── command.txt
│   ├── metrics.csv
│   └── checkpoints/step_N.pth
├── eval/step_N/{resolved_args.json,command.txt,metrics.csv}
└── slurm/{train,eval}_attemptN_JOBID.{out,err}
```

`latest` is only a selection alias. Before an eval is submitted it resolves to
an existing numbered checkpoint, and output is written under `eval/step_N/`.
The manifest is a lightweight pointer to jobs, commands, logs, and eval outputs;
the complete parameters and metrics remain in their existing files.

Retry is always user-explicit. `NODE_FAIL` and `TIMEOUT` can be resubmitted with
the same command and optional `--resume`; attempts remain under the same
experiment. OOM, NaN, semantic/data/code errors never trigger parameter changes,
GPU fallback, or automatic retry.

The HPC layer treats `train.py` and `evaluate.py` as black-box CLIs. Experiment
dict keys are forwarded generically (`some_parameter` becomes
`--some-parameter`). Architecture/name/inheritance choices belong in
`sweep_spec.py`; resource settings belong only in `resource_profiles.py`.
