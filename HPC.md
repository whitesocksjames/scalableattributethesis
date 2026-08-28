# HPC Workspace

本文件记录 project-level compute policy。N30操作见 [N30_GUIDE.md](N30_GUIDE.md)，FAU TinyGPU操作见 [HPC_GUIDE.md](HPC_GUIDE.md)。

## Dual-cluster compute policy

```text
PRIMARY:   N30R3 / RTX3090
           training, architecture screening, ablation, short sweeps

SECONDARY: FAU TinyGPU
           formal evaluation, released reference, overflow, continuity checks
```

- Training默认不再表示 FAU V100；默认训练平台是 N30 RTX3090。
- 一个 independent experiment默认使用一张3090，不启用DDP。
- 多组实验可以是独立单卡jobs，但concurrency必须显式配置；不得自动占满8张GPU。
- Final formal evaluation必须保持固定且记录清楚的 evaluation environment。不要把不同cluster/GPU产生的结果静默混为同一 protocol。
- FAU V100/RTX3080规则只是 TinyGPU-specific policy，不是全项目默认资源策略。

## N30R3 workspace

N30上本 thesis 的所有 source、dataset、environment、cache、checkpoint、log 和 output只能位于：

```text
/data/run01/scz0ade/Tanzeyu/
```

禁止进入、查看、枚举、修改、删除、移动、同步、清理或依赖共享账号中其他用户的个人目录。任何recursive/destructive command必须先验证规范化后的exact target严格位于`/data/run01/scz0ade/Tanzeyu/`内部；无法证明时必须停止。详细规则、资源、module、Slurm、`/dev/shm`和已验证环境见 [N30_GUIDE.md](N30_GUIDE.md)。

N30不复用下面的 TinyGPU scheduler adapter。后续如实现remote submission，应使用独立、很薄的 `scripts/n30/`。

## FAU TinyGPU workspace

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

## FAU TinyGPU rules verified on this account

- Submit with `sbatch.tinygpu` (equivalent to `sbatch -M tinygpu`).
- Query the live queue with `squeue.tinygpu`.
- Query accounting with `sacct -M tinygpu`; there is no `sacct.tinygpu` wrapper.
- Cancel with `scancel -M tinygpu`.
- Do not pass `--mem` for these GPU jobs; the cluster rejected it in a previous job.

FAU-specific resource profiles are kept only in `scripts/hpc/resource_profiles.py`:

```text
train:  partition=v100          gres=gpu:v100:1       cpus=8  time=1-00:00:00
train6: partition=v100          gres=gpu:v100:1       cpus=8  time=06:00:00
smokeV: partition=v100          gres=gpu:v100:1       cpus=4  time=00:30:00
eval:   partition=work,rtx3080,v100,a100  gres=gpu:1   cpus=8  time=02:00:00
smoke:  partition=work,rtx3080,v100,a100  gres=gpu:1   cpus=8  time=00:30:00
```

FAU evaluation and smoke jobs use `partition=work,rtx3080,v100,a100` with
`gres=gpu:1`, allowing Slurm to assign RTX2080Ti, RTX3080, V100, or A100.
Historical/explicit FAU training may remain typed V100 when continuity with an
existing V100 protocol is required. New project-default training belongs on N30.
Submitted jobs are never migrated silently; changing a pending request requires
explicit cancel and resubmit.

## FAU TinyGPU local-first workflow

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

## Cross-cluster experiment metadata

Every new N30 or FAU experiment must be traceable to its execution environment.
The lightweight manifest/runtime evidence must record:

```text
cluster, hostname/node, GPU model, GPU count, git commit,
initialization checkpoint, dataset split/manifest, environment fingerprint,
elapsed time, GPU-hours, optional estimated cost
```

Runtime facts such as hostname and GPU model must be collected inside the job.
These fields are a shared experiment contract; cluster-specific scheduler flags
remain in their separate thin adapters.
