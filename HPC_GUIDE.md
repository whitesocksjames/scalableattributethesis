# Thesis HPC 中文使用手册

这份手册用于日常操作。通常不需要手工登录 HPC：在本机修改 experiment spec 后，通过 SSH alias `tinyx` 完成 source sync、Slurm submission、状态查询、日志查看、取消和 retry。

稳定的 workspace 和 TinyGPU 规则见 [HPC.md](HPC.md)。

## 1. 路径分工

本机 repository 是 source of truth：

```text
/home/liltan/projects/Scalable-Attribute-Thesis
```

HPC 上的 source mirror：

```text
$HOME/Scalable-Attribute-Thesis
```

正式 data 和 experiment workspace：

```text
$WORK/scalable_attribute_thesis/
├── datasets/
├── checkpoints/
├── experiments/
└── logs/
```

Source sync 只传 code/scripts，不传 RWTT、checkpoints 和 experiment outputs。不要假设本机路径与 HPC 路径相同，也不要把 data 或 outputs 放入源码目录。

## 2. 配置一个短命令

先在本机 terminal 执行：

```bash
export THESIS_REPO=/home/liltan/projects/Scalable-Attribute-Thesis
thpc() { python "$THESIS_REPO/scripts/hpc/remote_submit.py" "$@"; }
```

以后可以直接使用：

```bash
thpc queue
thpc result STUDY/EXPERIMENT
thpc logs STUDY/EXPERIMENT --follow
```

如果希望每次打开 terminal 都能使用，可以手工把上面的两行加入 `~/.bashrc`。这只是 shell shortcut，不保存 experiment defaults。

不配置短命令也可以使用完整入口：

```bash
cd /home/liltan/projects/Scalable-Attribute-Thesis
python scripts/hpc/remote_submit.py queue
```

## 3. 一键提交 Training Sweep

Experiment parameters 集中修改：

```text
scripts/hpc/sweep_spec.py
```

先 dry-run：

```bash
thpc submit-train --study STUDY_NAME --dry-run
```

需要查看完整 resolved CLI 和 Slurm command 时：

```bash
thpc submit-train --study STUDY_NAME --dry-run --verbose
```

使用默认 V100 profile 正式提交：

```bash
thpc submit-train --study STUDY_NAME
```

指定约 6 小时的 V100 profile：

```bash
thpc submit-train --study STUDY_NAME --profile train_v100_6h
```

默认会先同步本机 source，再把每个 experiment 作为独立 Slurm job 提交。一个 submission failure 不会取消其他 experiments。

只提交一个 experiment：

```bash
thpc submit-train --study STUDY_NAME --only EXPERIMENT_NAME
```

只有已经明确同步过相同代码时，才加 `--no-sync`。

### Smoke Gate

让所有 training 仅在 smoke 成功后 eligible：

```bash
thpc submit-train --study STUDY_NAME --afterok SMOKE_JOB_ID
```

它会建立 `afterok:SMOKE_JOB_ID` dependency。Smoke `COMPLETED` 后 training 才能运行；smoke 失败时 training 不运行。各 training jobs 之间没有 dependency。

## 4. Hard Evaluation

先在 `sweep_spec.py` 中明确配置 `EVAL_EXPERIMENTS` 和 evaluation dataset，再运行：

```bash
thpc submit-eval --study STUDY_NAME --dry-run
thpc submit-eval --study STUDY_NAME
```

默认使用 RTX3080 hard-eval profile。添加 training dependency：

```bash
thpc submit-eval --study STUDY_NAME --afterok TRAIN_JOB_ID
```

也可以使用已经实现的 optional train-to-eval flow：

```bash
thpc submit-train --study STUDY_NAME --submit-eval-after-train
```

调用链是：

```text
V100 training
    └── afterok
        └── RTX3080 hard evaluation
```

`latest` 只用于选择 checkpoint。提交 eval 时会先解析到实际文件，例如 `step_3525.pth`，结果写入 `eval/step_3525/`；以后 `latest` 改变也不会覆盖旧结果。

开发和调参 validation 使用 RWTT Validation，最终 thesis Test 使用外部 8iVFB。没有明确 eval dataset 时应 fail fast，不允许 fallback 到完整 RWTT training root。

## 5. Query、Result 和 Logs

查看 TinyGPU queue：

```bash
thpc queue
```

查看 experiment 状态：

```bash
thpc status STUDY/EXPERIMENT
```

快速定位 checkpoint、metrics、hard-eval CSV、Job ID 和 output directory：

```bash
thpc result STUDY/EXPERIMENT
```

查看 training stdout：

```bash
thpc logs STUDY/EXPERIMENT
```

持续跟踪 stdout：

```bash
thpc logs STUDY/EXPERIMENT --follow
```

查看 stderr：

```bash
thpc logs STUDY/EXPERIMENT --stream stderr
```

查看某个 numbered checkpoint 的 eval log：

```bash
thpc logs STUDY/EXPERIMENT --kind eval --tag step_3525
```

常见 Slurm 状态：

| State / Reason | 含义 | 应对方式 |
|---|---|---|
| `PENDING (Priority)` | 等待调度优先级 | 正常排队 |
| `PENDING (Resources)` | 等待指定资源 | 正常排队，不换 GPU |
| `PENDING (Dependency)` | 等待 `afterok` job | 检查前置 job |
| `RUNNING` | 正在运行 | 查看日志或等待 |
| `COMPLETED` | 正常完成 | 用 `result` 找结果 |
| `NODE_FAIL` | node/infrastructure failure | 可显式 retry/resume |
| `TIMEOUT` | 超过 walltime | 保留 checkpoint，显式 resume |
| `FAILED` | 程序或环境失败 | 先检查 stderr |
| CUDA OOM | GPU memory 不足 | FAIL，不自动降低 batch size |

## 6. Cancel 和 Retry

取消 experiment 当前记录的 train job：

```bash
thpc cancel STUDY/EXPERIMENT
```

取消指定 eval：

```bash
thpc cancel STUDY/EXPERIMENT --kind eval --tag step_3525
```

Cancel 不删除 checkpoint、metrics 或 logs。

`NODE_FAIL` 或 `TIMEOUT` 后，从已有 checkpoint 显式 resume：

```bash
thpc retry STUDY/EXPERIMENT --kind train --resume
```

需要增加 walltime 时：

```bash
thpc retry STUDY/EXPERIMENT --kind train --resume --walltime 08:00:00
```

Retry 仍归属于原 experiment，manifest 只追加新的 Job ID 和 log pointer。工具不会后台自动 retry。

以下情况不要自动 retry：CUDA OOM、NaN/Inf、non-finite gradient、coordinate/checkpoint mismatch、Python exception、dataset 或 semantic error。也不要自动换 GPU、改 batch size、改 architecture 或 silent fallback。

## 7. 手工 Source Sync

Submission 默认会先 sync。需要单独同步时：

```bash
thpc sync
```

如果 sync 或 SSH 失败，工具不会继续提交 Slurm job。

## 8. Output 统一位置

给定 `STUDY/EXPERIMENT`，相关结果统一位于：

```text
$WORK/scalable_attribute_thesis/experiments/STUDY/EXPERIMENT/
├── manifest.json
├── train/
│   ├── resolved_args.json
│   ├── enhancement_config.json
│   ├── command.txt
│   ├── metrics.csv
│   └── checkpoints/step_N.pth
├── eval/
│   └── step_N/
│       ├── resolved_args.json
│       ├── command.txt
│       └── metrics.csv
└── slurm/
    └── {train,eval}_attemptN_JOBID.{out,err}
```

`manifest.json` 是轻量 index/pointer，不重复保存完整 config 和 metrics。完整 resolved parameters 在 `resolved_args.json`，实际训练曲线和 RD 结果在对应 metrics/CSV 中。

新一轮实验使用新的 `STUDY_NAME`，不要覆盖旧 study，也不要把结果临时写散到 `$WORK` 的其他位置。

## 9. 参数修改位置

| 参数类型 | 主要位置 | 示例 |
|---|---|---|
| Architecture defaults | model/config code | hidden/latent channels 默认值 |
| Training/evaluation CLI defaults | `train.py` / `evaluate.py` | optimizer 和执行默认值 |
| Experiment overrides | `scripts/hpc/sweep_spec.py` | `rd_lambda`、`lr`、batch、steps |
| HPC resources | `scripts/hpc/resource_profiles.py` | GPU、partition、CPU、walltime |
| Workspace/SSH paths | thin remote configuration | `tinyx`、HPC code/work root |
| Base convenience profile | experiment config | released checkpoint profile/override |

HPC tooling 把 `train.py` 和 `evaluate.py` 当作 black-box CLI，并把 generic key-value 转为 `--key value`。新增 model/training parameter 时，通常只改 model/entry point 和 `sweep_spec.py`；不要让 architecture semantics 渗入 scheduler layer。

## 10. Resource Profiles

| Profile | Partition | GRES | CPUs | Walltime | 用途 |
|---|---|---|---:|---:|---|
| `train_v100` | `v100` | `gpu:v100:1` | 8 | 1 day | 长 training |
| `train_v100_6h` | `v100` | `gpu:v100:1` | 8 | 6 hours | 约一 epoch training |
| `smoke_v100` | `v100` | `gpu:v100:1` | 4 | 30 min | V100 memory/throughput smoke |
| `eval_rtx3080` | `work,rtx3080` | `gpu:rtx3080:1` | 8 | 4 hours | hard evaluation |
| `smoke_rtx3080` | `work,rtx3080` | `gpu:rtx3080:1` | 8 | 30 min | RTX3080 smoke |

默认 policy：training 使用 V100；hard eval/smoke 使用 typed RTX3080 request。A100/2080Ti 只允许 explicit manual override，不做 automatic fallback。

## 11. 原生 TinyGPU 命令（排障备用）

日常优先使用 `thpc`。只有 wrapper 信息不足时才直接使用：

```bash
ssh tinyx 'squeue.tinygpu -u "$USER"'
ssh tinyx 'sacct -M tinygpu -j JOB_ID --format=JobID,State,ExitCode,Elapsed'
ssh tinyx 'scancel -M tinygpu JOB_ID'
```

经当前账户验证：submission 使用 `sbatch.tinygpu`，queue 使用 `squeue.tinygpu`，accounting 使用 `sacct -M tinygpu`，cancel 使用 `scancel -M tinygpu`。不要凭记忆换 flags，也不要为这些 GPU jobs 添加 `--mem`。

## 12. 最常用命令速查

```bash
# queue
thpc queue

# dry-run sweep
thpc submit-train --study STUDY_NAME --dry-run

# submit V100 training
thpc submit-train --study STUDY_NAME --profile train_v100_6h

# experiment result
thpc result STUDY_NAME/EXPERIMENT_NAME

# follow log / inspect stderr
thpc logs STUDY_NAME/EXPERIMENT_NAME --follow
thpc logs STUDY_NAME/EXPERIMENT_NAME --stream stderr

# cancel / infrastructure retry
thpc cancel STUDY_NAME/EXPERIMENT_NAME
thpc retry STUDY_NAME/EXPERIMENT_NAME --kind train --resume
```

底线：不生成 persistent `.sbatch` 文件，不在 resource layer 保存 model parameters，不自动换卡或改变实验条件，不把 dataset/checkpoint/output 放进 source directory。
