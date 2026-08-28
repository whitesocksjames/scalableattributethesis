# N30R3 Thesis 使用指南

本指南只适用于宁夏超算 N30R3 上的 Unicorn scalable attribute thesis 工作。项目级 compute policy 见 [HPC.md](HPC.md)，FAU TinyGPU 的操作见 [HPC_GUIDE.md](HPC_GUIDE.md)。

## 1. 平台角色与绝对路径规则

N30R3 / RTX3090 是本项目的 primary training、architecture screening 和 ablation 平台。当前使用实验室共享 N30 account，费用可暂按约 4 RMB/GPU-hour 估算；该数字仅用于预算参考，不作为账单依据。

N30 上所有 persistent project files，以及共享文件系统上的 project 操作，只能位于：

```text
/data/run01/scz0ade/Tanzeyu/
```

包括 source、dataset、environment、cache、checkpoint、log 和 experiment output。禁止进入、修改、清理或依赖共享账号中其他用户的个人目录，也不要把项目文件写入 `$HOME`、用户级 cache 或 shell profile。

唯一例外是 Slurm compute job 内可以使用 job-local `/dev/shm` 临时目录暂存 dataset。该例外只适用于第 6 节定义的目录和用途，不允许扩展到 source、checkpoint、metrics、logs 或 final outputs。

### Shared-account safety rule

N30 account由多名实验室用户共享。本 thesis 在共享文件系统上的全部 read、write、modify、delete 操作必须严格限制在：

```text
/data/run01/scz0ade/Tanzeyu/
```

不得进入、查看、枚举、修改、删除、移动、重命名、同步、清理或依赖共享账号下任何其他用户的个人目录。即使操作是只读调查，也不能把其他用户目录作为搜索范围。Slurm compute job 对自身 `/dev/shm/Tanzeyu_${SLURM_JOB_ID}/` 的临时操作是唯一例外。

对共享文件系统执行任何 destructive或recursive command之前，包括 `rm`、`find -delete`、`rsync --delete`、cleanup script和cache cleanup，必须先解析并验证 exact target：

1. target必须是绝对路径；
2. 规范化后的路径必须严格位于 `/data/run01/scz0ade/Tanzeyu/` 内；
3. target不得是 `/data/run01/scz0ade/Tanzeyu/` 根本身；
4. 不使用未验证的 environment variable、glob、symlink或command substitution决定删除范围；
5. 无法证明target满足以上条件时立即停止，不执行命令。

对 `/dev/shm` 的清理不套用上述 Tanzeyu路径前缀，但 target必须精确等于本 job创建的 `/dev/shm/Tanzeyu_${SLURM_JOB_ID}/`；不得清理 `/dev/shm` 根目录、使用宽泛 glob或触及其他 job的目录。

禁止对共享账号目录使用宽泛的 recursive scan或cleanup。Source sync默认不得使用 `--delete`；如以后确需使用，必须先对解析后的 Tanzeyu内部目标做只读核验，并确保不会涉及dataset、checkpoint、environment、cache或experiment output。

当前布局：

```text
/data/run01/scz0ade/Tanzeyu/
├── code/Scalable-Attribute-Thesis/
├── data/scalable_attribute_thesis/
│   ├── datasets/
│   ├── checkpoints/
│   └── experiments/              # 已迁移的稳定 artifacts
├── envs/unicorn-me-py38/
├── .cache/
│   ├── pip/
│   └── torch_extensions/
├── experiments/                  # N30 runs 与 environment evidence
├── scratch/
└── src/                          # dependency source used for builds
```

本机 repository 仍是 source of truth。同步 source 时不得同步或覆盖 N30 的 dataset、checkpoint、environment、cache 和 experiment output。

## 2. N30R3 资源规则

与 thesis 相关的节点资源：

```text
GPU:              NVIDIA RTX3090, 24 GB
GPU per node:     8
Allocation unit:  1 GPU = 6 CPU cores + 60 GB RAM
Slurm partition:  gpu
```

默认一个 independent experiment 使用一张 GPU，不启用 DDP。小型 sweep 可以提交多个独立单卡 jobs，但 concurrency 必须由用户显式设置；工具不得自动占满一个节点的 8 张 GPU，也不得 silent fallback 或自动扩卡。

建议策略：

- 单次 architecture/training run：1 GPU。
- 小型 sweep：多个 independent 1-GPU jobs。
- 默认 concurrency：必须显式配置，不在工具中假设 8。
- 需要多卡/DDP 时：作为单独实验设计审核，不从普通 sweep 自动推导。

## 3. Slurm 与 queue

提交使用标准 Slurm `sbatch`，查询可使用 `parajobs` 或 Slurm命令，取消使用 `scancel`。以当前 N30 实际配置为准，不把 FAU 的 `sbatch.tinygpu`、`squeue.tinygpu`、partition 或 GRES规则带到 N30。

```bash
# submit
sbatch job.sbatch

# user-friendly queue/status supplied by platform
parajobs

# standard queue/accounting
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList

# cancel
scancel JOB_ID
```

当前账号是实验室共享 N30 account；这不表示它属于手册所述的包节点专属 `vip_gpu_*` queue。实际 partition/queue 以当前 `sinfo`、`scontrol` 和成功 job evidence为准。不要假设必须使用 `vip_gpu_*`，也不要把费用、可用卡数或排队时间写成 scheduler fallback逻辑。

## 4. 正式 module/environment 初始化

Compute-node batch shell需要先初始化 Environment Modules：

```bash
source /etc/profile.d/modules.sh
module purge
module load miniforge/24.1.2
module load gcc/11.2
module load cuda/11.7

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scz0ade/Tanzeyu/envs/unicorn-me-py38

export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR=/data/run01/scz0ade/Tanzeyu/.cache/torch_extensions
export LD_LIBRARY_PATH=/data/apps/openblas/0.3.22/lib:${LD_LIBRARY_PATH:-}
```

不要修改 `~/.bashrc` 或依赖 interactive shell的隐式初始化。上述内容以后应由 project-local N30 activation/Slurm thin wrapper提供。

## 5. 已验证 environment

2026-08-28 已在 N30 RTX3090 compute node验证：

```text
Python                  3.8.20
PyTorch                 1.13.1+cu117
torch CUDA              11.7
MinkowskiEngine         0.5.4
PyTorch3D               0.7.2
torchac                 0.9.3
GPU                     RTX3090 24 GB
```

MinkowskiEngine 使用 official NVIDIA source exact commit：

```text
02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
```

PyTorch3D 0.7.2 使用 official tag commit：

```text
3145dd4d16edaceb394838364b8e87a440f83c10
```

最终 smoke：

```text
Job ID:       5434102
Node:         g0151
State:        COMPLETED
Exit code:    0:0
```

PASS项目包括 torch CUDA、MinkowskiEngine CUDA convolution、PyTorch3D CUDA operation、torchac actual round-trip、scalable training entrypoint import 和真实 RWTT H5读取。证据保存在：

```text
/data/run01/scz0ade/Tanzeyu/experiments/environment_bringup/
```

Open3D没有安装。Processed-H5 training不依赖其功能；Open3D-backed PLY函数采用 lazy import，只有实际调用时才要求 Open3D。

## 6. Dataset staging 到 `/dev/shm`

`/dev/shm` 是 compute job所在节点的 node-local temporary storage，可用于减少大量小 H5 的共享文件系统 I/O。它不是永久存储：job结束或节点回收后内容会消失。

使用原则：

- 永久 dataset仍保存在 `/data/run01/scz0ade/Tanzeyu/data/`。
- staging是 optional optimization，不是 training的前置要求；第一版可以直接从 `/data` 读取。
- 仅在 Slurm compute job内部把该 job需要的数据 staging到：

  ```text
  /dev/shm/Tanzeyu_${SLURM_JOB_ID}/
  ```

- 不使用 `/dev/shm/$USER/`：这是 shared account，不同用户或 jobs可能具有相同 `$USER`。
- `/dev/shm` 只用于临时 dataset staging，不放 source、checkpoint、metrics、logs 或 final outputs。
- staging成功后才启动训练；失败时 fail fast，不回写或删除永久 dataset。
- checkpoint、metrics、command、manifest和最终结果始终写回 Tanzeyu experiment root。
- job结束时只允许清理本 job明确创建的 `/dev/shm/Tanzeyu_${SLURM_JOB_ID}/`；不得使用宽泛 glob、模糊变量或 recursive cleanup影响任何其他目录。

只有 DataLoader throughput或 GPU utilization显示 I/O bottleneck时才启用 staging。判断时只需做简单的 direct-read/staged-read A/B measurement，不建立 benchmark framework。

## 7. Experiment output 与 manifest contract

每个 experiment使用稳定 output root，例如：

```text
/data/run01/scz0ade/Tanzeyu/experiments/STUDY/EXPERIMENT/
├── manifest.json
├── train/
│   ├── resolved_args.json
│   ├── command.txt
│   ├── metrics.csv
│   └── checkpoints/
├── eval/
└── slurm/
```

Manifest保持 lightweight pointer，但 N30 submission/runtime必须记录或补齐：

```text
cluster
hostname/node
GPU model
GPU count
git commit (允许同时标记 dirty worktree)
initialization checkpoint
dataset split/manifest
environment fingerprint
elapsed time
GPU-hours
estimated cost (optional)
```

其中 runtime-only字段（node/GPU/elapsed）应由 job实际运行时采集，不能由提交端猜测。`GPU-hours = elapsed_hours × GPU_count`；estimated cost只在明确提供费率时计算。

## 8. N30 thin tooling 规划

不要把 N30塞进现有 `scripts/hpc/resource_profiles.py`；该目录已经是 FAU TinyGPU-specific adapter。第一版只规划：

```text
scripts/n30/
├── activate.sh           # project-local module/conda/cache初始化
└── job_template.sbatch   # 简单的单卡 job模板
```

日常操作直接使用 `sbatch`、`parajobs`、`squeue`、`sacct` 和 `scancel`。不创建 scheduler abstraction、resource framework、database、自动 retry、自动占满8卡或 automatic GPU selection。只有后续确实出现大量重复 sweep submission时，再考虑增加一个很薄的 `submit_sweep.py`；它仍只负责 CLI forwarding和提交，不理解 architecture或 training semantics。

本轮只冻结以上规划，尚未实现 `scripts/n30/`。
