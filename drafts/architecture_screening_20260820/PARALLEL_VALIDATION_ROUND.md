# Parallel Validation Round

> 状态：进行中  
> 日期：2026-08-20  
> 本轮不做 Final Architecture Decision。

## Scope

- Track A：C2 Formal Validation，最高优先级。
- Track B：C3 stride-4 single-scale learned capacity probe。
- Track C：B1 isolated physical nested-coding feasibility/smoke。
- 不做 multi-lambda sweep，不做完整 C3，不训练 B1。

## Execution status

| Track | Step | Status | Evidence / artifact |
|---|---|---|---|
| A | C2 latent sparsity audit | PASS with warning | Job `1785450`；holdout 有 nonzero symbols，但仅 1/32 active channels |
| A | operating-point selection | selected | official R08 (`2k128`, `lambda_Base=256`)；保留明显 enhancement headroom，同时比 R09 更接近 low/mid-low working point |
| A | formal-path smoke Gate | PASS | all-GPU Job `1785276`；2-step train + 1-H5 actual hard round-trip；不自动放行 formal training |
| A | normal RWTT train/validation | human review gate | 四个 short jobs 全部汇总并经 Project Lead 明确放行后才可提交；正常 95/5 model split，不使用 2-H5 micro set |
| A | hard validation + official neighbors | pending | — |
| B | stride-4 capacity probe | FAIL | Job `1785452`；200 updates 后 4/4 H5 hard symbols 全零，Full=Base |
| C | isolated torchac cost audit | pass-to-smoke | 可由 experiment-only script 对 captured R02/r5 symbols 分层编码；无需修改 official traversal |
| C | physical smoke if lightweight | PASS | Job `1785451`；actual layered/original ratio `1.0068–1.0148`，exact nested symbol recovery |

后续所有结果、复杂度判断和最终 `GO/CONDITIONAL/FAIL` 状态继续写在本文件。

## Short-result provenance

### C2 formal-path smoke

```text
RESULT PROVENANCE

Result:
Gate PASS; Base 0.0199568 bpp, EL 0.0055693 bpp, Full 0.0255261 bpp;
Base 37.7823 dB, Full 37.4139 dB on the single hard-eval H5.

Source:
Base and EL actual hard-coded bytes; author pc_error per-H5 YUV-PSNR 6:1:1.

Data:
Training: 2 shuffled batches from the normal RWTT Train manifest, BS=1.
Hard evaluation: 1 RWTT Validation H5, RWT115/model_mesh_P0.h5, 68,949 points.

Training:
2 updates; R08 Base; rd_lambda=6500; lr=5e-5; seed=0.

Metric:
Per-H5 physical Base/EL/Full bpp and author pc_error YUV-PSNR 6:1:1.

Can be used for formal RD?
NO — path smoke only; 2 updates and one validation H5.
```

Additional hard diagnostic for this one H5:

- hard symbols: `2 / 633,408` nonzero (`3.1575e-6` fraction), range `[-1, 0]`;
- EL physical payload: `384 bits`;
- observed Full−Base: `-0.36845 dB` after only two updates;
- this negative delta is not an architecture verdict; it only confirms the complete train/checkpoint/actual-hard/metric path runs.

### Jobs 1785262 / 1785263 / 1785264

```text
RESULT PROVENANCE

Result:
NO SCIENTIFIC RESULT. Process exited before checkpoint/model execution.

Source:
Slurm stderr: torchac JIT import could not locate ninja.

Data:
None processed.

Training:
None performed.

Metric:
Not available.

Can be used for formal RD?
NO.
```

The missing PATH was an execution-environment error, not C2/C3/B1 evidence. Exact-parameter retries are `1785450 / 1785451 / 1785452`.

### C2 R09 latent sparsity audit

```text
RESULT PROVENANCE

Result:
All: 0.4142% nonzero (12,112 / 2,924,160 symbols).
Micro-train: 0.1805% nonzero.
Holdout: 0.6438% nonzero.
Only 1 / 32 latent channels active.

Source:
Rounded qE symbols recomputed from the saved C2 native-state checkpoint.

Data:
2 train-like + 2 holdout H5.

Training:
Saved R09 C2 capacity-probe checkpoint; 200 updates.

Metric:
Hard-symbol sparsity and magnitude distribution; not bitrate or RD.

Can be used for formal RD?
NO.
```

Magnitude detail over all nonzero symbols:

- `|q|=1`: 10,445 / 12,112 (`86.23%`);
- `|q|=2`: 1,398 / 12,112 (`11.54%`);
- `|q|>=3`: 269 / 12,112 (`2.22%`);
- range `[-5, 5]`.

Interpretation: holdout 不仅有 nonzero symbols，而且密度高于 micro-train，因此目前证据不支持“只记住两个 training H5”的简单 micro-overfit 解释。但 32 channels 中仅一个 channel active，是明确的 channel collapse / effective one-channel representation。四 H5 规模不足以证明 formal generalization。

### B1 isolated physical nested coding

```text
RESULT PROVENANCE

Result:
Layered/original physical-bit ratio = 1.0148 and 1.0068.
Exact qB round-trip and exact qE-conditioned recovery of original fine symbols.

Source:
Actual torchac bytes for isolated captured R02/r5 symbols.

Data:
2 representative H5: one train-like and one holdout-like sample.

Training:
None. Released R02 (32k8k, lambda=16384), native r5 captured symbols.

Metric:
Physical arithmetic payload only; no endpoint RD metric.

Can be used for formal RD?
NO — syntax feasibility evidence only.
```

Per sample:

| Sample | Original | Coarse qB | Refinement qE\|qB | Layered | Ratio |
|---|---:|---:|---:|---:|---:|
| RWT115/P0 | 6,480 bits | 3,152 | 3,424 | 6,576 | 1.0148 |
| RWT380/P15 | 120,592 bits | 70,536 | 50,880 | 121,416 | 1.0068 |

Interpretation: isolated physical nested syntax 是可行的，额外 arithmetic overhead 在这两个样本上约 `0.7–1.5%`。这不等于 independently decodable endpoint，也没有证明 Base RD。把它并入 native traversal、处理 signaling/ranges 并训练 endpoint 仍是后续中等到较高成本工作。

### C3 stride-4 capacity probe

```text
RESULT PROVENANCE

Result:
4 / 4 H5 have 100% zero hard symbols.
Hard payload = 8 bits/H5 coder minimum; about 0.00009–0.00012 bpp.
Full = Full_zero = Base within floating-point tolerance.

Source:
Actual torchac round-trip and tensor-domain reconstruction diagnostic.

Data:
2 train-like + 2 holdout H5.

Training:
200 updates; official R08 Base; rd_lambda=6500; lr=5e-5; seed=0.

Metric:
Hard-symbol activity, physical EL bpp, tensor PSNR. No author pc_error formal RD.

Can be used for formal RD?
NO — capacity falsification only.
```

Training estimated rate 从 `1.7025` 降到 `0.1491 bpp`，但 raw latent mean absolute value 始终约 `0.045`，最终 round 后全部为零。Hard evaluation 的 Full−Base 为 `0 dB`（一个样本仅有 `4.4e-7 dB` 浮点差）。按预设 Gate，这个 minimal single-scale stride-4 C3 probe 判定 `FAIL`，本轮不通过增加 layers/streams 继续救。

## Round-level status for review

| Track | Status | Meaning |
|---|---|---|
| C2 | **CONDITIONAL** | hard path PASS；holdout correction/symbol activity exists，但 effective one-channel collapse，formal generalization/RD 尚未训练验证 |
| C3 | **FAIL** | minimal stride-4 learned representation 在 200 updates 后没有任何 GT-dependent hard symbol 或 correction |
| B1 | **PHYSICAL SYNTAX PASS** | isolated nested torchac exact，physical overhead small；尚无 endpoint RD，future integration cost medium-to-high |

这是 Parallel Validation 的审核输入，不是 Final Architecture Decision。C2 one-epoch job 未提交，等待 Project Lead 明确决定。

## Evaluation aggregation blocker

- `per_h5.csv` directly records author `pc_error` channel PSNR and is valid.
- Current `aggregate_models()` receives normalized color MSE (peak `1`) but calls `psnr(..., peak=255)`.
- This adds `20 log10(255) = 48.1308 dB` to derived per-model PSNR. Example smoke per-H5 Base is `37.7823 dB`, while its one-model summary incorrectly becomes `85.9132 dB`.
- Therefore existing physical bpp remains usable, but current `per_model.csv`, `endpoint_summary.csv`, and derived reference curves must not be used as formal PSNR evidence until aggregation is corrected and re-derived from retained per-H5 CSVs.
- No frozen reference artifact has been overwritten during this round.

## Fixed execution choices

- C2 formal point：official R08，`2k128 + lambda_Base=256`。
- C2 normal training：RWTT model-level 95/5 split，Base frozen；single point only。
- C3：仅 single-scale stride-4 latent；无 stride-2 refinement、无 second stream。
- B1：仅 isolated captured-symbol syntax evidence；不训练、不侵入 `MultiscaleVAE` routing。
- 本轮所有脚本和结果均属于 experiment draft，不代表 Final Architecture Decision。

### Resource and runtime estimate

短诊断统一使用 `partition=work,rtx3080,v100,a100`、`gres=gpu:1`；不限定 100-series：

| Job | Work | Expected runtime | Requested walltime |
|---|---|---:|---:|
| `1785262` | 4-H5 C2 symbol audit | 2–6 min | 20 min |
| `1785263` | 2-H5 B1 actual torchac | 2–8 min | 20 min |
| `1785264` | C3 200-update + 4-H5 hard diagnostic | 5–12 min | 30 min |
| `1785276` | C2 2-step + 1-H5 hard smoke Gate | 3–10 min | 20 min |

若后续经人工审核放行，C2 normal RWTT one-epoch training 才使用 typed V100；预计为数小时级，需根据 smoke 实测再确定正式 walltime。

## Human review gate

- Job `1785276` PASS 只证明 C2 formal train/hard path 可运行，不自动触发 one-epoch training。
- 必须等待 `1785262 / 1785263 / 1785264 / 1785276` 四个 short jobs 全部形成统一结果汇总。
- 汇总提交给 Project Lead 审核后，只有收到明确放行指令才允许提交 C2 one-epoch job。
- 不使用 `afterok:1785276` 预先挂载正式 training job。

### Why R08 for C2 formal validation

R08 是本轮唯一正式点，不做 R08/R09 sweep。选择依据：

- R08 是 official `2k128 + lambda_Base=256`，属于 low/mid-low operating point；仍有足够 enhancement headroom。
- 既有 4-H5 C2 probe 在 R08 上已经出现 GT-dependent hard symbols、exact torchac round-trip 和 holdout correction。
- 与 R09 相比，R08 Base 稍强，可减少一次正式 generalization test 被极弱 Base 和过高 EL rate 主导的风险；与 R07 相比仍更接近本轮要验证的 low-rate scalable use case。
- 这不是 architecture recommendation，只是 single-point falsification 的实验控制选择。

Full RWTT Validation 上的 official neighboring endpoints（28-model equal average）：

| Point | Profile | lambda | mean bpp | author YUV-PSNR 6:1:1 |
|---|---|---:|---:|---:|
| R07 | 2k128 | 512 | 0.152390 | 80.654405 dB |
| R08 | 2k128 | 256 | 0.099592 | 79.668651 dB |
| R09 | 2k128 | 128 | 0.070640 | 78.833019 dB |
