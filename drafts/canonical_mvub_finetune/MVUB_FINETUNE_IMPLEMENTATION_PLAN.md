# MVUB Fine-tuning Implementation Plan

状态：**IMPLEMENTED; STATIC/IMPORT CHECKS ONLY, NO GPU TRAINING SUBMITTED**
日期：2026-08-31
限制：本轮不修改 model/training source，不提交训练任务。

## 0. 固定起点与审计结论

- Released Base profile：Official R01，`32k8k + lambda=32768`。
- Canonical Base：`step_5525.pth`，`x4_f4 / C=128 / ResNet K3 L2`。
- Enhancement 起点：D611 `+1762`，实际 checkpoint `step_3525.pth`。
- D611 checkpoint metadata 已核对：
  - `architecture=canonical_independent_enhancement`
  - `step=3525`
  - `conditioning_lambda=32768`
  - `rd_lambda=32768`
  - `distortion_weights=[6,1,1]`
- TAFA ZIP `scalable-geometry-roi-channel_split_lambda.zip` 可正常读取。它是 geometry codec，不能直接复制到 Attribute canonical model；可复用的是 manager 已确认的 training behavior：`trainer.py:209` 用 `random.randint(...)` 每 batch 选一个 level，随后只执行一次 forward、一次对应 loss、一次 backward/optimizer step（`trainer.py:212-237`）。
- 当前 canonical implementation **不能靠训练脚本直接 full-unfreeze**：Base path 同时受到 `requires_grad=False`、`eval()` 和多层 `@torch.no_grad()` 约束。Arm B/C 必须先做下述最小显式 refactor；只把 Base parameters 加入 optimizer 会产生“名义解冻、实际无梯度”的错误实验。

## A. 当前模型参数路径

### A1. 实际 module ownership

| 功能 | 当前真实 module path | 当前状态（D611 training） | 备注 |
|---|---|---|---|
| r1-r4 released prefix | `model.base.prefix.model` | frozen + eval + no_grad | 一个 shared `MultiscaleVAE(stage=1)`，不是四份 VAE |
| Prefix input projection | `model.base.prefix.model.linear_in.*` | frozen | r1 起始 decoder feature |
| Native Upscaler | `model.base.prefix.model.upscaler.*` | frozen | 同一 module 在 r1-r4 traversal 中复用，并负责 r4 后 `stride2 -> stride1` transition |
| Shared ResidualVAE | `model.base.prefix.model.VAE.*` | frozen | r1-r4 共享；其中 `VAE.fuseNet/outNet` 也用于 Canonical Base output |
| BaseSynthesis | `model.base.base_synthesis.backbone.*` | frozen | 从 `x4,f4` 生成 `c_B` |
| EnhancementVAE | `model.enhancement.vae.*` | **trainable** | 独立 copy，无参数共享 |
| Lambda Embedder | `model.base.prefix.model.embedder.lmb_embedding.*` | frozen | Base 与 Enhancement 共用其输出 embedding |
| Prefix AQL | `model.base.prefix.model.VAE.EQlayer.*` / `DQlayer.*` | frozen | r1-r4 共享 AQL |
| Prefix entropy prior | `model.base.prefix.model.VAE.block_prior.*`, `loc_net.*`, `scale_net.*` | frozen | `SymmetricConditional` 本身无 learned parameters |
| Enhancement AQL | `model.enhancement.vae.EQlayer.*` / `DQlayer.*` | trainable | D611 checkpoint 已包含 |
| Enhancement entropy prior | `model.enhancement.vae.block_prior.*`, `loc_net.*`, `scale_net.*` | trainable | `entropy_fn` 无 learned parameters |
| Enhancement transforms/output | `model.enhancement.vae.encoder.*`, `decoder.*`, `fuseNet.*`, `outNet.*` | trainable | D611 的实际 trained modules |

Source anchors：

- released checkpoint load/freeze：`scalable_attribute/canonical/prefix.py:37-61`
- Base checkpoint load/freeze：`scalable_attribute/canonical/scalable_model.py:17-33`
- independent Enhancement copy：`scalable_attribute/canonical/enhancement.py:6-20`
- native Base synthesis：`scalable_attribute/canonical/model.py:40-56`
- ResidualVAE parameter ownership：`lossy_attribute/model_resvae.py:20-63`

### A2. `full-unfreeze` 的精确定义

Arm B/C 的 `full-unfreeze` 应令以下 parameter prefixes 全部 `requires_grad=True`：

```text
base.prefix.model.linear_in.*
base.prefix.model.upscaler.*
base.prefix.model.VAE.*
base.prefix.model.embedder.*
base.base_synthesis.*
enhancement.vae.*
```

Pooling、unpooling、pruning 和 `SymmetricConditional` 没有 trainable parameters，不需要虚构 parameter group。

这意味着 full-unfreeze 会更新：released r1-r4 transforms、共享 entropy prior、EQ/DQ、native upscaler、released `fuseNet/outNet`、lambda embedder、BaseSynthesis 和 EnhancementVAE。更新后 hard decode 仍合法，但必须使用同一 fine-tuned checkpoint；不能再只靠 released checkpoint + old Base checkpoint 重建该模型。

## B. 当前 forward / loss 路径

### B1. 已验证 D611 training path

- entry point：`scripts/scalable_attribute/canonical/train_enhancement.py`
- model assembly：`CanonicalBaseModel -> load_frozen_base -> CanonicalScalableModel`（`train_enhancement.py:111-126`）
- Base reconstruction：`CanonicalBaseModel.reconstruct_from_state()`（`canonical/model.py:40-56`）
- Full reconstruction：`EnhancementVAE.forward()` 后的 `x_out`，由 `CanonicalScalableModel._result()` 暴露为 `Full`（`canonical/scalable_model.py:72-77,114-128`）
- current rate：

```text
R_E = -sum(log2(likelihood_E)) / N_full
```

  实现在 `train_enhancement.py:246-255`，denominator 是 `len(attribute)`。
- current D611 distortion：

```text
D_F611 = (6*MSE_Y + MSE_U + MSE_V) / 8
L_A = R_E + 32768 * D_F611
```

  `weighted_distortion()` 可直接复用（`train_enhancement.py:49-62`）。
- current `R_B`：training entry point **没有计算**，因为 Base 固定；hard evaluator 才统计 `x_low+r1+r2+r3+r4` physical bits（`evaluate_scalable_formal.py:172-217`）。

### B2. Full-unfreeze 所需的 existing source primitive

Official `MultiscaleVAE.forward(training=True, real_coding=False, max_residual_stages=4, return_state=True)` 已能返回：

```text
state after r4
likelihood_list for r1-r4
```

见 `lossy_attribute/model.py:109-207`。因此不需要新造 entropy surrogate，也不需要复制 traversal。

Full-unfreeze training 的 differentiable Base rate 定义为：

```text
R_B_est = sum_i[-sum(log2(likelihood_ri))] / N_full,  i=1..4
```

`x_low` G-PCC bits 不可微且没有 learned path；training objective 中它是 sample-dependent constant，不参与 gradient。正式 validation 仍必须用 hard physical：

```text
R_Base_physical = bits_xlow + bits_r1 + ... + bits_r4
R_Full_physical = R_Base_physical + bits_E
```

`D_B611` 可直接复用同一个 `weighted_distortion(GT, Base, (6,1,1))`；`D_F611` 同理作用于 `Full`。

### B3. 当前阻止 full-unfreeze 的位置

以下均必须在 implementation 时显式处理，默认 frozen behavior 必须保持不变：

1. `FrozenUnicornPrefix.__init__()` 固定 `requires_grad_(False)`（`prefix.py:54`）。
2. `FrozenUnicornPrefix.train()` 强制 eval（`prefix.py:58-61`）。
3. Prefix soft forward、state completion、embedding 均有 `@torch.no_grad()`（`prefix.py:63,97,109`）。
4. `load_frozen_base()` 冻结整个 Base（`scalable_model.py:31-32`）。
5. `CanonicalScalableModel.__init__()` 拒绝非 frozen Base（`scalable_model.py:39-46`）。
6. `CanonicalScalableModel.train()` 强制 Base eval（`scalable_model.py:48-52`）。
7. `base_forward()` 有 `@torch.no_grad()`（`scalable_model.py:54-66`）。

不能通过删除所有 no-grad 来粗暴修复；hard/deterministic evaluator 必须继续走现有 frozen paths。

## C. 三个 arm 的最小实现

### Common model API

保留现有 `forward()`、`deterministic_forward()`、`hard_reconstruct()` 的 frozen/hard semantics。新增两个命名清楚的 training path，避免一个塞满 mode flags 的超级函数：

```text
forward_base_train(attribute)
    -> Base, prefix_likelihoods

forward_full_train(attribute)
    -> Base, Full, prefix_likelihoods, likelihood_E
```

并新增单一明确的 trainability setup：

```text
set_trainable_scope("enhancement_only" | "full")
```

默认仍是 `enhancement_only/frozen Base`，保证当前 evaluator 和历史结果不变。

### Arm A — Enhancement-only fine-tuning

```text
frozen:   base.*
train:    enhancement.vae.*
forward:  existing Full path
loss:     R_E + 32768 * D_F611
```

- 直接复用当前 D611 objective。
- Base 在 `no_grad` 下计算。
- 不需要 prefix likelihood，也不改变 Base endpoint。
- 这是当前 `train_enhancement.py` semantics 的 MVUB data adaptation。

### Arm B — Full-unfreeze + Full-only

```text
train: all parameter prefixes listed in A2
forward: forward_full_train only
loss: R_B_est + R_E + 32768 * D_F611
```

- 每 batch 只生成 Full endpoint loss。
- Base 是 Full 的 differentiable intermediate，不单独加入 `D_B`。
- Prefix、BaseSynthesis、Enhancement 都收到 Full loss gradient。
- `R_B_est` 必须加入，否则 full-unfreeze 可以无约束增加 Base rate来改善 Full，所得 RD 不可解释。

### Arm C — Full-unfreeze + TAFA-style 50/50 random Base/Full

每个 batch 用 seeded Python RNG 做一次 uniform sample：

```text
endpoint = Base with p=0.5, Full with p=0.5
```

不 alternating，不在同一 iteration 相加两个 endpoint losses，也不强制每 epoch 精确各半。

Base batch：

```text
output = forward_base_train(attribute)
loss_B = R_B_est + 32768 * D_B611
```

- 只运行 r1-r4 + native transition + BaseSynthesis + released fuseNet/outNet。
- **不调用 EnhancementVAE**，所以没有无用 Enhancement forward。
- Base encoder 正常使用 GT 形成 r1-r4 codes，这是合法 encoder-side information；decoder endpoint 只依赖 x_low+r1-r4。
- 不访问 Enhancement latent、likelihood、feature 或 payload。

Full batch：

```text
output = forward_full_train(attribute)
loss_F = R_B_est + R_E + 32768 * D_F611
```

Optimizer 每次对当前全部 `requires_grad=True` parameters 调用 step：

- Base batch：Enhancement gradients 为 `None`，weight_decay=0 时 Enhancement 不更新。
- Full batch：Prefix、BaseSynthesis、Enhancement 全部更新。

TAFA ZIP 只作为“每 batch随机 endpoint、单 endpoint forward/backward”的行为证据；不复制其 geometry BCE、channel split 或 model code。

## D. Optimizer / checkpoint contract

### D1. Initialization

每个 arm 从完全相同 weights 开始：

1. released `32k8k/epoch_last.pth` -> `base.prefix.model`
2. canonical Base `step_5525.pth` -> `base.base_synthesis`
3. D611 `step_3525.pth` -> `enhancement.vae`

不加载 D611 的旧 Adam state。Human-domain fine-tuning 使用 **fresh Adam**：

```text
Adam(trainable_parameters, lr, betas=(0.9,0.999), weight_decay=0)
constant LR
```

原因：旧 optimizer moments 来自 RWTT，且只包含 Enhancement parameters；它与 Arm B/C 新 parameter set 不兼容，也会破坏三 arm 的公平起点。

Optimizer 必须在 `set_trainable_scope(...)` **之后**构造，并保存 resolved trainable parameter names/count；做一次 gate：

```text
set(optimizer params) == set(parameters with requires_grad=True)
```

### D2. Checkpoint

使用一个共同 schema 保存完整可重建状态：

```text
architecture = canonical_scalable_mvub_finetune_v1
scalable_model = model.state_dict()  # prefix + BaseSynthesis + Enhancement
optimizer
global_step
arm / trainable_scope
conditioning_lambda / rd_lambda / distortion_weights
released/base/enhancement initialization paths and steps
dataset manifests
resolved_args
```

Arm A 虽然 Base frozen，也使用同一完整 state schema。这样三个 arm 使用同一个 loader 和同一个 hard evaluator，不需要为 frozen/full-unfreeze 建两套 evaluation code。

Evaluator 只负责装载完整 state 后调用现有 `hard_reconstruct()`；hard codec contract仍是：

```text
Base = x_low+r1+r2+r3+r4
Full = Base streams + Enhancement stream
```

## E. MVUB data contract

### E1. Verified counts

当前 processed MVUB 数据已经 PASS：

| Subject | Frames | H5 blocks | Points | Role |
|---|---:|---:|---:|---|
| David10 | 216 | 4,784 | 338,764,450 | Train |
| Phil10 | 245 | 4,336 | 364,228,947 | Train |
| Ricardo10 | 216 | 3,456 | 216,535,389 | Train |
| Sarah10 | 207 | 3,312 | 240,840,773 | Train |
| **Train total** | **884** | **15,888** | **1,160,369,559** | Train |
| Andrew10 | 318 | 5,088 | 412,986,422 | Held-out validation |

注意实际 processed subject 名为 `Sarah10`，实现必须使用这个拼写。

### E2. Job-local staging

每个 Slurm job：

```text
scratch=/dev/shm/Tanzeyu_${SLURM_JOB_ID}
mkdir -p exact scratch
tar -xf /data/run01/scz0ade/Tanzeyu/data/MVUB/mvub10_h5.tar -C "$scratch"
data_root="$scratch/MVUB10_H5"
```

- 所有 persistent checkpoint/metrics/log 写回 `/data/run01/scz0ade/Tanzeyu/experiments/...`。
- job 结束只清理本 job 创建的 exact scratch path。
- 当前 N30 tar 已核对含五个 subject 的 H5，但 archive 内未发现 summary/manifest metadata；训练 manifests 应由 sorted filenames/固定 frame rule生成并持久保存到 experiment output，不依赖 archive metadata。

### E3. Fixed Andrew validation subset

第一版固定选择 8 个均匀覆盖 318-frame sequence 的 frame IDs：

```text
frame0000
frame0045
frame0091
frame0136
frame0181
frame0226
frame0272
frame0317
```

每个 selected frame 使用全部 16 个 H5 blocks，共 **8 frames / 128 H5**。规则是对 sorted frame indices `[0,317]` 做 8-point uniform spacing 后一次冻结；不按结果挑样本、不每次 resample。

Validation aggregation：先按 frame point-weighted aggregate MSE/rate，再 frame-equal average；formal hard metric继续使用 `pc_error YUV-PSNR 6:1:1`。Routine training 可记录 tensor-domain D611，但不得与 formal pc_error curve混为一条曲线。

## F. 建议的第一晚最小实验（本轮不执行）

为了真正比较三种策略，最小完整 set 是 A/B/C 三个 arm，而不是只跑 A/C；缺少 B 就无法区分“full-unfreeze 本身”与“random endpoint training”的影响。

建议统一：

```text
initial weights: D611 +1762 / step3525
conditioning_lambda = rd_lambda = 32768
distortion weights = 6,1,1
fresh Adam
LR = 1e-5 constant
batch size = 1
seed = 0
max_steps = 500
checkpoints = step0, step250, step500
train = 15,888 H5, shuffled
validation = fixed Andrew 8 frames / 128 H5
```

选择 `LR=1e-5` 是为保护 full-unfreeze 的 released prefix/AQL/entropy parameters；三 arm 同 LR 便于第一轮 causal comparison。选择 BS1 是因为 full-unfreeze 会保留 r1-r4 + Base + Enhancement 的完整 autograd graph，不能从 frozen-Base BS4 的 11.36 GiB 直接推断 24GB RTX3090 可承受 BS4。

正式 500-step 前只做 5-step BS1 memory/correctness gate；如 OOM，STOP 并人工复核，不动态改变 batch size。三个 arm 各用一张 RTX3090、独立 job，不使用 DDP。

第一轮判断只看：

- train/Andrew D611 trajectory；
- hard Base/Full physical bpp；
- Base/Full pc_error YUV 6:1:1；
- Arm C sampled Base/Full batch counts；
- prefix/Base/Enhancement gradient ownership；
- hard exact round-trip与 `Base bits subset Full bits`。

500 steps 是 domain-adaptation screening，不宣称收敛或 formal MVUB result。

## 预计修改文件（manager批准后）

最小 source diff：

1. `scalable_attribute/canonical/prefix.py`
   - 新增 differentiable r1-r4 training forward，返回 state + likelihoods。
   - 新增 differentiable state completion/embedding path。
   - 保留现有 frozen/hard methods不变。
2. `scalable_attribute/canonical/model.py`
   - 新增 differentiable Base training reconstruction入口；复用现有 `reconstruct_from_state()`。
3. `scalable_attribute/canonical/scalable_model.py`
   - 新增 explicit trainable scope。
   - 新增分开的 `forward_base_train()` / `forward_full_train()`。
   - 新增完整 fine-tune checkpoint loader；保留旧 checkpoint/evaluator compatibility。
4. `scripts/scalable_attribute/canonical/train_mvub_finetune.py`（新增）
   - 一个 entry point，三个明确 arm分支；不改稳定的 `train_enhancement.py`。
   - fresh Adam、seeded endpoint sampling、metrics/checkpoint。
5. `scripts/data/create_mvub_finetune_manifests.py`（新增、很薄）
   - 生成四 subject train list和固定 Andrew-8 list；不复制/移动 H5。
6. `scripts/scalable_attribute/canonical/evaluate_scalable_formal.py`
   - 仅扩展为可加载新的完整 scalable checkpoint；metric/hard coding/aggregation不改。

N30 submission 继续使用薄 sbatch/现有 activation，不增加 scheduler framework。

## Correctness gates

1. 初始化后，三个 arm step0 的 Base/Full 与 D611 step3525 exact一致。
2. Arm A：只有 `enhancement.vae.*` gradients；Base parameters grad全部 `None`。
3. Arm B：所有 declared full-unfreeze groups收到 gradient；optimizer无遗漏/额外参数。
4. Arm C Base batch：Enhancement完全不 forward且 grad全部 `None`。
5. Arm C Full batch：Prefix/Base/Enhancement均有 finite gradient。
6. 所有 batch support、coordinates、tensor stride一致。
7. 所有 rate/distortion/loss/gradients finite；异常 fail fast，不 skip。
8. hard Base只产生4个 residual streams；不访问 Enhancement payload。
9. hard Full满足 `Full_bits = Base_bits + Enhancement_bits`。
10. hard encode/decode exact round-trip。
11. Checkpoint reload后 deterministic/hard结果不变。
12. Andrew frames与任何 train subject overlap为0。

## 主要风险 / blockers

| Risk | 影响 | 最低成本控制 |
|---|---|---|
| 当前多层 no-grad/eval造成假解冻 | Arm B/C结论无效 | explicit differentiable API + gradient ownership gate |
| Full-unfreeze未计 `R_B_est` | Base rate可能无约束膨胀 | B/C Full loss纳入r1-r4 likelihood rate |
| Shared VAE在r1-r4重复使用 | 梯度累积、memory高 | BS1 5-step memory gate，不自动fallback |
| Embedder同时影响Prefix和Enhancement AQL | 少量更新也可能改变两层 rate | LR=1e-5；记录 Base/EL rate trajectory |
| MVUB frame高度相关 | H5-level validation偏乐观 | Andrew subject完全held-out；按frame固定subset/aggregate |
| N30 tar缺少metadata | split不可复现 | 从严格文件名规则生成并保存relative manifests |
| 新 full checkpoint只保存Enhancement | 无法复现fine-tuned Base | 保存完整 scalable model state |
| TAFA geometry代码被误搬到Attribute | objective/metric不匹配 | 只采用random single-endpoint behavior，不复制model/loss |

## Stop condition

本文件完成后停止。未修改 canonical source，未生成训练 manifest，未提交 Slurm job。
