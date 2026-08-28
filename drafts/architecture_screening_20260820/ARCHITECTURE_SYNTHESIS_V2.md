# Architecture Synthesis & Screening V2

> 状态：working draft，2026-08-20  
> 本轮所有 architecture、contribution、decision matrix 和淘汰理由统一维护在本文件。  
> 当前不构成最终 architecture commitment，也不授权正式 training、long sweep、
> Dev14/full evaluation 或 official source modification。

## 0. Hard contract 与证据边界

最终 codec 必须提供：

```text
Base:
  lower-rate payload -> full-resolution attribute B

Full:
  identical Base payload + Enhancement payload -> higher-quality attribute F

Base bits ⊂ Full bits
```

所有 Enhancement 必须携带 GT-dependent、sample-specific information：

```text
Encoder: GT A -> Enhancement representation -> Enhancement bits
Decoder: decoder-known Base state + Enhancement bits -> Full
```

只在 decoder 侧预测、没有传入新 sample-specific bits 的方案属于 Family A
no-bit continuation，不属于本轮 Family B/C 主候选。

### 已确认的 source contract

- Unicorn Attribute 是 `x_low -> r1 -> ... -> r5 -> reconstruction` 的顺序
  conditional coding。
- 每个 native stage 在读取当前 arithmetic payload 前，可从 decoded prefix、known
  geometry、fixed checkpoint/profile 和 lambda 得到 prior、`loc/scale`。
- `ResidualVAE` 已包含 residual encoder、AQL `EQ/DQ`、`block_prior`、
  `loc_net/scale_net`、`SymmetricConditional`、decoder、`fuseNet/outNet`。
- released RWTT `stage=1` 在五个 spatial stages 复用同一个 VAE/Upscaler 参数。
- native `torchac` stream 是 monolithic payload，没有 within-stream progressive API。
- Base/Full 实验允许沿用 official/in-memory contract；本阶段不设计 file container。

### Round-1 evidence

- Family A learned no-bit r5 continuation：micro-train 增益很小，holdout 不超过
  deterministic `round(loc)`/continuous `loc`，暂时降级。
- Family B R02/r5 offline oracle：step-2 coarse 约 `0.162 empirical bpp`，
  coarse+refinement marginal entropy 相对 original overhead 约 `0.8%`。
- Family C R09 residual oracle：stride-4 parent residual 在 quant step `4/255` 时约
  `0.321 empirical bpp`、平均保留 `+2.90 dB` tensor oracle gain；step `8/255`
  时约 `0.145 bpp`、`+1.82 dB`。
- 以上 B/C rate 都不是 physical arithmetic bits，不能直接画正式 RD curve。

---

# 1. Family B — Within-stream successive refinement

## B1. Deterministic Nested Quantization of a Native Late-stage Latent

### 一句话定义

把一个 native late-stage integer latent 拆成可独立解码的 coarse symbol stream 和
conditional refinement stream；Base 解 coarse，Full 解 coarse+refinement 并恢复 fine
native symbols。

### Encoder / decoder data flow

```text
Encoder
GT A
 │
 ├─ Unicorn native encode x_low+r1...r(K-1)
 │                         │
 │                         └─ decoded-prefix state S_K
 │
 └─ native residual encoder(A, S_K) -> EQ latent y_K
                                      │
                                      ├─ q_fine = round(y_K)
                                      ├─ q_B = NestedCoarse(q_fine)
                                      └─ q_E = q_fine - q_B

Base payload = x_low + r1...r(K-1) + stream(q_B | S_K)
Enhancement payload = stream(q_E | S_K, q_B)

Base decoder
Base payload -> S_K + q_B -> native DQ/decoder/fusion/out -> B, stride 1

Full decoder
Base payload + Enhancement payload
  -> q_fine = q_B + q_E
  -> same native DQ/decoder/fusion/out -> Full, stride 1
```

### Complete codec contract

1. **GT 使用位置**：native residual encoder 从 `A-x_in` 产生 `y_K/q_fine`；
   `q_B/q_E` 都是 GT-dependent。
2. **Base payload**：native prefix 加新的 coarse latent arithmetic bytes。
3. **Enhancement payload**：以 `q_B` 和 prefix state 为条件的 refinement bytes。
4. **Base decoder state**：geometry、profile/lambda、decoded native prefix、`q_B`。
5. **Full-resolution Base**：所有 spatial transitions 都执行；stage K 用 `q_B` 经
   native DQ/synthesis 输出 stride-1 reconstruction。
6. **Full 的新增信息**：只多接收 `q_E` bytes。
7. **Full reconstruction**：先恢复 `q_fine=q_B+q_E`，再走 native DQ/synthesis。
8. **Subset legality**：coarse 与 refinement 是两个有序 payload；Full 不替换 Base bytes。
9. **Encoder-only / decoder-known**：`A`、pre-quantization `y_K` 是 encoder-only；
   prefix state、`q_B` 在 Base decoder known；Full decoder 再获得 `q_E`。
10. **Unicorn changes**：新增 nested quantizer、coarse entropy head、refinement entropy
    head和双-stream encode/decode；native prefix、DQ、decoder/fusion/out 尽量不动。
11. **Checkpoint reuse**：prefix、residual encoder、DQ、decoder、feature path 可从
    released checkpoint 初始化；两个 entropy heads 是新增/改造部分。
12. **Representation change**：改变 quantization representation 和 entropy syntax；
    decoder topology 可以保持 native。
13. **Training endpoints**：共同优化 Base coarse reconstruction 与 Full fine
    reconstruction；第一阶段可冻结 prefix。
14. **Rate/Distortion**：

```text
R_B    = (bits_xlow + bits_prefix + bits_qB) / N0
R_E    = bits_qE / N0
R_Full = R_B + R_E
D_B    = MSE_YUV(A, B)
D_F    = MSE_YUV(A, Full)
L      = R_B + alpha*R_E + lambda_B*D_B + lambda_F*D_F
```

   `alpha` 是否需要独立存在由 prototype 决定，不能预先散落成多个默认值。
15. **解决的 limitation**：native stream 只能 whole-stage 截断，rate granularity 太粗。
16. **Thesis contribution**：为 Unicorn native latent 引入 nested quantization、
    conditional refinement coding 和 endpoint-aware optimization。
17. **Ablation**：whole-stage truncation；step-2 offline oracle；coarse-only training；
    joint Base/Full training；不同 nested quantizer。
18. **最大风险**：需要新的 physical entropy syntax；released Full bytes 无法原样保留。
19. **最便宜 falsification**：用 native `loc/scale` 和 captured symbols 计算
    `q_B/q_E` conditional estimated rate，检查 layered overhead 与 coarse RD 是否仍成立。

### Full preservation potential

如果 `q_B+q_E` exact recover released `q_fine`，Full tensor reconstruction 可以保持
released numerical result；但原 rK arithmetic bytes 会被两个新 streams 替换，因此是
reconstruction preservation，不是 bitstream-byte preservation。

### Contribution judgement

不是简单换 quantization step。只有同时具备 nested payload、conditional refinement 和
双-endpoint optimization，才构成主要 contribution。

---

## B2. Learned Coarse Latent + Conditional Refinement Latent

### 一句话定义

不要求 coarse/refinement 是同一个 integer symbol 的代数拆分，而是在一个 native late
stage 内学习两个 latent：`y_B` 专门优化 Base endpoint，`y_E` 在 `y_B` 条件下编码
剩余 GT information。

### Encoder / decoder data flow

```text
Encoder
A + decoded-prefix S_K
 │
 ├─ Analysis_B(A, S_K) -> y_B -> q_B -> Base stream
 │
 └─ Analysis_E(A, S_K, q_B) -> y_E -> q_E -> Enhancement stream

Base decoder
S_K + q_B -> Synthesis_B -> full-resolution B

Full decoder
S_K + q_B + q_E -> Synthesis_F -> full-resolution Full
```

Entropy dependencies：

```text
p(q_B | S_K)
p(q_E | S_K, q_B)
```

### Complete codec contract

1. `A` 同时进入 Base/Enhancement analysis；两个 latents 都是 sample-specific。
2. Base payload 是 native prefix + `q_B`。
3. Enhancement payload 是 conditional `q_E`。
4. Base decoder 只需 prefix state 与 `q_B`。
5. `Synthesis_B` 必须直接输出 stride-1 B。
6. Full 多收到 `q_E`。
7. `Synthesis_F` 使用 `q_B/q_E/S_K` 输出更高质量 Full。
8. 分离 streams 保证 `Base bits subset Full bits`。
9. `A/y_B/y_E` 是 encoder-only；quantized latents 与 prefix state 可由 decoder 重现。
10. 可复用 Backbone、ResidualVAE blocks、AQL 和 `SymmetricConditional`，但需新增
    dual analysis/synthesis/entropy heads。
11. released prefix 可完整复用；late-stage weights只能作为 initialization。
12. quantization 和 decoder topology 都改变，released Full 无法保证保留。
13. 必须 joint endpoint-aware training，不能只训练 Full loss。
14. `R_B/R_E/D_B/D_F` 与 B1 一致，但 rate 分别来自两个 learned distributions。
15. 解决 whole-stream granularity 与“coarse symbol 未针对 Base RD 优化”两个 limitation。
16. contribution 是 hierarchical learned latent 与 conditional refinement mechanism。
17. ablation：deterministic B1；unconditioned refinement；single endpoint loss；shared vs
    separate synthesis。
18. 最大风险：等于重做 late-stage learned codec，容易失去 Unicorn checkpoint advantage，
    并可能出现新的 latent collapse。
19. 最便宜 falsification：固定 native prefix，2-H5 micro-overfit dual latent，检查
    `q_B/q_E` 是否都 nonzero 且 Base/Full loss 可分别下降；不做 hard codec。

### Contribution judgement

潜在 novelty 高，但当前 Round-1 只支持“representation 有 headroom”，没有直接支持
dual learned latent。应保留在 backlog，不能先于 B1 legality probe 进入实现。

---

# 2. Family C — Learned Layered / Hybrid Enhancement

## ScalablePCAC connection：迁移什么，不迁移什么

ScalablePCAC 的公开论文级 contract 是：G-PCC 编码 downscaled thumbnail 形成 Base，
learned Enhancement 以 Base reconstruction 为条件恢复 full-resolution input，并通过
cross-layer rate allocation 在 Base resolution/QP 与 Enhancement quality factor 之间选择
operating point。资料来源：[IEEE DOI](https://doi.org/10.1109/TMM.2023.3331584)、
[作者 3DPCC 项目索引](https://github.com/3dpcc)。

本 thesis 可迁移：

- 两层 payload 与严格 Base/Full decode contract；
- Enhancement encoder 必须产生 GT-dependent bits；
- Enhancement analysis/prior/synthesis 以 decoded Base 为条件；
- 最终 rate 必须报告 `R_Base`、`R_E`、`R_Full`。

不能直接照搬：

- Unicorn Base 已是 full-resolution attribute，不是 spatial thumbnail；
- 不改变 geometry，也不重新引入 G-PCC Base；
- 第一版不做 Base resolution/QP 的 cross-layer combinatorial allocation；
- Unicorn 已提供 `F_U/D_U`、native residual VAE、AQL 和 conditional prior，忽略这些
  decoder-known state 会削弱 Unicorn coherence。

因此“ScalablePCAC-inspired”应作为合法 layered baseline；真正 contribution 必须来自
Unicorn-native representation/conditioning 或新的 multiscale Enhancement mechanism。

---

## C1. ScalablePCAC-inspired Base-conditioned External Enhancement

### 一句话定义

把一个 complete official Unicorn reconstruction 作为 frozen Base；独立 learned EL 编码
`A-B`，entropy prior 和 synthesis 只以 decoded B/geometry 为条件。

### Encoder / decoder data flow

```text
Encoder
A -> official Unicorn encode -> Base payload -> locally decoded B
A - B = E
(E, B, geometry) -> EL analysis -> y_E -> q_E -> Enhancement stream
                                  p(q_E | B, geometry)

Base decoder
Base payload -> official Unicorn decode -> B, stride 1

Full decoder
Base payload -> B
Enhancement stream + B + geometry -> EL synthesis -> delta
Full = B + delta
```

### Complete codec contract

1. GT `A` 用于 native Base encode 和 residual `E=A-B`。
2. Base payload 是完整 official Unicorn Attribute payload。
3. Enhancement payload 是 external EL latent bytes。
4. Base decoder 仅需 official Base bytes/config/geometry。
5. B 天然 full-resolution。
6. Full 多接收 EL bytes。
7. EL decode 产生 `delta`，Full=`B+delta`。
8. 两个独立有序 payload 保证 subset。
9. `A/E/y_E` encoder-only；B、geometry、quantized EL symbols decoder-known。
10. official Unicorn 完全不动；新增 external analysis/prior/synthesis/entropy path。
11. Base checkpoint 100% 复用，EL 无 released weights。
12. native entropy/quantization 不变；EL 有自己的 representation。
13. freeze Base，只优化 Full endpoint 的 `R_E + lambda_E D_F`；Base endpoint 固定。
14. `R_B` 来自 official hard codec，`R_Full=R_B+R_E`；D 分别比较 A/B、A/Full。
15. 解决 Unicorn 没有可附加 learned enhancement layer 的 limitation。
16. 若仅实现该结构，contribution 主要是 integration，不足以单独支撑 thesis。
17. ablation：unconditioned EL；B-conditioned；不同 Base point；zero-latent gate。
18. 最大风险：已有 EL V1 在相似 contract 下 hard latent collapse。
19. 最便宜 falsification：无需重跑；EL V1/zero-centered evidence 已使 C1 降级为 baseline。

### Contribution judgement

Scalability 合法、实现简单，但单纯照搬 layered philosophy 不是足够 novelty。保留为
Family C 的 baseline/control，不作为主 prototype。

---

## C2. Unicorn-native Appended Conditional Residual Stage

### 一句话定义

在 complete frozen Unicorn Base 之后追加一个 Unicorn-native residual coding call：复用
ResidualVAE/AQL/conditional prior semantics，用 decoder-known `B/F_U/D_U` 条件编码
`A-B`，形成新 Enhancement stream `r_E`。

### Encoder / decoder data flow

```text
Encoder
A -> frozen official Unicorn -> Base bytes + decoded (B, F_U, D_U)
residual E = A - B
ResidualVAE_E.encoder(E) -> EQ -> y_E -> q_E
prior_E = block_prior(B, F_U, D_U)
loc_E, scale_E = prior heads(prior_E)
q_E -> stream r_E under p(q_E | B, F_U, D_U)

Base decoder
Base bytes -> official B, F_U, D_U -> output B

Full decoder
Base bytes -> B, F_U, D_U
r_E -> q_E -> DQ -> decoder_E -> fuse_E(F_U, decoded_E)
    -> out_E -> delta
Full = B + delta
```

### Complete codec contract

1. GT `A` 产生 official Base 和 Enhancement residual latent。
2. Base payload 是完整 official Unicorn bytes。
3. Enhancement payload 是新增 `r_E` arithmetic bytes。
4. Base decoder state 是 `B/F_U/D_U`，全部从 Base decode 得到。
5. B 已是 stride-1 full-resolution。
6. Full 多收到 `r_E`。
7. Full 使用 native-style DQ/decoder/fusion 输出 sample-specific correction。
8. `r_E` 只追加，不改变 Base bytes。
9. `A/E/pre-quantization y_E` encoder-only；`B/F_U/D_U/prior/loc/scale` decoder-known。
10. official Base 不动；新增一套小型 `ResidualVAE_E` 或复用其 blocks，但不复制整个
    Unicorn codec。
11. Base checkpoint 100% 复用；EL 可以从 native ResidualVAE 初始化，但训练后独立。
12. 沿用 native EQ/DQ 与 `SymmetricConditional` semantics；新增 post-Base synthesis call。
13. freeze Base，优化 `R_E + lambda_E D_F`；可加入显式 hard-symbol gate，但不改变
    Base endpoint。
14. `R_B` official physical rate，`R_E` 新 stream rate，denominator 均为 `N0`；
    D 使用相同 author metric convention。
15. 解决 external EL 与 Unicorn internal state/entropy semantics 脱节的问题。
16. contribution 是 decoder-state-conditioned Unicorn-native Enhancement extension，
    不是“第六个 spatial scale”。
17. ablation：C1 only-B conditioning；`B+F_U`；`B+F_U+D_U`；random vs native init；
    native AQL vs fixed quantization。
18. 最大风险：若 native DQ bias/synthesis 仍形成 low-rate shortcut，可能重现 collapse；
    另一个风险是仅追加一套 ResidualVAE 被评价为 engineering extension。
19. 最便宜 falsification：固定四 H5，冻结 Base，只训练 appended residual call
    100–200 steps；检查 holdout nonzero symbols、conditional rate 与 Full-zero gap。

### Contribution judgement

Unicorn coherence、checkpoint reuse 和 time-to-result 都较好。若只做到“复制一个
ResidualVAE”，novelty 中等；必须证明 decoder-state conditioning/optimization 是解决
V1 collapse 的必要 mechanism，才能成为主 contribution。

---

## C3. Unicorn-conditioned Multiscale Spatial Enhancement

### 一句话定义

把 `A-B` 学成 geometry-derived coarse-to-fine Enhancement representation，在 stride-4
和 stride-2 support 上产生 GT-dependent latents，并用 Unicorn Base 的 multiscale
decoder state 条件熵编码和逐级 synthesis，最终恢复 stride-1 correction。

### Encoder / decoder data flow

```text
Encoder
A -> frozen Unicorn -> Base bytes + B + decoder feature pyramid S_B
E0 = A - B                                              stride 1
 │
 ├─ Analysis_coarse(E0, S_B) -> y4 -> q4 -> stream e4  stride 4
 │                              p(q4 | S_B^4)
 │
 └─ Analysis_refine(E0, decoded q4, S_B)
        -> y2 -> q2 -> stream e2                        stride 2
           p(q2 | S_B^2, q4)

Base payload = official Unicorn bytes
Enhancement payload = e4 + e2

Base decoder
Base bytes -> B + S_B -> output B, stride 1

Full decoder
Base bytes -> B + S_B
e4 -> q4 -> coarse correction feature
e2 conditioned on q4/S_B -> q2 -> fine correction feature
learned upsampling/fusion -> delta at stride 1
Full = B + delta
```

Geometry parent support由 decoded geometry deterministic 生成，不需要额外 signaling。

### Complete codec contract

1. GT `A` 通过 `E0=A-B` 进入 coarse/fine Enhancement analysis。
2. Base payload 是 complete official Unicorn bytes。
3. Enhancement payload 是两个内部有序 streams `e4+e2`；它们共同构成一个 EL。
4. Base decoder 只需 official bytes，得到 B 与 decoder feature pyramid。
5. B 天然 stride 1。
6. Full 多收到 `e4/e2`。
7. Full decoder 用 decoded latents 与 Base state 逐级 upsample/fuse 出 delta。
8. Base bytes 不变；Full 严格 append `e4/e2`。
9. `A/E0/y4/y2` encoder-only；geometry support、Base states、q4/q2 decoder-known。
10. official Base 不动；新增 multiscale analysis、two conditional priors、progressive
    synthesis。Backbone、native pooling/unpooling、ResidualVAE blocks和
    `SymmetricConditional` 可复用。
11. Base checkpoint 100% 复用；Enhancement blocks可由 native weights部分初始化，
    但 topology 不等同 released five stages。
12. Base entropy/quantization不变；EL 可先复用 native AQL/entropy semantics；decoder
    topology新增两级 correction synthesis。
13. freeze Base，训练一个 Full endpoint；EL 内可加入 coarse auxiliary endpoint/loss，
    但 thesis 对外仍只要求 Base 和 Full 两个 endpoints。
14. `R_E=(bits_e4+bits_e2)/N0`，`R_Full=R_B+R_E`；distortion 比较 A/B 与 A/Full。
15. 解决 full-resolution residual 直接 bottleneck 难编码、EL V1 未利用 residual spatial
    structure 的 limitation。
16. contribution 是 Unicorn-conditioned multiscale Enhancement representation 与
    coarse-to-fine entropy/synthesis mechanism。
17. ablation：C1 full-resolution EL；only e4；only e2；e4+e2；无 Base feature
    conditioning；不同 spatial support。
18. 最大风险：two-stream EL 增加 complexity，oracle average/broadcast 不保证 learned
    representation 或 physical RD 成功。
19. 最便宜 falsification：先只训练 stride-4 single-latent capacity probe，使用
    decoder-known Base/native state条件；检查 holdout correction、nonzero hard symbols和
    estimated/empirical rate。PASS 后才增加 stride-2 refinement。

### Contribution judgement

它最直接解释并利用 Round-1 residual evidence，novelty 和 thesis narrative 最强；但
第一版必须从 single-scale gate 开始，不能直接实现完整 two-stream network。

---

# 3. Contribution screening

评分定义：除 `Complexity`、`Failure risk` 外，`5` 表示更好；这两列 `5` 表示更高、
更不利。评分只用于当前 screening priority。

| Candidate | Legality | RD potential | Novelty | Rationale | Unicorn coherence | Checkpoint reuse | Training feasibility | Hard-codec feasibility | Ablation clarity | Complexity | Failure risk | Thesis strength | Time-to-result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 Nested native quantization | 5 | 4 | 4 | 5 | 5 | 5 | 3 | 2 | 5 | 4 | 4 | 5 | 3 |
| B2 Learned dual latent | 5 | 4 | 5 | 4 | 4 | 3 | 2 | 2 | 5 | 5 | 5 | 5 | 1 |
| C1 ScalablePCAC-style external | 5 | 3 | 2 | 3 | 2 | 5 | 4 | 4 | 5 | 3 | 4 | 2 | 4 |
| C2 Unicorn-native appended | 5 | 4 | 4 | 4 | 5 | 5 | 4 | 4 | 5 | 3 | 3 | 4 | 4 |
| C3 Multiscale spatial EL | 5 | 5 | 5 | 5 | 4 | 4 | 3 | 3 | 5 | 4 | 4 | 5 | 3 |

## Screening Gates

| Candidate | Gate 1 Legality | Gate 2 Contribution | Gate 3 Evidence | Gate 4 Feasibility | Decision |
|---|---|---|---|---|---|
| B1 | PASS | PASS | PASS: successive oracle | CONDITIONAL: entropy syntax | cheap legality/rate probe |
| B2 | PASS conceptually | PASS | PARTIAL | FAIL for first prototype | backlog |
| C1 | PASS | FAIL as main contribution | negative V1 evidence | PASS | baseline only |
| C2 | PASS | PASS if mechanism beats V1 | PASS: native state + residual headroom | PASS | cheap capacity probe |
| C3 | PASS | PASS | PASS: spatial oracle | CONDITIONAL | single-scale cheap probe |

## Identified limitation -> proposed mechanism -> measurable claim

### B1

```text
Limitation:
native stage stream 粒度太粗，不能在一个高-rate stage 内形成 optimized Base。

Mechanism:
nested native quantization + conditional refinement + endpoint-aware training。

Claim if successful:
在接近 original Full rate/reconstruction 的同时，新增有竞争力的 intermediate
full-resolution Base RD point。
```

### C2

```text
Limitation:
external EL V1 与 Unicorn decoder-state/entropy semantics 脱节并产生 collapse。

Mechanism:
使用 B/F_U/D_U 条件的 Unicorn-native residual stream。

Claim if successful:
在保持 official Base 不变的情况下产生 nonzero Enhancement payload 和可测 Full RD gain。
```

### C3

```text
Limitation:
full-resolution residual 的 direct bottleneck 忽略可压缩的 spatial structure。

Mechanism:
geometry-derived multiscale residual representation + Base-state conditional entropy/synthesis。

Claim if successful:
以更低 Enhancement rate 保留有效 correction，并避免 EL V1 hard collapse。
```

---

# 4. Round-1 淘汰、保留与下一步（已由第 5–8 节更新）

## 当前淘汰/降级

- **Family A learned no-bit continuation**：保留为 baseline/control，不进入主 prototype。
- **B2 learned dual latent**：不是理论淘汰；因需要同时重做 latent、entropy 和 synthesis，
  当前 evidence 不足，放入 backlog。
- **C1 ScalablePCAC-style external EL**：作为合法 layered baseline；单独使用 contribution
  太弱，且已有 V1 collapse negative evidence。

## 当前最值得保留的 architecture concepts

1. **B1 deterministic nested native quantization**：representation evidence 最直接，
   thesis mechanism 清楚；先解决 conditional-rate 与 physical nesting feasibility。
2. **C3 Unicorn-conditioned multiscale spatial Enhancement**：oracle evidence 与 contribution
   narrative 最强；先做 stride-4 single-latent gate，不直接实现 two-stream Full model。
3. **C2 Unicorn-native appended residual stage**：作为 C-family 的低复杂度 fallback 和
   ablation bridge；若 C3 gate 过于复杂，可优先得到 hard-symbol evidence。

## 推荐的下一轮 cheap evidence，尚未执行

### Probe P1 — B1 conditional-rate legality

- 固定 R02/r5、同四 H5；
- 计算 coarse/refinement 在 native `loc/scale` 条件下的 estimated rate；
- 明确 refinement CDF 需要哪些 decoder-known inputs；
- 不实现 `torchac` progressive payload，不训练。

Gate：coarse endpoint仍有用，layered conditional rate不过度膨胀，且 dependency 不要求
encoder-only state。

### Probe P2 — C3 stride-4 conditional capacity

- fixed R09 Base，同四 H5；
- 只做一个 stride-4 latent，不加 stride-2；
- condition 使用 decoder-known Base/native state；
- short micro-overfit + holdout；
- 检查 quantized nonzero fraction、estimated rate、Full-zero 和 Full gain。

Gate：holdout correction 明显、hard/rounded symbols 非零、rate-distortion signal 随
lambda/quantization direction 正常。PASS 后才讨论完整 C3；FAIL 时回退 C2，而不是继续
堆 network。

## Stop point

本草稿完成 architecture synthesis，但不代表已批准 P1/P2 或 prototype。进入任何正式
architecture implementation、长训练、sweep 或 Dev14/full evaluation 前必须再次汇报。

---

# 5. Architecture Feasibility & Complexity Screening

本节记录 2026-08-20 的第二轮结果。Design space 不再扩展：只审核 B1 与 C2；C3
仍是 C2 成功后的升级候选。所有代码均为 experiment-only draft，未修改 official
baseline path。

## 5.1 Jobs 与 authoritative artifacts

| Job | 内容 | Resource | State | Runtime |
|---:|---|---|---|---:|
| 1785177 | B1 R02 conditional-rate gate | all-GPU，分配 RTX3080/work | COMPLETED | 1 m 38 s |
| 1785178 | C2 first startup | V100 | FAILED | 18 s |
| 1785180 | C2 R09 matched controls | V100 | COMPLETED | 5 m 54 s |
| 1785186 | C2 R08 native-state sanity | V100 | COMPLETED | 约 3 m |

`1785178` 在任何 update 前因 CUDA MinkowskiEngine module 不能 `deepcopy` 而失败；修复仅
改成“重新实例化 ResidualVAE 并 load 相同 state_dict”，没有改变实验语义。

```text
$WORK/scalable_attribute_thesis/experiments/architecture_screening_20260820/
├── b1_conditional_r02/b1_conditional_rate_gate.csv
├── c2_minimal_r09/
│   ├── c2_training_trace.csv
│   ├── c2_hard_diagnostic.csv
│   ├── c2_runtime.json
│   ├── c2_b_only.pth
│   └── c2_native_state.pth
└── c2_minimal_r08/
    ├── c2_training_trace.csv
    ├── c2_hard_diagnostic.csv
    ├── c2_runtime.json
    └── c2_native_state.pth
```

---

# 6. B1 Conditional-rate Gate 与复杂度审计

## 6.1 最小 conditional model

本轮没有假设两个 learned entropy heads。对 native integer fine symbol `q` 定义
half-open nested cell：

```text
c  = floor((q + step/2) / step)
qB = c * step
qE = q - qB
qE ∈ {-step/2, ..., step/2-1}
```

native model给出每个 integer symbol 的 Laplace bin mass：

```text
p(q | S) = CDF(q+0.5 | loc,scale) - CDF(q-0.5 | loc,scale)
```

最小 nested factorization 为：

```text
p(qB | S) = sum_e p(qB+e | S)
p(qE | S,qB) = p(qB+qE | S) / p(qB | S)
```

所以：

```text
-log2 p(qB|S) - log2 p(qE|S,qB) = -log2 p(q|S)
```

这不是 learned result，而是对同一个 native probability mass 的 exact partition。

### Fact / estimate / future work 区分

- **直接 fact**：`qfine/loc/scale/S` 均由 current model 直接取得；qB 后 decoder 已知；
  qE dependency 不含 GT 或其他 encoder-only state。
- **近似 rate estimate**：CSV rate 是 native Laplace cross-entropy，未包含两条 physical
  byte streams 的 byte rounding、in-memory `min/max` 和 custom CDF implementation cost。
- **不需要 learned head 的部分**：legality、coarse cell probability、conditional
  refinement probability。
- **未来可能需要 learned/trained 的部分**：若 deterministic coarse endpoint 不够好，
  需要 endpoint-aware late-stage representation/decoder optimization；不是为了使
  factorization 本身合法。

## 6.2 R02 results

四 H5、full-resolution point-weighted rate，PSNR 为四 H5 mean author YUV 6:1:1：

| Nested step | Prefix + coarse estimated bpp | Prefix + layered estimated bpp | Coarse PSNR | Full PSNR | Layered/native ratio |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.5500 | 0.7964 | 39.3097 | 41.6155 | 1.000000003 |
| 4 | 0.3093 | 0.7964 | 38.7602 | 41.6155 | 0.999999996 |
| 8 | 0.2482 | 0.7964 | 38.4528 | 41.6155 | 1.000000015 |

其中 R02 prefix physical rate 是 `0.23851 bpp`。作为尺度参照，released R02 physical
Full 是 `0.78226 bpp`；estimated layered total 比 physical Full 高约 `1.8%`，不能把
estimated total 当作 actual bits。

同四 H5 official reference：

| Point | Physical bpp | Mean author PSNR |
|---|---:|---:|
| R02 | 0.7823 | 41.6155 |
| R03 | 0.5696 | 40.1976 |
| R04 | 0.2231 | 38.1446 |
| R05 | 0.1476 | 37.2429 |

### Gate judgement

- **Conditional legality：PASS。** 两层 rate 没有理论膨胀，dependency 全部
  decoder-known。
- **Coarse endpoint exists：PASS。** 三个 step 都输出合法 stride-1 Base。
- **Raw coarse RD value：WEAK / CONDITIONAL。** step-2 约 `0.55 bpp / 39.31 dB`，
  几乎被 R03 `0.57 bpp / 40.20 dB` 支配；step-4/8 只在 R03/R04 之间形成边际点，
  没有显示明显超越 official variable-rate curve。
- **Overall B1 gate：CONDITIONAL PASS。** Codec semantics 可行，但若不训练 coarse
  endpoint，thesis RD value 不够强。

## 6.3 Physical implementation 最小改动

不需要改变：

```text
native residual Analysis Transform
AQL EQ/DQ
native fine integer domain
native synthesis topology
r1-r4 payload
```

必须新增/修改：

1. nested quantizer：fine integer `<-> coarse index + bounded refinement`；
2. coarse categorical CDF：对每个 coarse cell 累加 native fine bin mass；
3. refinement categorical CDF：在 decoder-known `S+qB` 下归一化 cell 内 mass；
4. 两次 `torchac` encode/decode 与两个 in-memory payload；
5. `MultiscaleVAE`/last-stage routing，使 Base 停在 coarse、Full继续 refinement；
6. coordinate sorting、symbol range 和 Full exactness gates；
7. 若训练 coarse endpoint，增加 Base/Full joint loss 与 stage-5 specialization。

current `SymmetricConditional._get_cdf()` 只支持一个 consecutive Laplace alphabet，不能
直接表达 coarse-cell sums 与 bounded conditional refinement。最简做法是新增一个薄的
`NestedConditional`，复用 `_likelihood`，而不是改写整个 entropy model。

## 6.4 Parameter / training audit

Released checkpoint exact parameter count：

```text
whole Base       17,085,155
shared VAE       12,871,139
shared Upscaler   4,015,872
```

- **只做 deterministic B1 physical prototype**：0 new trainable parameters。
- **要优化 coarse endpoint**：至少 unshare/clone r5 VAE，约 `12.87M` trainable。
- 若 stage-5 transition也需调整，还要 unshare Upscaler，总计约 `16.89M`；否则更新
  shared Upscaler 会同时改变 r1-r4 prefix。
- prefix Base weights可完全 freeze；released r5 weights用于 initialization。
- 一个正式 point 预计至少一轮约 `3525 updates`。参考既有 BS4/V100 约 5 h/epoch，
  nested双 likelihood 与 endpoint loss 预计约 `5–7 h/epoch`，需实测确认。

## 6.5 Prototype -> formal RD steps

1. custom nested CDF unit/round-trip；
2. R02 四 H5 physical two-stream exactness；
3. last-stage VAE/Upscaler specialization boundary；
4. Base/Full endpoint-aware one-point training；
5. hard diagnostic；
6. Dev14；
7. 多 operating points 才能形成正式 curve。

最容易失败的是 CDF/symbol ordering 与 stage sharing，而不是普通 network forward。
physical bits、Full exactness与训练 RD 三类问题互相耦合，预计需要 `3–5` 个主要 debug
cycles。Fallback 是保留 native whole-stream truncation baseline 或转向 C2；没有一个
简单参数开关能把 B1 自动救成 competitive endpoint。

```text
Implementation complexity: High
Training complexity: High（若要求 competitive Base）；Low（只做未训练 syntax）
Debug risk: High
Time-to-first-hard-RD-point: C2 的约 2–3 倍开发周期；合理估计以“数天”而非“数小时”计
```

---

# 7. C2 Minimal Prototype 与复杂度审计

## 7.1 Matched topology

两个 controls 都从同一个 released R09 `ResidualVAE` state 初始化，Base 全部 frozen，
训练同样 `200 updates / Adam 5e-5 / rd_lambda=6500 / seed=0`：

```text
C1-like B-only:
prior/synthesis receives B + zero(F_U) + zero(D_U)

C2 native-state:
prior/synthesis receives B + F_U + D_U
```

Enhancement encoder始终读取 `A-B`，因此 qE 是 GT-dependent。前两个 H5 micro-train，
后两个 holdout。Training 使用 Uniform-noise；diagnostic 使用 rounded symbols；final 使用
actual `torchac compress/decompress`。

## 7.2 R09 matched result

四 H5 aggregate：

| Variant | Hard EL bpp | Base tensor PSNR | Full tensor PSNR | Full-zero PSNR | Full-Base | Full-Full-zero | Mean nonzero fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| B-only | 0.2743 | 35.2016 | 38.1281 | 35.1632 | +2.9265 | +2.9649 | 0.00404 |
| B+F_U+D_U | 0.1991 | 35.2016 | 38.1488 | 35.2546 | +2.9472 | +2.8943 | 0.00408 |

Facts：

- 两个 variants 都从 step 0 的 all-zero symbols 变成稳定 nonzero hard symbols；
- hard encode/decode `max_abs_diff=0`；
- native-state 在 mean Full quality 基本相同（约 `+0.021 dB`）时，把 hard EL rate
  从 `0.2743` 降至 `0.1991 bpp`，约减少 `27.4%`；
- `Full_zero` 接近 Base，主要 gain 来自 transmitted qE，不是 decoder-only correction；
- 从 step 20 到 200，estimated rate逐渐增加、distortion持续下降，方向正常。

Train/holdout：

| Variant | Split | Hard EL bpp | Full-Base | Full-Full-zero |
|---|---|---:|---:|---:|
| B-only | micro-train | 0.0791 | +3.9124 dB | +3.9401 dB |
| B-only | holdout | 0.4554 | +1.9405 dB | +1.9897 dB |
| native-state | micro-train | 0.0780 | +4.0608 dB | +3.8366 dB |
| native-state | holdout | 0.3115 | +1.8336 dB | +1.9519 dB |

Interpretation：native conditioning 有真实 entropy benefit，但 2-H5 micro-training 明显
overfit，holdout rate 高。

## 7.3 R08 operating-point sanity

按预先规则只增加相邻 R08、native-state、同样 200 updates：

| Base point | Hard EL bpp | Base tensor PSNR | Full tensor PSNR | Full-Base | Full-Full-zero |
|---|---:|---:|---:|---:|---:|
| R09 / lambda=128 | 0.1991 | 35.2016 | 38.1488 | +2.9472 | +2.8943 |
| R08 / lambda=256 | 0.1899 | 35.8253 | 38.4764 | +2.6511 | +2.6588 |

R08 EL rate只下降约 `4.6%`。因此 R09 rate较高不只是 extreme Base point；更可能来自
micro-training泛化不足或 single-stage representation效率有限。本轮不再试 R07、不改
network，也不启动 C3。

## 7.4 Mechanism gate

```text
1. Encoder产生 nonzero GT-dependent qE: PASS
2. qE可由 Base decoder-known state condition: PASS
3. hard symbols稳定非零、actual encode/decode exact: PASS
4. rate增加时 distortion下降: PASS
5. native conditioning相对 B-only有价值: PASS（约 27.4% rate reduction）
6. competitive dataset RD: NOT TESTED / NOT YET PASS
```

C2 已通过 feasibility/capacity gate，不等于正式 RD PASS。

## 7.5 Source / parameter / training audit

最小正式改动：

1. `lossy_attribute/model.py`：可选、backward-compatible 地 expose final `D_U`；
2. `BaseAdapter`：返回内部 Base state给 C2，公开 train API 仍可只暴露稳定结构；
3. 新增一个 `scalable_attribute/native_enhancement.py`，持有独立 ResidualVAE-sized EL；
4. 更新 `ScalableAttributeModel` 和 coder 使用 native EL forward/encode/decode；
5. train/evaluate 继续复用现有 CLI、loss、CSV 和 aggregation contract。

不需要改变：

```text
official Base Analysis Transform
official Base EQ/DQ
official Base entropy bytes
SymmetricConditional arithmetic syntax
geometry
released Base parameters
```

Exact新增 trainable parameters：`12,871,139`，约为 released Base 参数量的 `75.3%`。
这不是“小到可忽略”的 module，但只训练一套新增 weights；全部 `17.09M` Base weights
保持 freeze。若后续需要减参，应作为 ablation/optimization，不在 prototype 前改 topology。

Short probe实测（BS1、单 H5 updates，含少量 milestone diagnostic 的摊销）：

```text
R09 B-only       0.831 s/update, peak CUDA 2.11 GiB
R09 native-state 0.814 s/update, peak CUDA 2.02 GiB
R08 native-state 0.834 s/update, peak CUDA 2.02 GiB
```

不能把 BS1 memory/time线性当作 formal BS4 数字。结合既有 Base+EL BS4 V100 worst-case
PASS 和约 5 h/epoch 经验，C2 一个 `3525-update` formal point 的 realistic budget 是
约 `4–6 h/V100 epoch`；正式提交前仍需一次 BS4 worst-case gate。

## 7.6 Prototype -> formal RD steps

1. 最小 `D_U` exposure 与 native Enhancement module；
2. BS4 worst-case forward/backward；
3. 一个 R08/R09 formal one-epoch point；
4. hard round-trip + Dev14；
5. 根据结果决定继续训练/增加 operating points；
6. C2稳定后，C3才作为降低 EL rate 的升级 ablation。

最容易失败的是 rate仍高、generalization不足和 DQ/synthesis shortcut，而不是 arithmetic
codec；当前 hard path 已直接复用并通过。预计 `1–3` 个主要 debug cycles。Fallback清楚：
保留 C1-like control；若 C2 有 quality但 rate高，C3 multiscale 是有 evidence 的升级；
若 nonzero symbols再次 collapse，则停止该 formulation。

```text
Implementation complexity: Medium
Training complexity: Medium
Debug risk: Medium
Time-to-first-hard-RD-point: 已有 micro hard point；formal integration + one epoch 约 1–2 天
```

---

# 8. Architecture Decision Matrix V3

| Audit item | B1 Nested native latent | C2 Native appended Enhancement |
|---|---|---|
| Scalability legality | PASS | PASS |
| Current RD evidence | legal split，但 raw coarse endpoint弱 | strong capacity，formal RD未知 |
| Source modification | native last-stage traversal + nested coder，较侵入 | one optional Base state exposure + thesis modules |
| New trainable params | 0 untrained；12.87–16.89M optimized | 12.87M |
| Released checkpoint reuse | prefix全复用；late stage需clone/train | Base 100% freeze；EL native init |
| New entropy syntax | **需要** coarse/refinement custom CDF + 2 streams | **不需要**，直接复用 native syntax |
| Retrain native modules | competitive endpoint大概率需要 stage unsharing | 不需要；只训练 appended EL |
| Training cost / point | 约 5–7 h + joint endpoint complexity | 约 4–6 h，单 EL objective |
| Hard integration | High | Medium/已在 probe PASS |
| Expected debug cycles | 3–5 | 1–3 |
| Failure recoverability | 较差；syntax/training耦合 | 较好；可回退 C1或升级C3 |
| Novelty / story | High | Medium–High |
| Time-to-result | Slow | Fast |
| Overall risk/reward | high-risk/high-novelty，当前RD信号弱 | medium-risk，mechanism evidence强 |

## Final recommendation for formal prototype

**推荐 C2 进入正式 prototype，B1 保留为 secondary research candidate。**

理由不是 C2 已经达到好 RD，而是：

1. 它已经用 actual hard coding证明 GT-dependent Enhancement information 能被传输、
   decoder-only state足以解码，且 Full gain主要来自 bits；
2. `B/F_U/D_U` 相对 B-only 在 matched setting 下减少约 `27.4%` hard rate，给出了
   Unicorn-native conditioning 的直接 contribution evidence；
3. Base checkpoint/bytes可完全冻结，现有 entropy syntax可直接复用；
4. 剩余风险主要是可通过 formal training/Dev14观察的 RD/generalization问题，而不是先
   重写 nested arithmetic codec；
5. C2 成功后，C3 有清楚的升级动机：降低当前 single-stage EL 的高 rate，并可做
   `C1 -> C2 -> C3` ablation。

B1 scientific story仍然成立，但当前 deterministic endpoint接近或低于 official
variable-rate curve；要改善它需要同时承担 stage unsharing、joint endpoint training 和
custom arithmetic syntax。按“Contribution strength × RD likelihood × implementation
feasibility”综合判断，它不应先于 C2 消耗正式开发周期。

## Current stop point

本轮到此停止。没有实现正式 C2 architecture、没有运行 long training、multi-lambda
sweep、Dev14/full evaluation，也没有实现 C3。下一步需确认后才进入 C2 formal
implementation plan/code。
