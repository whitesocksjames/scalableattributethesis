# Formal static RGB evaluation handoff — 2026-09-09

> **HISTORICAL / SUPERSEDED OPERATIONAL SNAPSHOT.** This file records the
> preflight boundary on 2026-09-09. It is not current status. The final matrix
> is `320/320 FORMAL_REUSABLE`; use
> `FORMAL_STATIC_RGB_HANDOFF_20260910.md` and
> `results/comparisons/formal_static_rgb_final_20260911/`.

This file is the restart point if chat context is lost. Read it together with
`FORMAL_8I_OWLII_CTC_EXECUTION_PLAN_20260909.md`. The execution plan is the
scientific contract; this file records live operational state.

## Current state

| Item | State |
| --- | --- |
| Scientific-contract freeze | `e264168` |
| Runtime evaluator commit | `fdeff5bebae7a2df25e146564f47780285783a55` |
| Remote branch | `origin/refactor/readability-phase2`, confirmed at `fdeff5b` |
| N30 immutable source | `/data/run01/scz0ade/Tanzeyu/code/formal_static_rgb_fdeff5b` |
| N30 source mode | read-only (`dr-xr-xr-x`) |
| Source archive SHA-256 | `03a8099809ef2242a55b7a59526a614647b53d7e78db091bce9b7e106a6ab56d` |
| Frozen manifest validation | PASS: 20 samples and 19 checkpoint records |
| Local contract tests | PASS: 21 tests |
| Local CTC end-to-end synthetic smoke | PASS |
| N30 immutable-snapshot CPU smoke | PASS: 23 tests, including fail-closed CLI exit semantics |
| GPU/Slurm preflight | COMPLETE; affected-branch result recorded below |
| Full formal jobs | NOT SUBMITTED |

The hidden archive retained on N30 is
`/data/run01/scz0ade/Tanzeyu/code/.incoming_formal_static_rgb_fdeff5b.tar.gz`.
It was transferred resumably and its SHA-256 matched before extraction. The
snapshot contains `SOURCE_COMMIT` with the full runtime commit.

The repository may contain many unrelated untracked drafts/results owned by the
user. Do not stage, delete, or modify them. `AGENTS.md` is also untracked and was
not included in either formal commit.

## Frozen scope

- 8iVFB: 4 samples.
- Owlii: 4 author-compatible vox10 samples.
- CTC: exactly 12 samples listed in
  `configs/evaluation/formal_static_rgb_v1_samples.tsv`.
- Ours points: `32K / 16K / 8K / 4K / 2K / 1K / 512`.
- Original Unicorn: nine official physical-rate points; lambda 8192 always uses
  the frozen `32k8k@8192` policy.
- Excluded: ScanNet, lossless, dynamic, reflectance, and geometry evaluation.

CTC uses the released-code-consistent operational contract: named PLY
`x,y,z,red,green,blue` extraction, global coordinates, no recentering, no
requantization, no deduplication, pinned KD-tree chunks of at most 800,000
points, and serial processing. The source must equal the multiset union of all
chunks. The historical author-provided `dataset_processing.py` is evidence only
and its training-style 100k/local-recentering path is not the formal protocol.

N30 data already present:

- Owlii vox10: `/data/run01/scz0ade/Tanzeyu/data/Owlii_vox10_floor_v1/`
- CTC ten vox12 objects:
  `/data/run01/scz0ade/Tanzeyu/data/scalable_attribute_thesis/datasets/unicorn_official/ctc_vox12/`
- CTC native basketball/dancer vox11:
  `/data/run01/scz0ade/Tanzeyu/data/scalable_attribute_thesis/datasets/unicorn_official/owlii_vox11/`

Do not substitute Owlii vox10 basketball/dancer for the CTC native vox11
versions, and do not use same-named vox20 files.

## Implemented runtime semantics

The runtime commit adds the CTC adapter, frozen-contract validator, independent
upstream Unicorn physical runner, Ours formal evaluator, chunk aggregator, and
affected-branch preflight reducer under
`scripts/scalable_attribute/evaluation/`.

Each Ours `(sample, point)` formal task shares one physical Base computation and
emits both endpoints. Base is atomically published before Enhancement starts.
Base becomes `FORMAL_REUSABLE` after its own gates pass; a later Full failure
leaves Base reusable and makes the combined task `PARTIAL`. Full has separate
bit, reconstruction, metric, and runtime evidence.

Endpoint codec runtime is defined as:

- Base = Prefix encode + Prefix decode + Base synthesis.
- Full = complete Base runtime + Enhancement encode + Enhancement decode.

Model loading, metric, PLY I/O, peak VRAM, and task wall time are recorded
separately. CTC aggregation sums physical bits and runtime components over
chunks, merges global-coordinate reconstructions, and evaluates each merged
endpoint exactly once. It never averages per-chunk PSNR.

`--formal-stop-after-base` is preflight-only. Compare its Base bit components
and reconstruction hash/value with the Base emitted by the combined invocation.
The full formal batch must use the combined Base+Full path.

## Completed GPU preflight (2026-09-09/10)

Evidence roots:

- Original runtime attempt:
  `/data/run01/scz0ade/Tanzeyu/experiments/formal_static_rgb_preflight_6ad2e1d_a01`
- Final CTC compatibility rerun:
  `/data/run01/scz0ade/Tanzeyu/experiments/formal_static_rgb_preflight_fdeff5b_a02`

| Gate | Slurm ID | Result | Peak VRAM | Slurm elapsed |
| --- | ---: | --- | ---: | ---: |
| 8i Original retained-overlap reproduction | `5450247` | PASS; exact reconstruction/rate/YUV match | 6,553,411,584 B | 00:02:27 |
| Ours 8K combined plus isolated Base equality | `5450248` | Scientific PASS; wrapper false-negative exposed and fixed | 5,446,075,392 B | 00:03:38 |
| Ours rescued 2K branch | `5450249` | PASS | 5,446,075,392 B | 00:02:42 |
| Ours joint 4K branch | `5450250` | PASS | 5,446,621,184 B | 00:02:45 |
| CTC House 32K largest actual chunk on RTX3090 | `5450261` | PARTIAL: Base reusable; Full CUDA OOM | 21,429,910,528 B | 00:01:59 |
| CTC Thaidancer Original R03, final runtime | `5450307` | PASS; merged endpoint reusable | 6,468,605,952 B | 00:08:40 |
| CTC Thaidancer Ours 8K, final runtime | `5450308` | PASS; merged Base and Full reusable | 6,607,802,880 B | 00:14:09 |

The 8K Base isolated and combined paths are bit/reconstruction identical:
`69,320` physical bits and reconstruction SHA-256
`48b0f5a0e085b3ca5d77bee43dda3f7bf2aeff67593cde8432cd1f2da421be0b`.
The isolated task is intentionally `PARTIAL` because Full is not attempted;
its Base endpoint remains `FORMAL_REUSABLE`.

Final Thaidancer CTC preparation produced four chunks of
`782553/782554/782554/782554` points.  The source and chunk-union six-tuple
multiset SHA-256 both equal
`cfe126972beda481d2d9389793953cf6040d883593bead38be6981be26ff6ce3`.
The chunks retain global coordinates and exact integer-valued payloads while
declaring XYZ as float for MPEG `pc_error` 0.13.4 compatibility.  Original and
Ours used the same chunk manifest.  Merge verifies the full coordinate
multiset and evaluates the full 3,130,215-point source once per endpoint.

Two fail-closed runtime bugs discovered by preflight were fixed before the
final CTC rerun: evaluator/aggregator CLI return codes now propagate via
`SystemExit`, and the CTC metric-only canonical source plus merged PLY use
float-declared XYZ without changing content.  Runtime commits are `a4a9e1b`
and `fdeff5b`; the latter is the only final execution snapshot.

The House gate proves that RTX3090 is not sufficient for the tested 606,094
point 32K Full decode.  Per affected-branch policy, CTC 32K Full must use a
larger-memory HPC GPU while keeping the same <=800k scientific chunk contract.
Do not reduce chunk size merely to fit RTX3090.  The House Base result is
already `FORMAL_REUSABLE` evidence, but this preflight output is not a
substitute for the full formal task.

## Original mandatory stop point and proposed GPU preflight

The user explicitly required the agent to stop and present the plan before any
preflight submission. No `sbatch` has been run. The next agent must not submit a
GPU job until the user confirms the following proposal:

1. Reproduce one retained 8i Original Unicorn physical overlap point from the
   pinned independent upstream source and compare rate/metric evidence.
2. Run Ours standard 8K as combined Base+Full plus isolated Base and require
   identical Base components and reconstruction.
3. Exercise the rescued 2K, joint 4K, and sequential 32K checkpoint/loader
   branches independently.
4. Run converted/chunked Thaidancer through serial CTC aggregation to cover its
   exceptional PLY property order and both Original/Ours result envelopes.
5. Run an 800k `House_without_roof` chunk as the RTX3090 peak-VRAM/resource
   gate. If it is too large, move only that affected branch to a larger HPC GPU;
   do not change the 800k scientific contract.
6. Preserve every job's logs and endpoint evidence independently. A failure
   must not cancel unrelated jobs.

Affected-branch gating is mandatory. Dataset/sample/loader/operating-point
failures HOLD only matching branches. Only a shared metric, physical-rate,
identity, or other global scientific-contract failure may HOLD all 48 formal
branches (Original 9 points plus Ours 7 points across three datasets).

## Immediate next action after preflight review

Do not launch the full formal batch until the user reviews this preflight
evidence and explicitly approves the final N30/HPC split.  In particular,
route only the affected CTC 32K Full-capable work to a larger-memory HPC GPU;
keep independent N30 RTX3090 jobs for branches that passed their resource gate.

After preflight, report at minimum:

```text
Dataset | Data verified | Unicorn complete | Ours complete | BD-BR ready | Status
```

The final formal deliverables remain physical bpp, Y/U/V/YUV611, runtime,
per-sequence RD/BD-BR, dataset summaries, failures, and explicit
`FORMAL_REUSABLE` labels for both Base and Full.
