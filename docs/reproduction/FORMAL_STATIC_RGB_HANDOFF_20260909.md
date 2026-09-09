# Formal static RGB evaluation handoff — 2026-09-09

This file is the restart point if chat context is lost. Read it together with
`FORMAL_8I_OWLII_CTC_EXECUTION_PLAN_20260909.md`. The execution plan is the
scientific contract; this file records live operational state.

## Current state

| Item | State |
| --- | --- |
| Scientific-contract freeze | `e264168` |
| Runtime evaluator commit | `6ad2e1ddfdb598495e224b8e731b8a89e3566ffa` |
| Remote branch | `origin/refactor/readability-phase2`, confirmed at `6ad2e1d` |
| N30 immutable source | `/data/run01/scz0ade/Tanzeyu/code/formal_static_rgb_6ad2e1d` |
| N30 source mode | read-only (`dr-xr-xr-x`) |
| Source archive SHA-256 | `9a486124ddb869023de316e7a375c43c54448ed9308e32b88a8cbfee778b1d2b` |
| Frozen manifest validation | PASS: 20 samples and 19 checkpoint records |
| Local contract tests | PASS: 21 tests |
| Local CTC end-to-end synthetic smoke | PASS |
| N30 immutable-snapshot CPU smoke | PASS: 22 tests, including evaluator CLI imports |
| GPU/Slurm preflight | NOT SUBMITTED |
| Full formal jobs | NOT SUBMITTED |

The hidden archive retained on N30 is
`/data/run01/scz0ade/Tanzeyu/code/.incoming_formal_static_rgb_6ad2e1d.tar.gz`.
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

## Mandatory stop point and proposed GPU preflight

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

## Immediate next action after user confirmation

Create preflight commands/scripts that reference only the immutable N30 source
snapshot and frozen manifests, show the exact resolved job matrix once more,
then submit the independent preflight jobs. Do not launch the full formal batch
until preflight evidence has been reduced by `evaluate_preflight_gates.py`.

After preflight, report at minimum:

```text
Dataset | Data verified | Unicorn complete | Ours complete | BD-BR ready | Status
```

The final formal deliverables remain physical bpp, Y/U/V/YUV611, runtime,
per-sequence RD/BD-BR, dataset summaries, failures, and explicit
`FORMAL_REUSABLE` labels for both Base and Full.
