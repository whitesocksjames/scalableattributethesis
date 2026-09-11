# Formal static RGB evaluation handoff — 2026-09-10

This is the current restart point for the thesis `8iVFB + Owlii + CTC`
formal evaluation. Read it together with
`FORMAL_8I_OWLII_CTC_EXECUTION_PLAN_20260909.md`. The execution plan is the
scientific contract; this file is an operational snapshot.

Scheduler states in this document are historical observations, not live
claims. On recovery, inspect point artifacts and query each scheduler at most
once when the user requests a status audit. Do not poll continuously.

## Final closure — 2026-09-11

This section supersedes every incomplete-count or pending-job statement later
in this historical handoff.

- The deduplicated formal matrix is **320/320 `FORMAL_REUSABLE`**: 180/180
  Original Unicorn and 140/140 Ours logical points across 20/20 samples.
- House job `1809256` and Shiva job `1809401` each produced a complete
  16-point A100-SXM4-40GB set (Original 9 + Ours 7). These complete A100 sets
  are the final House/Shiva RD and runtime selections; older V100/RTX3090
  artifacts remain historical consistency evidence.
- House Ours old-versus-memory-optimized results match exactly on all 12
  shared Base/Full endpoints, including bits, reconstruction SHA-256, and
  Y/U/V/YUV611. Cross-GPU House/Shiva comparisons preserve exact checkpoint
  and chunk provenance and show only bounded numerical differences (at most
  56 bits, 0.002172%, and 0.0004 dB), so cross-GPU reconstruction hashes are
  not universally identical.
- The final classified artifacts are under
  `results/comparisons/formal_static_rgb_final_20260911/`.
- The raw-point competitiveness verdict is 19 competitive, one mixed
  (Facade), and zero clearly weaker samples.
- Under the frozen no-pruning/no-extrapolation PCHIP contract, 68/80
  per-sequence Base/Full × Y/YUV611 BD-BR values are available. Twelve remain
  explicitly unavailable: four for boxer (Ours nonmonotonic) and four each for
  CTC loot and redandblack (Official nonmonotonic).
- All 460 selected endpoint rows have complete current-formal encode/decode
  timings. Runtime remains hardware-labeled and House/Shiva use their complete
  A10040 sets.

The experiment-collection phase is frozen. Do not submit additional points to
make unavailable BD-BR entries computable, and do not delete or reorder raw
points to change those entries.

## Latest recovery checkpoint — A100 closure jobs submitted

This section is the newest operational state and supersedes older scheduler
observations later in this document. It records the most recent user-requested
FAU query on 2026-09-10; it is not a live scheduler claim.

### Formal matrix

- Current artifact-verified count: **318/320 logical points
  `FORMAL_REUSABLE`**.
- 8iVFB: Original 36/36 and Ours 28/28.
- Owlii: Original 36/36 and Ours 28/28.
- CTC: Original 108/108 and Ours 82/84.
- The only missing logical points are House Ours 32K and Shiva Ours 32K.
- A submitted replacement run does not increase this count. The matrix remains
  exactly 320 logical points, and complete-sample reruns must be deduplicated
  against the same logical IDs.

Facade is complete at 16/16. House and Shiva each have 15/16 reusable logical
points from retained evidence, but their complete-sample A100 reruns below are
still pending. Existing valid results remain preserved and are not overwritten.

### Runtime implementation and immutable source

- Memory-optimized evaluator commit:
  `6150f45a7144bb4d43f7196ae342f20b260d7fd6`.
- Immutable FAU runtime snapshot: `formal_static_rgb_6150f45`.
- The optimization moves validation-only tensor residency to CPU and releases
  encoded GPU outputs before decode; it does not change model, checkpoint,
  bitstream, reconstruction, chunk, endpoint, metric, or timing semantics.
- Old-vs-new equivalence smoke passed 25/25 checks, including exact physical
  bits and exact Base/Full reconstruction SHA values.

### Current FAU closure jobs

The latest single scheduler query found all four jobs below `PENDING`, with no
node assigned, zero elapsed runtime, and no point output yet:

| job | purpose | allocation contract | points | output root suffix | latest state |
| --- | --- | --- | ---: | --- | --- |
| `1809097` | House 32K fresh-process diagnostic | A100, default allocator | diagnostic only | `formal_house32k_chunk0007_diag_5f717fbb_a01` | PENDING |
| `1809098` | House 32K allocator diagnostic | A100, `max_split_size_mb:128` if baseline fails | diagnostic only | same diagnostic root | PENDING |
| `1809256` | House complete hardware-consistent rerun | **A100 40GB only**, one allocation | Original 9 + Ours 7 | `formal_static_rgb_house_6150f45_a100_a01` | PENDING, 0/16 |
| `1809401` | Shiva complete hardware-consistent rerun | **A100 40GB only**, one allocation | Original 9 + Ours 7 | `formal_static_rgb_shiva_6150f45_a100_a02` | PENDING, 0/16 |

Both complete-sample jobs use the frozen canonical input and shared chunk
manifest, pinned official Unicorn source, frozen seven-point Ours checkpoint
mapping, physical-bit and metric contract, failure-continue behavior, and
point-level resume/evidence. Each also has a job-side A100 40GB hardware gate.
Do not cancel, replace, or duplicate these jobs without a new reason supported
by scheduler or artifact evidence.

### Retained Shiva V100 attempt

Job `1809252` ran the same 16-point Shiva pack on a V100 32GB and ended
`FAILED` only at the pack level because one point failed:

- Original Unicorn R01-R09: 9/9 `FORMAL_REUSABLE`.
- Ours 16K through 512: 6/6 `FORMAL_REUSABLE`.
- Ours 32K: Base passed on chunk 0, but Full OOM requested an additional
  3.15 GiB with 3.06 GiB free; no valid 32K aggregate exists.
- The V100 run therefore produced 15/16 valid point aggregates and remains
  historical/formal evidence, but it is not the final hardware-consistent
  runtime set. The A100-only job `1809401` must run all 16 points.

### Read-only pre-final audits

The 318-point integrity audit found zero pipeline/scientific hard errors:
checkpoint mapping, pinned official provenance, endpoint completeness,
physical-bit identities, YUV611 identity, CTC aggregation, and runtime fields
all passed. House/Shiva incomplete points remain unresolved rather than being
treated as errors.

The classified RD competitiveness review is committed at
`ab01e86251de3d978d2aff6f9a964df5929336d5` under
`results/comparisons/formal_rd_competitiveness_20260910/`. Its primary entry is
the directory `README.md`, followed by `report/REVIEW.md`; figures, summary
tables, and normalized non-sensitive evidence are separated into dedicated
subdirectories. At the 318-point boundary it reports 17 competitive samples,
one mixed sample (Facade), no clearly weaker sample, and House/Shiva pending.
This is a descriptive performance review, not a replacement for the frozen
BD-BR contract.

### Exact continuation procedure

1. Do not query FAU again unless the user explicitly asks; then query once and
   inspect point aggregates/status, not scheduler state alone.
2. For `1809256` and `1809401`, require all 16 point aggregates to be `PASS`
   and `FORMAL_REUSABLE`; Ours requires both Base and Full endpoint status.
3. Confirm every point in each new sample set records the same A100 40GB model,
   then compare new RD, physical bits, and reconstruction evidence with the
   retained runs. Timing comparison for House/Shiva should use the complete
   A100-only sets, not mixed V100/RTX3090/A100 timing.
4. If both missing Ours 32K points pass, recount the deduplicated matrix as
   320/320. Do not count the complete-sample reruns as additional logical
   points.
5. Preserve all failed and superseded attempts as evidence. Do not delete
   inconvenient raw points or alter the frozen checkpoint, chunk, rate,
   metric, or BD-BR interpolation contract.

## Frozen scientific contract

- Contract ID: `formal_static_rgb_v1_20260909`.
- Original Unicorn source: pinned official `NJUVISION/Unicorn` commit
  `b50d6c1bd033185b9e893b755d5d316cca2d4448`.
- Ours points remain frozen at `512/1K/2K/4K/8K/16K/32K`.
- Relocation-validator implementation commit:
  `5f717fbb520018d6f2938a1852cc9413ca5dd09c`
  (`validate relocated formal checkpoint identity`).
- A previously reported long SHA for this change was erroneous. The commit
  above is the sole authoritative code provenance for the repair.
- Evaluated runtime checkpoint-manifest SHA-256:
  `d43641b2ba93536b9f5d95fd52fe2e796faa9eb18de2c03a76bd3e9a7b616484`.
- Publication-redacted checkpoint-manifest SHA-256:
  `f28fffcfe6ce1aa5dae5876ad7df6d2128d75a6ac26ff5b9f16efcdccdb501e3`.
  The publication copy changes only the private `origin_root`; checkpoint
  relative paths, sizes, content hashes, loaders, profiles, lambdas, and the
  frozen scientific selection are unchanged.

CTC uses the frozen KD-tree chunk contract: at most 800,000 points per chunk,
global coordinates, no recentering, no requantization, no deduplication, and
verified source-equals-chunk-union identity. Original Unicorn and Ours use the
same canonical input and chunk manifest.

Base and Full are endpoint-level results. An Ours logical point is complete
only when both endpoints independently pass all formal gates. A valid Base
endpoint is not invalidated by a later Full failure, but Base alone does not
complete the Base+Full logical point.

## Meaning of the 320-point matrix

The formal matrix contains exactly 320 logical points:

- Original Unicorn: 20 samples × 9 rates = 180 points.
- Ours: 20 samples × 7 operating points = 140 points. Each Ours point contains
  both Base and Full endpoints.

Runtime-only reruns do not add logical points to this matrix. In particular,
the 8i Original Unicorn runtime-completion run supplements the existing 36
8i Original points; it does not increase the denominator beyond 320.

## Most recent complete progress audit

The last complete artifact-level audit was performed after N30 array
`5451266_[0-3]` and the older FAU jobs had ended, but before the replacement
jobs listed below were submitted.

| dataset / subset | Original complete | Ours complete | reusable logical points |
| --- | ---: | ---: | ---: |
| 8iVFB, 4 samples | 36/36 | 28/28 | 64 |
| Owlii, 4 samples | 36/36 | 28/28 | 64 |
| CTC, ten N30 samples | 90/90 | 69/70 | 159 |
| CTC House | 9/9 | 0/7 | 9 |
| CTC Facade | 0/9 | 0/7 | 0 |
| **Total** | **171/180** | **125/140** | **296/320** |

At that audit boundary:

- `FORMAL_REUSABLE`: 296.
- `RUNNING`: 0.
- `PENDING`: 0.
- previously attempted `FAILED/PARTIAL`: 8 logical points (House Ours 7 and
  Shiva Ours 32K).
- not yet reached: 16 logical points (Facade 9 Original + 7 Ours).
- total work required to close the matrix: 24 logical points.

These categories describe the last complete artifact audit. Subsequent job
submission does not make a point reusable; only a valid point-level aggregate,
status, and provenance record can do that.

## Completed datasets and reusable results

- 8i Ours: 28/28 `FORMAL_REUSABLE`.
- Owlii Ours: 28/28 `FORMAL_REUSABLE`.
- Owlii Original Unicorn: 36/36 current-formal results complete.
- 8i Original Unicorn: 36/36 retained locally reproduced upstream physical
  RD results remain the frozen RD baseline, subject to the runtime distinction
  below.
- N30 CTC excluding House/Facade: Original 90/90 and Ours 69/70 complete.
- House Original Unicorn: 9/9 point aggregates are valid and retained; the
  replacement House job must not rerun them.

Do not count a point from Slurm state alone. Require `PASS` plus
`FORMAL_REUSABLE`; Ours additionally requires both Base and Full endpoint
statuses to be `FORMAL_REUSABLE`.

## Relocation-validator repair

The older `a299e8a` evaluator compared checkpoint lineage with absolute
`realpath()` values. Byte-identical FAU copies therefore failed closed even
though their frozen mapping and SHA-256 values were correct.

Commit `5f717fbb520018d6f2938a1852cc9413ca5dd09c` repairs the semantics without
weakening the identity gate:

- relocated artifacts must match the frozen manifest and content SHA-256;
- architecture, lambda/profile, loader branch, and parent lineage are checked;
- basename equality is never sufficient;
- wrong content, parent lineage, lambda/profile, or architecture fails closed;
- the scientific mapping and checkpoint bytes are unchanged.

The corresponding CPU/static suite passed 17 tests, including byte-identical
relocation PASS and wrong content/lineage/config FAIL cases. New runtime
snapshots were created without overwriting the snapshots used by old jobs.

## FAU replacement jobs

The following were the one-time immediate observations just after submission
on 2026-09-10:

| job | purpose | logical points | last observed state | output root suffix |
| --- | --- | ---: | --- | --- |
| `1809059` | House Ours 32K–512 only; retain old Original 9/9 | 7 | RUNNING | `formal_static_rgb_house_5f717fb_retry_a01` |
| `1809060` | Facade Original R01–R09 + Ours 32K–512 | 16 | RUNNING | `formal_static_rgb_facade_5f717fb_retry_a01` |

Facade first performs fail-closed reuse verification of the canonical chunks
and shared manifest. Its canonical input SHA-256 remains
`63f93f79aee5fccacd21b05a71b09aeff59a877a397cad1a4d65eb6478877be4`.
Both jobs permit either A100 or V100 32GB and record the actual GPU hardware.

Old reusable FAU evidence:

- House Original 9/9 is retained.
- Old House Ours 7/7 failed because of the absolute-path validator and needs
  the replacement run.
- Old Facade jobs produced no reusable logical point before failing chunk
  preparation/resume gates.

## Shiva 32K resource failure

Shiva Original is 9/9 complete and Ours is 6/7 complete. Ours 32K is not
reusable:

- the first 504,566-point chunk produced a valid Base endpoint;
- Full failed on RTX3090 with a CUDA out-of-memory error while requesting an
  additional 3.15 GiB;
- the logical point has no valid aggregate and remains `PARTIAL`/`HOLD`;
- an FAU large-memory retry is still required under the sample/platform
  assignment policy.

Do not count the valid partial Base as a completed Base+Full logical point, and
do not silently alter chunking, checkpoints, rate semantics, or metrics to make
the point fit.

## 8i Original Unicorn runtime completion

The retained 8i Original 36/36 artifacts are locally reproduced upstream
physical RD results. They are not historical author CSVs and remain the frozen
RD baseline. However, their legacy `enctime`/`dectime` measurements do not have
current-formal timing semantics: the old upstream code used wall-clock timing
without explicit CUDA synchronization.

The current official physical runner synchronizes CUDA around encode/decode
and excludes input reading, checkpoint loading, PLY writing, and `pc_error`
from those two timings. Therefore a runtime-completion run was submitted solely
to obtain comparable `encode_seconds` and `decode_seconds`. Its new physical
bits, reconstruction hashes, and RD values must be checked against the retained
reproduction; it does not automatically replace the frozen RD baseline.

| job | purpose | supplementary tasks | last observed state | output root suffix |
| --- | --- | ---: | --- | --- |
| `5453189_[0-3]` | four sample-level 8i Original runtime packs | 36 | 4/4 RUNNING | `formal_static_rgb_8i_runtime_5f717fb_a01` |

Owlii Original 36/36 already has current-formal encode/decode timing. Ours 8i
28/28 and Owlii 28/28 have complete Base and Full runtime fields.

## Recovery procedure

1. Read this file and the formal execution plan.
2. Treat the scheduler states above as historical only.
3. On an explicitly requested status audit, query N30 and FAU once each and
   inspect point-level output/status/provenance.
4. Recount the 320 logical points from valid artifacts, not job completion.
5. Preserve House Original 9/9 and all other `FORMAL_REUSABLE` points.
6. Determine whether jobs `1809059`, `1809060`, and `5453189_[0-3]` produced
   valid results; retry only missing or invalid points.
7. Arrange the still-required Shiva 32K large-memory retry without changing
   the scientific contract.
8. After all points pass, produce per-sequence Y/YUV611 RD and BD-BR plus the
   dataset-level summary, with runtime hardware clearly identified.
