# Formal 8iVFB + Owlii + CTC evaluation plan — 2026-09-09

Status: **SCIENTIFIC CONTRACT RETAINED / EXECUTION COMPLETE**

This is the frozen pre-execution contract, preserved verbatim below where it
describes the original preparation state. The matrix subsequently closed at
`320/320 FORMAL_REUSABLE`; use
`results/comparisons/formal_static_rgb_final_20260911/` for final values and
`docs/repository/CURRENT_RESULT_INDEX.md` for current navigation. Future-tense
submission and preflight statements in this document are historical, not live
task status.

## 1. Objective

Produce a formal static lossy RGB comparison on N30:

1. Original Unicorn physical;
2. Ours Base physical;
3. Ours Full physical.

The sample scope is 4 8iVFB + 4 Owlii + 12 CTC. Ours uses only
512/1K/2K/4K/8K/16K/32K. ScanNet, lossless, dynamic, reflectance, and geometry
experiments are outside this run.

## 2. Evidence established before submission

### Data

- Membership is 20/20: 4 8iVFB, 4 author-compatible Owlii vox10, and the fixed
  12-item CTC set.
- Four 8i PLYs and ten CTC vox12 PLYs match the bytes downloaded from the
  author-linked NJU Box archives.
- The four Owlii vox10 inputs re-derive byte-for-byte from native vox11 with the
  retained `floor(xyz/2)`, duplicate RGB mean, then round adapter and match the
  author CSV point counts.
- CTC basketball and dancer use native vox11, not the Owlii vox10 derivatives.
- The ten CTC archives were transferred directly from FAU to N30 by the reverse
  pull route and archive hashes match at both endpoints.

The exact file hashes and point counts are recorded in
`N30_8I_OWLII_CTC_SAMPLE_ALIGNMENT_20260909.md`.

### Official source and results

- Formal upstream source is `NJUVISION/Unicorn` commit
  `b50d6c1bd033185b9e893b755d5d316cca2d4448`.
- N30 contains an independent upstream snapshot and all three official object
  checkpoints. Critical source files match the pinned commit.
- Official formal points are R01--R09 with global first-occurrence
  deduplication; R03 is always `32k8k@8192`.
- The N30 8i official run is complete at 4 samples x 9 rates. It used only the
  independent upstream workspace, and its default and physical reconstructions
  are identical at all 36 points. Reuse it; do not rerun it.
- Released author CSV rows remain `AUTHOR_PROVIDED` paper-reference evidence.
  They use likelihood/default rates and are not mixed into physical BD-BR.
- Historical author-provided preprocessing evidence is retained at
  `scripts/data/dataset_processing.py` (SHA-256
  `2f2babef28af29df84dcf497c2913d2bf08e90d2014c11cfa7e7b125a435f7c7`).
  The file is unchanged from the thesis repository's initial prototype commit.
  Its color partition path is training-oriented: it ignores the CLI point limit,
  hard-codes 100,000-point KD-tree parts, and subtracts each part's XYZ minimum.
  It is evidence about the authors' historical preprocessing family, not a
  published CTC formal-evaluation manifest.

### Ours

- The current seven-point mapping exists in
  `configs/scalable_attribute/current_candidates.json`.
- All 16 unique released/bootstrap/effective checkpoint files required by the
  seven points exist on N30 and have newly measured SHA-256 values.
- Retained 8i/Owlii Base/Full rows cover the seven requested points and are useful
  regression oracles. The proposed formal run nevertheless re-evaluates Ours on
  all 20 samples so every result shares one code snapshot, manifest, evaluator,
  and output layout.
- The current branch is `refactor/readability-phase2` at
  `001681f5e0d825f52d97602413a14d4d7e8eaa46`. A new immutable N30 source
  snapshot must be created before evaluation; the mutable N30 code directory is
  not sufficient provenance.

### Metric and rate contract

- Both methods use the same canonical RGB content, MPEG `pc_error`, resolution
  1, Y/U/V PSNR, and `YUV611=(6Y+U+V)/8`.
- The denominator is the full-resolution input point count.
- Original Unicorn physical rate is exact entropy-string bytes plus physical
  G-PCC `x_low` bits under the pinned upstream convention.
- Ours Base is `x_low+r1+r2+r3+r4`; Ours Full is Base plus the physical
  enhancement payload. Integer bit identities and hard decode equality are run
  gates.

## 3. Scientific ambiguities requiring approval

### A. Candidate registry status — approved

The task calls the selected checkpoints frozen, but the only consolidated
registry currently says `working_candidates_not_final_freeze` and each entry says
`working_candidate`. The user explicitly approved freezing the following mapping
without using test performance to alter it:

| Point | Base | Full |
| --- | --- | --- |
| 512 | 2k128 Base step3525 | independent enhancement step1763 |
| 1K | 2k128 Base step3525 | independent enhancement step1763 |
| 2K | 2K-U-PATH rescue step500 | rescued enhancement step1500 |
| 4K | 8K-to-4K joint step3000 Base | same joint checkpoint Full |
| 8K | canonical Base step3525 | independent enhancement step1763 |
| 16K | canonical Base step3525 | independent enhancement step1763 |
| 32K | D111 Base step5525 | sequential D611 enhancement step3525 |

Before submission, copy the complete paths, file sizes, and SHA-256 values into
an immutable run-local checkpoint manifest.

### B. CTC input and chunk contract — approved

Six official vox12 files are binary little-endian, while the pinned test loader
calls an ASCII-only parser. Their PLY property layouts are also inconsistent:
Thaidancer places `red,green,blue` immediately after xyz; the other five place
normals before RGB. A positional conversion would therefore corrupt at least one
sample.

The official repository contains a named-color Open3D reader and a deterministic
KD-tree partitioner whose current pinned CLI default maximum is 800,000 points.
The paper states that large 11/12-bit samples were chunked and processed
serially, but it does not publish the exact CTC invocation or chunk manifest.
The older author-provided `dataset_processing.py` does not resolve this gap: its
100,000-point, per-part-local-coordinate path is training-style and materially
changes the test coordinate representation. It must not be promoted to the
formal CTC protocol merely because it is author-provided.

Frozen CTC policy, following the paper's explicit chunk-and-serial statement and
minimum intervention:

1. read named `x,y,z,red,green,blue` from every source PLY;
2. make only the binary-to-six-column-ASCII format adaptation; preserve global
   coordinates and RGB values exactly;
3. use the current pinned upstream KD-tree algorithm with `max_num=800000` for
   formal evaluation of all 12 CTC samples, with chunks processed serially. Do
   not recenter, requantize, or otherwise alter part coordinates;
4. freeze source/output hashes, point counts, properties, part ordering, and a
   lossless union check, then feed the exact same parts to Original Unicorn and
   Ours;
5. sum exact physical bits over parts, divide by the original
   full-sample point count, merge reconstructed parts in global coordinates, and
   run `pc_error` once against the full canonical source.

Whole-file execution may be used only as a resource diagnostic; it cannot select
or change the formal protocol. First preflight the 800,000-point contract on an
RTX3090. If it exceeds 24 GiB, retain exactly the same data/chunk/rate/metric
contract and move the affected evaluation to a larger-memory FAU HPC GPU. Do not
silently descend to the old 100,000-point/local-coordinate path. The frozen
contract is a defensible reproduction using pinned released code, but it cannot
be claimed byte-identical to the unpublished author chunk manifest. The
distinction must be stated in final provenance.

## 4. Implementation and preflight after approval

### Ours Base/Full endpoint and runtime contract

The formal Ours unit is one `(sample, operating point)` task which emits both
Base and Full.  It must not run Base and Full as independent Slurm tasks.
This follows the codec graph: Full contains the exact Base physical prefix and
is conditioned on the decoded Base tensors.  Sharing that computation prevents
input, checkpoint, or prefix drift and avoids encoding the same Base twice.

For 32K, 16K, 8K, 2K, 1K, and 512, the independent Enhancement checkpoint must
name the exact frozen Base checkpoint and released checkpoint before loading.
The paths and measured hashes must equal the run-local frozen checkpoint
manifest.  For 4K, the one frozen joint scalable checkpoint owns Prefix,
BaseSynthesis, and Enhancement; both endpoints must be taken from the same
single `hard_reconstruct()` result.

Every task must retain two independent endpoint records and two reconstruction
files when both endpoints complete.  Each record contains integer bit
components, physical bpp, Y/U/V/YUV611, runtime components, file hashes, and
point-count/round-trip gates.  Base is published before Enhancement is
attempted, so a Full failure leaves explicit `BASE_PASS_FULL_FAIL` evidence
rather than erasing the Base result.  Formal status is endpoint-level: a Base
that passes all of its own gates is `FORMAL_REUSABLE` even if Full later fails;
the combined task is then `PARTIAL`.  Full is `FORMAL_REUSABLE` only after all
Base-dependency and Full-specific gates pass.

The following equalities are hard gates, not descriptive claims:

1. `Base_bits = x_low_bits + r1_bits + r2_bits + r3_bits + r4_bits`;
2. `Full_bits = Base_bits + Enhancement_bits`;
3. the Base reconstruction written as the Base endpoint is exactly the Base
   tensor supplied to both Enhancement encode and Enhancement decode;
4. an isolated Base-only preflight invocation and the Base inside the combined
   invocation have identical bit components and reconstruction values/hashes;
5. encoded and decoded Full reconstructions are exactly equal.

Runtime excludes model/checkpoint loading, output PLY writing, and `pc_error`.
Those costs are retained separately.  CUDA work is synchronized at every
boundary.  Record at least:

- Base shared physical-prefix encode and decode;
- Base synthesis;
- Enhancement encode and decode;
- Base metric time, Full metric time, I/O time, peak VRAM, and task wall time.

Report standalone endpoint codec runtimes even though the combined task shares
work:

- `Base codec runtime = prefix encode + prefix decode + Base synthesis`;
- `Full codec runtime = Base codec runtime + Enhancement encode + Enhancement decode`.

Thus Full receives the complete cost required to obtain its Base dependency;
the saved task wall time is not substituted for either endpoint runtime.  For
CTC, sum each runtime component over serial chunks, then run and record the
full-sample metric once.  Never average per-chunk PSNR or per-chunk runtime.

The pre-existing evaluator's generic `seconds` field does not satisfy this
contract: Base timing starts after the shared prefix/Base computation, while
Full timing omits that shared computation and includes metric/I/O.  Retained
old `seconds` values are therefore regression evidence only, not formal runtime.

Before any long array:

1. Create a tracked run contract and immutable N30 source snapshot from the
   approved Git commit. Record code, evaluator, codec, dataset, adapter, chunk,
   checkpoint, environment, and command hashes.
2. Build a manifest-driven experiment-local harness. The Original Unicorn path
   imports only the pinned upstream modules and official checkpoints; it does not
   import thesis model code. A narrow adapter exposes exact integer physical bits
   because the released `model.test()` rounds bpp to 0.001.
3. Run CPU/static gates: exact 20-item membership, SHA-256, PLY schema, named RGB,
   chunk union, no duplicate/lost points, checkpoint metadata, task uniqueness,
   output non-overwrite, and BD-BR fixtures.
4. Run a small GPU preflight covering:
   - Original Unicorn 8i overlap against the retained formal run;
   - Original Unicorn on converted/chunked Thaidancer;
   - the largest CTC sample (`House_without_roof`) for memory/runtime;
   - Ours standard loader, rescued 2K loader, joint 4K loader, and sequential
     32K loader;
   - exact hard round-trip, integer bit identities, merged reconstruction point
     count, and peak VRAM.
5. Stop instead of launching the full batch if any identity, metric, rate,
   checkpoint, chunk-union, or whole-task resource gate fails.

Preflight failures use affected-branch gating.  A dataset-, loader-, sample-,
or operating-point-specific failure places only that branch on `HOLD` while
unaffected branches remain eligible.  A global hold is reserved for failures
in shared metric semantics, physical-rate semantics, source/checkpoint identity
machinery, or another scientific-contract invariant used by every branch.

Commit `e264168` freezes the scientific plan only.  After the evaluator is
implemented and its non-GPU contract and synthetic-pipeline smoke gates pass,
create and push a distinct runtime-code commit.  The immutable N30 source
snapshot must be created from that exact runtime-code commit; it must not use
`e264168` as executable-code provenance.  GPU preflight then runs from that
snapshot and remains subject to explicit pre-submission review.

The runtime-code commit tracks the executable contract in these entry points:

- `prepare_ctc_formal_chunks.py`: pinned 800k KD-tree partition, named PLY
  properties, global coordinates, and source/chunk multiset verification;
- `evaluate_unicorn_official_physical.py`: pinned upstream Original Unicorn
  physical coding, including normalized CTC `chunk_result.json` evidence;
- `evaluate_8ivfb_sequence.py --ours-formal`: one shared hard pass that
  atomically publishes Base before Full and records separate endpoint bits,
  reconstruction, quality, and component runtime;
- `evaluate_8ivfb_sequence.py --ours-formal --formal-stop-after-base`:
  preflight-only isolated Base invocation for the required equality check;
- `aggregate_formal_chunk_results.py`: serial-chunk bit/runtime sums followed
  by one merged full-sample metric per endpoint;
- `validate_formal_static_rgb_contract.py` and `evaluate_preflight_gates.py`:
  frozen-manifest validation and affected-branch launch eligibility.

## 5. Proposed full-batch shape

- Reuse: 8i Original Unicorn 4 x 9; zero new 8i upstream tasks.
- New upstream: Owlii + CTC = 16 samples x 9 rates = 144 independent tasks.
- New Ours: all 20 samples x 7 points = 140 independent tasks; each task emits
  Base and Full from the same frozen input/point contract.
- Use one GPU per task, no DDP. Use a manifest-driven Slurm array with a global
  concurrency cap of 16 RTX3090 tasks after preflight. Each task has its own
  output directory and atomic PASS/FAIL status. One failed task does not cancel
  any other task.
- Use conservative per-task resources initially: 6 CPUs, 60 GB host memory, and
  4 hours. Tighten or increase only from preflight evidence before the full array,
  not from measured RD performance.
- Submit aggregation with `afterany`, not `afterok`, so failures still produce a
  complete status matrix and retained evidence.

Historical 8i/Owlii evaluations were typically minutes per sample/point, but CTC
chunk timing must be established by preflight. No reliable completion estimate
is assigned before that evidence exists.

## 6. Aggregation and acceptance artifacts

The run root will be immutable and contain:

- dataset, chunk, checkpoint, code, environment, metric, and rate manifests;
- per-task command, Slurm ID, stdout/stderr, status, timing, VRAM, physical bit
  breakdown, and reconstruction checks;
- per-sequence physical RD tables for Original Unicorn, Ours Base, and Ours Full;
- per-sequence Y and YUV611 BD-BR using a predeclared log-rate PCHIP method over
  the shared quality interval with no extrapolation or result-dependent point
  removal;
- equal-sequence-weight dataset summaries and an explicit incomplete/unavailable
  status where a curve lacks enough valid overlap;
- a final matrix:
  `Dataset | Data verified | Unicorn complete | Ours complete | BD-BR ready | Status`.

Only rows that pass every provenance, identity, physical-rate, metric, and
round-trip gate receive `FORMAL_REUSABLE`.

## 7. Approval record

The user explicitly confirmed both:

1. freeze the seven checkpoint lineages in Section 3A despite the registry's
   current `working_candidate` label;
2. freeze shared no-recenter pinned KD-tree `max_num=800000` as the formal CTC
   protocol, never the historical 100k/local-coordinate path. Whole-file
   feasibility does not select the formal protocol.

Submit RTX3090 preflight first. Submit the full arrays only after all preflight
gates pass; otherwise stop and report the evidence. An RTX3090 OOM changes only
the execution site to a larger-memory FAU GPU, not the evaluation contract.
