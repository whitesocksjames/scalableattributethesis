# Unicorn Part II static lossy RGB reproduction audit

Status: **AUDIT COMPLETE / NO EXPERIMENTS STARTED**
Audit date: 2026-09-08
Scope: static point-cloud RGB attribute compression with lossless geometry.

This document converts the Unicorn Part II paper, the pinned upstream source,
and the released result artifacts into a concrete reproduction and data-acquisition
plan. It does not remove sparse samples, freeze the thesis primary comparison
set, or claim that existing thesis-repository compatibility results are an
independent upstream reproduction.

## 1. Evidence and precedence

Evidence was checked in this order:

1. Local paper: `references/unicorn_part_ii_attribute.pdf`, SHA256
   `4dd71c82f69c09c0a4d136d4c2540d804eb22b0283f263a7ff69ab0210eb8b86`.
   The paper is *A Versatile Point Cloud Compressor Using Universal Multiscale
   Conditional Coding — Part II: Attribute*, IEEE TPAMI 47(1), January 2025,
   DOI `10.1109/TPAMI.2024.3462945`.
2. Official upstream: <https://github.com/NJUVISION/Unicorn>, pinned at
   `b50d6c1bd033185b9e893b755d5d316cca2d4448`. The clean local checkout is
   `/home/liltan/projects/Unicorn-Clean-Source`; its origin is the official
   repository and its working tree was clean during this audit.
3. Released source/result artifacts under
   `Unicorn-family/Unicorn-v1/results/PCAC/` in that clean checkout.
4. Released attribute checkpoints from the official attribute checkpoint portal:
   <https://box.nju.edu.cn/f/7162590c2a46489291bd/>. The extracted local files
   currently live under the separate resource-bearing workspace
   `/home/liltan/projects/Unicorn-NJUVISION/.../local_resources/pretrained/`.

If the paper prose, current source, and retained CSV disagree, this report does
not silently choose one. The discrepancy is listed in Section 7. The formal
duplicate-lambda decision is now frozen in
`docs/reproduction/UNICORN_FORMAL_BASELINE_CONTRACT.md`.

The local availability search was deliberately bounded to
`/home/liltan/datasets`, `/home/liltan/scratch`, and `/home/liltan/projects`.
No broad `/mnt` scan, N30/HPC scan, download, or GPU evaluation was performed.

Status terms:

- `AVAILABLE`: the exact evaluation input was located and identified.
- `PARTIAL`: source material exists, but the required derived representation or
  its provenance is not yet frozen.
- `MISSING`: the exact expected item was not found in the audited roots.
- `UNKNOWN`: the paper/repository does not expose enough information to name or
  verify the exact required item.

Current readiness at a glance:

| Dataset | Local readiness | Immediate blocker |
| --- | --- | --- |
| 8iVFB | AVAILABLE, 4/4 exact frames | none; freeze canonical copy/hash |
| Owlii | PARTIAL, 4/4 raw vox11 frames | regenerate/freeze the paper's 10-bit derivatives |
| CTC | PARTIAL, 10/12 exact inputs | Thaidancer and Staue_Klimt PLYs |
| ScanNet | MISSING data, UNKNOWN exact manifest | exact 100 scan IDs and q2cm preprocessing contract |

## 2. Paper lossy experiment matrix

### 2.1 Static object RGB datasets

| Dataset | Exact paper/released sample | Geometry precision | Paper evidence | Released result evidence | Expected role |
| --- | --- | ---: | --- | --- | --- |
| 8iVFB | `longdress_vox10_1300.ply` | 10-bit | Table II; Fig. 8 8i average; Table III per-sequence | `ours/8ivfb/longdress_vox10_1300.csv` and matching G-PCC CSV | paper reproduction; thesis primary candidate |
| 8iVFB | `loot_vox10_1200.ply` | 10-bit | same | `ours/8ivfb/loot_vox10_1200.csv` and matching G-PCC CSV | same |
| 8iVFB | `redandblack_vox10_1550.ply` | 10-bit | same | `ours/8ivfb/redandblack_vox10_1550.csv` and matching G-PCC CSV | same |
| 8iVFB | `soldier_vox10_0690.ply` | 10-bit | same | `ours/8ivfb/soldier_vox10_0690.csv` and matching G-PCC CSV | same |
| Owlii | `basketball_player_vox11_00000200.ply`, converted/evaluated at 10-bit | 10-bit target | Table II; Fig. 8 Owlii average; Table III per-sequence | `ours/owlii_vox10/basketball_player_vox11_00000200.csv` and G-PCC CSV | paper reproduction; thesis primary candidate |
| Owlii | `dancer_vox11_00000001.ply`, converted/evaluated at 10-bit | 10-bit target | same | `ours/owlii_vox10/dancer_vox11_00000001.csv` and G-PCC CSV | same |
| Owlii | `model_vox11_00000001.ply`, converted/evaluated at 10-bit | 10-bit target | same | `ours/owlii_vox10/model_vox11_00000001.csv` and G-PCC CSV | same |
| Owlii | `exercise_vox11_00000001.ply`, converted/evaluated at 10-bit | 10-bit target | same | `ours/owlii_vox10/exercise_vox11_00000001.csv` and G-PCC CSV | same |

The exact CTC set is determined by the paper's “twelve point clouds”, Table III,
and the released plotting notebook together. The notebook indices `8..19`
provide the definitive 12-item retained-result ordering:

| # | Exact CTC sample | Precision | Paper/result evidence | Expected role |
| ---: | --- | ---: | --- | --- |
| 1 | `basketball_player_vox11_00000200.ply` | 11-bit | Table III; notebook index 8 | paper reproduction; thesis comparison pending predeclared rule |
| 2 | `dancer_vox11_00000001.ply` | 11-bit | Table III; notebook index 9 | same |
| 3 | `Thaidancer_viewdep_vox12.ply` | 12-bit | Table III; notebook index 10 | same |
| 4 | `longdress_viewdep_vox12.ply` | 12-bit | Table III; notebook index 11 | same |
| 5 | `loot_viewdep_vox12.ply` | 12-bit | Table III; notebook index 12 | same |
| 6 | `redandblack_viewdep_vox12.ply` | 12-bit | Table III; notebook index 13 | same |
| 7 | `soldier_viewdep_vox12.ply` | 12-bit | Table III; notebook index 14 | same |
| 8 | `boxer_viewdep_vox12.ply` | 12-bit | Table III; notebook index 15 | same |
| 9 | `Facade_00009_vox12.ply` | 12-bit | Table III; notebook index 16 | same |
| 10 | `House_without_roof_00057_vox12.ply` | 12-bit | Table III; notebook index 17 | same |
| 11 | `Shiva_00035_vox12.ply` | 12-bit | Table III; notebook index 18, marked sparse in notebook | paper reproduction; do not pre-emptively exclude |
| 12 | `Staue_Klimt_vox12.ply` | 12-bit | Table III; notebook index 19, marked sparse in notebook | paper reproduction; do not pre-emptively exclude |

The paper reports per-sample CTC BD-BR in Table III and average CTC Y/YUV curves
in Fig. 8. The released repository contains per-sequence Unicorn and G-PCC CSVs
for all 12. Sparse samples remain in `PAPER_REPRODUCTION_SET`; whether they later
belong to `THESIS_PRIMARY_COMPARISON_SET` must be declared before examining our
formal result.

### 2.2 Static scene RGB dataset

| Dataset | Exact sample set | Resolution | Paper evidence | Released result evidence | Expected role |
| --- | --- | ---: | --- | --- | --- |
| ScanNet | 100 test scans from 1,603 total scans; exact IDs not published in the checked paper/repo | 2 cm | Table II; Fig. 8 ScanNet average; Table III aggregate | `ours/scan2cm.csv` and `gpcc/raht21/scan2cm.csv` | paper reproduction, blocked on manifest/data |

The paper states 1,503 scans for training and 100 for testing following common
conditions, cited as MPEG document m58754. The current upstream config points at
`ScanNet/scans_test_q2cm/`, but no frozen list of those 100 scan IDs was found.
The ordinary ScanNet semantic benchmark split must not be assumed to be the same
set without evidence.

### 2.3 Explicitly out of scope

KITTI and Ford are static-scene **reflectance** experiments, not RGB. Dynamic
attribute coding, lossless attribute coding, lossy geometry+attribute coding,
and geometry experiments are also excluded. Fig. 9 is qualitative evidence for
the same lossy-attribute mode, not a separately specified quantitative dataset.

The paper also compares SparsePCAC, ScalablePCAC, and GDL-PCAC. The official
Unicorn result bundle checked here exposes reproducible curve inputs for Unicorn
and G-PCC, but not an independent pinned implementation/checkpoint workflow for
all those learned baselines. Their paper numbers may be retained as
`AUTHOR_PROVIDED`; they are not currently local-reproduction targets.

## 3. Local availability matrix

### 3.1 8iVFB

All four exact single frames are `AVAILABLE` and need no new download:

| Sample | Local path | Size |
| --- | --- | ---: |
| longdress 1300 | `/home/liltan/projects/Unicorn-NJUVISION/Unicorn-family/Unicorn-v1/local_resources/testdata/mpeg/longdress_vox10_1300/longdress_vox10_1300.ply` | 19,408,770 B |
| loot 1200 | `/home/liltan/projects/Unicorn-NJUVISION/Unicorn-family/Unicorn-v1/local_resources/testdata/mpeg/loot_vox10_1200/loot_vox10_1200.ply` | 17,543,324 B |
| redandblack 1550 | `/home/liltan/projects/Unicorn-NJUVISION/Unicorn-family/Unicorn-v1/local_resources/testdata/mpeg/redandblack_vox10_1550/redandblack_vox10_1550.ply` | 16,335,043 B |
| soldier 0690 | `/home/liltan/projects/Unicorn-NJUVISION/Unicorn-family/Unicorn-v1/local_resources/testdata/mpeg/soldier_vox10_0690/soldier_vox10_0690.ply` | 24,274,354 B |

Duplicate copies and full 8i sequences also exist under
`/home/liltan/datasets/pointcloud/8i/`. The formal manifest should select one
canonical copy and record its hash rather than mix roots.

### 3.2 Owlii

Owlii is `PARTIAL`, not missing. All four exact raw vox11 PLYs are present:

| Sample | Local path | Size | Remaining requirement |
| --- | --- | ---: | --- |
| basketball_player 0200 | `/home/liltan/datasets/pointcloud/Owlii/basketball_player_vox11_00000200.ply` | 171,109,244 B | freeze author-compatible 10-bit derivative and hash |
| dancer 0001 | `/home/liltan/datasets/pointcloud/Owlii/dancer_vox11_00000001.ply` | 145,265,491 B | same |
| exercise 0001 | `/home/liltan/datasets/pointcloud/Owlii/exercise_vox11_00000001.ply` | 123,321,863 B | same |
| model 0001 | `/home/liltan/datasets/pointcloud/Owlii/model_vox11_00000001.ply` | 139,467,504 B | same |

The project has previously validated the intended conversion as coordinate
`floor(xyz / 2)`, duplicate-voxel RGB mean, then rounding, including exact point
count fingerprints. However, the four derived formal PLYs were not found in the
audited local roots and the current upstream README does not freeze that exact
conversion. The formal manifest must therefore record the adapter commit,
source hashes, output hashes, and point counts. Do not download the raw data
again merely because the derived files are absent.

### 3.3 CTC 11/12-bit set

The set is `PARTIAL`: 10/12 exact inputs are locally available.

| Sample | Status | Local evidence / action |
| --- | --- | --- |
| basketball_player vox11 | AVAILABLE | same raw Owlii path above; use at native 11-bit for CTC |
| dancer vox11 | AVAILABLE | same raw Owlii path above; use at native 11-bit for CTC |
| Thaidancer viewdep vox12 | **MISSING** | obtain `Thaidancer_viewdep_vox12.ply` |
| longdress viewdep vox12 | AVAILABLE | local resource PLY, 195,058,948 B |
| loot viewdep vox12 | AVAILABLE | local resource PLY, 199,144,311 B |
| redandblack viewdep vox12 | AVAILABLE | local resource PLY, 174,548,982 B |
| soldier viewdep vox12 | AVAILABLE | local resource PLY, 264,119,264 B |
| boxer viewdep vox12 | AVAILABLE | local resource PLY, 230,547,109 B |
| Facade 00009 vox12 | AVAILABLE | local resource PLY, 84,579,046 B |
| House_without_roof 00057 vox12 | AVAILABLE | local resource PLY, 260,177,467 B |
| Shiva 00035 vox12 | AVAILABLE | local resource PLY, 53,850,001 B |
| Staue_Klimt vox12 | **MISSING** | obtain `Staue_Klimt_vox12.ply` |

The eight 12-bit available paths share the prefix
`/home/liltan/projects/Unicorn-NJUVISION/Unicorn-family/Unicorn-v1/local_resources/testdata/mpeg/`.

The upstream README identifies the MPEG static-object source and an official
NJU Box mirror: <https://box.nju.edu.cn/d/51327ae7c2644c0fa1c4/>. The mirror
listing observed during the audit reports compressed sizes of 74,360,616 B for
Thaidancer and 9,192,803 B for Staue_Klimt, about 79.7 MiB combined. Public MPEG
PCC mirrors also expose view-dependent files, but their ZIP sizes differ from
the NJU mirror. The downloaded PLY must therefore be verified by filename,
header/point count, and hash before it is admitted; matching names alone are
not sufficient.

The paper notes that large 11/12-bit samples were chunked and processed serially
under GPU-memory constraints. No frozen per-chunk manifest or aggregation rule
was found in the released source/results. Before reproduction, either reproduce
the released whole-file path on adequate-memory hardware or freeze one shared,
deterministic chunk contract for Original Unicorn and ours. Do not let each
method partition differently.

### 3.4 ScanNet

ScanNet q2cm is `MISSING` for data and `UNKNOWN` for the exact 100-scan manifest.
No actual `scans_test_q2cm` dataset was found in the audited roots. The released
Unicorn/G-PCC aggregate CSVs and ScanNet checkpoints are present, but they do
not reconstruct the sample identities.

ScanNet access requires accepting its Terms of Use with an institutional email;
the official source is <https://github.com/ScanNet/ScanNet>. Raw reconstructed
meshes use names such as `sceneXXXX_YY_vh_clean.ply`. The exact storage size for
the paper's 100 q2cm-derived scans cannot be responsibly estimated until the
scan-ID list and chosen raw mesh asset are known.

Required before download:

1. obtain m58754 or an author-provided `scans_test_q2cm` file list;
2. freeze the exact raw ScanNet asset (`*_vh_clean.ply` or another documented
   source) and the 2 cm quantization/color procedure;
3. only then estimate/download the required subset under ScanNet's license.

## 4. Original Unicorn provenance and operating points

### 4.1 Workspace rule

Formal `LOCALLY_REPRODUCED_UPSTREAM` results must come from an independent,
pinned official NJUVISION/Unicorn workspace. The clean checkout at commit
`b50d6c1...` is suitable after environment and assets are recorded. The modified
resource-bearing `/home/liltan/projects/Unicorn-NJUVISION` workspace is useful
as an asset store but must not itself be represented as a clean upstream run.
The thesis repository's `evaluate_unicorn_reference.py` also does not qualify.

### 4.2 Released checkpoint mapping

For object point clouds (`8iVFB`, `Owlii`, and CTC), upstream
`lossy_attribute/test.py:get_piecewise_variable_bitrates('object')` maps:

| Profile checkpoint | Lambdas | Local extracted checkpoint SHA256 |
| --- | --- | --- |
| `rwtt/32k8k/epoch_last.pth` | 32768, 16384, 8192 | `bdd40b608ca865de3a30595ab084dcaf48450c2c9843661790c61caf9885fc90` |
| `rwtt/8k256/epoch_last.pth` | 8192, 4096, 2048 | `c3e38cd37b9bcfae99cf87bbf3cb1001574497df9bab33be238a8927abe1d389` |
| `rwtt/2k128/epoch_last.pth` | 1024, 512, 256, 128 | `c8152c03adcaa0384ad4b168abda03d79da5f8474e9de2ac88d15395cbfe87aa` |

The current source globally deduplicates repeated lambdas by retaining the first
occurrence, producing nine unique object points. The formal policy therefore
uses `32k8k@8192` globally for 8iVFB, Owlii, and CTC; it never chooses a lambda
8192 family per sequence. The retained author CSVs are not uniform: 8i has nine
unique rows, while Owlii and most CTC CSVs retain ten rows including a family
boundary duplicate. Those rows remain unmodified `AUTHOR_PROVIDED` paper
references rather than formal checkpoint-selection instructions; see Section 7.

ScanNet uses independent checkpoints:

| Profile checkpoint | Source lambdas | Local extracted checkpoint SHA256 |
| --- | --- | --- |
| `scan2cm/variable/32k8k/epoch_last.pth` | 32768, 16384, 8192 | `2928dd39c031a45e13cec3518abfa37da18becd2c1e2efce39e7627b56cbc49d` |
| `scan2cm/variable/8k256/epoch_last.pth` | 8192, 4096, 2048, 1024 | `241e72a7a56b070cc2cb39c7cf481dc644f52ccbaa3aa2dff64f1dc4d3383d97` |
| `scan2cm/variable/2k256/epoch_last.pth` | 1024, 512, 256 | `adde0ec9f471e45074a6fe512883a44f63cf1b16dfcfdd014de6126a824984bf` |

These hashes identify the current local extracted files. A formal run register
should additionally record the downloaded checkpoint archive hash or otherwise
prove how these extracted files came from the official portal.

### 4.3 Rate convention

The upstream source exposes two distinct rate paths:

- Default `LossyAttributeCoder.test(..., real_coding=False)` reports `bpp` from
  likelihood-derived residual rate plus a physically measured G-PCC `x_low`
  attribute rate. The retained author CSVs use a `bpp` column and were generated
  by this default call path according to the checked script.
- `real_coding=True` calls `MultiscaleVAE.test()`, physically entropy-codes each
  residual stream, counts `len(strings) * 8`, and adds the physical G-PCC
  `x_low` bits. `min_v`/`max_v` are needed to decode but are not added to rate.

Both follow the upstream convention that the `x_low` tensor is passed into the
decode path after physical G-PCC rate measurement rather than requiring a
file-container/fresh-process G-PCC decode closure. Container headers and
`min_v`/`max_v` overhead are not thesis blockers, as already agreed.

This dual path matters: `AUTHOR_PROVIDED` curves from retained CSVs must not be
silently treated as physical arithmetic-stream measurements. For a formal
physical-rate comparison, rerun pinned upstream Unicorn with its hard path and
run ours under the matching convention. For a paper-figure reproduction based
on author CSVs, label those rows `AUTHOR_PROVIDED` and do not mix them into a
physical-rate BD-BR calculation unless equivalence is first demonstrated.

### 4.4 Metric and anchor convention

The paper specifies bpp, MPEG `pc_error`, Y-PSNR and YUV-PSNR, BD-BR, lossless
geometry, and G-PCC TMC13v21 with RAHT for the static lossy anchor. The released
source invokes `pc_error_d` with reference and reconstruction, resolution 1,
Hausdorff enabled, and color enabled. It forms:

`YUV-PSNR = (6 * Y + U + V) / 8`.

The upstream G-PCC helper uses color-space conversion and RAHT
`transformType=0` for lossy attribute coding. Original Unicorn and ours must use
the same input, denominator, rate mode, pc_error binary/config, rounding policy,
and sequence aggregation before formal RD/BD-BR comparison.

## 5. Released result evidence

The clean upstream result bundle contains:

- four 8i Unicorn CSVs and four matching G-PCC RAHT21 CSVs;
- four Owlii vox10 Unicorn CSVs and four matching G-PCC CSVs;
- two Owlii native-vox11 CSV pairs used as the first two CTC samples;
- ten 12-bit CTC Unicorn CSVs and ten matching G-PCC CSVs;
- ScanNet q2cm aggregate Unicorn and G-PCC CSVs;
- `results_object.ipynb`, which constructs 8i, Owlii, 12-sample CTC, and ScanNet
  curves; and `utils.py:mean_file()`, which averages numeric rows by row index.

The G-PCC RAHT21 CSVs have six QP rows (`22, 28, 34, 40, 46, 51`) for every
listed object sequence and ScanNet. Unicorn has nine rows for each 8i sequence,
ten for each Owlii/CTC sequence, and ten for ScanNet. In the retained notebook,
the plotted Unicorn length is nine for 8i/Owlii and ten otherwise. Consequently,
the Owlii paper figure uses the first nine retained rows: it includes both
lambda-8192 family points and omits the final lambda-128 row.

The result bundle is strong `AUTHOR_PROVIDED` evidence and supplies the exact
CTC sequence identity missing from the paper prose. It is not a substitute for
an independent local run when the provenance label is
`LOCALLY_REPRODUCED_UPSTREAM`.

## 6. Proposed set semantics

No result-dependent exclusion is made in this audit.

- `PAPER_REPRODUCTION_SET`: 8i 4 + Owlii 4 + all CTC 12 + the paper's exact
  ScanNet 100-scan q2cm set once recovered.
- `THESIS_PRIMARY_COMPARISON_SET`: **not frozen yet**. The candidate scope is
  8i/Owlii/CTC. Any CTC exclusion (including Shiva or Staue_Klimt) must use a
  written, content-independent rule frozen before our formal result is examined.
- `SUPPLEMENTARY_SET`: **not assigned yet**. Do not move inconvenient paper
  samples here retroactively.

ScanNet should remain a paper-reproduction target. Whether it is also a thesis
primary comparison requires a separate decision because it uses ScanNet-specific
released checkpoints and its exact test manifest is currently absent.

## 7. Unresolved paper/source/result inconsistencies

1. **8i frame order typo:** paper prose lists sequences as longdress, loot,
   redandblack, soldier but frames as 1300, 1550, 1200, 0690 “respectively”. The
   released filenames and standard selection establish loot 1200 and
   redandblack 1550. Use filenames, and document the paper prose error.
2. **CTC count:** paper prose says 12 but names only 11, omitting boxer. Table III
   and the released notebook/CSVs include `boxer_viewdep_vox12`, resolving the
   operational set to 12.
3. **Owlii naming versus precision:** paper calls the four tests 10-bit, while
   released CSV filenames retain `vox11`. The exact 11-to-10 preprocessing is
   not frozen in the current upstream evaluation source.
4. **Current source versus retained object curves:** current source deduplicates
   repeated lambdas and yields nine unique object points. The frozen formal
   baseline follows this behavior and uses `32k8k@8192` globally. Retained 8i
   CSVs have nine rows; retained Owlii vox10 CSVs have ten rows with two
   lambda-8192 family points. The notebook plots only their first nine, so the
   released Owlii curve is not the current source's nine-unique-lambda curve.
   Both are preserved with different provenance and must not be combined by
   per-sequence best-point selection.
5. **CTC row identity:** retained CTC CSVs generally have ten rows but duplicated
   boundary lambdas are not fully uniform: House duplicates 2048/512, Shiva
   duplicates 32768/512, while most duplicate 8192/512. Lambda alone is therefore
   not a sufficient rate-point key; preserve row index, profile, lambda, and
   measured bpp.
6. **CTC aggregation:** the notebook performs row-wise arithmetic mean. Given the
   nonuniform duplicate-lambda lineage above, reproducing the published average
   requires preserving author row order; a new normalized nine-point curve is a
   different contract and must be labeled as such.
7. **ScanNet manifest:** the exact 100 scan IDs and exact q2cm generation path are
   not present in the checked paper/source/result bundle.
8. **Estimated versus physical rate:** retained author CSV `bpp` follows the
   default likelihood-rate path, while the upstream hard path can report actual
   arithmetic payload bytes. These are related but not automatically identical.
9. **Checkpoint archive provenance:** local extracted checkpoint hashes are
   recorded, but the official downloaded archive hash/extraction record is not
   yet in the formal run register.

## 8. MISSING_DATA_ACTION_LIST

### Must obtain or resolve

| Item | Why | Source/action | Approximate size |
| --- | --- | --- | ---: |
| `Thaidancer_viewdep_vox12.ply` | one of 12 paper CTC samples | official NJU Box/MPEG source; validate PLY and hash | 74,360,616 B compressed on NJU mirror |
| `Staue_Klimt_vox12.ply` | one of 12 paper CTC samples | official NJU Box/MPEG source; validate PLY and hash | 9,192,803 B compressed on NJU mirror |
| Exact ScanNet 100-scan IDs | paper ScanNet result cannot be independently reproduced without it | obtain MPEG m58754 or author manifest | negligible list size |
| ScanNet q2cm input/derivation contract | prevents substituting an unrelated ScanNet split/preprocess | identify source mesh and exact 2 cm quantization/color procedure | unknown |
| ScanNet data for frozen 100 scans | required after the two preceding contracts are fixed | request ScanNet access under Terms of Use and download only required assets | unknown until manifest/asset fixed |
| Owlii formal vox10 derivatives | raw data exists, but formal target files are absent locally | regenerate once with frozen adapter; record source/output hashes and point counts | no new raw download |

### Recommended before formal runs

- Create immutable dataset manifests containing exact file, hash, point count,
  geometry precision, source URL/license, and preprocessing identity.
- Copy or symlink the released checkpoint assets into the independent pinned
  upstream workspace through machine-specific configuration; record extracted
  checkpoint hashes and official package provenance.
- Preserve the author's retained row sequence as `PAPER_REFERENCE`; use the
  current source's globally deduplicated points as `FORMAL_UNICORN_BASELINE`.
  Do not call both the same curve or select between them per sequence.
- Freeze one CTC large-sample execution/partition contract shared by Original
  Unicorn and ours.
- Verify one 8i result end-to-end before launching all rates, including whether
  the intended formal comparison uses estimated or physical rate.

### No download currently needed

- 8iVFB four static frames.
- Owlii four raw vox11 frames.
- The ten already available CTC inputs.
- KITTI/Ford, dynamic sequences, reflectance assets, lossless-only assets, or
  geometry datasets outside this audit scope.

## 9. Recommended reproduction order

1. **Freeze provenance/contracts:** pinned clean upstream commit, environment,
   checkpoint package record, dataset manifests, pc_error/TMC13v21 binaries,
   rate mode, and aggregation.
2. **8i pilot:** one frame × one rate for Original Unicorn and G-PCC, then compare
   against the corresponding author CSV. This cheaply detects metric/rate drift.
3. **8i complete:** all four frames and the frozen curve points. It is fully
   locally available and has no preprocessing blocker.
4. **Owlii preprocess freeze and complete:** generate the four vox10 derivatives,
   validate fingerprints, then run the same frozen protocol.
5. **CTC complete:** acquire the two missing PLYs, freeze large-sample handling,
   then run all 12, including Shiva and Staue_Klimt. Report per-sequence before
   any aggregate.
6. **ScanNet last:** only after the exact 100-scan manifest and q2cm procedure are
   recovered. Do not substitute the common semantic split.
7. **Ours comparison:** admit Ours Base/Full results into
   `results/comparisons/static_rgb_v1/` only after their sample/rate/metric/
   aggregation contracts match the locally reproduced upstream baseline.

This order minimizes wasted computation: the immediately actionable gap is only
two CTC archives plus Owlii deterministic preprocessing. ScanNet is the sole
large unresolved data/provenance blocker.
