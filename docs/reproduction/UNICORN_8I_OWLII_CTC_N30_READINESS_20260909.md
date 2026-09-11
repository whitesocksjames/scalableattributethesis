# Original Unicorn 8iVFB / Owlii / CTC N30 readiness — 2026-09-09

## Decision

The nominal 20-sample input set is now complete on N30: the 8iVFB baseline is
already complete, Owlii vox10 is ready to run, and all 12 CTC source samples are
present. The CTC sweep is still not execution-contract-complete because six
official view-dependent PLYs are binary little-endian while the pinned upstream
loader calls `read_ply_ascii()`, and a whole-file VRAM smoke test is still
required. Data availability is no longer the blocker.

The frozen formal baseline is `NJUVISION/Unicorn` commit
`b50d6c1bd033185b9e893b755d5d316cca2d4448`, with nine points:

| Point | Profile | Lambda |
| --- | --- | ---: |
| R01 | `rwtt/32k8k` | 32768 |
| R02 | `rwtt/32k8k` | 16384 |
| R03 | `rwtt/32k8k` | 8192 |
| R04 | `rwtt/8k256` | 4096 |
| R05 | `rwtt/8k256` | 2048 |
| R06 | `rwtt/2k128` | 1024 |
| R07 | `rwtt/2k128` | 512 |
| R08 | `rwtt/2k128` | 256 |
| R09 | `rwtt/2k128` | 128 |

`lambda=8192` is always the `32k8k` checkpoint. The formal comparison rate is
physical G-PCC `x_low` bits plus physical entropy-string bytes. Released author
CSV rates remain `AUTHOR_PROVIDED` paper-reference evidence and must not be
silently relabelled as physical rates.

## Data and run readiness

| Dataset | Paper samples | Released author CSVs | FAU HPC inputs | N30 status | Action |
| --- | ---: | --- | --- | --- | --- |
| 8iVFB vox10 | 4 | 4 x 9 rows | 4/4 | 4 x 9 upstream default and physical runs complete; aggregate PASS | Do not rerun unless a same-hardware timing rerun is wanted |
| Owlii vox10 | 4 | 4 x 10 retained rows; paper plots first 9 | 4/4 author-compatible derivatives | 4/4 derivatives already present at `/REDACTED/N30_PROJECT_ROOT/data/Owlii_vox10_floor_v1/` | Ready for pinned 9-point run |
| CTC | 12 | 12 x 10 retained rows, aggregated by row index | 10/10 object ZIPs downloaded from the official NJU Box share; two native Owlii vox11 inputs were already available | 12/12 source inputs present | Freeze binary-input handling, run a representative VRAM smoke, then launch the 12 x 9 sweep |

Before this audit, FAU was not a complete source: its project workspace
contained 183 GiB of datasets overall, but the exact CTC search found none of
the ten 12-bit object inputs. On 2026-09-09 all ten were downloaded to FAU from
the official NJU Box public share and transferred to N30.

## NJU Box acquisition and HPC-to-N30 transfer

The public Seafile share permits anonymous downloads. Its ten CTC object ZIPs
total 603,664,732 bytes. Every archive passed `unzip -t`, contains exactly one
same-named PLY, and has matching SHA-256 on FAU and N30.

| ZIP | Bytes | SHA-256 |
| --- | ---: | --- |
| `Facade_00009_vox12.zip` | 29,108,120 | `ef5fe5b85a2ef46d9ce3bd683479a2c2e5d926bbe2d23e7ebf56158c1fb64459` |
| `House_without_roof_00057_vox12.zip` | 83,746,746 | `c25ca48e072ab950f7c70050d5a525fdbe74dc4856eb3acc04c181a9d67eab1c` |
| `Shiva_00035_vox12.zip` | 19,423,490 | `46bc86b198fa16b526d79aee4dc3ca48d741caafe64bbad9873cf25e506fe712` |
| `Staue_Klimt_vox12.zip` | 9,192,803 | `2e246b2adf310c472e668c5ca63f24ccc6623e489deeb35182ed4a65609fd5bc` |
| `Thaidancer_viewdep_vox12.zip` | 74,360,616 | `b78e7a69e3c08e1a12753326ac023f734aca2d1a522f2192d303312419d43455` |
| `boxer_viewdep_vox12.zip` | 86,597,776 | `1b36a94d6b83a84d8314bfeab40d790f98c85845a0b037366c805e8500c478e5` |
| `longdress_viewdep_vox12.zip` | 80,604,607 | `31cc3aba73b3e7b47068e3fbb5d2802d438b05ce02eee459536d9dd55877fb55` |
| `loot_viewdep_vox12.zip` | 62,480,436 | `692ce93c9863e9f6dd161de2a53104020ee6682ab9326c55306bcb8b030490e1` |
| `redandblack_viewdep_vox12.zip` | 65,820,231 | `2b22bf38a3d5365b636917ca5a4f9d269da09e6cf0372951aac408826d069567` |
| `soldier_viewdep_vox12.zip` | 92,329,907 | `518d4e127e6a9ba1fd78d7bf428ab9a4e4b47ce4111909039905181c22298d58` |

Locations:

- FAU ZIPs: `/REDACTED/FAU_PROJECT_ROOT/datasets/unicorn_official/ctc_zips/`
- N30 ZIPs: `/REDACTED/N30_PROJECT_ROOT/data/scalable_attribute_thesis/datasets/unicorn_official/ctc_zips/`
- N30 extracted PLYs: `/REDACTED/N30_PROJECT_ROOT/data/scalable_attribute_thesis/datasets/unicorn_official/ctc_vox12/`
- N30 native Owlii vox11 inputs: `/REDACTED/N30_PROJECT_ROOT/data/scalable_attribute_thesis/datasets/unicorn_official/owlii_vox11/`

The resulting N30 CTC input inventory matches the 12-sample author CSV
contract:

| Official sample | Bytes | Points | PLY format | N30 |
| --- | ---: | ---: | --- | --- |
| `basketball_player_vox11_00000200.ply` | 171,109,244 | 2,925,514 | ASCII | present |
| `dancer_vox11_00000001.ply` | 145,265,491 | 2,592,758 | ASCII | present |
| `Thaidancer_viewdep_vox12.ply` | 206,597,700 | 3,130,215 | binary little-endian | present |
| `longdress_viewdep_vox12.ply` | 195,058,948 | 3,096,122 | binary little-endian | present |
| `loot_viewdep_vox12.ply` | 199,144,311 | 3,017,285 | binary little-endian | present |
| `redandblack_viewdep_vox12.ply` | 174,548,982 | 2,770,567 | binary little-endian | present |
| `soldier_viewdep_vox12.ply` | 264,119,264 | 4,001,754 | binary little-endian | present |
| `boxer_viewdep_vox12.ply` | 230,547,109 | 3,493,085 | binary little-endian | present |
| `Facade_00009_vox12.ply` | 84,579,046 | 1,596,085 | ASCII | present |
| `House_without_roof_00057_vox12.ply` | 260,177,467 | 4,848,745 | ASCII | present |
| `Shiva_00035_vox12.ply` | 53,850,001 | 1,009,132 | ASCII | present |
| `Staue_Klimt_vox12.ply` | 26,162,100 | 499,660 | ASCII | present |

FAU could not initiate SSH to the N30 gateway: the connection timed out before
authentication during banner exchange. The working no-local-data route is the
reverse direction: N30 uses the FAU key and `HPC_JUMP_HOST` as an SSH jump
host, then pulls from `HPC_LOGIN_HOST` with `rsync --append-verify`. The pull
completed at about 9.66 MB/s and did not route file payloads through the local
workstation. The authorized FAU credential is isolated at
`/REDACTED/N30_PROJECT_ROOT/.ssh_transfer/hpc_identity` with mode 600; its
directory has mode 700.

N30 already contains the pinned upstream source and all three released RWTT
checkpoints. Their SHA-256 values are:

| Profile | SHA-256 |
| --- | --- |
| `32k8k` | `bdd40b608ca865de3a30595ab084dcaf48450c2c9843661790c61caf9885fc90` |
| `8k256` | `c3e38cd37b9bcfae99cf87bbf3cb1001574497df9bab33be238a8927abe1d389` |
| `2k128` | `c8152c03adcaa0384ad4b168abda03d79da5f8474e9de2ac88d15395cbfe87aa` |

## Reproduction alignment with released author evidence

The existing N30 8iVFB run contains 4 sequences x 9 points x 3 row roles = 108
comparison rows. Default and physical paths reconstruct byte-identical point
clouds for every point.

| Check | Result |
| --- | ---: |
| Maximum absolute default-minus-author bpp, all rows | 0.0240 |
| Maximum absolute default-minus-author YUV611, all rows | 0.4363 dB |
| Maximum absolute default-minus-author bpp, excluding R03 | 0.0010 |
| Maximum absolute default-minus-author YUV611, excluding R03 | 0.0124 dB |
| Physical/default reconstruction identity | 36/36 |

The R03 outliers are expected provenance evidence: the released 8i CSVs used
different historical R3/R4 choices at the duplicate `lambda=8192` boundary,
whereas the frozen formal baseline globally chooses `32k8k@8192`. They are not
evidence that the pinned upstream reproduction is broken.

## Existing thesis Ours versus official-curve interpolation

These are diagnostics already present in
`results/multirate_256_32k_audit_20260904/MULTIRATE_256_32K_SUMMARY.csv`.
Positive delta means Ours is above the local official Unicorn interpolation at
the same physical bpp; no extrapolation is used.

| Point | 8i Base delta | 8i Full delta | Owlii Base delta | Owlii Full delta |
| --- | ---: | ---: | ---: | ---: |
| 256 | +0.089 | +0.081 | +0.020 (2/4) | +0.006 (2/4) |
| 512 | -0.020 | +0.076 | -0.009 | -0.006 |
| 1K | -0.136 | +0.029 | -0.100 | -0.058 |
| 2K | -0.072 | +0.109 | -0.326 | -0.208 |
| 4K | +0.059 | +0.066 | -0.365 | -0.238 |
| 8K | +0.234 | -0.109 | -0.255 | -0.179 |
| 16K | +0.223 | -0.316 | -0.223 | +0.035 |
| 32K | +0.113 | unavailable | -0.288 | -0.446 |

Except where shown, interpolation covers 4/4 sequences. This table is a useful
screening comparison, not the final independent-upstream BD-BR table.

## CTC blockers

All 12 official source samples are now present on N30. Of the ten 12-bit object
PLYs from NJU Box, `Facade`, `House_without_roof`, `Shiva`, and `Staue_Klimt`
are ASCII. `Thaidancer`, `boxer`, `longdress`, `loot`, `redandblack`, and
`soldier` are binary little-endian, while the pinned upstream
`load_sparse_tensor()` path calls `read_ply_ascii()`. A deterministic conversion
or narrowly audited reader change must preserve the intended first RGB triplet
and record source/output hashes, point counts, and metric behavior. Large CTC
samples may also exceed the 24 GiB RTX3090 whole-file path; the paper says large
11/12-bit samples were processed serially, but the released source does not
freeze a chunk manifest or aggregation rule.

Therefore the safe execution order is:

1. retain the completed 8i result;
2. run Owlii vox10 4 x 9;
3. freeze binary PLY handling and run one representative binary CTC smoke while
   recording peak VRAM;
4. launch the 12 x 9 CTC sweep only after the smoke passes, preserving failures
   explicitly if whole-file execution exceeds memory.

The released CTC curve must preserve its original ten rows and row-wise
aggregation; the separately reproduced formal curve uses the frozen nine-point
profile and lambda mapping above.
