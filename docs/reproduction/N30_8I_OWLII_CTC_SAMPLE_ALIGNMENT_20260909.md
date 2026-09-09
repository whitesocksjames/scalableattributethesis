# N30 8iVFB / Owlii / CTC sample alignment audit — 2026-09-09

## Verdict

The N30 sample membership matches the released Unicorn evidence for all 20
dataset roles: 4 8iVFB, 4 Owlii vox10, and 12 CTC samples.

| Evidence class | Count | Meaning |
| --- | ---: | --- |
| Official downloadable bytes match | 14/20 | Four 8i and ten CTC object PLY SHA-256 values match the PLY bytes inside the author-linked NJU Box ZIPs |
| Deterministic derived contract matches | 4/20 | Four Owlii vox10 files reproduce byte-for-byte from N30 native vox11 sources and match the author CSV point-count fingerprints |
| Name/frame/precision match only | 2/20 | Native Owlii vox11 basketball and dancer used by CTC match the paper and released CSV contract, but the author publishes no source PLY hashes |

The data contract and execution contract are not the same. Six byte-exact
official CTC PLYs are binary little-endian, but the released loader calls
`read_ply_ascii()`. They are the correct downloadable samples but cannot be fed
unchanged through the pinned released loader.

## Frozen official evidence

- Paper: `references/unicorn_part_ii_attribute.pdf`, Section IV-A and Tables II–III.
- Released source/result snapshot: `NJUVISION/Unicorn` commit
  `b50d6c1bd033185b9e893b755d5d316cca2d4448`.
- N30 critical source files and released CSVs were checked against the same Git
  commit. SHA-256 matched for `lossy_attribute/test.py`,
  `attribute_dataloader.py`, and representative 8i/Owlii/CTC CSVs.
- Official dataset mirror: the NJU Box link referenced by the released README.

The released runner recursively sorts every `*.ply` below `--testdata`; it does
not hard-code a dataset manifest. Therefore the released result CSV filenames
and paper must define the sample set, while the run directory must be curated.

## 8iVFB — 4/4 official byte matches

All files are ASCII PLY. N30 point counts match the first row of each released
author PCAC CSV, and each N30 PLY SHA-256 matches the PLY bytes inside the
author-linked NJU Box ZIP.

| Sample | N30 points | N30/official PLY SHA-256 | Verdict |
| --- | ---: | --- | --- |
| `longdress_vox10_1300.ply` | 857,966 | `da32b3ee5e3c3d83407c31a75acc08c22ed137ece83339d056547c41a266b001` | byte match |
| `loot_vox10_1200.ply` | 805,285 | `64d9a6c368c85fba0218a066518124ced0e174b7264eda9bf4a61f6ad70c86e6` | byte match |
| `redandblack_vox10_1550.ply` | 757,691 | `6091282374f7e396e6b9f01ae4a3e0d5041d0ee3a0cb22556353ff5f600906f8` | byte match |
| `soldier_vox10_0690.ply` | 1,089,091 | `873b9e187d10314501daa041b52fbf1b1355db6974f40f9eb5d4da47ebfcd1a6` | byte match |

N30 roots are `/data/run01/scz0ade/Tanzeyu/data/8iVFB/<sequence>/`.
The four official ZIPs are retained on FAU at
`/home/woody/iwnt/iwnt193h/scalable_attribute_thesis/datasets/unicorn_official/8ivfb_zips/`.

## Owlii vox10 — 4/4 deterministic contract matches

The paper evaluates four 10-bit Owlii frames. The available source PLY names
retain `vox11`; the author `owlii_vox10` CSVs report the post-quantization point
counts. The frozen adapter performs
`floor(xyz / 2) + duplicate-voxel RGB mean + round`.

On N30, all four native inputs were reprocessed in a new temporary directory.
The generated files matched the retained `Owlii_vox10_floor_v1` files
byte-for-byte, and all point counts matched the author CSVs.

| Sample | Native points | Author/N30 vox10 points | Re-derived/N30 SHA-256 | Verdict |
| --- | ---: | ---: | --- | --- |
| `basketball_player_vox11_00000200.ply` | 2,925,514 | 796,217 | `c0856f10c30c8b1d69e310f77a004a172c1cc35d7d39510f74c9ad39842c86e0` | derived contract match |
| `dancer_vox11_00000001.ply` | 2,592,758 | 702,038 | `c88373584a850612e4b754c14c6bf1d964883177947c3a25c2837a47f6e9ae48` | derived contract match |
| `exercise_vox11_00000001.ply` | 2,391,718 | 645,135 | `6d3aaae352d090b603d4d80f13a0bf65baca98fe4717627aadf98fab5a3ca82e` | derived contract match |
| `model_vox11_00000001.ply` | 2,458,429 | 657,755 | `979c17678e269c9ad97b0530a52890bcf7ffc9c29732388224aac5294ba5bbc4` | derived contract match |

N30 input root:
`/data/run01/scz0ade/Tanzeyu/data/Owlii_vox10_floor_v1/`.

This is stronger than a filename-only check but weaker than an official-file
byte match: the released repository does not publish these four derived PLYs or
their hashes, and does not freeze the conversion recipe in its README. The
claim is therefore “author-compatible deterministic derivation,” not “downloaded
official bytes.”

## CTC — 12/12 membership matches

### Native 11-bit Owlii members

| Sample | Points | Format | N30 SHA-256 | Verdict |
| --- | ---: | --- | --- | --- |
| `basketball_player_vox11_00000200.ply` | 2,925,514 | ASCII | `a73faf6bd858b74eb236a57f0f00d161bc7aa0342f883e8d71eaff97c5549fd3` | name/frame/11-bit source contract match; no published official hash |
| `dancer_vox11_00000001.ply` | 2,592,758 | ASCII | `29bcbee86a531e509d0fe030e8623f144c853199e159e737248f6463568fe95d` | name/frame/11-bit source contract match; no published official hash |

N30 root:
`/data/run01/scz0ade/Tanzeyu/data/scalable_attribute_thesis/datasets/unicorn_official/owlii_vox11/`.
That directory also contains `exercise` and `model`; those two are **not** CTC
members. Do not pass this whole directory to the released recursive-glob runner.

### Official NJU Box 12-bit object members

Each extracted N30 PLY was hashed separately and compared with `unzip -p` of its
official NJU Box archive. All ten comparisons are exact.

| Sample | Points | Format | N30/official PLY SHA-256 | Verdict |
| --- | ---: | --- | --- | --- |
| `Thaidancer_viewdep_vox12.ply` | 3,130,215 | binary little-endian | `ddce408b6b35387e8b44fa9ecd2e63370f27f47edab179cb84490d58e539a3ac` | byte match; loader-incompatible format |
| `longdress_viewdep_vox12.ply` | 3,096,122 | binary little-endian | `3a10b360f03f220d95908cdfa72ec41b2c5a5a163d611ec3c98d894540b759e0` | byte match; loader-incompatible format |
| `loot_viewdep_vox12.ply` | 3,017,285 | binary little-endian | `a8bb72e36b40587953aca76f27802d877ddd33e8c300a28a6d61f4a34e72edee` | byte match; loader-incompatible format |
| `redandblack_viewdep_vox12.ply` | 2,770,567 | binary little-endian | `d9f701585150e45158b4f8b22c81e0c824ae2593eb5e633414d83d720d982e4c` | byte match; loader-incompatible format |
| `soldier_viewdep_vox12.ply` | 4,001,754 | binary little-endian | `b245f380e66daa1da272e4b54c5b3a4714e3d0de7e4a0a1dc5ae4fd497bf8346` | byte match; loader-incompatible format |
| `boxer_viewdep_vox12.ply` | 3,493,085 | binary little-endian | `7f80cdc21aec82a7139b075c6ae8af1af4b95b5c5a5f98b64fd1e81037ad1ba7` | byte match; loader-incompatible format |
| `Facade_00009_vox12.ply` | 1,596,085 | ASCII | `63f93f79aee5fccacd21b05a71b09aeff59a877a397cad1a4d65eb6478877be4` | byte match; directly loader-compatible |
| `House_without_roof_00057_vox12.ply` | 4,848,745 | ASCII | `82966dfcfda208ae5edecbc8b7d37ebc5f3c034cec778af338a5e1c540920a05` | byte match; directly loader-compatible |
| `Shiva_00035_vox12.ply` | 1,009,132 | ASCII | `d05f7f4c32c82cd17fabd0e42493407371abbf27651cff08b923ec38963ab6e3` | byte match; directly loader-compatible |
| `Staue_Klimt_vox12.ply` | 499,660 | ASCII | `3d05ed590f17d72115a1305c994ee2e9a50bd0d764d82aae72f78cb38b3aac46` | byte match; directly loader-compatible |

N30 root:
`/data/run01/scz0ade/Tanzeyu/data/scalable_attribute_thesis/datasets/unicorn_official/ctc_vox12/`.

## Paper/code inconsistencies resolved by the released data

1. The paper prose lists 8i sequences as longdress, loot, redandblack, soldier,
   then gives frame numbers 1300, 1550, 1200, 0690 “respectively.” Taken
   literally, loot and redandblack are swapped. The released CSVs and official
   NJU Box files use `loot_1200` and `redandblack_1550`; N30 matches them.
2. The paper says CTC contains twelve point clouds but its prose enumeration
   names only eleven and omits `boxer`. The released result bundle/notebook has
   twelve, including `boxer`; N30 matches the released 12-item contract.

When paper prose and released author data conflict in these two cases, use the
released file/CSV contract and record the paper typo rather than renaming N30
inputs.

## Run gate

- 8i: sample identity PASS; completed 4 x 9 reproduction remains valid.
- Owlii vox10: sample derivation PASS; ready for the pinned 4 x 9 run.
- CTC ASCII subset: identity PASS, but the 12-sample aggregate must remain one
  declared contract.
- CTC binary subset: identity PASS as official data, execution FAIL with the
  unmodified released loader. Freeze either a lossless binary-to-ASCII adapter
  with source/output hashes or a narrowly audited binary reader before running.
- Run CTC from a curated 12-item manifest/directory. Never point the released
  recursive-glob runner at the four-file native Owlii directory and assume it
  will select only basketball and dancer.
