# Pretrained checkpoints and delivery status

Complete handover requires **all seven files on each platform**, independently of the other platform. They are external assets, never ordinary Git objects. No checkpoint has been deleted from the existing source storage, and no re-export, retraining or precision conversion is needed.

## Exact assets

| Point | File | Bytes |
| --- | --- | ---: |
| 512 | `ours_512.pth` | 135996037 |
| 1k | `ours_1k.pth` | 135995717 |
| 2k | `ours_2k.pth` | 135995717 |
| 4k | `ours_4k.pth` | 136030085 |
| 8k | `ours_8k.pth` | 135995717 |
| 16k | `ours_16k.pth` | 135996037 |
| 32k | `ours_32k.pth` | 135996037 |

Total: **952,005,347 bytes**. Each file contains the complete Prefix/BaseSynthesis/Enhancement inference state for Base and Full, without optimizer state or an additional Original Unicorn checkpoint. [checkpoints.json](../configs/checkpoints.json) records file hashes, source hashes and the original audit results. [SHA256SUMS](../configs/SHA256SUMS) is the portable seven-file checksum list.

The audited models are those of `b70c25b`. All seven passed strict CPU loading, exact tensor preservation and same-backend original/export equivalence on the synthetic example. GPU and fresh-process GPU decoding checks covered 8k, 2k and 4k. Other GPU points and a fresh dependency installation were not certified. These are previous audit results, not claims of testing downloads from either platform.

## Platform downloads

| Platform | Storage | Status |
| --- | --- | --- |
| [GitHub Release](https://github.com/whitesocksjames/scalableattributethesis/releases/tag/thesis-handover-v1) | Seven individual model assets, `manifest.json` and `SHA256SUMS` | Available |
| GitLab | Planned Generic Package Registry files with same-platform Release links | Repository URL not supplied; no upload performed |

Each GitHub asset is linked from the manifest. The portable checksum list is identical in the source repository and Release. GitLab delivery remains incomplete until the target repository is supplied and all seven files are independently hosted and verified there.

## Download and verify

From the cloned repository root, use the Python standard-library downloader:

```bash
python -m scalable_attribute.download --point 8k --output checkpoints
```

It checks both size and SHA-256, including when reusing an existing file. Download and independently verify all seven with:

```bash
python -m scalable_attribute.download --point all --output checkpoints
(cd checkpoints && sha256sum -c ../configs/SHA256SUMS)
```

Require seven `OK` lines. Individual files can also be downloaded from the Release page and checked against [SHA256SUMS](../configs/SHA256SUMS). Each checkpoint is about 136 MB; the seven-file total is 952,005,347 bytes. No personal HPC access or cross-platform download service is required for GitHub delivery.

Follow the [README Quick Test](../README.md#quick-start). It writes Base/Full reconstruction, payloads, physical rates and quality metrics. The output directory must be new; encode and decode on the same backend. The recorded CPU/GPU checks above predate this upload and do not certify a fresh installation.

## Attribution and remaining platform work

The owner approved GitHub publication. Applicable upstream redistribution permission for inherited code and derived weights remains unconfirmed; [THIRD_PARTY.md](../THIRD_PARTY.md) records that distinction and attribution. Publication does not grant additional rights to inherited material.

When the actual GitLab target is supplied, use the same source commit/tag and seven unchanged files. GitLab Generic Packages and its Release links provide a native storage/access route. Check the actual instance's limits and credentials before uploading; [GitLab.com limits](https://docs.gitlab.com/user/gitlab_com/#package-registry-limits) and [self-managed limits](https://docs.gitlab.com/administration/instance_limits/#package-registry-limits) may differ. Do not substitute GitHub links for the required second platform copy.
