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

## Current platform availability

| Platform | Intended storage | Actual status |
| --- | --- | --- |
| [GitHub](https://github.com/whitesocksjames/scalableattributethesis) | Seven individual Release assets, plus manifest and SHA256SUMS | No Release/assets published; permission unresolved |
| GitLab | Seven individual Generic Package Registry files with same-platform Release links | Repository URL not supplied; no upload performed |

There are **no working platform model downloads yet**. No placeholder asset URL is advertised. Existing research storage is an internal source for the final files, not the supervisor delivery mechanism. Neither platform is complete until its own seven assets can be independently downloaded and verified.

## Publication requirements

Confirm permission covering both inherited source and derived weights before uploading to any public or private third-party destination; see [THIRD_PARTY.md](../THIRD_PARTY.md). Establish the actual GitLab target and upload permissions rather than guessing a project URL.

After permission is established, publish the same source commit/tag on both platforms. Attach the seven unchanged model files and copies of the manifest/checksum list to the GitHub Release. Store a second full copy in the GitLab project's Generic Package Registry and link those local package files from its Release. Publish directly from the existing model source host or through an approved server-to-server route; no workstation relay is needed.

GitHub permits each Release file below 2 GiB ([documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)); each model is about 136 MB. GitLab.com permits Generic Package files up to 5 GB ([documentation](https://docs.gitlab.com/user/gitlab_com/#package-registry-limits)). A self-managed GitLab instance can impose different limits and quotas, which must be checked against the actual target ([instance limits](https://docs.gitlab.com/administration/instance_limits/#package-registry-limits)). GitLab [Generic Packages](https://docs.gitlab.com/user/packages/generic_packages/) and [Release links](https://docs.gitlab.com/user/project/releases/release_fields/) provide the intended native storage/access route.

Only after uploading should the README gain exact, tested asset download commands. Download all seven from each destination into separate verification directories; compare both sizes and SHA-256 against this source manifest, without fetching from the other platform. Confirm remote source trees match the tagged commit and run the README Quick Test using a downloaded checkpoint. A local/source-storage checksum pass does not establish platform download success.

## Verify downloaded files

For all seven, run from the cloned repository root:

```bash
(cd checkpoints && sha256sum -c ../configs/SHA256SUMS)
```

Require seven `OK` lines. For a single model, use `--ignore-missing` and explicitly require that model's `OK` line. Follow the [README Quick Test](../README.md#verify-a-checkpoint-and-run-the-quick-test) after verification. It writes Base/Full reconstruction, payloads, physical rates and quality metrics. The output directory must be new; encode and decode on the same backend.
