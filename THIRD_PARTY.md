# Reference, attribution and redistribution

The implementation extends [NJUVISION/Unicorn](https://github.com/NJUVISION/Unicorn), pinned for the reference evaluation at commit `b50d6c1bd033185b9e893b755d5d316cca2d4448`. Retained source under `basic_models/`, `lossy_attribute/`, `data_utils/`, `cfg/` and the codec wrappers derives from that implementation. Existing author/copyright headers are preserved. Cite *A Versatile Point Cloud Compressor Using Universal Multiscale Conditional Coding — Part II: Attribute Compression* ([paper](https://ieeexplore.ieee.org/document/10682566)).

OURS adds full-resolution BaseSynthesis, conditional Enhancement composition and associated runtime/evaluation support. Model weights are not included or downloaded by this repository. Their frozen identifiers document the evaluation states, not a distribution service or permission grant.

## Inherited code permission

The complete pinned Unicorn tree has no general code/weight license identified by this delivery review. Its explicit PCQM license files concern PCQM, which is not retained here. The official repository license endpoint returned 404. Public source access and an academic citation do not establish permission to redistribute inherited source or derivative weights. No blanket license is assigned to them here.

**A public push containing the retained Unicorn-derived code is blocked pending confirmation of applicable redistribution permission.** Omitting checkpoints does not resolve the source-code permission issue. This condition applies to every distribution destination.

Available delivery alternatives are author-owned results/figures/analysis only, an already authorized controlled evaluation, or an arrangement where the evaluator independently obtains upstream material through an authorized channel. A complete public source repository containing these inherited modules requires an applicable license or rightsholder authorization.

## Codec tools and dependencies

| Retained tool | Upstream reference | Notices |
| --- | --- | --- |
| `tmc3_v21` | [TMC13 release-v21.0-rc1](https://github.com/MPEGGroup/mpeg-pcc-tmc13/tree/release-v21.0-rc1) | [Root BSD notice](third_party/licenses/tmc3-COPYING), [dependency notices](third_party/licenses/DEPENDENCIES.txt) |
| `pc_error_d` | [dmetric release-v0.13.4](https://github.com/MPEGGroup/mpeg-pcc-dmetric/tree/release-v0.13.4) | [Original bundled notice](third_party/licenses/pc_error-LICENSE), [MERL and dependency notices](third_party/licenses/DEPENDENCIES.txt) |

The identified components provide conditional BSD/MIT redistribution terms. TMC13 includes nanoflann and program-options-lite BSD notices and a Schroedinger component offering an MIT option. Dmetric source headers carry MERL copyright and BSD-style terms. Preserve all notices when distributing supplied or rebuilt tools. Supplied binary identities and separate verified source-build references are recorded in [tools.json](third_party/tools.json); the supplied executables' complete original build provenance remains unverified. [INSTALL.md](docs/INSTALL.md) gives the checked upstream build commands.

`third_party/unicorn_partition.py.txt` preserves the pinned upstream partition reference with SHA-256 `ec3c5c347e289b364acb1c41bbd388c172bff6b142fc87f630a7460d67389dec`. The preprocessing gate hashes/parses it and verifies the equivalent local function before execution, avoiding its unused Open3D import. It remains part of the inherited-source permission check above.

Python runtime packages are installed separately and retain their own upstream terms. The repository does not distribute an environment or dependency wheels.
