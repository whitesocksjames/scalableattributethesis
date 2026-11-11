# Attribution and third-party notices

The proposed model extends [NJUVISION/Unicorn](https://github.com/NJUVISION/Unicorn), pinned for the original reference evaluation at commit `b50d6c1bd033185b9e893b755d5d316cca2d4448`. Retained modules under `basic_models/`, `lossy_attribute/`, `data_utils/`, `cfg/`, and codec wrappers derive from the official source. Their author/copyright headers are retained. Cite *A Versatile Point Cloud Compressor Using Universal Multiscale Conditional Coding — Part II: Attribute Compression*, [IEEE publication](https://ieeexplore.ieee.org/document/10682566).

The OURS contribution adds the full-resolution BaseSynthesis path, conditional Enhancement composition, and their evaluation/runtime support. Pretrained inference files retain the relevant upstream-derived Prefix state; merging weights does not change their provenance. Each published model records source SHA-256 identities.

The pinned Unicorn source does not supply an explicit repository-wide license in the inspected tree. This handover does not assign an invented blanket license to that code or its pretrained weights. Attribution is complete; an explicit upstream redistribution grant remains unverified and should be resolved before public release. The current branch and its model files are prepared for private supervisor review, with public release pending acceptance.

| Retained tool | Source/version evidence | License notice |
| --- | --- | --- |
| `third_party/tmc3_v21` | [MPEGGroup/mpeg-pcc-tmc13, release-v21.0-rc1](https://github.com/MPEGGroup/mpeg-pcc-tmc13/tree/release-v21.0-rc1), matching embedded binary build paths | [BSD notice](third_party/licenses/tmc3-COPYING), retrieved from that tag |
| `third_party/pc_error_d` | MPEG `pc_error` 0.13.4, retained unchanged from the research tree; upstream [release-v0.13.4](https://github.com/MPEGGroup/mpeg-pcc-dmetric/tree/release-v0.13.4) | [BSD notice](third_party/licenses/pc_error-LICENSE), preserved from the original bundled `d3metric` source archive |

The original binary build provenance is not fully reproducible from the retained executable alone. Binary SHA-256 values are recorded in [tools.json](third_party/tools.json), allowing the supplied versions to be identified exactly. Platform-specific rebuilds must use the same evaluation options and pass parity checks before replacing a tool used for formal comparison.

External runtime packages (PyTorch, MinkowskiEngine, torchac, NumPy and other installed dependencies) retain their own upstream licenses; they are installed separately. Removed third-party projects are not dependencies of this handover.

`third_party/unicorn_partition.py.txt` is the original pinned Unicorn partition source (SHA-256 `ec3c5c347e289b364acb1c41bbd388c172bff6b142fc87f630a7460d67389dec`). The preprocessing gate hashes and parses this file, then executes the equivalent local function in `data_utils/attribute/partition.py`. This preserves the frozen function and avoids importing the upstream file's unused Open3D dependency.
