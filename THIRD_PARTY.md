# Attribution and third-party notices

The proposed model extends [NJUVISION/Unicorn](https://github.com/NJUVISION/Unicorn), pinned for the original reference evaluation at commit `b50d6c1bd033185b9e893b755d5d316cca2d4448`. Retained modules under `basic_models/`, `lossy_attribute/`, `data_utils/`, `cfg/`, and codec wrappers derive from the official source. Their author/copyright headers are retained. Cite *A Versatile Point Cloud Compressor Using Universal Multiscale Conditional Coding — Part II: Attribute Compression*, [IEEE publication](https://ieeexplore.ieee.org/document/10682566).

The OURS contribution adds the full-resolution BaseSynthesis path, conditional Enhancement composition, and their evaluation/runtime support. Pretrained inference files retain the relevant upstream-derived Prefix state; merging weights does not change their provenance. Each published model records source SHA-256 identities.

The pinned Unicorn source does not supply an explicit repository-wide license in the inspected tree. This handover does not assign an invented blanket license to that code or its pretrained weights. Attribution is complete; an explicit upstream redistribution grant remains unverified and should be resolved before public release. The files remain in the existing controlled research setting. Neither public availability nor a private handover establishes a redistribution grant; no public Release or weight upload is authorized by this audit.

| Retained tool | Source/version evidence | License notice |
| --- | --- | --- |
| `third_party/tmc3_v21` | [MPEGGroup/mpeg-pcc-tmc13, release-v21.0-rc1](https://github.com/MPEGGroup/mpeg-pcc-tmc13/tree/release-v21.0-rc1), matching embedded binary build paths | [BSD notice](third_party/licenses/tmc3-COPYING), retrieved from that tag |
| `third_party/pc_error_d` | MPEG `pc_error` 0.13.4, retained unchanged from the research tree; upstream [release-v0.13.4](https://github.com/MPEGGroup/mpeg-pcc-dmetric/tree/release-v0.13.4) | [BSD notice](third_party/licenses/pc_error-LICENSE), preserved from the original bundled `d3metric` source archive |

The original binary build provenance is not fully reproducible from the retained executable alone. Binary SHA-256 values are recorded in [tools.json](third_party/tools.json), allowing the supplied versions to be identified exactly. Platform-specific rebuilds must use the same evaluation options and pass parity checks before replacing a tool used for formal comparison.

External runtime packages (PyTorch, MinkowskiEngine, torchac, NumPy and other installed dependencies) retain their own upstream licenses; they are installed separately. Removed third-party projects are not dependencies of this handover.

`third_party/unicorn_partition.py.txt` is the original pinned Unicorn partition source (SHA-256 `ec3c5c347e289b364acb1c41bbd388c172bff6b142fc87f630a7460d67389dec`). The preprocessing gate hashes and parses this file, then executes the equivalent local function in `data_utils/attribute/partition.py`. This preserves the frozen function and avoids importing the upstream file's unused Open3D dependency.

Final distribution audit (2026-11-11): the exact TMC13 v21.0-rc1 tree includes nanoflann and program-options-lite BSD notices and a Schroedinger component offering an MIT license option. Dmetric v0.13.4 carries MERL copyright and BSD-style redistribution terms in source headers, plus its dependency notices. The original bundled `pc_error-LICENSE` alone did not contain these MERL/dependency notices. They are now preserved in [DEPENDENCIES.txt](third_party/licenses/DEPENDENCIES.txt). The original executables are unchanged; their full build provenance remains unverified.

| Distribution item | Verified facts and action |
| --- | --- |
| Own thesis results, figures, and analysis scripts | These can be prepared as a separate results-only delivery, subject to the author's rights; they do not require copying pretrained weights. |
| Retained TMC13/dmetric tools | Their identified upstream components provide conditional BSD/MIT redistribution terms; retain all notices. Rebuilding from the identified upstream release is preferable if exact binary provenance is required. |
| Copied Unicorn-derived code | No general license/redistribution grant was found in the complete pinned tree. PCQM license files cover PCQM, which is absent from this handover. Resolve authorization before further distribution. |
| Merged inference weights | They include upstream-derived Prefix parameters. A new container, removal of optimizer state, or private recipient does not establish redistribution permission. Keep URLs unset pending authorization. |
| Installed Python dependencies | They are obtained separately under their own package terms. This handover does not distribute an environment or dependency wheels. |

Concrete alternatives are: (1) provide only author-owned results and analysis while authorization is pending; (2) let an authorized evaluator run the existing controlled installation; or (3) have the evaluator independently acquire upstream source/weights through a verified authorized official channel, then run the local exporter with the frozen source identities. The upstream project page is a reference, not a substitute for a verified model-download grant. Obtain explicit permission from the relevant rightsholder before supplying copied Unicorn code or derived weights through a new distribution channel. No permission request has been sent automatically.

The final FAU handover preparation rechecked the official Unicorn license endpoint (no repository license returned), the pinned v1 README and license/readme files under the actual released-weight directory. No additional redistribution grant was established. Seven model assets are staged only in the existing controlled FAU research account; publication URLs remain unset. No upstream author has been contacted automatically.
