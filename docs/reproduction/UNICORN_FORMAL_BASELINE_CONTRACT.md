# Original Unicorn formal baseline contract

Status: **FROZEN FOR STATIC LOSSY RGB COMPARISON**
Contract ID: `unicorn_static_lossy_rgb_formal_v1`
Upstream: `NJUVISION/Unicorn@b50d6c1bd033185b9e893b755d5d316cca2d4448`

This contract separates preservation of the authors' released results from the
checkpoint policy used for a reproducible thesis baseline. It applies to
8iVFB, Owlii, and CTC unless an explicit dataset-specific behavior is later
found in the pinned official implementation. Such evidence must be reported
before changing this contract.

## PAPER_REFERENCE

Released author CSV rows are retained unchanged and labeled
`AUTHOR_PROVIDED`. Their original row order and rate labels remain part of the
provenance. In particular, historical R3/R4 usage at the duplicated lambda
boundary must not be normalized retrospectively.

These rows reproduce the released paper evidence. They are not used to choose
a checkpoint independently for each test sequence and are not treated as a
physical arithmetic-stream curve unless their rate contract says so.

## FORMAL_UNICORN_BASELINE

The formal baseline follows the pinned current official object's global
first-occurrence lambda deduplication. Its nine operating points are:

| Formal point | Released profile | Lambda |
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

Therefore, lambda 8192 is **always `32k8k@8192`**, for every dataset and
sequence. Selecting between `32k8k@8192` and `8k256@8192` according to measured
sequence performance is forbidden. The locally extracted `32k8k` checkpoint
used by the 8i reproduction has SHA256
`bdd40b608ca865de3a30595ab084dcaf48450c2c9843661790c61caf9885fc90`.

For Ours-versus-Unicorn comparisons, the formal baseline uses the pinned
upstream physical-rate path: physical G-PCC `x_low` rate plus physical entropy
string bytes under the upstream accounting convention. The upstream default
likelihood-estimated result is retained for comparison with author CSVs, not
substituted into a physical-rate comparison.

## Change control

The policy is global and fixed before Owlii/CTC formal evaluation. A different
mapping is allowed only if the pinned official implementation contains an
explicit dataset-specific contract. It must be documented as a new material
comparison-contract version; observed test performance alone is not evidence
for changing the mapping.
