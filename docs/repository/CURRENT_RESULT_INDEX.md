# Current results — start here

Working candidates, not a final freeze. 256 remains in evaluation; 4K joint
step3000 remains the main provisional candidate regardless of Base rate ordering.

| Need | Artifact |
| --- | --- |
| Current 256–32K master rows | [MASTER_REVIEW.csv](../../results/multirate_256_32k_audit_20260904/MULTIRATE_256_32K_MASTER_REVIEW.csv) |
| Compact multi-rate summary | [SUMMARY.csv](../../results/multirate_256_32k_audit_20260904/MULTIRATE_256_32K_SUMMARY.csv) |
| RWTT-28Lite RD figure | [RWTT figure](../../results/multirate_256_32k_audit_20260904/RWTT_28LITE_MULTIRATE_RD.png) |
| 8i per-sequence four panels | [8i figure](../../results/multirate_256_32k_audit_20260904/8IVFB_MULTIRATE_RD_FOUR_PANEL.png) |
| Owlii per-sequence four panels | [Owlii figure](../../results/multirate_256_32k_audit_20260904/OWLII_MULTIRATE_RD_FOUR_PANEL.png) |
| Underlying written analysis | [20260904 audit](../../results/multirate_256_32k_audit_20260904/MULTIRATE_256_32K_FINAL_AUDIT.md) — historical recommendation labels, not current freeze authority |
| Current checkpoint identities | [candidate provenance](CURRENT_OPERATING_POINTS.md) |
| 4K all-checkpoint tradeoffs | [4K review](../../results/joint_8k_to_4k_external_20260904/all_checkpoints_rd_review/JOINT_4K_ALL_CHECKPOINTS_RD_REVIEW.md) |
| 2K Enhancement comparison | [2K review](../../results/rescued_2k_enh_external_20260903/RESCUED_2K_ENH_EXTERNAL_ANALYSIS.md) |
| Base rescue historical screening | [rescue report](../../results/base_rescue_screening_20260903/BASE_RESCUE_SCREENING_REPORT.md) |
| Native family/content diagnosis | [internal diagnosis](../../drafts/20260903_fixed_model_internal_trace/4K2K_FAMILY_CONTENT_INTERNAL_DIAGNOSIS.md) |

The older candidate inventory reported missing 4K RWTT evidence. That statement
is superseded by the above master table and retained joint-step3000 raw output.
All eight candidate pairs have retained RWTT-28Lite, 8i and Owlii evidence in the
current assembled tables; this is not a claim of newly rerun or independent tests.

Historical reports and results are preserved unchanged. FREEZE/REJECT and ladder
flags in them are dated analysis, not automatic current selection changes.
In particular, 4K Base > 8K Base is not a codec failure. Author interpolation is
diagnostic, without extrapolation. Earlier external sample use for selection
must remain visible in eventual comparison reporting.

Some reconstruction scripts read untracked historical raw/summary tables outside
the snapshot. Re-running a plot from a clean clone is not yet guaranteed; see
[inventory](INVENTORY.md). Do not delete these inputs because newer plots exist.
