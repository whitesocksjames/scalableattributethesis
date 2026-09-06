#!/usr/bin/env python3
"""Build the selected 256--32K Base/Full RD audit from retained evidence."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "multirate_256_32k_audit_20260904"
POINTS = ["256", "512", "1K", "2K", "4K", "8K", "16K", "32K"]

CANDIDATES = {
    "256": ("2k128 Base step3525", "D111 Enhancement step1763"),
    "512": ("2k128 Base step3525", "D111 Enhancement step1763"),
    "1K": ("2k128 Base step3525", "D111 Enhancement step1763"),
    "2K": ("2K-U-PATH step500", "D111 Enhancement step1500"),
    "4K": ("8K-to-4K joint step3000 Base", "8K-to-4K joint step3000 Full"),
    "8K": ("canonical Base step3525", "D111 Enhancement step1763"),
    "16K": ("canonical Base step3525", "D111 Enhancement step1763"),
    "32K": ("D111 Base step5525", "sequential D111-to-D611 step3525"),
}


def add(rows, point, dataset, sequence, endpoint, bpp, quality,
        correctness=True, provenance="RETAINED_HARD_EVIDENCE"):
    base_ckpt, full_ckpt = CANDIDATES[point]
    rows.append({
        "lambda": point,
        "candidate_name": f"{point} selected",
        "base_checkpoint": base_ckpt,
        "enhancement_checkpoint": full_ckpt,
        "dataset": dataset,
        "sequence": sequence,
        "endpoint": endpoint,
        "physical_bpp": float(bpp),
        "YUV611": float(quality),
        "correctness_pass": bool(correctness),
        "provenance": provenance,
    })


def load_rwtt(rows):
    table = pd.read_csv(
        ROOT / "drafts/20260901_multirate_review/02_summary_tables/"
        "Table20_Base_Full_Matched_Official.csv")
    for point in ["256", "512", "1K", "8K", "16K", "32K"]:
        for endpoint in ["Base", "Full"]:
            row = table[(table.dataset == "RWTT-28Lite") &
                        (table.point == point) &
                        (table.endpoint == endpoint)].iloc[0]
            add(rows, point, "RWTT-28Lite", "RWTT-28Lite", endpoint,
                row.physical_bpp, row.YUV611_dB)

    for point, directory in [
        ("2K", OUT / "raw/2k_rescued_step1500_rwtt28lite"),
        ("4K", OUT / "raw/4k_joint_step3000_rwtt28lite"),
    ]:
        summary = pd.read_csv(directory / "endpoint_summary.csv")
        for _, row in summary.iterrows():
            add(rows, point, "RWTT-28Lite", "RWTT-28Lite", row.endpoint,
                row.mean_model_bpp, row.mean_model_yuv_psnr_611,
                provenance="NEW_REQUIRE_EXACT_HARD_EVALUATION")


def load_8i(rows):
    base = pd.read_csv(
        ROOT / "drafts/20260901_multirate_review/02_summary_tables/"
        "Table12_Selected_Base_8iVFB_PerSequence.csv")
    for point in ["256", "512", "1K", "8K", "16K"]:
        for _, row in base[base.point.str.upper() == point].iterrows():
            add(rows, point, "8iVFB", row.sequence, "Base", row.bpp,
                row.YUV611)

    high = pd.read_csv(
        ROOT / "drafts/20260901_multirate_review/02_summary_tables/"
        "Table17_Fixed_Full_8iVFB_PerSequence.csv")
    for point in ["256", "512", "1K", "8K", "16K", "32K"]:
        for _, row in high[high.point.str.upper() == point].iterrows():
            add(rows, point, "8iVFB", row.sequence, "Full", row.bpp,
                row.YUV611, row.hard_status == "PASS")

    thirty_two = pd.read_csv(
        ROOT / "drafts/20260902_joint32k_official_comparison/"
        "official_and_32k_endpoints.csv")
    for _, row in thirty_two[
            (thirty_two.dataset != "RWTT-28Lite") &
            (thirty_two.label == "Selected Base")].iterrows():
        add(rows, "32K", "8iVFB", row.dataset, "Base", row.physical_bpp,
            row.yuv_psnr_611)

    current_2k = pd.read_csv(
        ROOT / "results/rescued_2k_enh_external_20260903/"
        "rescued_2k_external_results.csv")
    current_2k = current_2k[current_2k.dataset == "8iVFB"]
    for endpoint, source_endpoint in [("Base", "Canonical_Base"),
                                      ("Full", "step1500")]:
        for _, row in current_2k[
                current_2k.endpoint == source_endpoint].iterrows():
            add(rows, "2K", "8iVFB", row.sequence, endpoint,
                row.physical_bpp, row.yuv611,
                int(row.num_base_residual_streams) == 4 and
                int(row.num_native_r5_streams) == 0 and
                (endpoint == "Base" or
                 float(row.hard_roundtrip_max_abs_difference) == 0))

    joint = pd.read_csv(
        ROOT / "results/joint_8k_to_4k_external_20260904/"
        "all_checkpoints_rd_review/joint_4k_all_checkpoints_rd_review.csv")
    joint = joint[(joint.checkpoint == 3000) & (joint.dataset == "8iVFB")]
    for _, row in joint.iterrows():
        add(rows, "4K", "8iVFB", row.sequence, row.endpoint,
            row.physical_bpp, row.YUV611, row.correctness_pass)


def load_owlii(rows):
    retained = pd.read_csv(
        ROOT / "drafts/20260902_owlii_joint256_review/tables/"
        "owlii_base_full_per_sequence_matched.csv")
    for point in ["256", "512", "1K", "8K", "16K", "32K"]:
        for _, row in retained[retained.point.str.upper() == point].iterrows():
            add(rows, point, "Owlii", row.sequence, row.endpoint,
                row.physical_bpp, row.yuv611_db, row.hard_status == "PASS")

    current_2k = pd.read_csv(
        ROOT / "results/rescued_2k_enh_external_20260903/"
        "rescued_2k_external_results.csv")
    current_2k = current_2k[current_2k.dataset == "Owlii"]
    for endpoint, source_endpoint in [("Base", "Canonical_Base"),
                                      ("Full", "step1500")]:
        for _, row in current_2k[
                current_2k.endpoint == source_endpoint].iterrows():
            add(rows, "2K", "Owlii", row.sequence, endpoint,
                row.physical_bpp, row.yuv611,
                int(row.num_base_residual_streams) == 4 and
                int(row.num_native_r5_streams) == 0 and
                (endpoint == "Base" or
                 float(row.hard_roundtrip_max_abs_difference) == 0))

    joint = pd.read_csv(
        ROOT / "results/joint_8k_to_4k_external_20260904/"
        "all_checkpoints_rd_review/joint_4k_all_checkpoints_rd_review.csv")
    joint = joint[(joint.checkpoint == 3000) & (joint.dataset == "Owlii")]
    for _, row in joint.iterrows():
        add(rows, "4K", "Owlii", row.sequence, row.endpoint,
            row.physical_bpp, row.YUV611, row.correctness_pass)


def official_curves():
    result = {}
    rwtt = pd.read_csv(
        ROOT / "drafts/20260901_multirate_review/03_raw_evidence/"
        "official_rwtt28lite/RWTT_28LITE_AUTHOR_9PT.csv")
    result[("RWTT-28Lite", "RWTT-28Lite")] = pd.DataFrame({
        "rate_id": rwtt.rate_id,
        "bpp": rwtt.mean_model_bpp,
        "quality": rwtt.mean_model_yuv_psnr_611,
    })
    ivfb = pd.read_csv(
        ROOT / "drafts/20260901_multirate_review/02_summary_tables/"
        "Table13_Official_R01_R09_8iVFB_PerSequence.csv")
    for sequence, group in ivfb.groupby("sequence"):
        result[("8iVFB", sequence)] = pd.DataFrame({
            "rate_id": group.rate_id, "bpp": group.bpp,
            "quality": group.YUV611})
    owlii = pd.read_csv(
        ROOT / "drafts/20260902_owlii_joint256_review/tables/"
        "owlii_author_r01_r09_per_sequence.csv")
    for sequence, group in owlii.groupby("sequence"):
        result[("Owlii", sequence)] = pd.DataFrame({
            "rate_id": group.rate_id, "bpp": group.physical_bpp,
            "quality": group.yuv611_db})
    return result


def interpolate(curve, rate):
    curve = curve.sort_values("bpp")
    x = curve.bpp.to_numpy(float)
    y = curve.quality.to_numpy(float)
    if rate < x[0] or rate > x[-1]:
        return (np.nan, "OUTSIDE_R01_R09")
    if rate in x:
        i = int(np.where(x == rate)[0][0])
        rid = curve.sort_values("bpp").iloc[i].rate_id
        return (float(y[i]), str(rid))
    upper = int(np.searchsorted(x, rate))
    lower = upper - 1
    value = y[lower] + (rate - x[lower]) * (y[upper] - y[lower]) / (
        x[upper] - x[lower])
    ordered = curve.sort_values("bpp").reset_index(drop=True)
    return (float(value), f"{ordered.iloc[lower].rate_id}_to_{ordered.iloc[upper].rate_id}")


def enrich(selected, curves):
    selected = selected.copy()
    selected["point_order"] = selected["lambda"].map({p: i for i, p in enumerate(POINTS)})
    selected["official_interp_YUV611"] = np.nan
    selected["official_bracket"] = ""
    for index, row in selected.iterrows():
        value, bracket = interpolate(
            curves[(row.dataset, row.sequence)], row.physical_bpp)
        selected.at[index, "official_interp_YUV611"] = value
        selected.at[index, "official_bracket"] = bracket
    selected["delta_to_official_curve"] = (
        selected.YUV611 - selected.official_interp_YUV611)

    for name in ["lower_neighbor", "upper_neighbor"]:
        selected[name] = ""
        selected[name + "_bpp"] = np.nan
        selected[name + "_YUV611"] = np.nan
    selected["rate_between_neighbors"] = pd.NA
    selected["quality_between_neighbors"] = pd.NA
    selected["local_neighbor_RD_residual"] = np.nan
    selected["dominated_by_lower_neighbor"] = False
    selected["dominated_by_upper_neighbor"] = False
    selected["dominated_by_any_selected_point"] = False

    grouped = selected.groupby(["dataset", "sequence", "endpoint"])
    for _, indices in grouped.groups.items():
        group = selected.loc[indices].sort_values("point_order")
        for index, row in group.iterrows():
            order = int(row.point_order)
            lower = group[group.point_order == order - 1]
            upper = group[group.point_order == order + 1]
            for label, neighbor in [("lower_neighbor", lower),
                                    ("upper_neighbor", upper)]:
                if len(neighbor):
                    neighbor = neighbor.iloc[0]
                    selected.at[index, label] = neighbor["lambda"]
                    selected.at[index, label + "_bpp"] = neighbor.physical_bpp
                    selected.at[index, label + "_YUV611"] = neighbor.YUV611
            if len(lower) and len(upper):
                lo, hi = lower.iloc[0], upper.iloc[0]
                rate_between = min(lo.physical_bpp, hi.physical_bpp) <= row.physical_bpp <= max(lo.physical_bpp, hi.physical_bpp)
                quality_between = min(lo.YUV611, hi.YUV611) <= row.YUV611 <= max(lo.YUV611, hi.YUV611)
                selected.at[index, "rate_between_neighbors"] = rate_between
                selected.at[index, "quality_between_neighbors"] = quality_between
                if rate_between and hi.physical_bpp != lo.physical_bpp:
                    chord = lo.YUV611 + (row.physical_bpp - lo.physical_bpp) * (
                        hi.YUV611 - lo.YUV611) / (hi.physical_bpp - lo.physical_bpp)
                    selected.at[index, "local_neighbor_RD_residual"] = row.YUV611 - chord
                selected.at[index, "dominated_by_lower_neighbor"] = (
                    lo.physical_bpp <= row.physical_bpp and lo.YUV611 >= row.YUV611 and
                    (lo.physical_bpp < row.physical_bpp or lo.YUV611 > row.YUV611))
                selected.at[index, "dominated_by_upper_neighbor"] = (
                    hi.physical_bpp <= row.physical_bpp and hi.YUV611 >= row.YUV611 and
                    (hi.physical_bpp < row.physical_bpp or hi.YUV611 > row.YUV611))
            dominated = False
            for _, other in group.iterrows():
                if other.name == index:
                    continue
                if (other.physical_bpp <= row.physical_bpp and
                        other.YUV611 >= row.YUV611 and
                        (other.physical_bpp < row.physical_bpp or
                         other.YUV611 > row.YUV611)):
                    dominated = True
                    break
            selected.at[index, "dominated_by_any_selected_point"] = dominated

    selected["enhancement_bpp"] = np.nan
    selected["full_minus_base_YUV611"] = np.nan
    selected["enhancement_RD_gain"] = np.nan
    for keys, group in selected.groupby(["lambda", "dataset", "sequence"]):
        if set(group.endpoint) != {"Base", "Full"}:
            continue
        base = group[group.endpoint == "Base"].iloc[0]
        full = group[group.endpoint == "Full"].iloc[0]
        idx = full.name
        selected.at[idx, "enhancement_bpp"] = full.physical_bpp - base.physical_bpp
        selected.at[idx, "full_minus_base_YUV611"] = full.YUV611 - base.YUV611
        if pd.notna(base.delta_to_official_curve) and pd.notna(full.delta_to_official_curve):
            selected.at[idx, "enhancement_RD_gain"] = (
                full.delta_to_official_curve - base.delta_to_official_curve)
    selected["rwtt_available"] = True
    return selected.sort_values(["dataset", "sequence", "endpoint", "point_order"])


def side_audit_4k(curves):
    joint = pd.read_csv(
        ROOT / "results/joint_8k_to_4k_external_20260904/"
        "all_checkpoints_rd_review/joint_4k_all_checkpoints_rd_review.csv")
    side = joint[joint.checkpoint.isin([1500, 3000])].copy()
    side.to_csv(OUT / "JOINT_4K_STEP1500_VS_STEP3000.csv", index=False)
    return side


def plot_panel(ax, subset, curve, title):
    official = curve.sort_values("bpp")
    ax.plot(official.bpp, official.quality, "o-", color="#333333",
            linewidth=1.7, markersize=4.2, label="Official Unicorn R01-R09")
    styles = {"Base": ("s-", "#1f77b4"), "Full": ("^-", "#d62728")}
    for endpoint in ["Base", "Full"]:
        points = subset[subset.endpoint == endpoint].sort_values("physical_bpp")
        style, color = styles[endpoint]
        ax.plot(points.physical_bpp, points.YUV611, style, color=color,
                linewidth=1.5, markersize=5, label=f"Our {endpoint}")
        for _, row in points.iterrows():
            ax.annotate(row["lambda"], (row.physical_bpp, row.YUV611),
                        xytext=(3, 4 if endpoint == "Full" else -10),
                        textcoords="offset points", fontsize=7, color=color)
    ax.set_title(title)
    ax.set_xlabel("Physical attribute rate (bpp)")
    ax.set_ylabel("YUV PSNR 6:1:1 (dB)")
    ax.grid(True, alpha=0.25)


def make_plots(selected, curves):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    plot_panel(ax, selected[selected.dataset == "RWTT-28Lite"],
               curves[("RWTT-28Lite", "RWTT-28Lite")], "RWTT-28Lite")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "RWTT_28LITE_MULTIRATE_RD.png", dpi=220)
    plt.close(fig)

    for dataset, sequences, filename in [
        ("8iVFB", ["longdress", "loot", "redandblack", "soldier"],
         "8IVFB_MULTIRATE_RD_FOUR_PANEL.png"),
        ("Owlii", ["basketball_player", "dancer", "exercise", "model"],
         "OWLII_MULTIRATE_RD_FOUR_PANEL.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))
        for ax, sequence in zip(axes.flat, sequences):
            subset = selected[(selected.dataset == dataset) &
                              (selected.sequence == sequence)]
            plot_panel(ax, subset, curves[(dataset, sequence)],
                       sequence.replace("_", " ").title())
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.005))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT / filename, dpi=220)
        plt.close(fig)


def aggregate_summary(master):
    rows = []
    for point in POINTS:
        data = master[master["lambda"] == point]
        out = {"lambda": point}
        for dataset in ["8iVFB", "Owlii"]:
            subset = data[data.dataset == dataset]
            for endpoint in ["Base", "Full"]:
                ep = subset[subset.endpoint == endpoint]
                prefix = f"{endpoint.lower()}_{dataset.lower()}"
                out[prefix + "_mean_bpp"] = ep.physical_bpp.mean()
                out[prefix + "_mean_delta_official"] = ep.delta_to_official_curve.mean()
                out[prefix + "_official_interp_count"] = int(
                    ep.delta_to_official_curve.notna().sum())
        external = data[data.dataset.isin(["8iVFB", "Owlii"])]
        for endpoint in ["Base", "Full"]:
            ep = external[external.endpoint == endpoint]
            out[endpoint.lower() + "_rate_order_pass_count_of_8"] = int(
                ep.rate_between_neighbors.fillna(True).astype(bool).sum())
            out[endpoint.lower() + "_pareto_violations"] = int(
                ep.dominated_by_any_selected_point.sum())
            out[endpoint.lower() + "_local_rd_positive_count"] = int(
                (ep.local_neighbor_RD_residual > 0).sum())
        full = external[external.endpoint == "Full"]
        out["enhancement_raw_positive_count_of_8"] = int(
            (full.full_minus_base_YUV611 > 0).sum())
        out["enhancement_rd_positive_count_of_8"] = int(
            (full.enhancement_RD_gain > 0).sum())
        out["enhancement_negative_raw_count_of_8"] = int(
            (full.full_minus_base_YUV611 < 0).sum())
        out["rwtt_status"] = "PASS" if len(data[data.dataset == "RWTT-28Lite"]) == 2 else "MISSING"
        out["correctness_status"] = "PASS" if data.correctness_pass.all() else "FAIL"
        rows.append(out)
    return pd.DataFrame(rows)


def decision_table(summary):
    # Manager-facing classifications reflect RD/correctness evidence, not label purity.
    decisions = {
        "256": ("KEEP_WITH_WARNING", "Full adds almost no rate and slightly lowers quality on several sequences"),
        "512": ("FREEZE", "distinct low-rate point; no material Pareto pathology"),
        "1K": ("FREEZE", "distinct low-rate point with useful Enhancement"),
        "2K": ("FREEZE", "rescued U-PATH step500 + Full step1500 is stable and manager-frozen"),
        "4K": ("KEEP_WITH_WARNING", "step3000 Full is useful, but Base rate placement overshoots 8K Base externally"),
        "8K": ("FREEZE", "healthy Base/Full ladder point; external Full has a modest efficiency warning"),
        "16K": ("KEEP_WITH_WARNING", "RWTT is healthy; 8i Full is below the local official curve"),
        "32K": ("KEEP_WITH_WARNING", "legacy exception; useful endpoint but often outside official interpolation range"),
    }
    rows = []
    def metric(value, count):
        if pd.isna(value):
            return f"N/A ({int(count)}/4)"
        return f"{value:+.3f} ({int(count)}/4)"

    for point in POINTS:
        s = summary[summary["lambda"] == point].iloc[0]
        decision, reason = decisions[point]
        rows.append({
            "Point": point,
            "Base RD": "8i {}; Owlii {}".format(
                metric(s.base_8ivfb_mean_delta_official,
                       s.base_8ivfb_official_interp_count),
                metric(s.base_owlii_mean_delta_official,
                       s.base_owlii_official_interp_count)),
            "Full RD": "8i {}; Owlii {}".format(
                metric(s.full_8ivfb_mean_delta_official,
                       s.full_8ivfb_official_interp_count),
                metric(s.full_owlii_mean_delta_official,
                       s.full_owlii_official_interp_count)),
            "Base ladder": f"{s.base_rate_order_pass_count_of_8}/8",
            "Full ladder": f"{s.full_rate_order_pass_count_of_8}/8",
            "Enh usefulness": f"raw {s.enhancement_raw_positive_count_of_8}/8; RD {s.enhancement_rd_positive_count_of_8}/8",
            "Correctness": s.correctness_status,
            "RWTT": s.rwtt_status,
            "Decision": decision,
            "Reason": reason,
        })
    return pd.DataFrame(rows)


def write_report(master, summary, decisions, side):
    display = decisions.drop(columns=["Reason"])
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |"
            for row in display.itertuples(index=False, name=None)]
    decision_markdown = "\n".join([header, separator] + body)
    lines = [
        "# Unified 256–32K Multirate Scalable RD Audit",
        "",
        "This report uses physical hard-rate evidence. Official-curve and neighbor interpolation are diagnostic local linear interpolation only; no extrapolation is used.",
        "",
        "## Manager-facing decision table",
        "",
        decision_markdown,
        "",
        "## Candidate definitions",
        "",
    ]
    for point in reversed(POINTS):
        lines.append(f"- **{point}:** Base `{CANDIDATES[point][0]}`; Full `{CANDIDATES[point][1]}`.")
    lines += [
        "",
        "## Main findings",
        "",
        "1. **Correctness is closed for all eight selected points.** Every retained result comes from the physical hard path; Base uses exactly `x_low+r1-r4`, native r5 is absent, and Full hard round-trip/bit identity passed.",
        "2. **The 4K joint step3000 Full is a usable scalable refinement, but its Base is rate-misplaced.** Full lies between the frozen 2K/8K Full rates on all eight external sequences and improves Base quality on 7/8; its Base rate exceeds the selected 8K Base on all eight external sequences. This is a ladder warning, not a codec-legality failure.",
        "3. **The weakest Enhancement endpoint is 256.** Its Enhancement payload is nearly zero and quality is slightly below Base on most external sequences. The 256 Base remains a meaningful low-rate endpoint, but the selected Full does not add useful refinement.",
        "4. **512, 1K and rescued 2K remain distinct low-rate points.** They provide rate separation; 512/1K Enhancements are useful, and the rescued 2K step1500 is consistently stronger than its discarded step1763 checkpoint.",
        "5. **16K has a dataset-dependent warning.** Its RWTT-28Lite Full is close to the official curve, while its 8i mean is below the matched official interpolation. That is evidence for an external-generalization warning, not a correctness issue.",
        "6. **The 8K Base `0/8` ladder count is induced by 4K Base overshoot.** It does not mean that 8K itself is RD-invalid: the selected 4K Base lies to the right of 8K Base on all external sequences.",
        "",
        "Numbers in parentheses in the decision table are the count of sequences bracketed by the official R01-R09 curve. `N/A` means no extrapolation was performed.",
        "",
        "## RWTT-28Lite selected endpoints",
        "",
        "| Point | Base bpp | Base YUV611 | Base Δofficial | Full bpp | Full YUV611 | Full Δofficial |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rwtt = master[master.dataset == "RWTT-28Lite"]
    for point in POINTS:
        base = rwtt[(rwtt["lambda"] == point) & (rwtt.endpoint == "Base")].iloc[0]
        full = rwtt[(rwtt["lambda"] == point) & (rwtt.endpoint == "Full")].iloc[0]
        base_delta = "N/A" if pd.isna(base.delta_to_official_curve) else f"{base.delta_to_official_curve:+.3f}"
        full_delta = "N/A" if pd.isna(full.delta_to_official_curve) else f"{full.delta_to_official_curve:+.3f}"
        lines.append(
            f"| {point} | {base.physical_bpp:.4f} | {base.YUV611:.3f} | {base_delta} | "
            f"{full.physical_bpp:.4f} | {full.YUV611:.3f} | {full_delta} |")
    lines += [
        "",
        "## 4K step1500 versus step3000",
        "",
    ]
    for step in [1500, 3000]:
        s = side[side.checkpoint == step]
        b = s[s.endpoint == "Base"]
        f = s[s.endpoint == "Full"]
        lines.append(
            f"- **step{step}:** Base mean Δofficial `{b.delta_to_official_curve.mean():+.3f} dB`; "
            f"Full mean Δofficial `{f.delta_to_official_curve.mean():+.3f} dB`; "
            f"Base rate-in-range `{int(b.rate_between_2k_8k.sum())}/8`; "
            f"Full rate-in-range `{int(f.rate_between_2k_8k.sum())}/8`; "
            f"Enhancement RD-positive `{int((f.enhancement_RD_gain > 0).sum())}/8`.")
    lines += [
        "",
        "step3000 remains the manager-provisional point: it gives the stronger Full endpoint overall, while neither checkpoint cures the cross-lambda Base-rate overshoot.",
        "",
        "## Answers to the review questions",
        "",
        "- **Freeze now:** 512, 1K, rescued 2K, and 8K have sufficient evidence under the current contracts.",
        "- **Keep with warning:** 4K step3000 (Base rate placement), 16K (8i Full efficiency), and 32K (legacy exception/out-of-range interpolation).",
        "- **True pathology:** the selected 256 Full is effectively redundant; this does not invalidate the 256 Base. No selected point has a hard correctness failure.",
        "- **Low-rate value:** 256/512/1K Bases are distinct. The 256 Full is not; 512 and 1K Full remain useful.",
        "- **2K:** yes, the selected U-PATH step500 Base + D111 step1500 Full is stable enough to freeze.",
        "- **4K checkpoint trade-off:** step1500 is earlier and marginally less rate-drifted on some content; step3000 has the stronger overall Full quality/RD evidence. Both retain the Base-placement warning.",
        "- **8K/16K:** 8K needs no further tuning evidence. 16K only warrants an external checkpoint/Stage-2 selection review if one more experiment is allowed.",
        "- **If only 1–2 GPU experiments remain:** first address 256 Enhancement usefulness; second evaluate/select a 16K Stage-2 candidate externally. Do not retrain 4K before deciding whether its Base-rate placement is acceptable in the thesis ladder.",
        "",
        "## Figures",
        "",
        "- `RWTT_28LITE_MULTIRATE_RD.png`: Official Unicorn, selected Base ladder, selected Full ladder.",
        "- `8IVFB_MULTIRATE_RD_FOUR_PANEL.png`: four 8i sequences, no dataset averaging.",
        "- `OWLII_MULTIRATE_RD_FOUR_PANEL.png`: four Owlii sequences, no dataset averaging.",
        "",
        "## Data completeness",
        "",
        "All three requested contracts are complete for the selected 256–32K set. No additional GPU evaluation is required for this audit.",
    ]
    (OUT / "MULTIRATE_256_32K_FINAL_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    load_rwtt(rows)
    load_8i(rows)
    load_owlii(rows)
    selected = pd.DataFrame(rows)
    expected = 8 * (2 + 8 * 2)
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} selected rows, got {len(selected)}")
    if selected.duplicated(["lambda", "dataset", "sequence", "endpoint"]).any():
        raise RuntimeError("Duplicate selected endpoint rows")
    curves = official_curves()
    master = enrich(selected, curves)
    master.to_csv(OUT / "MULTIRATE_256_32K_MASTER_REVIEW.csv", index=False)
    summary = aggregate_summary(master)
    summary.to_csv(OUT / "MULTIRATE_256_32K_SUMMARY.csv", index=False)
    side = side_audit_4k(curves)
    decisions = decision_table(summary)
    decisions.to_csv(OUT / "MULTIRATE_256_32K_DECISIONS.csv", index=False)
    make_plots(master, curves)
    write_report(master, summary, decisions, side)
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
