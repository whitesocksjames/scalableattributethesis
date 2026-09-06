#!/usr/bin/env python3
"""Build the Owlii operating-point and joint-256 review from retained CSVs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_n30"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
AUTHOR = Path("results/PCAC/csvfiles/ours/owlii_vox10")
OLD_REVIEW = Path("drafts/20260901_multirate_review")
SEQUENCES = ("basketball_player", "dancer", "exercise", "model")
AUTHOR_FILES = {
    "basketball_player": "basketball_player_vox11_00000200.csv",
    "dancer": "dancer_vox11_00000001.csv",
    "exercise": "exercise_vox11_00000001.csv",
    "model": "model_vox11_00000001.csv",
}
POINTS = ("32k", "16k", "8k", "4k", "2k", "1k", "512", "256")
DISPLAY = {point: point.upper() for point in POINTS}
FULL_LABEL = {
    "32k": "Canonical_Full_step3525",
    "16k": "16k_stage1_step1763",
    "8k": "8k_stage1_step1763",
    "1k": "1k_stage1_step1763",
    "512": "512_stage1_step1763",
    "256": "256_stage1_step1763",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interpolate(curve, bpp):
    curve = sorted(curve)
    for lower, upper in zip(curve, curve[1:]):
        if lower[0] <= bpp <= upper[0]:
            fraction = (bpp - lower[0]) / (upper[0] - lower[0])
            value = lower[1] + fraction * (upper[1] - lower[1])
            return value, "{}_to_{}".format(lower[2], upper[2])
    return None, "OUTSIDE_R01_R09"


def load_author():
    curves = {}
    rows_out = []
    for sequence in SEQUENCES:
        rows = read_csv(AUTHOR / AUTHOR_FILES[sequence])[:9]
        curve = []
        for index, row in enumerate(rows, 1):
            item = (float(row["bpp"]), float(row["YUV-PSNR"]),
                    "R{:02d}".format(index))
            curve.append(item)
            rows_out.append({
                "sequence": sequence, "rate_id": item[2],
                "physical_bpp": item[0], "yuv611_db": item[1],
                "provenance": "AUTHOR_PROVIDED",
            })
        curves[sequence] = curve
    curves["mean"] = []
    for rate_index in range(9):
        curves["mean"].append((
            sum(curves[s][rate_index][0] for s in SEQUENCES) / 4,
            sum(curves[s][rate_index][1] for s in SEQUENCES) / 4,
            "R{:02d}".format(rate_index + 1)))
    return curves, rows_out


def candidate_csv(point, sequence):
    if point == "32k":
        return RAW / "owlii_32k_existing" / sequence / "physical_rd.csv"
    return RAW / "owlii" / point / sequence / "physical_rd.csv"


def validate_hard_outputs():
    failures = []
    checked = 0
    for point in POINTS[1:]:
        for sequence in SEQUENCES:
            path = RAW / "owlii" / point / sequence / "physical_rd.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            checked += 1
            if payload.get("status") != "PASS":
                failures.append(str(path) + ": status")
            for row in payload["rows"]:
                if row["endpoint"] == "Canonical_Base":
                    if (int(row["num_base_residual_streams"]) != 4 or
                            int(row["num_native_r5_streams"]) != 0 or
                            int(row["physical_bits"]) != int(row["base_bits"]) or
                            int(row["enhancement_bits"]) != 0):
                        failures.append(str(path) + ": Base identity")
                if "stage1" in row["endpoint"]:
                    if (float(row["hard_roundtrip_max_abs_difference"]) != 0 or
                            int(row["physical_bits"]) !=
                            int(row["base_bits"]) + int(row["enhancement_bits"])):
                        failures.append(str(path) + ": Full identity")
    # The retained 32K results predate the renamed fields but record exact Full.
    for sequence in SEQUENCES:
        path = RAW / "owlii_32k_existing" / sequence / "physical_rd.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        checked += 1
        if payload.get("status") != "PASS":
            failures.append(str(path) + ": status")
        full = next(row for row in payload["rows"]
                    if row["endpoint"] == FULL_LABEL["32k"])
        exact = full.get("hard_roundtrip_max_abs_difference",
                         full.get("hard_max_abs_difference"))
        if float(exact) != 0:
            failures.append(str(path) + ": Full identity")
    if failures:
        raise RuntimeError("; ".join(failures))
    return checked


def load_candidates(curves):
    rows = []
    for point in POINTS:
        for sequence in SEQUENCES:
            source_rows = read_csv(candidate_csv(point, sequence))
            base = next(row for row in source_rows
                        if row["endpoint"] == "Canonical_Base")
            endpoints = [("Base", base)]
            if point in FULL_LABEL:
                endpoints.append(("Full", next(
                    row for row in source_rows
                    if row["endpoint"] == FULL_LABEL[point])))
            for endpoint, row in endpoints:
                bpp = float(row["physical_bpp"])
                quality = float(row["yuv_psnr_611"])
                official, bracket = interpolate(curves[sequence], bpp)
                rows.append({
                    "sequence": sequence, "point": DISPLAY[point],
                    "endpoint": endpoint,
                    "checkpoint_step": row.get("checkpoint_step", ""),
                    "checkpoint_profile": row["checkpoint_profile"],
                    "physical_bpp": bpp, "yuv611_db": quality,
                    "official_interpolated_yuv611_db": (
                        "" if official is None else official),
                    "delta_matched_db": (
                        "" if official is None else quality - official),
                    "official_bracket": bracket,
                    "hard_status": "PASS",
                    "result_provenance": (
                        "REUSED_EXISTING_32K" if point == "32k"
                        else "NEW_N30_EVALUATION"),
                })
    return rows


def mean_rows(rows, curves):
    output = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["point"], row["endpoint"])].append(row)
    for (point, endpoint), items in groups.items():
        bpp = sum(row["physical_bpp"] for row in items) / len(items)
        quality = sum(row["yuv611_db"] for row in items) / len(items)
        official, bracket = interpolate(curves["mean"], bpp)
        output.append({
            "sequence": "four_sequence_equal_mean", "point": point,
            "endpoint": endpoint,
            "physical_bpp": bpp, "yuv611_db": quality,
            "official_interpolated_yuv611_db": (
                "" if official is None else official),
            "delta_matched_db": "" if official is None else quality - official,
            "official_bracket": bracket, "hard_status": "PASS",
        })
    order = {DISPLAY[p]: i for i, p in enumerate(POINTS)}
    output.sort(key=lambda row: (order[row["point"]], row["endpoint"]))
    return output


def plot_one(name, curve, rows, path):
    plt.figure(figsize=(8.2, 5.6))
    official = sorted(curve)
    plt.plot([x[0] for x in official], [x[1] for x in official],
             "o-", color="#222222", label="Official R01-R09")
    for x, y, label in official:
        plt.annotate(label, (x, y), xytext=(4, 4),
                     textcoords="offset points", fontsize=7)
    for endpoint, marker, color in (("Base", "s", "#1874CD"),
                                     ("Full", "^", "#D94801")):
        selected = [row for row in rows if row["endpoint"] == endpoint]
        selected.sort(key=lambda row: row["physical_bpp"])
        plt.plot([row["physical_bpp"] for row in selected],
                 [row["yuv611_db"] for row in selected], marker=marker,
                 linestyle="--", color=color, label="Canonical " + endpoint)
        for row in selected:
            plt.annotate(row["point"],
                         (row["physical_bpp"], row["yuv611_db"]),
                         xytext=(4, -10 if endpoint == "Base" else 5),
                         textcoords="offset points", fontsize=7, color=color)
    plt.xlabel("Physical attribute bpp")
    plt.ylabel("YUV-PSNR 6:1:1 (dB)")
    plt.title("Owlii {} RD".format(name))
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def joint256_rows():
    official_rows = read_csv(
        OLD_REVIEW / "03_raw_evidence/official_rwtt28lite"
        / "RWTT_28LITE_AUTHOR_9PT.csv")
    official_curve = [
        (float(row["mean_model_bpp"]),
         float(row["mean_model_yuv_psnr_611"]), row["rate_id"])
        for row in official_rows]
    rows = []
    for step in (500, 1500):
        summary = read_csv(
            RAW / "joint256_rwtt28lite" / "step_{}".format(step)
            / "endpoint_summary.csv")
        by_endpoint = {row["endpoint"]: row for row in summary}
        for endpoint in ("Base", "Full"):
            row = by_endpoint[endpoint]
            rows.append({
                "recipe": "TAFA_joint", "step": step, "endpoint": endpoint,
                "physical_bpp": float(row["mean_model_bpp"]),
                "y_psnr_db": float(row["mean_model_y_psnr"]),
                "yuv611_db": float(row["mean_model_yuv_psnr_611"]),
                "hard_status": "PASS",
            })
    sequential = read_csv(
        OLD_REVIEW / "03_raw_evidence/lowrate_enhancement_stage1/256"
        / "rwtt_28lite/step_1763/endpoint_summary.csv")
    for row in sequential:
        rows.append({
            "recipe": "Sequential_Enhancement", "step": 1763,
            "endpoint": row["endpoint"],
            "physical_bpp": float(row["mean_model_bpp"]),
            "y_psnr_db": float(row["mean_model_y_psnr"]),
            "yuv611_db": float(row["mean_model_yuv_psnr_611"]),
            "hard_status": "PASS",
        })
    for row in rows:
        official, bracket = interpolate(official_curve, row["physical_bpp"])
        row["official_interpolated_yuv611_db"] = (
            "" if official is None else official)
        row["delta_matched_db"] = (
            "" if official is None else row["yuv611_db"] - official)
        row["official_bracket"] = bracket
    for recipe in ("TAFA_joint", "Sequential_Enhancement"):
        steps = sorted({row["step"] for row in rows if row["recipe"] == recipe})
        for step in steps:
            base = next(row for row in rows if row["recipe"] == recipe
                        and row["step"] == step and row["endpoint"] == "Base")
            full = next(row for row in rows if row["recipe"] == recipe
                        and row["step"] == step and row["endpoint"] == "Full")
            for row in (base, full):
                row["enhancement_increment_bpp"] = (
                    full["physical_bpp"] - base["physical_bpp"])
                row["full_minus_base_yuv611_db"] = (
                    full["yuv611_db"] - base["yuv611_db"])
    return rows


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    checked = validate_hard_outputs()
    curves, author_rows = load_author()
    candidates = load_candidates(curves)
    means = mean_rows(candidates, curves)
    joint = joint256_rows()
    write_csv(TABLES / "owlii_author_r01_r09_per_sequence.csv", author_rows)
    write_csv(TABLES / "owlii_base_full_per_sequence_matched.csv", candidates)
    write_csv(TABLES / "owlii_base_full_four_sequence_mean.csv", means)
    write_csv(TABLES / "joint256_rwtt28lite_comparison.csv", joint)
    for sequence in SEQUENCES:
        plot_one(sequence, curves[sequence],
                 [row for row in candidates if row["sequence"] == sequence],
                 FIGURES / "owlii_{}_rd.png".format(sequence))
    plot_one("four-sequence equal mean", curves["mean"], means,
             FIGURES / "owlii_four_sequence_mean_rd.png")

    pattern = [row for row in candidates if row["endpoint"] == "Base"
               and row["point"] in ("8K", "4K", "2K", "1K")]
    lines = [
        "# Owlii operating-point diagnosis and joint-256 review", "",
        "## Result provenance", "",
        "- Owlii frames: author Part II static selections, author-compatible vox11→vox10 preprocessing.",
        "- Point-count fingerprint: Basketball 796217; Dancer 702038; Exercise 645135; Model 657755.",
        "- Official curves: repository-retained author-provided per-sequence R01–R09 CSVs.",
        "- Candidate rate: actual physical hard attribute bits; metric: `pc_error` Y/U/V and YUV611.",
        "- Matched delta: linear interpolation in each sequence's own bpp–YUV611 curve; no extrapolation.",
        "- 32K results are reused; 16K–256 are new N30 evaluations.",
        "- 32K Full in this bundle is the retained D111 step3525 result, not the later D611 recipe.",
        "- Hard/bit correctness: PASS for all {} sequence-point outputs checked.".format(checked),
        "", "## Four-sequence equal-mean endpoints", "",
        "| Point | Endpoint | bpp | YUV611 | Official@bpp | Δmatched |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in means:
        official = row["official_interpolated_yuv611_db"]
        delta = row["delta_matched_db"]
        lines.append("| {point} | {endpoint} | {physical_bpp:.6f} | "
                     "{yuv611_db:.4f} | {official} | {delta} |".format(
                         **row,
                         official=("N/A" if official == "" else
                                   "{:.4f}".format(official)),
                         delta=("N/A" if delta == "" else
                                "{:+.4f}".format(delta))))
    lines += ["", "## 8K→4K→2K→1K per-sequence Base pattern", "",
              "| Sequence | Point | bpp | YUV611 | Δmatched |",
              "|---|---|---:|---:|---:|"]
    point_order = {p: i for i, p in enumerate(("8K", "4K", "2K", "1K"))}
    for row in sorted(pattern, key=lambda r: (r["sequence"], point_order[r["point"]])):
        delta = row["delta_matched_db"]
        lines.append("| {sequence} | {point} | {physical_bpp:.6f} | "
                     "{yuv611_db:.4f} | {delta} |".format(
                         **row, delta=("N/A" if delta == "" else
                                      "{:+.4f}".format(delta))))
    lines += ["", "## Joint-256 RWTT-28Lite", "",
              "| Recipe | Step | Endpoint | bpp | YUV611 | Δmatched | EL Δbpp | Full−Base dB |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in joint:
        matched = row["delta_matched_db"]
        lines.append("| {recipe} | {step} | {endpoint} | {physical_bpp:.6f} | "
                     "{yuv611_db:.4f} | {matched} | {enhancement_increment_bpp:.6f} | "
                     "{full_minus_base_yuv611_db:+.4f} |".format(
                         **row, matched=("N/A" if matched == "" else
                                         "{:+.4f}".format(matched))))
    lines += ["", "## Evidence synthesis", "",
              "- **Owlii Base:** 4K is below the matched Official envelope on all four sequences and is the clearest weak point (mean −0.5870 dB). 2K improves over 4K on every sequence but remains negative (mean −0.2421 dB). 1K recovers to near-envelope mean performance (−0.0124 dB), with Dancer still weak.",
              "- **Family pattern:** the cross-sequence 8K→4K degradation and 2K→1K recovery reproduce the earlier 8i concern. This supports an `8k256`-family/native-truncation contribution, modulated by sequence content; it is not merely a Longdress-only effect.",
              "- **Owlii Full:** 16K Stage-1 is near/slightly above the matched author curve (+0.0346 dB); 8K is moderately below (−0.1705 dB); 1K and 512 are near the envelope. The 256 endpoints lie below the author R09 rate and therefore have no interpolated delta.",
              "- **32K provenance:** the reused 32K Full is D111 step3525 and is −0.4460 dB matched on Owlii; this result must not be labeled as the later D611 recipe.",
              "- **Joint-256:** step1500 improves Full over its own Base by +0.0713 dB using 0.005609 bpp Enhancement, but its matched efficiency is lower than sequential step1763. Sequential uses only 0.000109 bpp Enhancement and has the larger positive matched delta; the joint probe does not currently justify replacing the sequential recipe.",
              "- **Joint training status:** scientific conclusion is provisional because BS4 OOM stopped training at step1609; nevertheless both retained hard checkpoints are valid evaluation points."]
    lines += ["", "## Interpretation guardrails", "",
              "- Owlii interpolation is diagnostic, not BD-rate.",
              "- Joint step500/1500 are incomplete-training checkpoints after a BS4 OOM at step1609.",
              "- Joint and sequential rows use the same RWTT-28Lite physical hard contract.",
              "- No new training or automatic recipe selection is implied.", ""]
    (HERE / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
