#!/usr/bin/env python3
"""Build the read-only d707ce7 multirate review bundle from retained CSVs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURES = ROOT / "01_figures"
TABLES = ROOT / "02_summary_tables"
RAW = ROOT / "03_raw_evidence"
CURRENT = RAW / "current_round"
PRIOR = RAW / "prior_selected/evaluation"
OFFICIAL_RWTT = RAW / "official_rwtt28lite/RWTT_28LITE_AUTHOR_9PT.csv"
OFFICIAL_8I = ROOT.parent / "canonical_scalable_formal_r01/8ivfb_external"

FIGURE_NAMES = {
    "enhancement_16k_rd_trajectory": "Fig01_16K_Enhancement_RD_RWTT28Lite",
    "enhancement_8k_rd_trajectory": "Fig02_8K_Enhancement_RD_RWTT28Lite",
    "base_1k_512_256_training_trajectory":
        "Fig03_LowRate_Base_Training_Trajectory_RWTT28Lite",
    "lowrate_base_rd_rwtt28lite": "Fig04_Selected_Base_RD_RWTT28Lite",
    "lowrate_base_rd_8i_fixed4": "Fig05_Selected_Base_RD_8iVFB",
    "selected_base_rd_8i_per_sequence":
        "Fig06_Selected_Base_RD_8iVFB_PerSequence",
}

POINT_ORDER = ["32k", "16k", "8k", "4k", "2k", "1k", "512", "256"]
POINT_RATE_ID = {
    "32k": "R01", "16k": "R02", "8k": "R03", "4k": "R04",
    "2k": "R05", "1k": "R06", "512": "R07", "256": "R08",
}
POINT_PROFILE = {
    "32k": "32k8k", "16k": "32k8k", "8k": "32k8k",
    "4k": "8k256", "2k": "8k256", "1k": "2k128",
    "512": "2k128", "256": "2k128",
}
SEQUENCES = ("longdress", "loot", "redandblack", "soldier")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields=None):
    if not rows:
        raise ValueError("refusing to write empty CSV: {}".format(path))
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row, field):
    return float(row[field])


def official_rwtt():
    return read_csv(OFFICIAL_RWTT)


def official_8i():
    by_rate = {}
    for sequence in SEQUENCES:
        path = OFFICIAL_8I / sequence / "physical_rd.csv"
        for row in read_csv(path):
            rate = row["endpoint"]
            if rate not in {"R{:02d}".format(i) for i in range(1, 10)}:
                continue
            by_rate.setdefault(rate, []).append(row)
    result = []
    for rate in sorted(by_rate):
        rows = by_rate[rate]
        if len(rows) != 4:
            raise ValueError("{} does not cover fixed-4".format(rate))
        result.append({
            "rate_id": rate,
            "mean_bpp": sum(f(row, "physical_bpp") for row in rows) / 4,
            "mean_y": sum(f(row, "y_psnr") for row in rows) / 4,
            "mean_u": sum(f(row, "u_psnr") for row in rows) / 4,
            "mean_v": sum(f(row, "v_psnr") for row in rows) / 4,
            "mean_yuv611": sum(f(row, "yuv_psnr_611") for row in rows) / 4,
            "num_sequences": 4,
        })
    return result


def official_8i_per_sequence():
    result = []
    for sequence in SEQUENCES:
        for row in read_csv(OFFICIAL_8I / sequence / "physical_rd.csv"):
            if row["endpoint"] not in {"R{:02d}".format(i) for i in range(1, 10)}:
                continue
            result.append({
                "sequence": sequence, "rate_id": row["endpoint"],
                "bpp": f(row, "physical_bpp"),
                "Y": f(row, "y_psnr"), "U": f(row, "u_psnr"),
                "V": f(row, "v_psnr"), "YUV611": f(row, "yuv_psnr_611"),
            })
    return result


def selected_base_rwtt():
    rows = []
    for point in ("16k", "8k", "4k", "2k"):
        path = PRIOR / "base_{}".format(point) / "screening" / (
            "BASE_{}_28LITE_SHORTLIST.json".format(point.upper()))
        selected = read_json(path)["shortlist"][0]
        rows.append({
            "point": point,
            "rate_id": POINT_RATE_ID[point],
            "profile": POINT_PROFILE[point],
            "checkpoint_step": int(selected["checkpoint_step"]),
            "bpp": f(selected, "mean_model_bpp"),
            "y": f(selected, "mean_model_y_psnr"),
            "u": f(selected, "mean_model_u_psnr"),
            "v": f(selected, "mean_model_v_psnr"),
            "yuv611": f(selected, "mean_model_yuv_psnr_611"),
            "role": "canonical_selected",
        })
    for point, label in (("1k", "base_1k"), ("512", "base_512"),
                         ("256", "base_256")):
        path = CURRENT / "branch_b" / label / "screen" / (
            "BASE_{}_28LITE_SHORTLIST.json".format(point.upper()))
        selected = read_json(path)["shortlist"][0]
        rows.append({
            "point": point,
            "rate_id": POINT_RATE_ID[point],
            "profile": POINT_PROFILE[point],
            "checkpoint_step": int(selected["checkpoint_step"]),
            "bpp": f(selected, "mean_model_bpp"),
            "y": f(selected, "mean_model_y_psnr"),
            "u": f(selected, "mean_model_u_psnr"),
            "v": f(selected, "mean_model_v_psnr"),
            "yuv611": f(selected, "mean_model_yuv_psnr_611"),
            "role": "diagnostic_candidate",
        })
    return rows


def selected_base_8i():
    rows = []
    for point in ("16k", "8k", "4k", "2k"):
        path = PRIOR / "base_{}".format(point) / "8ivfb_summary" / (
            "BASE_{}_8I_RESULTS.csv".format(point.upper()))
        row = read_csv(path)[0]
        rows.append({
            "point": point, "rate_id": POINT_RATE_ID[point],
            "profile": POINT_PROFILE[point],
            "checkpoint_step": int(float(row.get("checkpoint_step", 0) or 0)),
            "bpp": f(row, "mean_physical_bpp"),
            "y": f(row, "mean_y_psnr"), "u": f(row, "mean_u_psnr"),
            "v": f(row, "mean_v_psnr"),
            "yuv611": f(row, "mean_yuv_psnr_611"),
            "role": "canonical_selected",
        })
    for point, label in (("1k", "base_1k"), ("512", "base_512"),
                         ("256", "base_256")):
        path = CURRENT / "branch_b" / label / "8ivfb" / "summary" / (
            "BASE_{}_8I_RESULTS.csv".format(point.upper()))
        row = read_csv(path)[0]
        rows.append({
            "point": point, "rate_id": POINT_RATE_ID[point],
            "profile": POINT_PROFILE[point],
            "checkpoint_step": int(row["candidate"].split("step_")[-1]),
            "bpp": f(row, "mean_physical_bpp"),
            "y": f(row, "mean_y_psnr"), "u": f(row, "mean_u_psnr"),
            "v": f(row, "mean_v_psnr"),
            "yuv611": f(row, "mean_yuv_psnr_611"),
            "role": "diagnostic_candidate",
        })
    return rows


def selected_base_8i_per_sequence():
    result = []
    for point in ("16k", "8k", "4k", "2k"):
        path = PRIOR / "base_{}".format(point) / "8ivfb_summary" / (
            "BASE_{}_8I_PER_SEQUENCE.csv".format(point.upper()))
        for row in read_csv(path):
            result.append({
                "sequence": row["sequence"], "point": point,
                "rate_id": POINT_RATE_ID[point], "profile": POINT_PROFILE[point],
                "checkpoint_step": int(row["checkpoint_step"]),
                "bpp": f(row, "physical_bpp"),
                "Y": f(row, "y_psnr"), "U": f(row, "u_psnr"),
                "V": f(row, "v_psnr"), "YUV611": f(row, "yuv_psnr_611"),
                "role": "canonical_selected",
            })
    for point, label in (("1k", "base_1k"), ("512", "base_512"),
                         ("256", "base_256")):
        path = CURRENT / "branch_b" / label / "8ivfb/summary" / (
            "BASE_{}_8I_PER_SEQUENCE.csv".format(point.upper()))
        for row in read_csv(path):
            result.append({
                "sequence": row["sequence"], "point": point,
                "rate_id": POINT_RATE_ID[point], "profile": POINT_PROFILE[point],
                "checkpoint_step": int(row["checkpoint_step"]),
                "bpp": f(row, "physical_bpp"),
                "Y": f(row, "y_psnr"), "U": f(row, "u_psnr"),
                "V": f(row, "v_psnr"), "YUV611": f(row, "yuv_psnr_611"),
                "role": "diagnostic_candidate",
            })
    return result


def matched_official_delta_8i(selected, official):
    result = []
    included = {"8k", "4k", "2k", "1k", "512", "256"}
    for base in selected:
        if base["point"] not in included:
            continue
        curve = sorted(
            (row["bpp"], row["YUV611"], row["rate_id"])
            for row in official if row["sequence"] == base["sequence"])
        bracket = next(((left, right) for left, right in zip(curve, curve[1:])
                        if left[0] <= base["bpp"] <= right[0]), None)
        if bracket is None:
            raise ValueError("Base bpp lies outside official curve: {} {}".format(
                base["sequence"], base["point"]))
        lower, upper = bracket
        interpolated = lower[1] + (
            (base["bpp"] - lower[0]) / (upper[0] - lower[0])
            * (upper[1] - lower[1]))
        result.append({
            "sequence": base["sequence"], "point": base["point"],
            "base_bpp": base["bpp"], "base_YUV611": base["YUV611"],
            "official_lower": lower[2], "official_upper": upper[2],
            "official_interpolated_YUV611": interpolated,
            "delta_matched_db": base["YUV611"] - interpolated,
        })
    return result


def enhancement_trajectory():
    rows = []
    official = {row["rate_id"]: row for row in official_rwtt()}
    for point in ("16k", "8k"):
        path = CURRENT / "branch_a" / point / "rwtt_28lite" / "summary" / (
            "enhancement_28lite_trajectory.csv")
        target = official[POINT_RATE_ID[point]]
        for row in read_csv(path):
            rows.append({
                "point": point, "checkpoint_step": int(row["checkpoint_step"]),
                "base_bpp": f(row, "base_bpp"),
                "enhancement_bpp": f(row, "enhancement_bpp"),
                "full_bpp": f(row, "full_bpp"),
                "base_Y": f(row, "base_y_psnr"),
                "base_U": f(row, "base_u_psnr"),
                "base_V": f(row, "base_v_psnr"),
                "base_YUV611": f(row, "base_yuv_psnr_611"),
                "full_Y": f(row, "full_y_psnr"),
                "full_U": f(row, "full_u_psnr"),
                "full_V": f(row, "full_v_psnr"),
                "full_YUV611": f(row, "full_yuv_psnr_611"),
                "pareto_candidate": row["pareto_candidate"],
                "hard_status": row["hard_status"],
                "official_rate_id": POINT_RATE_ID[point],
                "official_bpp": f(target, "mean_model_bpp"),
                "official_YUV611": f(target, "mean_model_yuv_psnr_611"),
            })
    return rows


def base_trajectory():
    native = {}
    for point, dirname in (("1k", "R06_2k128_L1024"),
                           ("512", "R07_2k128_L512"),
                           ("256", "R08_2k128_L256")):
        path = CURRENT / "branch_c/rwtt_28lite" / dirname / "endpoint_summary.csv"
        row = next(row for row in read_csv(path) if row["endpoint"] == "B_native")
        native[point] = f(row, "mean_model_yuv_psnr_611")
    result = []
    for point, label in (("1k", "base_1k"), ("512", "base_512"),
                         ("256", "base_256")):
        path = CURRENT / "branch_b" / label / "rwtt_28lite/endpoint_summary.csv"
        for row in read_csv(path):
            quality = f(row, "mean_model_yuv_psnr_611")
            result.append({
                "point": point, "checkpoint_step": int(row["checkpoint_step"]),
                "base_bpp": f(row, "mean_model_bpp"),
                "Y": f(row, "mean_model_y_psnr"),
                "U": f(row, "mean_model_u_psnr"),
                "V": f(row, "mean_model_v_psnr"),
                "YUV611": quality, "B_native_YUV611": native[point],
                "synthesis_gain_db": quality - native[point],
                "hard_status": (
                    "PASS" if row["bit_identity_pass"] == "True" else "FAIL"),
            })
    return result


def base_8i_per_sequence():
    result = []
    for point, label in (("1k", "base_1k"), ("512", "base_512"),
                         ("256", "base_256")):
        for sequence in SEQUENCES:
            path = CURRENT / "branch_b" / label / "8ivfb" / sequence / "physical_rd.csv"
            rows = read_csv(path)
            row = next(row for row in rows if row["endpoint"].startswith("Diagnostic_"))
            result.append({
                "point": point, "sequence": sequence,
                "base_bpp": f(row, "physical_bpp"),
                "Y": f(row, "y_psnr"), "U": f(row, "u_psnr"),
                "V": f(row, "v_psnr"), "YUV611": f(row, "yuv_psnr_611"),
                "hard_status": "PASS",
                "bit_identity": (
                    int(row["physical_bits"]) == int(row["base_bits"])),
            })
    return result


def residual_share():
    result = []
    for index in range(1, 10):
        rate = "R{:02d}".format(index)
        path = next((CURRENT / "branch_c/rwtt_28lite").glob(rate + "_*"))
        row = next(row for row in read_csv(path / "endpoint_summary.csv")
                   if row["endpoint"] == "Official_Full")
        result.append({
            "rate_id": rate, "checkpoint_profile": row["checkpoint_profile"],
            "lambda": int(row["base_lambda"]),
            **{"r{}_bits".format(i): int(row["r{}_bits".format(i)])
               for i in range(1, 6)},
            **{"r{}_share".format(i): f(row, "r{}_residual_share".format(i))
               for i in range(1, 6)},
        })
    return result


def overlap_rwtt():
    result = []
    for dirname in ("R03_32k8k_L8192", "DIAGNOSTIC_OVERLAP_8k256_L8192"):
        path = CURRENT / "branch_c/rwtt_28lite" / dirname / "endpoint_summary.csv"
        for row in read_csv(path):
            result.append({
                "dataset": "RWTT-28Lite", "profile": row["checkpoint_profile"],
                "lambda": int(row["base_lambda"]), "endpoint": row["endpoint"],
                "bpp": f(row, "mean_model_bpp"),
                "Y": f(row, "mean_model_y_psnr"),
                "U": f(row, "mean_model_u_psnr"),
                "V": f(row, "mean_model_v_psnr"),
                "YUV611": f(row, "mean_model_yuv_psnr_611"),
                "hard_roundtrip_max_abs_difference": f(
                    row, "hard_roundtrip_max_abs_difference"),
            })
    return result


def overlap_8i():
    result = []
    base = CURRENT / "branch_c/overlap_8192_8ivfb"
    for profile in ("32k8k", "8k256"):
        directory = base / "DIAGNOSTIC_OVERLAP_{}_L8192".format(profile)
        for sequence in SEQUENCES:
            path = directory / sequence / "endpoint_summary.csv"
            for row in read_csv(path):
                result.append({
                    "dataset": "8i fixed-4", "sequence": sequence,
                    "profile": profile, "lambda": 8192,
                    "endpoint": row["endpoint"],
                    "bpp": f(row, "mean_model_bpp"),
                    "Y": f(row, "mean_model_y_psnr"),
                    "U": f(row, "mean_model_u_psnr"),
                    "V": f(row, "mean_model_v_psnr"),
                    "YUV611": f(row, "mean_model_yuv_psnr_611"),
                    "hard_roundtrip_max_abs_difference": f(
                        row, "hard_roundtrip_max_abs_difference"),
                })
    return result


def decomposition_rwtt():
    selected = {row["point"]: row for row in selected_base_rwtt()}
    dirs = {"8k": "R03_32k8k_L8192", "4k": "R04_8k256_L4096",
            "2k": "R05_8k256_L2048"}
    result = []
    for point, dirname in dirs.items():
        rows = read_csv(CURRENT / "branch_c/rwtt_28lite" / dirname /
                        "endpoint_summary.csv")
        native = next(row for row in rows if row["endpoint"] == "B_native")
        full = next(row for row in rows if row["endpoint"] == "Official_Full")
        canonical = selected[point]
        result.append({
            "point": point, "profile": POINT_PROFILE[point],
            "base_bpp": canonical["bpp"],
            "B_native_YUV611": f(native, "mean_model_yuv_psnr_611"),
            "Canonical_Base_YUV611": canonical["yuv611"],
            "Official_Full_bpp": f(full, "mean_model_bpp"),
            "Official_Full_YUV611": f(full, "mean_model_yuv_psnr_611"),
            "Synthesis_Gain_db": canonical["yuv611"] - f(
                native, "mean_model_yuv_psnr_611"),
            "Missing_r5_Gap_db": f(full, "mean_model_yuv_psnr_611") - f(
                native, "mean_model_yuv_psnr_611"),
        })
    return result


def decomposition_8i():
    official = {}
    for sequence in SEQUENCES:
        for row in read_csv(OFFICIAL_8I / sequence / "physical_rd.csv"):
            if row["endpoint"] in ("R03", "R04", "R05"):
                official[(sequence, row["endpoint"])] = row
    result = []
    for point in ("8k", "4k", "2k"):
        for sequence in SEQUENCES:
            rows = read_csv(PRIOR / "base_{}".format(point) / "8ivfb" /
                            sequence / "physical_rd.csv")
            native = next(row for row in rows if row["endpoint"] == "B_native")
            canonical = next(row for row in rows
                             if row["endpoint"] == "Canonical_Base")
            full = official[(sequence, POINT_RATE_ID[point])]
            result.append({
                "point": point, "sequence": sequence,
                "profile": POINT_PROFILE[point],
                "base_bpp": f(canonical, "physical_bpp"),
                "B_native_YUV611": f(native, "yuv_psnr_611"),
                "Canonical_Base_YUV611": f(canonical, "yuv_psnr_611"),
                "Official_Full_bpp": f(full, "physical_bpp"),
                "Official_Full_YUV611": f(full, "yuv_psnr_611"),
                "Synthesis_Gain_db": f(canonical, "yuv_psnr_611") - f(
                    native, "yuv_psnr_611"),
                "Missing_r5_Gap_db": f(full, "yuv_psnr_611") - f(
                    native, "yuv_psnr_611"),
            })
    return result


def style_axis(axis, title, xlabel="Physical attribute rate (bpp)"):
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
    axis.grid(True, alpha=0.25)


def save_figure(figure, stem, rect=None):
    figure.tight_layout(rect=rect)
    output_stem = FIGURE_NAMES[stem]
    figure.savefig(FIGURES / (output_stem + ".png"), dpi=190)
    figure.savefig(FIGURES / (output_stem + ".svg"))
    plt.close(figure)


def plot_enhancement(point, trajectory, official, selected):
    figure, (axis, zoom) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    ox = [f(row, "mean_model_bpp") for row in official]
    oy = [f(row, "mean_model_yuv_psnr_611") for row in official]
    axis.plot(ox, oy, "o-", color="#777777", label="Official R01–R09 Full")
    for row, x, y in zip(official, ox, oy):
        axis.annotate(row["rate_id"], (x, y), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    sx = [row["bpp"] for row in selected]
    sy = [row["yuv611"] for row in selected]
    axis.plot(sx, sy, "^--", color="#2c7fb8", alpha=0.75,
              label="Selected/candidate Base endpoints")
    for row in selected:
        axis.annotate(row["point"], (row["bpp"], row["yuv611"]),
                      xytext=(4, -10), textcoords="offset points", fontsize=8)
    current = [row for row in trajectory if row["point"] == point]
    axis.plot([row["full_bpp"] for row in current],
              [row["full_YUV611"] for row in current], "s-",
              color="#d95f0e", label="{} Enhancement Stage-1".format(point))
    base = current[0]
    axis.scatter([base["base_bpp"]], [base["base_YUV611"]], marker="X",
                 s=90, color="#1b9e77", label="{} Base".format(point))
    style_axis(axis, "Full official-curve context")
    axis.legend(fontsize=8)

    target = next(row for row in official if row["rate_id"] == POINT_RATE_ID[point])
    zoom.scatter([base["base_bpp"]], [base["base_YUV611"]], marker="X",
                 s=110, color="#1b9e77", label="{} Base".format(point))
    zoom.plot([row["full_bpp"] for row in current],
              [row["full_YUV611"] for row in current], "-",
              color="#d95f0e", alpha=0.7)
    markers = ("s", "D", "P", "v")
    for row, marker in zip(current, markers):
        zoom.scatter(
            [row["full_bpp"]], [row["full_YUV611"]], marker=marker, s=60,
            label="step{} ({:.4f} bpp, {:.3f} dB)".format(
                row["checkpoint_step"], row["full_bpp"], row["full_YUV611"]))
    zoom.scatter([f(target, "mean_model_bpp")],
                 [f(target, "mean_model_yuv_psnr_611")], marker="o", s=90,
                 color="#777777", label="Official {} Full".format(
                     target["rate_id"]))
    zoom.annotate(target["rate_id"],
                  (f(target, "mean_model_bpp"),
                   f(target, "mean_model_yuv_psnr_611")),
                  xytext=(5, 5), textcoords="offset points", fontsize=9)
    local_x = [base["base_bpp"], f(target, "mean_model_bpp")] + [
        row["full_bpp"] for row in current]
    local_y = [base["base_YUV611"], f(target, "mean_model_yuv_psnr_611")] + [
        row["full_YUV611"] for row in current]
    x_pad = (max(local_x) - min(local_x)) * 0.08
    y_pad = (max(local_y) - min(local_y)) * 0.08
    zoom.set_xlim(min(local_x) - x_pad, max(local_x) + x_pad)
    zoom.set_ylim(min(local_y) - y_pad, max(local_y) + y_pad)
    style_axis(zoom, "Target operating-point zoom")
    zoom.legend(fontsize=8)
    figure.suptitle("{} Enhancement trajectory — RWTT-28Lite".format(point),
                    fontsize=16)
    save_figure(figure, "enhancement_{}_rd_trajectory".format(point))


def plot_training(trajectory):
    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    for point, color in (("1k", "#1b9e77"), ("512", "#d95f02"),
                         ("256", "#7570b3")):
        rows = [row for row in trajectory if row["point"] == point]
        axis.plot([row["checkpoint_step"] for row in rows],
                  [row["YUV611"] for row in rows], "o-", color=color,
                  label=point)
    style_axis(axis, "Low-rate Base training trajectory — RWTT-28Lite",
               xlabel="Training step")
    axis.legend()
    save_figure(figure, "base_1k_512_256_training_trajectory")


def plot_base_rd(selected, official, dataset):
    figure, axis = plt.subplots(figsize=(8.4, 5.8))
    if dataset == "rwtt":
        ox = [f(row, "mean_model_bpp") for row in official]
        oy = [f(row, "mean_model_yuv_psnr_611") for row in official]
    else:
        ox = [row["mean_bpp"] for row in official]
        oy = [row["mean_yuv611"] for row in official]
    axis.plot(ox, oy, "o-", color="#777777", label="Official R01–R09 Full")
    for row, x, y in zip(official, ox, oy):
        axis.annotate(row["rate_id"], (x, y), xytext=(3, 4),
                      textcoords="offset points", fontsize=8)
    colors = {"32k8k": "#1b9e77", "8k256": "#d95f02", "2k128": "#7570b3"}
    for profile in ("32k8k", "8k256", "2k128"):
        rows = [row for row in selected if row["profile"] == profile]
        axis.plot([row["bpp"] for row in rows], [row["yuv611"] for row in rows],
                  "^--", color=colors[profile], label="{} Base".format(profile))
        for row in rows:
            axis.annotate(row["point"], (row["bpp"], row["yuv611"]),
                          xytext=(4, -11), textcoords="offset points", fontsize=8)
    title = ("Selected low-rate Base progression — RWTT-28Lite" if dataset == "rwtt"
             else "Selected low-rate Base progression — 8i fixed-4")
    style_axis(axis, title)
    axis.legend(fontsize=8)
    save_figure(figure, "lowrate_base_rd_{}".format(
        "rwtt28lite" if dataset == "rwtt" else "8i_fixed4"))


def plot_base_rd_8i_per_sequence(selected, official):
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    colors = {"32k8k": "#1b9e77", "8k256": "#d95f02", "2k128": "#7570b3"}
    display = {"longdress": "Longdress 1300", "loot": "Loot 1200",
               "redandblack": "Redandblack 1550", "soldier": "Soldier 0690"}
    for axis, sequence in zip(axes.flat, SEQUENCES):
        official_rows = [row for row in official if row["sequence"] == sequence]
        axis.plot([row["bpp"] for row in official_rows],
                  [row["YUV611"] for row in official_rows], "o-",
                  color="#777777", label="Official R01–R09 Full")
        for row in official_rows:
            axis.annotate(row["rate_id"], (row["bpp"], row["YUV611"]),
                          xytext=(3, 4), textcoords="offset points", fontsize=7)
        selected_rows = [row for row in selected if row["sequence"] == sequence]
        for profile in ("32k8k", "8k256", "2k128"):
            rows = [row for row in selected_rows if row["profile"] == profile]
            axis.plot([row["bpp"] for row in rows], [row["YUV611"] for row in rows],
                      "^--", color=colors[profile], label="{} Base".format(profile))
            for row in rows:
                axis.annotate(row["point"], (row["bpp"], row["YUV611"]),
                              xytext=(3, -10), textcoords="offset points", fontsize=7)
        style_axis(axis, display[sequence])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
                  bbox_to_anchor=(0.5, 0.955))
    figure.suptitle("Selected Base vs Official Unicorn — 8iVFB per sequence",
                    fontsize=16, y=0.995)
    save_figure(figure, "selected_base_rd_8i_per_sequence",
                rect=(0, 0, 1, 0.91))


def main():
    enhancement = enhancement_trajectory()
    base_traj = base_trajectory()
    base_8i = base_8i_per_sequence()
    selected_rwtt = selected_base_rwtt()
    selected_8i = selected_base_8i()
    official_rwtt_rows = official_rwtt()
    official_8i_rows = official_8i()
    selected_8i_sequence_rows = selected_base_8i_per_sequence()
    official_8i_sequence_rows = official_8i_per_sequence()

    write_csv(TABLES / "Table01_Enhancement_16K_8K_Trajectory.csv", enhancement)
    write_csv(TABLES / "Table02_Base_1K_512_256_Trajectory.csv", base_traj)
    write_csv(TABLES / "Table03_Base_1K_512_256_8iVFB.csv", base_8i)
    write_csv(TABLES / "Table04_Selected_Base_RWTT28Lite.csv", selected_rwtt)
    write_csv(TABLES / "Table05_Selected_Base_8iVFB.csv", selected_8i)
    write_csv(TABLES / "Table06_Official_R01_R09_8iVFB.csv", official_8i_rows)
    write_csv(TABLES / "Table07_Overlap_8192_RWTT28Lite.csv", overlap_rwtt())
    write_csv(TABLES / "Table08_Overlap_8192_8iVFB.csv", overlap_8i())
    write_csv(TABLES / "Table09_Decomposition_8K_4K_2K_RWTT28Lite.csv",
              decomposition_rwtt())
    write_csv(TABLES / "Table10_Decomposition_8K_4K_2K_8iVFB.csv",
              decomposition_8i())
    write_csv(TABLES / "Table11_Residual_Share_R01_R09.csv", residual_share())
    write_csv(TABLES / "Table12_Selected_Base_8iVFB_PerSequence.csv",
              selected_8i_sequence_rows)
    write_csv(TABLES / "Table13_Official_R01_R09_8iVFB_PerSequence.csv",
              official_8i_sequence_rows)
    write_csv(TABLES / "Table14_Base_Matched_Official_Delta_8iVFB_PerSequence.csv",
              matched_official_delta_8i(selected_8i_sequence_rows,
                                        official_8i_sequence_rows))

    plot_enhancement("16k", enhancement, official_rwtt_rows, selected_rwtt)
    plot_enhancement("8k", enhancement, official_rwtt_rows, selected_rwtt)
    plot_training(base_traj)
    plot_base_rd(selected_rwtt, official_rwtt_rows, "rwtt")
    plot_base_rd(selected_8i, official_8i_rows, "8i")
    plot_base_rd_8i_per_sequence(selected_8i_sequence_rows,
                                official_8i_sequence_rows)

    info = {
        "source_commit": "d707ce7ebfa3426e52f2838070b36847c6552c8e",
        "new_evaluation_submitted": False,
        "checkpoints_copied": False,
        "rwtt_contract": "RWTT-28Lite; 28 models/28 H5; model-equal; pc_error YUV611",
        "8i_contract": "Longdress1300/Loot1200/Redandblack1550/Soldier0690; arithmetic mean",
        "official_curves": {
            "rwtt": str(OFFICIAL_RWTT.relative_to(ROOT)),
            "8i": "locally retained official physical R01-R09 per-sequence CSVs",
        },
        "missing": [
            "32K selected Base under the RWTT-28Lite contract",
            "32K selected Base under the current fixed-4 summary table",
        ],
    }
    (ROOT / "review_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
