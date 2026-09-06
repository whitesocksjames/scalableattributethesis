#!/usr/bin/env python3
"""Build fixed six-point Full comparisons from retained physical hard results."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "03_raw_evidence"
TABLES = ROOT / "02_summary_tables"
FIGURES = ROOT / "01_figures"
SEQUENCES = ("longdress", "loot", "redandblack", "soldier")
POINTS = ("32K", "16K", "8K", "1K", "512", "256")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_rwtt():
    specs = {
        "32K": RAW / "highrate_contract_completion/rwtt_28lite/32k_d611_step3525/endpoint_summary.csv",
        "16K": RAW / "current_round/branch_a/16k/rwtt_28lite/step_1763/endpoint_summary.csv",
        "8K": RAW / "current_round/branch_a/8k/rwtt_28lite/step_1763/endpoint_summary.csv",
        "1K": RAW / "lowrate_enhancement_stage1/1k/rwtt_28lite/step_1763/endpoint_summary.csv",
        "512": RAW / "lowrate_enhancement_stage1/512/rwtt_28lite/step_1763/endpoint_summary.csv",
        "256": RAW / "lowrate_enhancement_stage1/256/rwtt_28lite/step_1763/endpoint_summary.csv",
    }
    steps = {"32K": 3525, "16K": 1763, "8K": 1763,
             "1K": 1763, "512": 1763, "256": 1763}
    result = []
    for point in POINTS:
        row = next(row for row in read_csv(specs[point]) if row["endpoint"] == "Full")
        result.append({"point": point, "checkpoint_step": steps[point],
                       "bpp": float(row["mean_model_bpp"]),
                       "YUV611": float(row["mean_model_yuv_psnr_611"]),
                       "num_models": int(row["num_models"]),
                       "num_h5": int(row["num_h5"]), "hard_status": "PASS"})
    return result


def full_8i_per_sequence():
    result = []
    summary32 = {row["sequence"]: row for row in read_csv(
        Path("drafts/canonical_scalable_formal_r01/8ivfb_external/8ivfb_external_summary.csv"))}
    for sequence in SEQUENCES:
        row = summary32[sequence]
        result.append({"sequence": sequence, "point": "32K", "checkpoint_step": 3525,
                       "bpp": float(row["canonical_full3525_bpp"]),
                       "YUV611": float(row["canonical_full3525_yuv611"]),
                       "hard_status": "PASS"})
    paths = {
        "16K": RAW / "highrate_contract_completion/8ivfb/16k",
        "8K": RAW / "highrate_contract_completion/8ivfb/8k",
        "1K": RAW / "multirate_full_step1763/8ivfb/1k",
        "512": RAW / "multirate_full_step1763/8ivfb/512",
        "256": RAW / "multirate_full_step1763/8ivfb/256",
    }
    labels = {point: point.lower() + "_step1763" for point in paths}
    for point, base in paths.items():
        for sequence in SEQUENCES:
            row = next(row for row in read_csv(base / sequence / "physical_rd.csv")
                       if row["endpoint"] == labels[point])
            exact = float(row["hard_roundtrip_max_abs_difference"]) == 0
            result.append({"sequence": sequence, "point": point,
                           "checkpoint_step": 1763,
                           "bpp": float(row["physical_bpp"]),
                           "YUV611": float(row["yuv_psnr_611"]),
                           "hard_status": "PASS" if exact else "FAIL"})
    return result


def average_8i(rows):
    by_point = defaultdict(list)
    for row in rows:
        by_point[row["point"]].append(row)
    return [{"point": point, "checkpoint_step": group[0]["checkpoint_step"],
             "bpp": sum(row["bpp"] for row in group) / 4,
             "YUV611": sum(row["YUV611"] for row in group) / 4,
             "num_sequences": 4,
             "hard_status": "PASS" if all(row["hard_status"] == "PASS" for row in group)
             else "FAIL"}
            for point in POINTS for group in (by_point[point],)]


def interpolate(curve, bpp):
    curve = sorted(curve)
    for lower, upper in zip(curve, curve[1:]):
        if lower[0] <= bpp <= upper[0]:
            value = lower[1] + ((bpp - lower[0]) / (upper[0] - lower[0])
                                * (upper[1] - lower[1]))
            return value, lower[2], upper[2]
    return None


def matched_rows(dataset, full, official):
    curve = [(row["bpp"], row["YUV611"], row["rate_id"]) for row in official]
    result = []
    for row in full:
        matched = interpolate(curve, row["bpp"])
        result.append({
            "dataset": dataset, "point": row["point"], "bpp": row["bpp"],
            "Full_YUV611": row["YUV611"],
            "Official_interpolated_YUV611": "" if matched is None else matched[0],
            "delta_matched_db": "" if matched is None else row["YUV611"] - matched[0],
            "bracket": "OUTSIDE_R01_R09" if matched is None
            else "{}_to_{}".format(matched[1], matched[2]),
        })
    return result


def style(axis, title):
    axis.set_title(title)
    axis.set_xlabel("Physical attribute rate (bpp)")
    axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
    axis.grid(True, alpha=0.25)


def plot_one(axis, official, base, full, title, legend=True):
    axis.plot([row["bpp"] for row in official], [row["YUV611"] for row in official],
              "o-", color="#777777", label="Official R01–R09 Full")
    for row in official:
        axis.annotate(row["rate_id"], (row["bpp"], row["YUV611"]),
                      xytext=(3, 4), textcoords="offset points", fontsize=7)
    axis.plot([row["bpp"] for row in base], [row["YUV611"] for row in base],
              "^--", color="#2c7fb8", label="Selected Base")
    for row in base:
        axis.annotate(row["point"], (row["bpp"], row["YUV611"]),
                      xytext=(3, -11), textcoords="offset points", fontsize=7)
    axis.plot([row["bpp"] for row in full], [row["YUV611"] for row in full],
              "s-", color="#d95f02", label="Fixed canonical Full")
    for row in full:
        axis.annotate(row["point"], (row["bpp"], row["YUV611"]),
                      xytext=(4, 5), textcoords="offset points", fontsize=8)
    style(axis, title)
    if legend:
        axis.legend(fontsize=8)


def main():
    official_rwtt = [{"rate_id": row["rate_id"], "bpp": float(row["mean_model_bpp"]),
                      "YUV611": float(row["mean_model_yuv_psnr_611"])}
                     for row in read_csv(RAW / "official_rwtt28lite/RWTT_28LITE_AUTHOR_9PT.csv")]
    base_rwtt = [{"point": row["point"], "bpp": float(row["bpp"]),
                  "YUV611": float(row["yuv611"])}
                 for row in read_csv(TABLES / "Table04_Selected_Base_RWTT28Lite.csv")]
    official_8i_seq = read_csv(TABLES / "Table13_Official_R01_R09_8iVFB_PerSequence.csv")
    base_8i_seq = read_csv(TABLES / "Table12_Selected_Base_8iVFB_PerSequence.csv")
    full_rwtt_rows = full_rwtt()
    full_8i_seq_rows = full_8i_per_sequence()
    full_8i_mean_rows = average_8i(full_8i_seq_rows)
    write_csv(TABLES / "Table15_Fixed_Full_RWTT28Lite.csv", full_rwtt_rows)
    write_csv(TABLES / "Table16_Fixed_Full_8iVFB_Mean.csv", full_8i_mean_rows)
    write_csv(TABLES / "Table17_Fixed_Full_8iVFB_PerSequence.csv", full_8i_seq_rows)

    fig, axis = plt.subplots(figsize=(9, 6.2))
    plot_one(axis, official_rwtt, base_rwtt, full_rwtt_rows,
             "Official, selected Base and fixed Full — RWTT-28Lite")
    fig.tight_layout(); fig.savefig(FIGURES / "Fig07_Fixed_Full_RD_RWTT28Lite.png", dpi=190)
    fig.savefig(FIGURES / "Fig07_Fixed_Full_RD_RWTT28Lite.svg"); plt.close(fig)

    official_mean = []
    for rate in ("R{:02d}".format(i) for i in range(1, 10)):
        rows = [row for row in official_8i_seq if row["rate_id"] == rate]
        official_mean.append({"rate_id": rate,
                              "bpp": sum(float(row["bpp"]) for row in rows) / 4,
                              "YUV611": sum(float(row["YUV611"]) for row in rows) / 4})
    base_mean = []
    for point in (row["point"] for row in base_rwtt):
        rows = [row for row in base_8i_seq if row["point"] == point]
        base_mean.append({"point": point,
                          "bpp": sum(float(row["bpp"]) for row in rows) / 4,
                          "YUV611": sum(float(row["YUV611"]) for row in rows) / 4})
    write_csv(TABLES / "Table18_Fixed_Full_Matched_Official_Mean.csv",
              matched_rows("RWTT-28Lite", full_rwtt_rows, official_rwtt)
              + matched_rows("8iVFB-fixed4-mean", full_8i_mean_rows, official_mean))
    per_sequence_matched = []
    for sequence in SEQUENCES:
        official = [{"rate_id": row["rate_id"], "bpp": float(row["bpp"]),
                     "YUV611": float(row["YUV611"])} for row in official_8i_seq
                    if row["sequence"] == sequence]
        full = [row for row in full_8i_seq_rows if row["sequence"] == sequence]
        per_sequence_matched += matched_rows("8iVFB-" + sequence, full, official)
    write_csv(TABLES / "Table19_Fixed_Full_Matched_Official_8iVFB_PerSequence.csv",
              per_sequence_matched)
    fig, axis = plt.subplots(figsize=(9, 6.2))
    plot_one(axis, official_mean, base_mean, full_8i_mean_rows,
             "Official, selected Base and fixed Full — 8iVFB fixed-4 mean")
    fig.tight_layout(); fig.savefig(FIGURES / "Fig08_Fixed_Full_RD_8iVFB_Mean.png", dpi=190)
    fig.savefig(FIGURES / "Fig08_Fixed_Full_RD_8iVFB_Mean.svg"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    names = {"longdress": "Longdress 1300", "loot": "Loot 1200",
             "redandblack": "Redandblack 1550", "soldier": "Soldier 0690"}
    for axis, sequence in zip(axes.flat, SEQUENCES):
        official = [{"rate_id": row["rate_id"], "bpp": float(row["bpp"]),
                     "YUV611": float(row["YUV611"])} for row in official_8i_seq
                    if row["sequence"] == sequence]
        base = [{"point": row["point"], "bpp": float(row["bpp"]),
                 "YUV611": float(row["YUV611"])} for row in base_8i_seq
                if row["sequence"] == sequence]
        full = [row for row in full_8i_seq_rows if row["sequence"] == sequence]
        plot_one(axis, official, base, full, names[sequence], legend=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Official, selected Base and fixed Full — 8iVFB per sequence",
                 y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "Fig09_Fixed_Full_RD_8iVFB_PerSequence.png", dpi=190)
    fig.savefig(FIGURES / "Fig09_Fixed_Full_RD_8iVFB_PerSequence.svg"); plt.close(fig)


if __name__ == "__main__":
    main()
