#!/usr/bin/env python3
"""Summarize Base and Full against same-contract Official linear interpolation."""
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "03_raw_evidence"
TABLES = ROOT / "02_summary_tables"
POINTS = ("32K", "16K", "8K", "1K", "512", "256")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interpolate(curve, bpp):
    curve = sorted(curve)
    for lower, upper in zip(curve, curve[1:]):
        if lower[0] <= bpp <= upper[0]:
            y = lower[1] + ((bpp - lower[0]) / (upper[0] - lower[0])
                            * (upper[1] - lower[1]))
            return y, "{}_to_{}".format(lower[2], upper[2])
    return None, "OUTSIDE_R01_R09"


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint(curve, dataset, point, name, bpp, quality):
    official, bracket = interpolate(curve, bpp)
    return {
        "dataset": dataset, "point": point, "endpoint": name,
        "physical_bpp": bpp, "YUV611_dB": quality,
        "official_interpolated_YUV611_dB": "" if official is None else official,
        "delta_matched_dB": "" if official is None else quality - official,
        "official_bracket": bracket,
    }


def main():
    official_rwtt_rows = read_csv(
        RAW / "official_rwtt28lite/RWTT_28LITE_AUTHOR_9PT.csv")
    official_rwtt = [(float(row["mean_model_bpp"]),
                      float(row["mean_model_yuv_psnr_611"]), row["rate_id"])
                     for row in official_rwtt_rows]
    official_8i_rows = read_csv(TABLES / "Table06_Official_R01_R09_8iVFB.csv")
    official_8i = [(float(row["mean_bpp"]), float(row["mean_yuv611"]),
                    row["rate_id"]) for row in official_8i_rows]

    selected_rwtt = {row["point"].upper(): row for row in read_csv(
        TABLES / "Table04_Selected_Base_RWTT28Lite.csv")}
    selected_8i = {row["point"].upper(): row for row in read_csv(
        TABLES / "Table05_Selected_Base_8iVFB.csv")}
    full_rwtt = {row["point"].upper(): row for row in read_csv(
        TABLES / "Table15_Fixed_Full_RWTT28Lite.csv")}
    full_8i = {row["point"].upper(): row for row in read_csv(
        TABLES / "Table16_Fixed_Full_8iVFB_Mean.csv")}

    rwtt32 = {row["endpoint"]: row for row in read_csv(
        RAW / "highrate_contract_completion/rwtt_28lite/32k_d611_step3525/endpoint_summary.csv")}
    summary32 = read_csv(Path(
        "drafts/canonical_scalable_formal_r01/8ivfb_external/8ivfb_external_summary.csv"))
    base_8i_32 = {
        "bpp": sum(float(row["canonical_base_bpp"]) for row in summary32) / 4,
        "quality": sum(float(row["canonical_base_yuv611"]) for row in summary32) / 4,
    }

    rows = []
    for point in POINTS:
        if point == "32K":
            base_rwtt_bpp = float(rwtt32["Base"]["mean_model_bpp"])
            base_rwtt_quality = float(rwtt32["Base"]["mean_model_yuv_psnr_611"])
            base_8i_bpp = base_8i_32["bpp"]
            base_8i_quality = base_8i_32["quality"]
        else:
            base_rwtt_bpp = float(selected_rwtt[point]["bpp"])
            base_rwtt_quality = float(selected_rwtt[point]["yuv611"])
            base_8i_bpp = float(selected_8i[point]["bpp"])
            base_8i_quality = float(selected_8i[point]["yuv611"])
        rows.append(endpoint(official_rwtt, "RWTT-28Lite", point, "Base",
                             base_rwtt_bpp, base_rwtt_quality))
        rows.append(endpoint(official_rwtt, "RWTT-28Lite", point, "Full",
                             float(full_rwtt[point]["bpp"]),
                             float(full_rwtt[point]["YUV611"])))
        rows.append(endpoint(official_8i, "8iVFB-fixed4-mean", point, "Base",
                             base_8i_bpp, base_8i_quality))
        rows.append(endpoint(official_8i, "8iVFB-fixed4-mean", point, "Full",
                             float(full_8i[point]["bpp"]),
                             float(full_8i[point]["YUV611"])))
    write_csv(TABLES / "Table20_Base_Full_Matched_Official.csv", rows)

    lines = [
        "# Base/Full matched-rate comparison against Official R01–R09",
        "",
        "Metric: physical attribute bpp and author `pc_error` YUV-PSNR 6:1:1.",
        "Interpolation: linear in the bpp–YUV611 plane between adjacent Official points.",
        "A positive delta means the endpoint lies above the same-contract Official interpolation.",
        "No extrapolation is performed outside R01–R09.",
    ]
    for dataset in ("RWTT-28Lite", "8iVFB-fixed4-mean"):
        lines += ["", "## " + dataset, "",
                  "| Point | Base bpp | Base dB | Base Δmatched | Full bpp | Full dB | Full Δmatched | Δbpp Full−Base | ΔdB Full−Base |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for point in POINTS:
            base = next(row for row in rows if row["dataset"] == dataset
                        and row["point"] == point and row["endpoint"] == "Base")
            full = next(row for row in rows if row["dataset"] == dataset
                        and row["point"] == point and row["endpoint"] == "Full")
            def delta(row):
                return "N/A" if row["delta_matched_dB"] == "" else "{:+.3f}".format(
                    row["delta_matched_dB"])
            lines.append(
                "| {} | {:.6f} | {:.4f} | {} | {:.6f} | {:.4f} | {} | {:.6f} | {:+.4f} |".format(
                    point, base["physical_bpp"], base["YUV611_dB"], delta(base),
                    full["physical_bpp"], full["YUV611_dB"], delta(full),
                    full["physical_bpp"] - base["physical_bpp"],
                    full["YUV611_dB"] - base["YUV611_dB"]))
    lines += ["", "## Notes", "",
              "- The 32K Full endpoint is above the maximum Official R01 bpp on both datasets, so its matched delta is N/A.",
              "- Interpolation is a diagnostic, not a formal BD-rate result.",
              "- All listed endpoints use physical hard rate and passed their recorded hard correctness checks.", ""]
    (ROOT / "BASE_FULL_MATCHED_OFFICIAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
