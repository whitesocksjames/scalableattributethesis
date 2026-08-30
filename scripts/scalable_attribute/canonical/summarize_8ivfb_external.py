#!/usr/bin/env python3
"""Merge and summarize the four-sequence external 8iVFB audit."""

import argparse
import csv
import json
import os
import re

import matplotlib.pyplot as plt


SEQUENCES = {
    "longdress": "longdress_vox10_1300.csv",
    "loot": "loot_vox10_1200.csv",
    "redandblack": "redandblack_vox10_1550.csv",
    "soldier": "soldier_vox10_0690.csv",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def by_endpoint(rows, endpoint):
    matches = [row for row in rows if row["endpoint"] == endpoint]
    if len(matches) != 1:
        raise RuntimeError("Expected one {} row, got {}".format(endpoint, len(matches)))
    return matches[0]


def number(row, key):
    return float(row[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--longdress-formal", required=True)
    parser.add_argument("--author-csv-root", required=True)
    args = parser.parse_args()
    os.makedirs(args.audit_root, exist_ok=True)

    all_rows = {}
    formal_longdress = read_csv(args.longdress_formal)
    diagnostic_longdress = read_csv(os.path.join(
        args.audit_root, "longdress", "diagnostic.csv"))
    longdress = [row for row in formal_longdress if row["endpoint"].startswith("R")]
    longdress += [by_endpoint(diagnostic_longdress, name) for name in (
        "B_native", "Canonical_Base", "Canonical_Full_step1763")]
    full3525 = dict(by_endpoint(formal_longdress, "Canonical_Full"))
    full3525["endpoint"] = "Canonical_Full_step3525"
    longdress.append(full3525)
    all_rows["longdress"] = longdress
    write_csv(os.path.join(args.audit_root, "longdress", "physical_rd.csv"), longdress)

    for sequence in ("loot", "redandblack", "soldier"):
        all_rows[sequence] = read_csv(os.path.join(
            args.audit_root, sequence, "physical_rd.csv"))

    manager = []
    comparisons = []
    base_native = []
    trajectory = []
    author_comparison = []
    author_summary = {}
    channel_gaps = []
    for sequence, rows in all_rows.items():
        r01 = by_endpoint(rows, "R01")
        native = by_endpoint(rows, "B_native")
        base = by_endpoint(rows, "Canonical_Base")
        early = by_endpoint(rows, "Canonical_Full_step1763")
        final = by_endpoint(rows, "Canonical_Full_step3525")
        manager.append({
            "sequence": sequence,
            "official_r01_bpp": number(r01, "physical_bpp"),
            "official_r01_yuv611": number(r01, "yuv_psnr_611"),
            "canonical_base_bpp": number(base, "physical_bpp"),
            "canonical_base_yuv611": number(base, "yuv_psnr_611"),
            "canonical_full3525_bpp": number(final, "physical_bpp"),
            "canonical_full3525_yuv611": number(final, "yuv_psnr_611"),
        })
        comparisons.append({
            "sequence": sequence,
            "full_vs_r01_delta_bpp": number(final, "physical_bpp") - number(r01, "physical_bpp"),
            "full_vs_r01_delta_bpp_percent": 100.0 * (
                number(final, "physical_bpp") / number(r01, "physical_bpp") - 1.0),
            "full_vs_r01_delta_yuv611": number(final, "yuv_psnr_611") - number(r01, "yuv_psnr_611"),
            "base_gain_vs_native_yuv611": number(base, "yuv_psnr_611") - number(native, "yuv_psnr_611"),
        })
        base_native.append({
            "sequence": sequence,
            "shared_base_bpp": number(base, "physical_bpp"),
            **{"delta_" + channel: number(base, channel + "_psnr") - number(native, channel + "_psnr")
               for channel in ("y", "u", "v")},
            "delta_yuv611": number(base, "yuv_psnr_611") - number(native, "yuv_psnr_611"),
        })
        trajectory.append({
            "sequence": sequence,
            "step1763_bpp": number(early, "physical_bpp"),
            "step3525_bpp": number(final, "physical_bpp"),
            "delta_bpp_3525_minus_1763": number(final, "physical_bpp") - number(early, "physical_bpp"),
            **{"delta_" + channel + "_psnr_3525_minus_1763":
               number(final, channel + "_psnr") - number(early, channel + "_psnr")
               for channel in ("y", "u", "v")},
            "delta_yuv611_3525_minus_1763": number(final, "yuv_psnr_611") - number(early, "yuv_psnr_611"),
        })
        channel_gaps.append({
            "sequence": sequence,
            **{"full_vs_r01_delta_" + channel + "_psnr":
               number(final, channel + "_psnr") - number(r01, channel + "_psnr")
               for channel in ("y", "u", "v")},
        })

        author = read_csv(os.path.join(args.author_csv_root, SEQUENCES[sequence]))
        author_by_lambda = {int(row["lmb"]): row for row in author}
        differences = []
        compatible_differences = []
        for rate in [by_endpoint(rows, "R{:02d}".format(index))
                     for index in range(1, 10)]:
            lmb = int(float(rate["lambda"]))
            source = author_by_lambda[lmb]
            quality_delta = number(rate, "yuv_psnr_611") - float(source["YUV-PSNR"])
            differences.append(abs(quality_delta))
            match = re.search(r"_R([0-9]+)\.ply$", source["filedir_rec"])
            author_rate_label = "R" + match.group(1) if match else "UNKNOWN"
            contract_note = "MATCHED_OFFICIAL_MAPPING"
            if lmb == 8192 and author_rate_label == "R4":
                contract_note = "AUTHOR_OVERLAP_USES_8k256_L8192"
            else:
                compatible_differences.append(abs(quality_delta))
            author_comparison.append({
                "sequence": sequence,
                "rate_id": rate["endpoint"],
                "lambda": lmb,
                "author_estimated_bpp": float(source["bpp"]),
                "local_physical_bpp": number(rate, "physical_bpp"),
                "physical_minus_estimated_bpp": number(rate, "physical_bpp") - float(source["bpp"]),
                "author_yuv611": float(source["YUV-PSNR"]),
                "local_yuv611": number(rate, "yuv_psnr_611"),
                "local_minus_author_yuv611": quality_delta,
                "author_output_rate_label": author_rate_label,
                "contract_note": contract_note,
            })
        author_summary[sequence] = {
            "mean_abs_yuv611_difference": sum(differences) / len(differences),
            "max_abs_yuv611_difference": max(differences),
            "matched_contract_mean_abs_yuv611_difference": (
                sum(compatible_differences) / len(compatible_differences)),
            "matched_contract_max_abs_yuv611_difference": max(
                compatible_differences),
        }

        official = [by_endpoint(rows, "R{:02d}".format(index)) for index in range(1, 10)]
        figure, axis = plt.subplots(figsize=(7.2, 5.2))
        axis.plot([number(row, "physical_bpp") for row in official],
                  [number(row, "yuv_psnr_611") for row in official], "o-",
                  label="Official Unicorn physical R01-R09")
        for row in official:
            axis.annotate(row["endpoint"],
                          (number(row, "physical_bpp"), number(row, "yuv_psnr_611")),
                          xytext=(4, 4), textcoords="offset points", fontsize=8)
        for row, label, marker in ((base, "Canonical Base", "*"),
                                   (early, "Canonical Full step1763", "D"),
                                   (final, "Canonical Full step3525", "*")):
            axis.scatter(number(row, "physical_bpp"), number(row, "yuv_psnr_611"),
                         s=130, marker=marker, label=label, zorder=5)
        axis.set_xlabel("Physical attribute rate (bpp)")
        axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
        axis.set_title("{} — 8iVFB physical attribute RD".format(sequence.title()))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        png = os.path.join(args.audit_root, sequence, "physical_rd.png")
        figure.savefig(png, dpi=180)
        figure.savefig(png[:-4] + ".svg")
        plt.close(figure)
        with open(os.path.join(args.audit_root, sequence, "physical_rd.json"),
                  "w", encoding="utf-8") as handle:
            json.dump({"sequence": sequence, "rows": rows}, handle, indent=2)

    write_csv(os.path.join(args.audit_root, "8ivfb_external_summary.csv"), manager)
    write_csv(os.path.join(args.audit_root, "endpoint_comparison.csv"), comparisons)
    write_csv(os.path.join(args.audit_root, "base_native_comparison.csv"), base_native)
    write_csv(os.path.join(args.audit_root, "enhancement_checkpoint_trajectory.csv"), trajectory)
    write_csv(os.path.join(args.audit_root, "author_vs_physical_reference.csv"), author_comparison)
    write_csv(os.path.join(args.audit_root, "full_vs_r01_channel_gaps.csv"), channel_gaps)
    with open(os.path.join(args.audit_root, "8ivfb_external_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "manager_table": manager,
            "endpoint_comparison": comparisons,
            "base_native": base_native,
            "trajectory": trajectory,
            "author_comparison": author_summary,
            "channel_gaps": channel_gaps,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
