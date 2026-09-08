#!/usr/bin/env python3
"""Plot an official Unicorn curve with canonical Base and Full endpoints."""

import argparse
import csv

import matplotlib.pyplot as plt


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row, candidates):
    for key in candidates:
        if key in row and row[key] != "":
            return float(row[key])
    raise KeyError("None of {} is present".format(candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-curve", required=True)
    parser.add_argument("--canonical-endpoints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    official = [row for row in rows(args.official_curve)
                if row.get("source", "OFFICIAL_RELEASED") == "OFFICIAL_RELEASED"]
    canonical = [row for row in rows(args.canonical_endpoints)
                 if row.get("endpoint") in ("Base", "Full", "Canonical_Base",
                                             "Canonical_Full")]
    ox = [value(row, ("mean_model_bpp", "physical_bpp")) for row in official]
    oy = [value(row, ("mean_model_yuv_psnr_611", "yuv_psnr_611"))
          for row in official]
    labels = [row.get("rate_id") or row.get("endpoint") for row in official]
    cx = [value(row, ("mean_model_bpp", "physical_bpp")) for row in canonical]
    cy = [value(row, ("mean_model_yuv_psnr_611", "yuv_psnr_611"))
          for row in canonical]
    clabels = [row["endpoint"].replace("Canonical_", "Canonical ")
               for row in canonical]

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(ox, oy, "o-", color="#3568b8", label="Official Unicorn R01-R09")
    for x, y, label in zip(ox, oy, labels):
        axis.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points",
                      fontsize=8)
    colors = ("#e08214", "#c73535")
    for index, (x, y, label) in enumerate(zip(cx, cy, clabels)):
        axis.scatter([x], [y], marker="*", s=150, color=colors[index % 2],
                     label=label, zorder=5)
    if len(cx) == 2:
        axis.plot(cx, cy, "--", color="#777777", linewidth=1)
    axis.set_xlabel("Physical attribute rate (bpp)")
    axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
    axis.set_title(args.title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    if args.output.lower().endswith(".png"):
        figure.savefig(args.output[:-4] + ".svg")


if __name__ == "__main__":
    main()
