#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
with (root / "DIRECT_D611_TRAJECTORY.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

direct = [row for row in rows if row["checkpoint"].startswith("Direct_")]
two_stage = [row for row in rows if row["checkpoint"].startswith("TwoStage_")]
direct.sort(key=lambda row: int(row["checkpoint"].split("s")[-1]))

figure, axis = plt.subplots(figsize=(7.2, 5.2))
axis.plot(
    [float(row["physical_bpp"]) for row in direct],
    [float(row["yuv_psnr_611"]) for row in direct],
    "o--", color="#d66b19", alpha=0.8, label="Direct D611 trajectory")
for row in direct:
    axis.annotate(
        row["checkpoint"].replace("Direct_s", "s"),
        (float(row["physical_bpp"]), float(row["yuv_psnr_611"])),
        xytext=(4, 4), textcoords="offset points", fontsize=8)
for row in two_stage:
    axis.scatter(
        [float(row["physical_bpp"])], [float(row["yuv_psnr_611"])],
        marker="D", s=75, color="#3568b8", label="Two-stage D111→D611")
    axis.annotate(
        "Two-stage s3525",
        (float(row["physical_bpp"]), float(row["yuv_psnr_611"])),
        xytext=(5, 5), textcoords="offset points", fontsize=8)
axis.set_xlabel("Physical Full rate (bpp)")
axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
axis.set_title("Direct-D611 matched-budget screening on 8i fixed-4")
axis.grid(True, alpha=0.25)
axis.legend()
figure.tight_layout()
figure.savefig(root / "direct_d611_8i_trajectory.png", dpi=180)
figure.savefig(root / "direct_d611_8i_trajectory.svg")
