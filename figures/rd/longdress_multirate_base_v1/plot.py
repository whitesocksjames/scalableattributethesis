#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "longdress_official_r01_r09_and_base_5pt.csv"


with CSV_PATH.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

official = [row for row in rows if row["series"] == "Official Unicorn Full"]
base = [row for row in rows if row["series"] == "Canonical Base"]
official.sort(key=lambda row: float(row["physical_bpp"]))
base.sort(key=lambda row: float(row["physical_bpp"]))

figure, axis = plt.subplots(figsize=(7.4, 5.3))
axis.plot(
    [float(row["physical_bpp"]) for row in official],
    [float(row["yuv_psnr_611"]) for row in official],
    "o-", color="#3568b8", label="Official Unicorn Full (R01–R09)")
axis.plot(
    [float(row["physical_bpp"]) for row in base],
    [float(row["yuv_psnr_611"]) for row in base],
    "*--", markersize=11, color="#d66b19", label="Canonical Base")

for row in official:
    axis.annotate(
        row["endpoint"],
        (float(row["physical_bpp"]), float(row["yuv_psnr_611"])),
        xytext=(4, 4), textcoords="offset points", fontsize=8)
for row in base:
    axis.annotate(
        row["endpoint"],
        (float(row["physical_bpp"]), float(row["yuv_psnr_611"])),
        xytext=(4, -12), textcoords="offset points", fontsize=8)

axis.set_xlabel("Physical attribute rate (bpp)")
axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
axis.set_title("Longdress frame 1300: Official Unicorn vs Canonical Base")
axis.grid(True, alpha=0.25)
axis.legend()
figure.tight_layout()
figure.savefig(ROOT / "longdress_official_r01_r09_vs_base_5pt.png", dpi=180)
figure.savefig(ROOT / "longdress_official_r01_r09_vs_base_5pt.svg")
