#!/usr/bin/env python3
"""Plot selected rescued 2K-U-PATH step500 against author Unicorn and G-PCC."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


merged = read(ROOT / "results/base_rescue_screening_20260903/base_rescue_merged_results.csv")
candidate = [r for r in merged if r["record_type"] == "candidate_external"
             and r["arm"] == "2K-U-PATH" and r["step"] == "500"]
if len(candidate) != 8:
    raise RuntimeError("Expected eight selected 2K-U-PATH step500 rows")

official_8i = read(ROOT / "drafts/20260901_multirate_review/02_summary_tables/Table13_Official_R01_R09_8iVFB_PerSequence.csv")
official_owlii = read(ROOT / "drafts/20260902_owlii_joint256_review/tables/owlii_author_r01_r09_per_sequence.csv")
gpcc = read(ROOT / "results/best_base_256_32k_external_20260903/author_gpcc_raht21_per_sequence.csv")

with (OUT / "selected_2k_u_path_step500_per_sequence.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = ["dataset", "sequence", "arm", "step", "profile", "lambda", "bpp", "Y", "U", "V", "YUV611", "official_interp_YUV611", "delta_to_official_curve"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows([{k:r[k] for k in fields} for r in candidate])


def panel(dataset, sequences, official_rows, xfield, yfield, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    for ax, sequence in zip(axes.flat, sequences):
        off = sorted([r for r in official_rows if r["sequence"].lower() == sequence],
                     key=lambda r: float(r[xfield]))
        g = sorted([r for r in gpcc if r["dataset"] == dataset and r["sequence"].lower() == sequence],
                   key=lambda r: float(r["attribute_bpp"]))
        point = next(r for r in candidate if r["dataset"] == dataset and r["sequence"].lower() == sequence)
        ax.plot([float(r[xfield]) for r in off], [float(r[yfield]) for r in off],
                "o-", color="#4c78a8", linewidth=1.8, markersize=4.5, label="Official Unicorn R01–R09")
        ax.plot([float(r["attribute_bpp"]) for r in g], [float(r["yuv_psnr"]) for r in g],
                "s--", color="#8a8a8a", linewidth=1.4, markersize=4, label="G-PCC RAHT")
        x, y = float(point["bpp"]), float(point["YUV611"])
        ax.scatter([x], [y], marker="*", s=180, color="#e45756", edgecolor="black",
                   linewidth=.6, zorder=5, label="Rescued 2K Base (U-PATH step500)")
        ax.annotate("2K Base\n{:.4f} bpp, {:.3f} dB".format(x, y), (x, y),
                    xytext=(8, 9), textcoords="offset points", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#e45756", alpha=.88))
        ax.set_xscale("log")
        ax.set_title(sequence.replace("_", " ").title())
        ax.set_xlabel("Physical attribute rate (bpp)")
        ax.set_ylabel("YUV PSNR 6:1:1 (dB)")
        ax.grid(True, which="both", alpha=.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945),
               ncol=3, frameon=False)
    fig.suptitle("{}: selected rescued 2K Base vs author references".format(
        "8iVFB" if dataset == "8ivfb" else "Owlii"), fontsize=15, y=.988)
    fig.subplots_adjust(left=.075, right=.985, bottom=.075, top=.875,
                        hspace=.28, wspace=.19)
    fig.savefig(OUT / output, dpi=220, bbox_inches="tight")
    plt.close(fig)


panel("8ivfb", ["longdress", "loot", "redandblack", "soldier"],
      official_8i, "bpp", "YUV611", "selected_2k_base_8ivfb_four_panel.png")
panel("owlii", ["basketball_player", "dancer", "exercise", "model"],
      official_owlii, "physical_bpp", "yuv611_db", "selected_2k_base_owlii_four_panel.png")
