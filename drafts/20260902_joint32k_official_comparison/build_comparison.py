#!/usr/bin/env python3
"""Build five same-contract Official-vs-32K comparison figures."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-joint32k")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SEQUENCES = ("longdress", "loot", "redandblack", "soldier")
THESIS_ORDER = (
    "Selected Base",
    "Sequential D111 Full",
    "Sequential D611 Full",
    "Joint32K Base",
    "Joint32K Full",
)
COLORS = {
    "Selected Base": "#1f77b4",
    "Sequential D111 Full": "#2ca02c",
    "Sequential D611 Full": "#9467bd",
    "Joint32K Base": "#ff7f0e",
    "Joint32K Full": "#d62728",
}


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def endpoint(rows, name):
    return next(row for row in rows if row["endpoint"] == name)


def point(dataset, label, row, provenance):
    return {
        "dataset": dataset,
        "label": label,
        "physical_bpp": float(row["physical_bpp"]),
        "yuv_psnr_611": float(row["yuv_psnr_611"]),
        "provenance": provenance,
    }


def load_8ivfb(sequence):
    existing = csv_rows(
        REPO / "drafts/canonical_scalable_formal_r01/8ivfb_external" /
        sequence / "physical_rd.csv")
    joint = csv_rows(ROOT / "raw_n30/8ivfb" / sequence / "physical_rd.csv")
    official = [point(sequence, row["endpoint"], row, "OFFICIAL_RELEASED")
                for row in existing if row["endpoint"].startswith("R")]
    thesis = [
        point(sequence, "Selected Base", endpoint(existing, "Canonical_Base"),
              "CURRENT_CANONICAL"),
        point(sequence, "Sequential D111 Full",
              endpoint(existing, "Canonical_Full_step1763"),
              "CURRENT_CANONICAL"),
        point(sequence, "Sequential D611 Full",
              endpoint(existing, "Canonical_Full_step3525"),
              "CURRENT_CANONICAL"),
        point(sequence, "Joint32K Base",
              endpoint(joint, "Joint32K_step3525_Base"),
              "CURRENT_JOINT32K"),
        point(sequence, "Joint32K Full",
              endpoint(joint, "Joint32K_step3525_Full"),
              "CURRENT_JOINT32K"),
    ]
    return official, thesis


def summary_endpoint(path, name):
    data = json.loads(path.read_text(encoding="utf-8"))
    row = next(item for item in data["endpoints"] if item["endpoint"] == name)
    return {
        "physical_bpp": row["mean_model_bpp"],
        "yuv_psnr_611": row["mean_model_yuv_psnr_611"],
    }


def load_rwtt():
    official_path = (REPO / "drafts/20260901_multirate_review/03_raw_evidence/"
                     "official_rwtt28lite/RWTT_28LITE_AUTHOR_9PT.csv")
    official = []
    for row in csv_rows(official_path):
        official.append({
            "dataset": "RWTT-28Lite", "label": row["rate_id"],
            "physical_bpp": float(row["mean_model_bpp"]),
            "yuv_psnr_611": float(row["mean_model_yuv_psnr_611"]),
            "provenance": "OFFICIAL_RELEASED",
        })
    seq_path = (REPO / "drafts/20260901_multirate_review/03_raw_evidence/"
                "highrate_contract_completion/rwtt_28lite/"
                "32k_d611_step3525/endpoint_summary.json")
    d111_path = ROOT / "raw_n30/rwtt_28lite/d111_step1763/endpoint_summary.json"
    joint_path = ROOT / "raw_fau/rwtt_28lite/step_3525/endpoint_summary.json"
    base = summary_endpoint(seq_path, "Base")
    thesis_values = [
        ("Selected Base", base, "CURRENT_CANONICAL"),
        ("Sequential D111 Full", summary_endpoint(d111_path, "Full"),
         "CURRENT_CANONICAL"),
        ("Sequential D611 Full", summary_endpoint(seq_path, "Full"),
         "CURRENT_CANONICAL"),
        ("Joint32K Base", summary_endpoint(joint_path, "Base"),
         "CURRENT_JOINT32K"),
        ("Joint32K Full", summary_endpoint(joint_path, "Full"),
         "CURRENT_JOINT32K"),
    ]
    thesis = [{"dataset": "RWTT-28Lite", "label": label,
               "physical_bpp": value["physical_bpp"],
               "yuv_psnr_611": value["yuv_psnr_611"],
               "provenance": provenance}
              for label, value, provenance in thesis_values]
    return official, thesis


def plot(dataset, official, thesis, filename):
    official = sorted(official, key=lambda row: row["physical_bpp"])
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.plot([row["physical_bpp"] for row in official],
            [row["yuv_psnr_611"] for row in official],
            color="#222222", marker="o", linewidth=1.8,
            markersize=4.5, label="Official R01–R09")
    for row in official:
        ax.annotate(row["label"], (row["physical_bpp"], row["yuv_psnr_611"]),
                    xytext=(3, 4), textcoords="offset points", fontsize=7)
    for label in THESIS_ORDER:
        row = next(item for item in thesis if item["label"] == label)
        marker = "s" if "Base" in label else "D"
        ax.scatter(row["physical_bpp"], row["yuv_psnr_611"], s=62,
                   marker=marker, color=COLORS[label], edgecolor="white",
                   linewidth=0.7, zorder=4, label=label)
        ax.annotate(label, (row["physical_bpp"], row["yuv_psnr_611"]),
                    xytext=(5, -10 if "Full" in label else 6),
                    textcoords="offset points", fontsize=7.5,
                    color=COLORS[label])
    ax.set_xlabel("Physical attribute bpp")
    ax.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
    ax.set_title(dataset + ": Official curve and 32K scalable endpoints")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / filename, dpi=180)
    fig.savefig(ROOT / filename.replace(".png", ".svg"))
    plt.close(fig)


def main():
    all_rows = []
    rwtt_official, rwtt_thesis = load_rwtt()
    plot("RWTT-28Lite", rwtt_official, rwtt_thesis,
         "rwtt28lite_official_and_32k_endpoints.png")
    all_rows.extend(rwtt_official + rwtt_thesis)
    for sequence in SEQUENCES:
        official, thesis = load_8ivfb(sequence)
        plot("8iVFB " + sequence, official, thesis,
             "8ivfb_{}_official_and_32k_endpoints.png".format(sequence))
        all_rows.extend(official + thesis)
    with (ROOT / "official_and_32k_endpoints.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "dataset", "label", "physical_bpp", "yuv_psnr_611", "provenance"))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
