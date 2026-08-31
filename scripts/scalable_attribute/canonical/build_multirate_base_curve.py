#!/usr/bin/env python3
"""Combine manager-selected Base checkpoints into the five-point dev curve."""

import argparse
import csv
import json
import os


EXPECTED_POINTS = ("2k", "4k", "8k", "16k", "32k")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-curve", required=True)
    parser.add_argument(
        "--selection", action="append", required=True,
        metavar="POINT,CANDIDATE,ENDPOINT_SUMMARY_CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selection(value):
    pieces = value.split(",", 2)
    if len(pieces) != 3 or not all(pieces):
        raise ValueError(
            "selection must be POINT,CANDIDATE,ENDPOINT_SUMMARY_CSV")
    return pieces[0].lower(), pieces[1], pieces[2]


def plot(official, selected, path):
    import matplotlib.pyplot as plt

    ox = [float(row["mean_model_bpp"]) for row in official]
    oy = [float(row["mean_model_yuv_psnr_611"]) for row in official]
    bx = [float(row["mean_model_bpp"]) for row in selected]
    by = [float(row["mean_model_yuv_psnr_611"]) for row in selected]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(ox, oy, "o-", color="#3568b8", label="Official Unicorn Full")
    axis.plot(bx, by, "*--", markersize=11, color="#d66b19",
              label="Canonical Base")
    for x, y, row in zip(ox, oy, official):
        axis.annotate(row["rate_id"], (x, y), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    for x, y, row in zip(bx, by, selected):
        axis.annotate(row["point"], (x, y), xytext=(4, -11),
                      textcoords="offset points", fontsize=8)
    axis.set_xlabel("Physical attribute rate (bpp)")
    axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
    axis.set_title("Canonical multi-rate Base on RWTT-28Lite")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)


def main():
    args = parse_args()
    choices = [selection(value) for value in args.selection]
    if {point for point, _, _ in choices} != set(EXPECTED_POINTS):
        raise ValueError("Exactly one selection is required for each of "
                         + ", ".join(EXPECTED_POINTS))
    selected = []
    for point in EXPECTED_POINTS:
        _, candidate, path = next(item for item in choices if item[0] == point)
        matches = [row for row in read_csv(path)
                   if row.get("candidate") == candidate]
        if len(matches) != 1:
            raise ValueError(
                "{}:{} matched {} rows".format(point, candidate, len(matches)))
        selected.append({"point": point, **matches[0]})
    rates = [float(row["mean_model_bpp"]) for row in selected]
    qualities = [float(row["mean_model_yuv_psnr_611"]) for row in selected]
    anomalies = []
    for previous, current, previous_rate, current_rate, previous_quality, current_quality in zip(
            selected, selected[1:], rates, rates[1:], qualities, qualities[1:]):
        if current_rate <= previous_rate or current_quality < previous_quality:
            anomalies.append({
                "from": previous["point"], "to": current["point"],
                "rate_change": current_rate - previous_rate,
                "quality_change_db": current_quality - previous_quality,
            })

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(
        args.output_dir, "CANONICAL_BASE_MULTIRATE_28LITE.csv")
    json_path = os.path.join(
        args.output_dir, "CANONICAL_BASE_MULTIRATE_28LITE.json")
    plot_path = os.path.join(
        args.output_dir, "CANONICAL_BASE_MULTIRATE_28LITE.png")
    paths = [csv_path, json_path] + ([plot_path] if args.plot else [])
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    with open(csv_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    with open(json_path, "x", encoding="utf-8") as handle:
        json.dump({
            "status": "ANOMALOUS" if anomalies else "PASS",
            "anomalies": anomalies,
            "selected": selected,
        }, handle, indent=2)
        handle.write("\n")
    if args.plot:
        plot(read_csv(args.author_curve), selected, plot_path)


if __name__ == "__main__":
    main()
