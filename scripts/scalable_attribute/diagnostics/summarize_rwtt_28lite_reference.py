#!/usr/bin/env python3
"""Freeze the nine released Unicorn points evaluated on RWTT-28Lite."""

import argparse
import csv
import json
import os

from scalable_attribute.evaluation import aggregate_models, average_models
from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", required=True,
        help="Directory containing R01/per_h5.csv through R09/per_h5.csv")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty CSV: " + path)
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows, path):
    import matplotlib.pyplot as plt

    x = [float(row["mean_model_bpp"]) for row in rows]
    y = [float(row["mean_model_yuv_psnr_611"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(x, y, "o-", color="#3568b8", label="Official Unicorn R01-R09")
    for x_value, y_value, row in zip(x, y, rows):
        axis.annotate(
            row["rate_id"], (x_value, y_value), xytext=(4, 4),
            textcoords="offset points", fontsize=8)
    axis.set_xlabel("Physical attribute rate (bpp)")
    axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
    axis.set_title("Official Unicorn on RWTT-28Lite")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)


def main():
    args = parse_args()
    names = (
        "RWTT_28LITE_AUTHOR_9PT.csv",
        "RWTT_28LITE_AUTHOR_9PT_PER_H5.csv",
        "RWTT_28LITE_AUTHOR_9PT_PER_MODEL.csv",
        "RWTT_28LITE_AUTHOR_9PT.json",
    )
    if args.plot:
        names += ("RWTT_28LITE_AUTHOR_RD_CURVE.png",)
    os.makedirs(args.output_dir, exist_ok=True)
    existing = [name for name in names
                if os.path.exists(os.path.join(args.output_dir, name))]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    with open(args.manifest, encoding="utf-8") as handle:
        expected_samples = [line.strip() for line in handle if line.strip()]
    with open(args.manifest_metadata, encoding="utf-8") as handle:
        manifest_metadata = json.load(handle)
    if len(expected_samples) != 28 or len(set(expected_samples)) != 28:
        raise ValueError("RWTT-28Lite manifest must contain 28 unique H5")

    all_h5 = []
    all_models = []
    curve = []
    for rate_id, profile, base_lambda in OFFICIAL_RWTT_REFERENCE_POINTS:
        path = os.path.join(args.runs_root, rate_id, "per_h5.csv")
        rows = read_csv(path)
        actual = [row["sample"] for row in rows]
        if set(actual) != set(expected_samples) or len(actual) != 28:
            raise RuntimeError(rate_id + " does not exactly cover RWTT-28Lite")
        if any(row["checkpoint_profile"] != profile for row in rows):
            raise RuntimeError(rate_id + " profile mismatch")
        if any(int(row["base_lambda"]) != base_lambda for row in rows):
            raise RuntimeError(rate_id + " lambda mismatch")
        models = aggregate_models(rows)
        if len(models) != 28:
            raise RuntimeError(rate_id + " is not one H5 per model")
        all_h5.extend(rows)
        all_models.extend(models)
        curve.append(average_models(models))

    write_csv(os.path.join(
        args.output_dir, "RWTT_28LITE_AUTHOR_9PT_PER_H5.csv"), all_h5)
    write_csv(os.path.join(
        args.output_dir, "RWTT_28LITE_AUTHOR_9PT_PER_MODEL.csv"), all_models)
    write_csv(os.path.join(
        args.output_dir, "RWTT_28LITE_AUTHOR_9PT.csv"), curve)
    with open(os.path.join(
            args.output_dir, "RWTT_28LITE_AUTHOR_9PT.json"), "x",
            encoding="utf-8") as handle:
        json.dump({
            "status": "PASS",
            "benchmark": "RWTT-28Lite",
            "num_models": 28,
            "num_h5": 28,
            "metric": "author pc_error YUV-PSNR 6:1:1",
            "aggregation": "model-equal average; one fixed H5 per model",
            "rate": "physical x_low plus arithmetic residual streams",
            "manifest": os.path.abspath(args.manifest),
            "manifest_metadata": manifest_metadata,
            "points": curve,
        }, handle, indent=2)
        handle.write("\n")
    if args.plot:
        plot_curve(curve, os.path.join(
            args.output_dir, "RWTT_28LITE_AUTHOR_RD_CURVE.png"))


if __name__ == "__main__":
    main()
