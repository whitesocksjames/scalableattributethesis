#!/usr/bin/env python3
"""Merge one B1 source point and derive corrected Full28/Dev14 endpoints."""
import argparse
import csv
import glob
import os

from scalable_attribute.evaluation import aggregate_models, average_models


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if os.path.exists(path):
        raise FileExistsError("Refusing to overwrite: " + path)
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint(rows, step, source_rate_id, source_lambda):
    selected = [row for row in rows if int(row["step"]) == step]
    adapted = [{**row, "rate_id": "{}_B1_step{}".format(source_rate_id, step),
                "checkpoint_profile": "B1_from_{}".format(source_rate_id),
                "base_lambda": source_lambda} for row in selected]
    base = aggregate_models(adapted, "base_bits", "base_")
    full = aggregate_models(adapted, "full_bits", "full_")
    base_avg, full_avg = average_models(base), average_models(full)
    full_by_id = {row["model_id"]: row for row in full}
    per_model, ratios = [], []
    for base_row in base:
        full_row = full_by_id[base_row["model_id"]]
        source = [row for row in selected if row["model_id"] == base_row["model_id"]]
        original_bits = sum(int(row["original_bits"]) for row in source)
        ratio = int(full_row["total_bits"]) / original_bits
        ratios.append(ratio)
        per_model.append({
            "source_rate_id": source_rate_id, "step": step,
            "model_id": base_row["model_id"], "num_h5": base_row["num_h5"],
            "total_points": base_row["total_points"],
            "base_bpp": base_row["bpp"],
            "base_yuv_psnr_611": base_row["yuv_psnr_611"],
            "full_bpp": full_row["bpp"],
            "full_yuv_psnr_611": full_row["yuv_psnr_611"],
            "layered_over_original_ratio": ratio,
        })
    return per_model, {
        "source_rate_id": source_rate_id, "source_lambda": source_lambda,
        "step": step, "num_models": len(base), "num_h5": len(selected),
        "base_bpp": base_avg["mean_model_bpp"],
        "base_yuv_psnr_611": base_avg["mean_model_yuv_psnr_611"],
        "full_bpp": full_avg["mean_model_bpp"],
        "full_yuv_psnr_611": full_avg["mean_model_yuv_psnr_611"],
        "full_minus_base_bpp": full_avg["mean_model_bpp"] - base_avg["mean_model_bpp"],
        "full_minus_base_yuv_psnr_611": full_avg["mean_model_yuv_psnr_611"] - base_avg["mean_model_yuv_psnr_611"],
        "mean_model_layered_over_original_ratio": sum(ratios) / len(ratios),
        "qB_roundtrip_exact": min(int(row["qB_roundtrip_exact"]) for row in selected),
        "qB_plus_qE_exact_q": min(int(row["qB_plus_qE_exact_q"]) for row in selected),
        "max_full_abs_difference": max(float(row["full_max_abs_difference"]) for row in selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-rate-id", required=True)
    parser.add_argument("--source-lambda", type=int, required=True)
    args = parser.parse_args()
    rows = [row for path in sorted(glob.glob(
        os.path.join(args.shards_root, "shard_*", "per_h5.csv")))
            for row in read_csv(path)]
    with open(args.full_manifest, encoding="utf-8") as handle:
        full_entries = [line.strip() for line in handle if line.strip()]
    expected = {(entry, step) for entry in full_entries for step in (4, 8)}
    actual = {(row["sample"], int(row["step"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("Full28 shard coverage mismatch")
    with open(args.dev_manifest, encoding="utf-8") as handle:
        dev_entries = {line.strip() for line in handle if line.strip()}
    for split, selected in (("full", rows),
                            ("dev", [row for row in rows if row["sample"] in dev_entries])):
        output = os.path.join(args.output_root, split)
        os.makedirs(output, exist_ok=True)
        write_csv(os.path.join(output, "per_h5.csv"), selected)
        models, summaries = [], []
        for step in (4, 8):
            current_models, summary = endpoint(
                selected, step, args.source_rate_id, args.source_lambda)
            models.extend(current_models)
            summaries.append(summary)
        write_csv(os.path.join(output, "per_model.csv"), models)
        write_csv(os.path.join(output, "endpoint_summary.csv"), summaries)


if __name__ == "__main__":
    main()
