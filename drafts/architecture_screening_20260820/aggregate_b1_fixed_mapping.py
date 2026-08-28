#!/usr/bin/env python3
"""Merge B1 shards and derive Full28/Dev14 model-equal endpoint summaries."""
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


def aggregate_endpoint(rows, step):
    selected = [row for row in rows if int(row["step"]) == step]
    adapted = []
    for row in selected:
        adapted.append({
            **row, "rate_id": "B1_step{}".format(step),
            "checkpoint_profile": "official_R02_32k8k_l16384",
            "base_lambda": 16384,
        })
    base_models = aggregate_models(
        adapted, bits_field="base_bits", metric_prefix="base_")
    full_models = aggregate_models(
        adapted, bits_field="full_bits", metric_prefix="full_")
    base_average = average_models(base_models)
    full_average = average_models(full_models)
    full_by_model = {row["model_id"]: row for row in full_models}
    per_model = []
    ratios = []
    for base in base_models:
        full = full_by_model[base["model_id"]]
        source = [row for row in selected if row["model_id"] == base["model_id"]]
        original_bits = sum(int(row["official_r02_bits"]) for row in source)
        ratio = int(full["total_bits"]) / original_bits
        ratios.append(ratio)
        per_model.append({
            "step": step, "model_id": base["model_id"],
            "num_h5": base["num_h5"], "total_points": base["total_points"],
            "base_bits": base["total_bits"], "base_bpp": base["bpp"],
            "base_yuv_psnr_611": base["yuv_psnr_611"],
            "full_bits": full["total_bits"], "full_bpp": full["bpp"],
            "full_yuv_psnr_611": full["yuv_psnr_611"],
            "official_r02_bits": original_bits,
            "layered_over_original_ratio": ratio,
        })
    summary = {
        "step": step, "num_models": len(base_models),
        "num_h5": len(selected),
        "mean_model_base_bpp": base_average["mean_model_bpp"],
        "mean_model_base_yuv_psnr_611": base_average["mean_model_yuv_psnr_611"],
        "mean_model_full_bpp": full_average["mean_model_bpp"],
        "mean_model_full_yuv_psnr_611": full_average["mean_model_yuv_psnr_611"],
        "full_minus_base_bpp": (
            full_average["mean_model_bpp"] - base_average["mean_model_bpp"]),
        "full_minus_base_yuv_psnr_611": (
            full_average["mean_model_yuv_psnr_611"]
            - base_average["mean_model_yuv_psnr_611"]),
        "mean_model_layered_over_original_ratio": sum(ratios) / len(ratios),
        "max_full_abs_difference": max(
            float(row["full_max_abs_difference"]) for row in selected),
    }
    return per_model, summary


def official_neighbors(path):
    rows = read_csv(path)
    selected = [row for row in rows if row["rate_id"] in {"R02", "R03", "R04"}]
    if len(selected) != 3:
        raise RuntimeError("Expected official R02/R03/R04 in " + path)
    return [{
        "endpoint": row["rate_id"],
        "kind": "official",
        "physical_bpp": row["mean_model_bpp"],
        "direct_yuv_psnr_611": row["mean_model_yuv_psnr_611"],
        "note": "same-dataset official curve; no interpolation",
    } for row in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--full-reference-curve", required=True)
    parser.add_argument("--dev-reference-curve", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.shards_root, "shard_*", "per_h5.csv")))
    if not paths:
        raise FileNotFoundError("No shard per_h5.csv files")
    rows = [row for path in paths for row in read_csv(path)]
    with open(args.full_manifest, encoding="utf-8") as handle:
        full_entries = [line.strip() for line in handle if line.strip()]
    expected = {(entry, step) for entry in full_entries for step in (4, 8)}
    actual = {(row["sample"], int(row["step"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(
            "Shard coverage mismatch: expected {}, got {}".format(
                len(expected), len(rows)))
    with open(args.dev_manifest, encoding="utf-8") as handle:
        dev_entries = {line.strip() for line in handle if line.strip()}
    for split, selected, reference in (
            ("full", rows, args.full_reference_curve),
            ("dev", [row for row in rows if row["sample"] in dev_entries],
             args.dev_reference_curve)):
        output = os.path.join(args.output_root, split)
        os.makedirs(output, exist_ok=True)
        write_csv(os.path.join(output, "per_h5.csv"), selected)
        models, summaries = [], []
        for step in (4, 8):
            step_models, summary = aggregate_endpoint(selected, step)
            models.extend(step_models)
            summaries.append(summary)
        write_csv(os.path.join(output, "per_model.csv"), models)
        write_csv(os.path.join(output, "endpoint_summary.csv"), summaries)
        comparisons = official_neighbors(reference)
        for summary in summaries:
            comparisons.extend(({
                "endpoint": "B1_step{}_Base".format(summary["step"]),
                "kind": "B1 fixed mapping",
                "physical_bpp": summary["mean_model_base_bpp"],
                "direct_yuv_psnr_611": summary[
                    "mean_model_base_yuv_psnr_611"],
                "note": "measured endpoint; no interpolation",
            }, {
                "endpoint": "B1_step{}_Full".format(summary["step"]),
                "kind": "B1 fixed mapping",
                "physical_bpp": summary["mean_model_full_bpp"],
                "direct_yuv_psnr_611": summary[
                    "mean_model_full_yuv_psnr_611"],
                "note": "exact R02 reconstruction; layered physical syntax",
            }))
        write_csv(os.path.join(output, "official_neighbor_comparison.csv"),
                  comparisons)


if __name__ == "__main__":
    main()
