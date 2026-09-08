#!/usr/bin/env python3
"""Summarize 8i/Owlii Base+Full trajectories against author RD curves."""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", action="append", required=True,
        metavar="DATASET:SEQUENCE=CSV")
    parser.add_argument("--official-8i", required=True)
    parser.add_argument("--official-owlii", required=True)
    parser.add_argument("--neighbor-reference-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty summary")
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_evaluation(value):
    label, path = value.split("=", 1)
    dataset, sequence = label.split(":", 1)
    if dataset not in ("8ivfb", "owlii"):
        raise ValueError("dataset must be 8ivfb or owlii")
    return dataset, sequence.lower(), Path(path).expanduser().resolve()


def official_point(row, dataset):
    if dataset == "8ivfb":
        return float(row["bpp"]), float(row["YUV611"])
    return float(row["physical_bpp"]), float(row["yuv611_db"])


def interpolate(points, rate):
    points = sorted(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= rate <= x1:
            if x0 == x1:
                return y0, "equal-rate"
            value = y0 + (rate - x0) * (y1 - y0) / (x1 - x0)
            return value, "local-linear"
    return None, "out-of-range"


def step_from_source(source):
    # Evaluator labels are required to be stepNNN_Base / stepNNN_Full.
    label = source.rsplit("_", 1)[0]
    if not label.startswith("step"):
        raise ValueError("Unexpected scalable checkpoint label: " + source)
    return int(label[4:])


def endpoint_rows(rows):
    selected = [row for row in rows
                if row["endpoint"].endswith(("_Base", "_Full"))]
    result = []
    for row in selected:
        endpoint = "Base" if row["endpoint"].endswith("_Base") else "Full"
        result.append((step_from_source(row["endpoint"]), endpoint, row))
    return result


def dominates(left, right):
    return (left[0] <= right[0] and left[1] >= right[1] and
            (left[0] < right[0] or left[1] > right[1]))


args = parse_args()
output = Path(args.output_dir).expanduser().resolve()
output.mkdir(parents=True, exist_ok=True)
official = {
    "8ivfb": read_csv(args.official_8i),
    "owlii": read_csv(args.official_owlii),
}
neighbor_rows = read_csv(args.neighbor_reference_csv)
neighbor = {}
for row in neighbor_rows:
    key = (row["point"].upper(), row["dataset"], row["sequence"].lower(),
           row["endpoint"])
    if key in neighbor:
        raise ValueError("Duplicate neighbor reference: " + repr(key))
    neighbor[key] = (float(row["physical_bpp"]), float(row["YUV611"]))
rows = []
for dataset, sequence, path in map(parse_evaluation, args.evaluation):
    evaluation = read_csv(path)
    curves = [official_point(row, dataset) for row in official[dataset]
              if row["sequence"].lower() == sequence]
    if len(curves) != 9:
        raise ValueError("Official curve must have nine points: " + sequence)
    for step, endpoint, row in endpoint_rows(evaluation):
        neighbor_key_2k = ("2K", dataset, sequence, endpoint)
        neighbor_key_8k = ("8K", dataset, sequence, endpoint)
        if neighbor_key_2k not in neighbor or neighbor_key_8k not in neighbor:
            raise ValueError("Missing frozen 2K/8K neighbor reference for " +
                             repr((dataset, sequence, endpoint)))
        if (int(row["num_base_residual_streams"]) != 4 or
                int(row["num_native_r5_streams"]) != 0):
            raise ValueError("Hard Base syntax contract failed")
        if endpoint == "Full" and float(
                row["hard_roundtrip_max_abs_difference"]) != 0.0:
            raise ValueError("Hard Full round-trip failed")
        if endpoint == "Full" and int(row["physical_bits"]) != (
                int(row["base_bits"]) + int(row["enhancement_bits"])):
            raise ValueError("Full physical bit identity failed")
        if endpoint == "Base" and (int(row["enhancement_bits"]) != 0 or
                                   int(row["physical_bits"]) !=
                                   int(row["base_bits"])):
            raise ValueError("Base physical bit identity failed")
        rate = float(row["physical_bpp"])
        quality = float(row["yuv_psnr_611"])
        target, bracket = interpolate(curves, rate)
        rate_2k, quality_2k = neighbor[neighbor_key_2k]
        rate_8k, quality_8k = neighbor[neighbor_key_8k]
        dominated_2k = dominates((rate_2k, quality_2k), (rate, quality))
        dominated_8k = dominates((rate_8k, quality_8k), (rate, quality))
        rows.append({
            "dataset": dataset, "sequence": sequence, "step": step,
            "endpoint": endpoint, "physical_bpp": rate,
            "Y": float(row["y_psnr"]), "U": float(row["u_psnr"]),
            "V": float(row["v_psnr"]), "YUV611": quality,
            "base_bpp": int(row["base_bits"]) / int(row["points"]),
            "enhancement_bpp": (
                int(row["enhancement_bits"]) / int(row["points"])),
            "official_interp_YUV611": target if target is not None else "N/A",
            "delta_to_official_curve": (
                quality - target if target is not None else "N/A"),
            "interpolation": bracket,
            "neighbor_2k_bpp": rate_2k,
            "neighbor_2k_YUV611": quality_2k,
            "neighbor_8k_bpp": rate_8k,
            "neighbor_8k_YUV611": quality_8k,
            "ladder_rate_region_pass": rate_2k <= rate <= rate_8k,
            "dominated_by_2k_neighbor": dominated_2k,
            "dominated_by_8k_neighbor": dominated_8k,
            "neighbor_pareto_dominated": dominated_2k or dominated_8k,
            "hard_status": "PASS",
        })

expected_steps = sorted({row["step"] for row in rows})
if len(expected_steps) != 7:
    raise ValueError("Expected seven saved checkpoints")
expected = len(expected_steps) * 8 * 2
if len(rows) != expected:
    raise ValueError("Expected {} endpoint rows, found {}".format(
        expected, len(rows)))

means = []
for dataset in ("8ivfb", "owlii"):
    for step in expected_steps:
        for endpoint in ("Base", "Full"):
            group = [row for row in rows if row["dataset"] == dataset
                     and row["step"] == step and row["endpoint"] == endpoint]
            if len(group) != 4:
                raise ValueError("Each dataset mean requires four sequences")
            numeric_delta = [row["delta_to_official_curve"] for row in group
                             if row["delta_to_official_curve"] != "N/A"]
            means.append({
                "dataset": dataset, "step": step, "endpoint": endpoint,
                "physical_bpp_mean": sum(r["physical_bpp"] for r in group) / 4,
                "Y_mean": sum(r["Y"] for r in group) / 4,
                "YUV611_mean": sum(r["YUV611"] for r in group) / 4,
                "delta_to_official_curve_mean": (
                    sum(numeric_delta) / len(numeric_delta)
                    if numeric_delta else "N/A"),
                "interpolated_sequences": len(numeric_delta),
            })

violations = []
for dataset in ("8ivfb", "owlii"):
    for sequence in sorted({r["sequence"] for r in rows
                            if r["dataset"] == dataset}):
        for endpoint in ("Base", "Full"):
            points = [r for r in rows if r["dataset"] == dataset
                      and r["sequence"] == sequence and r["endpoint"] == endpoint]
            for candidate in points:
                dominators = [other["step"] for other in points
                              if other is not candidate and dominates(
                                  (other["physical_bpp"], other["YUV611"]),
                                  (candidate["physical_bpp"], candidate["YUV611"]))]
                if dominators:
                    violations.append({"dataset": dataset, "sequence": sequence,
                        "endpoint": endpoint, "dominated_step": candidate["step"],
                        "dominating_steps": ",".join(map(str, sorted(dominators)))})

write_csv(output / "joint_external_per_sequence.csv", rows)
write_csv(output / "joint_external_group_means.csv", means)
if violations:
    write_csv(output / "joint_external_pareto_violations.csv", violations)
with open(output / "joint_external_summary.json", "x", encoding="utf-8") as handle:
    json.dump({"status": "PASS", "steps": expected_steps,
               "num_endpoint_rows": len(rows), "group_means": means,
               "pareto_violations": violations,
               "neighbor_reference_csv": str(
                   Path(args.neighbor_reference_csv).expanduser().resolve()),
               "neighbor_pareto_dominated_rows": sum(
                   bool(row["neighbor_pareto_dominated"]) for row in rows),
               "ladder_rate_region_failure_rows": sum(
                   not bool(row["ladder_rate_region_pass"]) for row in rows),
               "selection_performed": False}, handle, indent=2)
