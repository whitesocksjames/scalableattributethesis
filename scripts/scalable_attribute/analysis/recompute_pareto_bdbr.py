#!/usr/bin/env python3
"""Recompute formal static-RGB BD-BR on uniform Pareto RD frontiers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


METRICS = ("Y", "YUV611")
ENDPOINTS = ("Original Unicorn", "Ours Base", "Ours Full")
OURS_ENDPOINTS = ("Ours Base", "Ours Full")


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _pchip_derivatives(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """Return PCHIP derivatives using the Fritsch-Carlson/SciPy convention."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("PCHIP requires equal arrays with at least two points")
    h = [x[index + 1] - x[index] for index in range(len(x) - 1)]
    if any(step <= 0.0 for step in h):
        raise ValueError("PCHIP abscissae must be strictly increasing")
    slopes = [(y[index + 1] - y[index]) / h[index] for index in range(len(h))]
    if len(x) == 2:
        return [slopes[0], slopes[0]]

    derivatives = [0.0] * len(x)
    first = ((2.0 * h[0] + h[1]) * slopes[0] - h[0] * slopes[1]) / (h[0] + h[1])
    if _sign(first) != _sign(slopes[0]):
        first = 0.0
    elif _sign(slopes[0]) != _sign(slopes[1]) and abs(first) > 3.0 * abs(slopes[0]):
        first = 3.0 * slopes[0]
    derivatives[0] = first

    for index in range(1, len(x) - 1):
        left = slopes[index - 1]
        right = slopes[index]
        if left == 0.0 or right == 0.0 or _sign(left) != _sign(right):
            derivatives[index] = 0.0
            continue
        weight_left = 2.0 * h[index] + h[index - 1]
        weight_right = h[index] + 2.0 * h[index - 1]
        derivatives[index] = (weight_left + weight_right) / (
            weight_left / left + weight_right / right
        )

    last = (
        (2.0 * h[-1] + h[-2]) * slopes[-1] - h[-1] * slopes[-2]
    ) / (h[-1] + h[-2])
    if _sign(last) != _sign(slopes[-1]):
        last = 0.0
    elif _sign(slopes[-1]) != _sign(slopes[-2]) and abs(last) > 3.0 * abs(slopes[-1]):
        last = 3.0 * slopes[-1]
    derivatives[-1] = last
    return derivatives


def _pchip_integral(
    x: Sequence[float], y: Sequence[float], lower: float, upper: float
) -> float:
    if upper <= lower:
        raise ValueError("integration interval must be non-empty")
    derivatives = _pchip_derivatives(x, y)
    total = 0.0
    for index in range(len(x) - 1):
        left = max(lower, x[index])
        right = min(upper, x[index + 1])
        if right <= left:
            continue
        width = x[index + 1] - x[index]
        slope = (y[index + 1] - y[index]) / width
        c2 = (3.0 * slope - 2.0 * derivatives[index] - derivatives[index + 1]) / width
        c3 = (derivatives[index] + derivatives[index + 1] - 2.0 * slope) / (width * width)

        def primitive(offset: float) -> float:
            return (
                y[index] * offset
                + derivatives[index] * offset**2 / 2.0
                + c2 * offset**3 / 3.0
                + c3 * offset**4 / 4.0
            )

        total += primitive(right - x[index]) - primitive(left - x[index])
    return total


def pareto_frontier(points: Sequence[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    dominators: dict[str, list[dict]] = {}
    retained: list[dict] = []
    for point in points:
        dominating_points = []
        for other in points:
            if other is point:
                continue
            no_higher_rate = other["rate"] <= point["rate"]
            no_lower_quality = other["quality"] >= point["quality"]
            strict = other["rate"] < point["rate"] or other["quality"] > point["quality"]
            if no_higher_rate and no_lower_quality and strict:
                dominating_points.append(
                    {
                        "operating_point": other["operating_point"],
                        "physical_bpp": other["rate"],
                        "quality": other["quality"],
                    }
                )
        if dominating_points:
            dominators[point["operating_point"]] = sorted(
                dominating_points, key=lambda item: item["operating_point"]
            )
        else:
            retained.append(point)

    retained.sort(key=lambda item: (item["quality"], item["rate"], item["operating_point"]))
    if len(retained) < 2:
        raise ValueError("Pareto frontier has fewer than two points")
    for left, right in zip(retained, retained[1:]):
        if right["quality"] <= left["quality"] or right["rate"] <= left["rate"]:
            raise ValueError("Pareto frontier is not strictly increasing in quality and rate")
    return retained, dominators


def bd_br(reference: Sequence[dict], test: Sequence[dict]) -> tuple[float, float, float]:
    quality_min = max(reference[0]["quality"], test[0]["quality"])
    quality_max = min(reference[-1]["quality"], test[-1]["quality"])
    if quality_max <= quality_min:
        raise ValueError("no shared quality overlap")
    ref_quality = [point["quality"] for point in reference]
    ref_log_rate = [math.log(point["rate"]) for point in reference]
    test_quality = [point["quality"] for point in test]
    test_log_rate = [math.log(point["rate"]) for point in test]
    ref_area = _pchip_integral(ref_quality, ref_log_rate, quality_min, quality_max)
    test_area = _pchip_integral(test_quality, test_log_rate, quality_min, quality_max)
    mean_difference = (test_area - ref_area) / (quality_max - quality_min)
    return 100.0 * (math.exp(mean_difference) - 1.0), quality_min, quality_max


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.raw.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    with args.strict.open(newline="", encoding="utf-8") as handle:
        strict_rows = list(csv.DictReader(handle))
    if len(raw_rows) != 460:
        raise ValueError(f"expected 460 endpoint rows, found {len(raw_rows)}")

    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        for metric in METRICS:
            grouped[
                (row["dataset"], row["sample"], row["method"], row["endpoint"], metric)
            ].append(
                {
                    "operating_point": row["operating_point"],
                    "rate": float(row["physical_bpp"]),
                    "quality": float(row[metric]),
                }
            )

    frontiers: dict[tuple[str, str, str, str, str], list[dict]] = {}
    curve_audit = []
    point_audit = []
    for key in sorted(grouped):
        dataset, sample, method, endpoint, metric = key
        points = grouped[key]
        frontier, dominators = pareto_frontier(points)
        frontiers[key] = frontier
        removed = [point["operating_point"] for point in points if point["operating_point"] in dominators]
        curve_audit.append(
            {
                "dataset": dataset,
                "sample": sample,
                "method": method,
                "endpoint": endpoint,
                "metric": metric,
                "raw_point_count": len(points),
                "retained_pareto_point_count": len(frontier),
                "removed_operating_points": json.dumps(removed, separators=(",", ":")),
                "dominating_points": json.dumps(dominators, sort_keys=True, separators=(",", ":")),
                "reason": "pareto_dominated" if removed else "none",
            }
        )
        for point in sorted(points, key=lambda item: item["operating_point"]):
            point_audit.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "method": method,
                    "endpoint": endpoint,
                    "metric": metric,
                    "operating_point": point["operating_point"],
                    "physical_bpp": point["rate"],
                    "quality": point["quality"],
                    "pareto_retained": point["operating_point"] not in dominators,
                    "dominating_points": json.dumps(
                        dominators.get(point["operating_point"], []), separators=(",", ":")
                    ),
                }
            )

    strict_by_key = {
        (row["dataset"], row["sample"], row["ours_endpoint"], row["metric"]): row
        for row in strict_rows
    }
    sequence_rows = []
    comparison_rows = []
    samples = sorted({(row["dataset"], row["sample"]) for row in raw_rows})
    for dataset, sample in samples:
        for endpoint in OURS_ENDPOINTS:
            for metric in METRICS:
                key = (dataset, sample, endpoint, metric)
                strict = strict_by_key[key]
                try:
                    value, quality_min, quality_max = bd_br(
                        frontiers[
                            (dataset, sample, "Original_Unicorn", "Original Unicorn", metric)
                        ],
                        frontiers[(dataset, sample, "Ours", endpoint, metric)],
                    )
                    status = "AVAILABLE"
                    reason = ""
                except ValueError as error:
                    value = quality_min = quality_max = ""
                    status = "UNAVAILABLE"
                    reason = str(error)
                sequence_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "ours_endpoint": endpoint,
                        "metric": metric,
                        "status": status,
                        "BD_BR_percent": value,
                        "shared_quality_min": quality_min,
                        "shared_quality_max": quality_max,
                        "reason": reason,
                    }
                )
                strict_value = (
                    float(strict["BD_BR_percent"])
                    if strict["status"] == "AVAILABLE"
                    else ""
                )
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "ours_endpoint": endpoint,
                        "metric": metric,
                        "strict_status": strict["status"],
                        "strict_BD_BR_percent": strict_value,
                        "pareto_status": status,
                        "pareto_BD_BR_percent": value,
                        "pareto_minus_strict_percentage_points": (
                            value - strict_value
                            if status == "AVAILABLE" and strict_value != ""
                            else ""
                        ),
                        "newly_resolved": strict["status"] != "AVAILABLE" and status == "AVAILABLE",
                    }
                )

    dataset_rows = []
    for dataset in sorted({row["dataset"] for row in sequence_rows}):
        for endpoint in OURS_ENDPOINTS:
            for metric in METRICS:
                selected = [
                    row for row in sequence_rows
                    if row["dataset"] == dataset
                    and row["ours_endpoint"] == endpoint
                    and row["metric"] == metric
                    and row["status"] == "AVAILABLE"
                ]
                total = sum(
                    1 for sample_dataset, _sample in samples if sample_dataset == dataset
                )
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "ours_endpoint": endpoint,
                        "metric": metric,
                        "available_sequences": len(selected),
                        "total_sequences": total,
                        "equal_sequence_mean_BD_BR_percent": mean(
                            float(row["BD_BR_percent"]) for row in selected
                        ),
                        "status": "AVAILABLE" if len(selected) == total else "AVAILABLE_SUBSET",
                    }
                )

    output = args.output
    _write_csv(
        output / "tables" / "curve_frontier_audit.csv",
        list(curve_audit[0]),
        curve_audit,
    )
    _write_csv(
        output / "evidence" / "pareto_frontier_points.csv",
        list(point_audit[0]),
        point_audit,
    )
    _write_csv(
        output / "tables" / "bdbr_per_sequence.csv",
        list(sequence_rows[0]),
        sequence_rows,
    )
    _write_csv(
        output / "tables" / "bdbr_dataset_summary.csv",
        list(dataset_rows[0]),
        dataset_rows,
    )
    _write_csv(
        output / "tables" / "strict_vs_pareto.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )

    available = sum(row["status"] == "AVAILABLE" for row in sequence_rows)
    summary = {
        "protocol": "uniform_pareto_frontier_pchip_log_physical_bpp_shared_quality_no_extrapolation",
        "raw_endpoint_rows": len(raw_rows),
        "curve_count": len(curve_audit),
        "comparison_count": len(sequence_rows),
        "available_comparisons": available,
        "unavailable_comparisons": len(sequence_rows) - available,
        "curves_with_removals": sum(row["reason"] == "pareto_dominated" for row in curve_audit),
        "removed_curve_points": sum(
            row["raw_point_count"] - row["retained_pareto_point_count"] for row in curve_audit
        ),
        "newly_resolved_comparisons": sum(row["newly_resolved"] for row in comparison_rows),
        "newly_resolved_samples": sorted(
            {
                f'{row["dataset"]}/{row["sample"]}'
                for row in comparison_rows if row["newly_resolved"]
            }
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
