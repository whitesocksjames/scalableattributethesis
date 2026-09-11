#!/usr/bin/env python3
"""Render all final static-RGB raw RD panels from frozen evidence."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


DATASET_ORDER = ("8iVFB", "Owlii", "CTC")
METRICS = ("Y", "YUV611")
ENDPOINTS = ("Original Unicorn", "Ours Base", "Ours Full")
STYLE = {
    "Original Unicorn": ("#3568b8", "o"),
    "Ours Base": ("#e08214", "s"),
    "Ours Full": ("#c73535", "^"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "dataset", "sample", "endpoint", "operating_point",
        "physical_bpp", "Y", "YUV611",
    }
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError("raw RD table is missing columns: " + ", ".join(sorted(missing)))
    if len(rows) != 460:
        raise ValueError(f"expected 460 endpoint rows, found {len(rows)}")
    return rows


def validate(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = defaultdict(list)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset, sample, endpoint = row["dataset"], row["sample"], row["endpoint"]
        if dataset not in DATASET_ORDER or endpoint not in ENDPOINTS:
            raise ValueError(f"unexpected curve identity: {dataset}/{sample}/{endpoint}")
        if sample not in samples[dataset]:
            samples[dataset].append(sample)
        grouped[(dataset, sample, endpoint)].append(row)

    for dataset, expected in {"8iVFB": 4, "Owlii": 4, "CTC": 12}.items():
        if len(samples[dataset]) != expected:
            raise ValueError(f"{dataset}: expected {expected} samples, found {len(samples[dataset])}")
        for sample in samples[dataset]:
            for endpoint in ENDPOINTS:
                expected_points = 9 if endpoint == "Original Unicorn" else 7
                actual = len(grouped[(dataset, sample, endpoint)])
                if actual != expected_points:
                    raise ValueError(
                        f"{dataset}/{sample}/{endpoint}: expected "
                        f"{expected_points} points, found {actual}"
                    )
    return samples


def _marker(x: float, y: float, endpoint: str) -> str:
    color, shape = STYLE[endpoint]
    if shape == "o":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.3" fill="{color}"/>'
    if shape == "s":
        return (
            f'<rect x="{x - 3.2:.2f}" y="{y - 3.2:.2f}" width="6.4" '
            f'height="6.4" fill="{color}"/>'
        )
    points = f"{x:.2f},{y - 3.8:.2f} {x - 3.8:.2f},{y + 3.2:.2f} {x + 3.8:.2f},{y + 3.2:.2f}"
    return f'<polygon points="{points}" fill="{color}"/>'


def _panel(
    curves: dict[str, list[dict[str, str]]], metric: str, sample: str,
    x0: float, y0: float, width: float, height: float, show_legend: bool,
) -> list[str]:
    left, right, top, bottom = 60.0, 18.0, 32.0, 46.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_rows = [row for endpoint in ENDPOINTS for row in curves[endpoint]]
    rates = [float(row["physical_bpp"]) for row in all_rows]
    qualities = [float(row[metric]) for row in all_rows]
    x_min, x_max = min(rates), max(rates)
    y_min, y_max = min(qualities), max(qualities)
    x_pad = max((x_max - x_min) * 0.05, 1e-6)
    y_pad = max((y_max - y_min) * 0.08, 0.05)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(value: float) -> float:
        return x0 + left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return y0 + top + (y_max - value) / (y_max - y_min) * plot_height

    result = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" fill="white"/>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="12">{escape(sample)} — {metric}</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = x0 + left + fraction * plot_width
        gy = y0 + top + fraction * plot_height
        rate = x_min + fraction * (x_max - x_min)
        quality = y_max - fraction * (y_max - y_min)
        result.extend([
            f'<line x1="{gx:.2f}" y1="{y0 + top:.2f}" x2="{gx:.2f}" y2="{y0 + top + plot_height:.2f}" stroke="#dddddd"/>',
            f'<line x1="{x0 + left:.2f}" y1="{gy:.2f}" x2="{x0 + left + plot_width:.2f}" y2="{gy:.2f}" stroke="#dddddd"/>',
            f'<text x="{gx:.2f}" y="{y0 + top + plot_height + 17:.2f}" text-anchor="middle" font-size="9">{rate:.3f}</text>',
            f'<text x="{x0 + left - 7:.2f}" y="{gy + 3:.2f}" text-anchor="end" font-size="9">{quality:.2f}</text>',
        ])
    result.extend([
        f'<line x1="{x0 + left:.2f}" y1="{y0 + top + plot_height:.2f}" x2="{x0 + left + plot_width:.2f}" y2="{y0 + top + plot_height:.2f}" stroke="#222"/>',
        f'<line x1="{x0 + left:.2f}" y1="{y0 + top:.2f}" x2="{x0 + left:.2f}" y2="{y0 + top + plot_height:.2f}" stroke="#222"/>',
        f'<text x="{x0 + left + plot_width / 2:.2f}" y="{y0 + height - 7:.2f}" text-anchor="middle" font-size="10">Physical attribute rate (bpp)</text>',
        f'<text x="{x0 + 13:.2f}" y="{y0 + top + plot_height / 2:.2f}" text-anchor="middle" font-size="10" transform="rotate(-90 {x0 + 13:.2f} {y0 + top + plot_height / 2:.2f})">{metric} PSNR (dB)</text>',
    ])
    for endpoint in ENDPOINTS:
        curve = sorted(curves[endpoint], key=lambda item: float(item["physical_bpp"]))
        coordinates = [
            (sx(float(row["physical_bpp"])), sy(float(row[metric]))) for row in curve
        ]
        color, _ = STYLE[endpoint]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        result.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        result.extend(_marker(x, y, endpoint) for x, y in coordinates)
    if show_legend:
        legend_x, legend_y = x0 + left + 8, y0 + top + 10
        for index, endpoint in enumerate(ENDPOINTS):
            ly = legend_y + index * 16
            color, _ = STYLE[endpoint]
            result.extend([
                f'<line x1="{legend_x:.2f}" y1="{ly:.2f}" x2="{legend_x + 20:.2f}" y2="{ly:.2f}" stroke="{color}" stroke-width="1.5"/>',
                _marker(legend_x + 10, ly, endpoint),
                f'<text x="{legend_x + 27:.2f}" y="{ly + 3:.2f}" font-size="9">{endpoint}</text>',
            ])
    return result


def render(rows: list[dict[str, str]], samples: dict[str, list[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["sample"], row["endpoint"])].append(row)

    filenames = {
        "8iVFB": "8i_raw_rd_curves.svg",
        "Owlii": "owlii_raw_rd_curves.svg",
        "CTC": "ctc_raw_rd_curves.svg",
    }
    for dataset in DATASET_ORDER:
        dataset_samples = samples[dataset]
        panel_width, panel_height = 560.0, 250.0
        width, height = panel_width * 2, panel_height * len(dataset_samples) + 38
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width / 2:.1f}" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">{dataset}: measured raw RD points</text>',
            '<g font-family="sans-serif" fill="#222">',
        ]
        for row_index, sample in enumerate(dataset_samples):
            for column_index, metric in enumerate(METRICS):
                curves = {
                    endpoint: grouped[(dataset, sample, endpoint)] for endpoint in ENDPOINTS
                }
                parts.extend(_panel(
                    curves, metric, sample,
                    column_index * panel_width, 38 + row_index * panel_height,
                    panel_width, panel_height,
                    show_legend=(row_index == 0 and column_index == 0),
                ))
        parts.extend(["</g>", "</svg>"])
        (output_dir / filenames[dataset]).write_text(
            "\n".join(parts) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True,
                        help="Frozen final_raw_rd_points.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Destination; never defaults to the frozen package")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate the 460-row matrix without rendering")
    args = parser.parse_args()
    rows = read_rows(args.raw)
    samples = validate(rows)
    if args.check_only:
        print("PASS: 460 endpoints, 20 samples, 40 sample-metric panels")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is used")
    render(rows, samples, args.output_dir)
    print(f"Wrote three dataset figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
