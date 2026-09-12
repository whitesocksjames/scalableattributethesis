#!/usr/bin/env python3
"""Build presentation-only per-sequence BD-BR rows from frozen CSV evidence."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "results/comparisons/formal_static_rgb_pareto_bdbr_20260911/tables"
)
OUTPUT = (
    ROOT
    / "results/comparisons/formal_static_rgb_thesis_results_20260911"
    / "bdbr_per_sequence_compact.csv"
)

SAMPLE_ORDER = {
    "8iVFB": ["longdress", "loot", "redandblack", "soldier"],
    "Owlii": ["basketball_player", "dancer", "exercise", "model"],
    "CTC": [
        "basketball_player_vox11_00000200",
        "dancer_vox11_00000001",
        "Thaidancer_viewdep_vox12",
        "longdress_viewdep_vox12",
        "loot_viewdep_vox12",
        "redandblack_viewdep_vox12",
        "soldier_viewdep_vox12",
        "boxer_viewdep_vox12",
        "Facade_00009_vox12",
        "House_without_roof_00057_vox12",
        "Shiva_00035_vox12",
        "Staue_Klimt_vox12",
    ],
}

COLUMNS = {
    ("Ours Base", "Y"): "base_y_bdbr_percent",
    ("Ours Base", "YUV611"): "base_yuv611_bdbr_percent",
    ("Ours Full", "Y"): "full_y_bdbr_percent",
    ("Ours Full", "YUV611"): "full_yuv611_bdbr_percent",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


per_sequence = read_csv(SOURCE / "bdbr_per_sequence.csv")
summary = read_csv(SOURCE / "bdbr_dataset_summary.csv")
if len(per_sequence) != 80 or any(row["status"] != "AVAILABLE" for row in per_sequence):
    raise RuntimeError("frozen per-sequence source must contain 80 AVAILABLE rows")

values: dict[tuple[str, str], dict[str, str]] = {}
for row in per_sequence:
    column = COLUMNS[(row["ours_endpoint"], row["metric"])]
    values.setdefault((row["dataset"], row["sample"]), {})[column] = row["BD_BR_percent"]

averages = {
    (row["dataset"], COLUMNS[(row["ours_endpoint"], row["metric"])]):
        row["equal_sequence_mean_BD_BR_percent"]
    for row in summary
}
if len(averages) != 12 or any(row["status"] != "AVAILABLE" for row in summary):
    raise RuntimeError("frozen dataset summary must contain 12 AVAILABLE rows")

fields = ["dataset", "sample", "row_type", *COLUMNS.values()]
rows = []
for dataset, samples in SAMPLE_ORDER.items():
    for sample in samples:
        measurements = values.get((dataset, sample))
        if measurements is None or set(measurements) != set(COLUMNS.values()):
            raise RuntimeError(f"missing frozen BD-BR values for {dataset}/{sample}")
        rows.append({"dataset": dataset, "sample": sample,
                     "row_type": "sequence", **measurements})
    rows.append({
        "dataset": dataset,
        "sample": "Average",
        "row_type": "dataset_average",
        **{column: averages[(dataset, column)] for column in COLUMNS.values()},
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"PASS: wrote {len(rows)} rows from frozen per-sequence and summary CSVs")
