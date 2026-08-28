#!/usr/bin/env python3
"""Derive Full and fixed Dev14 reference tables from Full per-H5 CSVs."""

import argparse
import csv
import json
import os

from scalable_attribute.evaluation import aggregate_models, average_models
from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--dev-manifest", required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_new_csv(path, rows):
    if os.path.exists(path):
        raise FileExistsError("Refusing to overwrite reference CSV: " + path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    full_root = os.path.join(args.study_root, "reference_full")
    dev_root = os.path.join(args.study_root, "reference_dev")
    with open(args.dev_manifest, encoding="utf-8") as handle:
        dev_samples = {line.strip() for line in handle if line.strip()}

    full_curve = []
    dev_curve = []
    selected_models = set()
    for rate_id, _, _ in OFFICIAL_RWTT_REFERENCE_POINTS:
        rows = read_csv(os.path.join(full_root, rate_id, "per_h5.csv"))
        full_models = aggregate_models(rows)
        full_curve.append(average_models(full_models))

        dev_rows = [row for row in rows if row["sample"] in dev_samples]
        if {row["sample"] for row in dev_rows} != dev_samples:
            raise RuntimeError("Full per-H5 CSV does not cover the fixed Dev manifest")
        dev_models = aggregate_models(dev_rows)
        selected_models.update(row["model_id"] for row in dev_models)
        write_new_csv(
            os.path.join(dev_root, rate_id, "per_model.csv"), dev_models)
        dev_curve.append(average_models(dev_models))

    write_new_csv(os.path.join(full_root, "reference_curve.csv"), full_curve)
    write_new_csv(os.path.join(dev_root, "reference_dev_curve.csv"), dev_curve)

    with open(os.path.join(full_root, "reference_info.json"), encoding="utf-8") as handle:
        full_info = json.load(handle)
    dev_info = dict(full_info)
    dev_info.update({
        "validation_manifest": os.path.abspath(args.dev_manifest),
        "dev_manifest": os.path.abspath(args.dev_manifest),
        "num_models": len(selected_models),
        "num_h5": len(dev_samples),
        "selected_models": sorted(
            selected_models, key=lambda value: int(value[3:])),
        "derived_from": os.path.abspath(full_root),
    })
    info_path = os.path.join(dev_root, "reference_info.json")
    if os.path.exists(info_path):
        raise FileExistsError("Refusing to overwrite reference metadata: " + info_path)
    with open(info_path, "x", encoding="utf-8") as handle:
        json.dump(dev_info, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
