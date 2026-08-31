#!/usr/bin/env python3
"""Summarize the four fixed 8i sequences for shortlisted Base checkpoints."""

import argparse
import csv
import json
import os


EXPECTED_SEQUENCES = ("longdress", "loot", "redandblack", "soldier")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", required=True)
    parser.add_argument(
        "--input", action="append", required=True,
        help="physical_rd.csv from one fixed 8i sequence")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = []
    for path in args.input:
        rows.extend(read_csv(path))
    sequences = {row["sequence"].lower() for row in rows}
    if sequences != set(EXPECTED_SEQUENCES):
        raise ValueError(
            "Expected fixed-4 sequences {}, got {}".format(
                EXPECTED_SEQUENCES, sorted(sequences)))
    candidates = []
    per_sequence = []
    for label in args.candidate:
        selected = [row for row in rows if row["endpoint"] == label]
        if len(selected) != 4:
            raise RuntimeError(
                "Expected four rows for {}, got {}".format(label, len(selected)))
        if any(int(float(row["num_base_residual_streams"])) != 4
               for row in selected):
            raise RuntimeError(label + " did not use exactly r1-r4")
        if any(int(float(row["num_native_r5_streams"])) != 0
               for row in selected):
            raise RuntimeError(label + " accessed native r5")
        if any(float(row["hard_max_abs_difference"]) != 0.0
               for row in selected):
            raise RuntimeError(label + " hard exactness failed")
        per_sequence.extend(selected)
        candidates.append({
            "point": args.point,
            "candidate": label,
            "num_sequences": 4,
            "mean_physical_bpp": sum(
                float(row["physical_bpp"]) for row in selected) / 4,
            "mean_y_psnr": sum(float(row["y_psnr"]) for row in selected) / 4,
            "mean_u_psnr": sum(float(row["u_psnr"]) for row in selected) / 4,
            "mean_v_psnr": sum(float(row["v_psnr"]) for row in selected) / 4,
            "mean_yuv_psnr_611": sum(
                float(row["yuv_psnr_611"]) for row in selected) / 4,
            "hard_exact": True,
        })
    candidates.sort(key=lambda row: -row["mean_yuv_psnr_611"])
    for rank, row in enumerate(candidates, start=1):
        row["external_quality_rank"] = rank

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = "BASE_{}_8I".format(args.point.upper())
    result_path = os.path.join(args.output_dir, prefix + "_RESULTS.csv")
    raw_path = os.path.join(args.output_dir, prefix + "_PER_SEQUENCE.csv")
    json_path = os.path.join(args.output_dir, prefix + "_RESULTS.json")
    existing = [path for path in (result_path, raw_path, json_path)
                if os.path.exists(path)]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    write_csv(result_path, candidates)
    write_csv(raw_path, per_sequence)
    with open(json_path, "x", encoding="utf-8") as handle:
        json.dump({
            "point": args.point,
            "role": "external generalization sanity/tie-break",
            "primary_selection_remains": "RWTT-28Lite",
            "results": candidates,
        }, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
