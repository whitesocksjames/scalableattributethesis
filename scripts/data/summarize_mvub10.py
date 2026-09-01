#!/usr/bin/env python3
"""Validate and summarize a completed MVUB10_H5 dataset."""

import argparse
import csv
import json
import os

import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.dataset_root)
    subjects = sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)))
    if not subjects:
        raise RuntimeError("No subject directories under " + root)
    all_rows = []
    subject_rows = []
    seen_paths = set()
    for subject in subjects:
        directory = os.path.join(root, subject)
        with open(os.path.join(directory, "summary.json"),
                  encoding="utf-8") as handle:
            declared = json.load(handle)
        with open(os.path.join(directory, "manifest.csv"),
                  encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        frames = set()
        points = 0
        for row in rows:
            relative = row["relative_path"]
            if relative in seen_paths:
                raise RuntimeError("Duplicate H5 path: " + relative)
            seen_paths.add(relative)
            path = os.path.join(root, relative)
            with h5py.File(path, "r") as h5:
                coords = h5["coords"]
                feats = h5["feats"]
                if coords.shape != feats.shape or coords.ndim != 2 or coords.shape[1] != 3:
                    raise RuntimeError("Invalid coords/feats shape: " + path)
                if str(coords.dtype) != "int16" or str(feats.dtype) != "uint8":
                    raise RuntimeError("Invalid H5 dtype: " + path)
                count = len(coords)
            if count != int(row["points"]) or count > 100000:
                raise RuntimeError("Invalid point count: " + path)
            frames.add(row["frame"])
            points += count
            all_rows.append(row)
        if (len(frames) != int(declared["frames"]) or
                len(rows) != int(declared["h5_blocks"]) or
                points != int(declared["source_points"]) or
                points != int(declared["processed_points"])):
            raise RuntimeError("Subject summary mismatch: " + subject)
        subject_rows.append({
            "subject": subject,
            "frames": len(frames),
            "h5_blocks": len(rows),
            "total_points": points,
        })

    with open(os.path.join(root, "manifest.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(root, "all_h5.txt"), "w",
              encoding="utf-8") as handle:
        handle.writelines(row["relative_path"] + "\n" for row in all_rows)
    with open(os.path.join(root, "subject_summary.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=subject_rows[0].keys())
        writer.writeheader()
        writer.writerows(subject_rows)
    summary = {
        "status": "PASS",
        "subjects": len(subject_rows),
        "frames": sum(row["frames"] for row in subject_rows),
        "h5_blocks": len(all_rows),
        "total_points": sum(row["total_points"] for row in subject_rows),
        "max_points_per_block": 100000,
        "all_paths_unique": True,
        "all_subject_point_totals_conserved": True,
    }
    with open(os.path.join(root, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
