#!/usr/bin/env python3
"""Partition official MVUB depth-10 ASCII PLY frames into Unicorn HDF5."""

import argparse
import csv
import json
import os

import numpy as np

from data_utils.attribute.inout import write_h5
from data_utils.attribute.partition import kdtree_partition


REQUIRED_PROPERTIES = ("x", "y", "z", "red", "green", "blue")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-points", type=int, default=100000)
    return parser.parse_args()


def ply_schema(path):
    header_lines = 0
    vertex_count = None
    properties = []
    in_vertex = False
    format_name = None
    with open(path, "rb") as handle:
        for raw in handle:
            header_lines += 1
            line = raw.decode("ascii").strip()
            words = line.split()
            if words[:1] == ["format"]:
                format_name = words[1]
            elif words[:1] == ["element"]:
                in_vertex = len(words) == 3 and words[1] == "vertex"
                if in_vertex:
                    vertex_count = int(words[2])
            elif in_vertex and words[:1] == ["property"] and len(words) == 3:
                properties.append(words[2])
            if line == "end_header":
                break
    if format_name != "ascii":
        raise ValueError("MVUB adapter requires ASCII PLY: " + path)
    if vertex_count is None:
        raise ValueError("PLY vertex count missing: " + path)
    missing = [name for name in REQUIRED_PROPERTIES if name not in properties]
    if missing:
        raise ValueError("PLY properties missing {}: {}".format(missing, path))
    return header_lines, vertex_count, [properties.index(x) for x in REQUIRED_PROPERTIES]


def read_mvub_ascii(path):
    header_lines, vertex_count, columns = ply_schema(path)
    values = np.loadtxt(
        path, dtype=np.float64, skiprows=header_lines,
        max_rows=vertex_count, usecols=columns)
    if values.shape != (vertex_count, 6):
        raise ValueError("PLY vertex rows do not match header: " + path)
    xyz = values[:, :3]
    if not np.array_equal(xyz, np.rint(xyz)):
        raise ValueError("MVUB geometry is not integer lattice data: " + path)
    coords = np.rint(xyz).astype(np.int32)
    if coords.min() < 0 or coords.max() > 1023:
        raise ValueError("MVUB depth-10 coordinates outside [0,1023]: " + path)
    rgb = values[:, 3:6]
    if not np.array_equal(rgb, np.rint(rgb)) or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError("MVUB RGB is not uint8-compatible: " + path)
    return coords, np.rint(rgb).astype(np.uint8)


def main():
    args = parse_args()
    if args.max_points != 100000:
        raise ValueError("Canonical MVUB preprocessing requires max_points=100000")
    input_dir = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_root)
    subject_dir = os.path.join(output_root, args.subject)
    if os.path.exists(subject_dir) and os.listdir(subject_dir):
        raise FileExistsError("Refusing to overwrite non-empty " + subject_dir)
    os.makedirs(subject_dir, exist_ok=True)
    frames = sorted(
        name for name in os.listdir(input_dir) if name.lower().endswith(".ply"))
    if not frames:
        raise RuntimeError("No PLY frames in " + input_dir)

    rows = []
    total_source_points = 0
    for frame_index, name in enumerate(frames):
        frame = os.path.splitext(name)[0]
        coords, rgb = read_mvub_ascii(os.path.join(input_dir, name))
        total_source_points += len(coords)
        points = np.hstack((coords, rgb.astype(np.int32)))
        parts = kdtree_partition(points, max_num=args.max_points)
        for part_index, part in enumerate(parts):
            part = part.copy()
            part[:, :3] -= part[:, :3].min(axis=0)
            filename = "{}_{}_P{:03d}.h5".format(
                args.subject, frame, part_index)
            path = os.path.join(subject_dir, filename)
            write_h5(path, part[:, :3], part[:, 3:])
            rows.append({
                "subject": args.subject,
                "frame": frame,
                "part": part_index,
                "points": len(part),
                "relative_path": os.path.relpath(path, output_root),
            })
        print("{}/{} {} points={} blocks={}".format(
            frame_index + 1, len(frames), name, len(coords), len(parts)),
            flush=True)

    manifest = os.path.join(subject_dir, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "subject": args.subject,
        "frames": len(frames),
        "h5_blocks": len(rows),
        "source_points": total_source_points,
        "processed_points": sum(row["points"] for row in rows),
        "max_points": args.max_points,
        "coordinate_translation": "per-block minimum subtraction",
        "h5_schema": "coords:int16, feats:uint8",
    }
    with open(os.path.join(subject_dir, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
