#!/usr/bin/env python3
"""Derive author-compatible Owlii vox10 frames from read-only vox11 PLYs."""

import argparse
import csv
import os

import numpy as np


FRAMES = (
    ("basketball_player", "basketball_player_vox11_00000200.ply", 796217),
    ("dancer", "dancer_vox11_00000001.ply", 702038),
    ("exercise", "exercise_vox11_00000001.ply", 645135),
    ("model", "model_vox11_00000001.ply", 657755),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def ply_header(path):
    header_lines = 0
    vertex_count = None
    properties = []
    in_vertex = False
    with open(path, "rb") as handle:
        for raw in handle:
            header_lines += 1
            line = raw.decode("ascii").strip()
            if line.startswith("element "):
                words = line.split()
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    vertex_count = int(words[2])
            elif in_vertex and line.startswith("property "):
                words = line.split()
                if len(words) == 3:
                    properties.append(words[2])
            if line == "end_header":
                break
    required = ("x", "y", "z", "red", "green", "blue")
    if vertex_count is None or any(name not in properties for name in required):
        raise ValueError("PLY lacks required vertex properties: " + path)
    return header_lines, vertex_count, [properties.index(name) for name in required]


def derive(path):
    skiprows, source_points, usecols = ply_header(path)
    values = np.loadtxt(
        path, dtype=np.float32, skiprows=skiprows, max_rows=source_points,
        usecols=usecols)
    coords = np.floor(values[:, :3] / 2.0).astype(np.int32)
    rgb = values[:, 3:6].astype(np.float64)
    unique, inverse, counts = np.unique(
        coords, axis=0, return_inverse=True, return_counts=True)
    averaged = np.empty((len(unique), 3), dtype=np.uint8)
    for channel in range(3):
        sums = np.bincount(inverse, weights=rgb[:, channel], minlength=len(unique))
        averaged[:, channel] = np.clip(
            np.rint(sums / counts), 0, 255).astype(np.uint8)
    return source_points, unique, averaged


def write_ply(path, coords, rgb):
    with open(path, "w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("element vertex {}\n".format(len(coords)))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\n")
        handle.write("property uchar blue\nend_header\n")
        np.savetxt(
            handle, np.column_stack((coords, rgb)),
            fmt="%d %d %d %d %d %d")


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.realpath(input_dir) == os.path.realpath(output_dir):
        raise ValueError("Output directory must differ from read-only source")
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for sequence, filename, target in FRAMES:
        output = os.path.join(output_dir, filename)
        if os.path.exists(output):
            raise FileExistsError("Refusing to overwrite " + output)
        source_points, coords, rgb = derive(os.path.join(input_dir, filename))
        derived_points = len(coords)
        exact = derived_points == target
        rows.append({
            "sequence": sequence,
            "source_points": source_points,
            "derived_points": derived_points,
            "author_target_points": target,
            "exact_match": exact,
            "mode": "floor(xyz/2)+duplicate_rgb_mean_round",
            "output": output,
        })
        if not exact:
            raise RuntimeError(sequence + " point-count fingerprint mismatch")
        write_ply(output, coords, rgb)
        print(sequence, source_points, "->", derived_points, flush=True)
    with open(os.path.join(output_dir, "owlii_preprocessing_fingerprint.csv"),
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
