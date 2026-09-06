#!/usr/bin/env python3
"""Block-level content statistics for the 4K/2K distribution diagnosis.

This is deliberately independent of training and codec code.  Its hierarchy is
the NumPy equivalent of the released Attribute model's five occupied-voxel
MinkowskiAvgPooling(stride=2) operations.  At each level, the parent average is
broadcast back only to the occupied child support when measuring GT residual
energy.  Native residual names are reversed: the finest residual is r5.
"""

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np


def rgb2yuv(rgb):
    rgb = rgb.astype(np.float64, copy=False)
    out = np.empty_like(rgb, dtype=np.float64)
    out[:, 0] = 0.257 * rgb[:, 0] + 0.504 * rgb[:, 1] + 0.098 * rgb[:, 2] + 16
    out[:, 1] = -0.148 * rgb[:, 0] - 0.291 * rgb[:, 1] + 0.439 * rgb[:, 2] + 128
    out[:, 2] = 0.439 * rgb[:, 0] - 0.368 * rgb[:, 1] - 0.071 * rgb[:, 2] + 128
    return out / 255.0


def read_h5(path):
    with h5py.File(path, "r") as handle:
        return handle["coords"][:].astype(np.int64), handle["feats"][:]


def ply_layout(path):
    lines = 0
    count = None
    properties = []
    in_vertex = False
    with open(path, "rb") as handle:
        for raw in handle:
            lines += 1
            line = raw.decode("ascii").strip()
            words = line.split()
            if words[:1] == ["format"] and words[1] != "ascii":
                raise ValueError("Only ASCII PLY is supported: " + path)
            if words[:1] == ["element"]:
                in_vertex = len(words) == 3 and words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
            elif in_vertex and words[:1] == ["property"] and len(words) == 3:
                properties.append(words[2])
            if line == "end_header":
                break
    required = ("x", "y", "z", "red", "green", "blue")
    if count is None or any(name not in properties for name in required):
        raise ValueError("Missing PLY vertex fields: " + path)
    return lines, count, [properties.index(name) for name in required]


def read_ply(path):
    skip, count, cols = ply_layout(path)
    values = np.loadtxt(path, dtype=np.float64, skiprows=skip,
                        max_rows=count, usecols=cols)
    return values[:, :3].astype(np.int64), values[:, 3:6]


def kdtree_partition(points, max_num=100000):
    """Exact iterative equivalent of data_utils.attribute.partition."""
    parts, pending = [], [points]
    while pending:
        data = pending.pop()
        if len(data) <= max_num:
            parts.append(data)
            continue
        axis = int(np.argmax(np.var(data[:, :3], axis=0)))
        order = np.argsort(data[:, axis], kind="stable")
        middle = len(data) // 2
        pending.append(data[order[middle:]])
        pending.append(data[order[:middle]])
    return parts


def pool_average(coords, feats):
    parent_coords = np.floor_divide(coords, 2)
    unique, inverse, counts = np.unique(
        parent_coords, axis=0, return_inverse=True, return_counts=True)
    sums = np.column_stack([
        np.bincount(inverse, weights=feats[:, c], minlength=len(unique))
        for c in range(3)
    ])
    return unique, sums / counts[:, None], inverse


def tv6(coords, feats):
    origin = coords.min(axis=0)
    shifted = coords - origin
    shape = shifted.max(axis=0) + 1
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    codes = shifted @ strides
    order = np.argsort(codes)
    sorted_codes = codes[order]
    total = np.zeros(3, dtype=np.float64)
    edges = 0
    for axis, stride in enumerate(strides):
        valid = shifted[:, axis] + 1 < shape[axis]
        source = np.nonzero(valid)[0]
        targets = codes[source] + stride
        pos = np.searchsorted(sorted_codes, targets)
        hit = pos < len(sorted_codes)
        hit[hit] &= sorted_codes[pos[hit]] == targets[hit]
        source = source[hit]
        target = order[pos[hit]]
        diff = feats[source] - feats[target]
        total += np.square(diff).sum(axis=0)
        edges += len(source)
    return total, edges


def infer_model(path):
    p = Path(path)
    for part in reversed(p.parts[:-1]):
        if part.startswith("RWT") or part.lower().startswith("model"):
            return part
    return p.parent.name


def compute_one(item):
    dataset, sequence, block_id, source, coords, rgb = item
    if coords is None:
        coords, rgb = read_h5(source)
    coords = coords - coords.min(axis=0)
    yuv = rgb2yuv(rgb)
    n = len(coords)
    channel_sum = yuv.sum(axis=0)
    channel_sumsq = np.square(yuv).sum(axis=0)
    row = {
        "dataset": dataset, "sequence": sequence, "block_id": block_id,
        "source": source, "original_model": infer_model(source),
        "point_count": n,
    }
    labels = "YUV"
    for i, label in enumerate(labels):
        row["sum_" + label] = channel_sum[i]
        row["sumsq_" + label] = channel_sumsq[i]
        row["mean_" + label] = channel_sum[i] / n
        row["var_" + label] = channel_sumsq[i] / n - (channel_sum[i] / n) ** 2
        row["min_" + label] = yuv[:, i].min()
        row["max_" + label] = yuv[:, i].max()
    var_total = sum(row["var_" + c] for c in labels)
    row["Y_variance_share"] = row["var_Y"] / var_total if var_total else 0.0
    row["UV_variance_share"] = ((row["var_U"] + row["var_V"]) / var_total
                                if var_total else 0.0)

    tv_sum, edges = tv6(coords, yuv)
    row["tv6_edge_count"] = edges
    row["tv6_mean_degree"] = 2.0 * edges / n
    for i, label in enumerate(labels):
        row["tv6_sse_" + label] = tv_sum[i]
        row["tv6_" + label] = tv_sum[i] / edges if edges else 0.0
    row["tv6_D111"] = tv_sum.sum() / (3 * edges) if edges else 0.0

    current_coords, current_feats = coords, yuv
    energies = {}
    for down_level in range(1, 6):
        parent_coords, parent_feats, inverse = pool_average(
            current_coords, current_feats)
        residual = current_feats - parent_feats[inverse]
        native_stage = 6 - down_level
        row["N_level{}".format(down_level)] = len(parent_coords)
        row["retention_level{}".format(down_level)] = (
            len(parent_coords) / len(current_coords))
        row["collision_level{}".format(down_level)] = (
            1.0 - len(parent_coords) / len(current_coords))
        for i, label in enumerate(labels):
            sse = np.square(residual[:, i]).sum()
            row["r{}_sse_{}".format(native_stage, label)] = sse
            row["r{}_E_{}".format(native_stage, label)] = sse / len(residual)
        d111 = np.square(residual).sum() / (3 * len(residual))
        row["r{}_E_D111".format(native_stage)] = d111
        energies[native_stage] = d111
        current_coords, current_feats = parent_coords, parent_feats
    energy_sum = sum(energies.values())
    for stage, value in energies.items():
        row["r{}_energy_share".format(stage)] = value / energy_sum if energy_sum else 0.0
    row["r45_energy_share"] = ((energies[4] + energies[5]) / energy_sum
                               if energy_sum else 0.0)
    row["r5_over_r4"] = energies[5] / energies[4] if energies[4] else float("nan")
    return row


def external_items(manifest, whole_frame=False):
    for line in Path(manifest).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        dataset, sequence, path = line.split("\t")
        coords, rgb = read_ply(path)
        joined = np.column_stack((coords, rgb))
        if whole_frame:
            yield (dataset, sequence, sequence + ":whole", path, coords, rgb)
            continue
        for index, part in enumerate(kdtree_partition(joined, 100000)):
            yield (dataset, sequence, "{}:P{:03d}".format(sequence, index), path,
                   part[:, :3].astype(np.int64), part[:, 3:6])


def h5_items(manifest, dataset):
    for line in Path(manifest).read_text().splitlines():
        path = line.strip()
        if path:
            yield (dataset, infer_model(path), Path(path).stem, path, None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("h5", "external", "external_whole"),
                        required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    items = (h5_items(args.manifest, args.dataset) if args.mode == "h5"
             else external_items(args.manifest,
                                 whole_frame=args.mode == "external_whole"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(compute_one, items, chunksize=1), 1):
            rows.append(row)
            if index % 100 == 0:
                print("processed", index, flush=True)
    rows.sort(key=lambda x: (x["dataset"], x["sequence"], x["block_id"]))
    if not rows:
        raise RuntimeError("No input rows")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", len(rows), output, flush=True)


if __name__ == "__main__":
    main()
