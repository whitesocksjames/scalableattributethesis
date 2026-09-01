#!/usr/bin/env python3
"""Experiment-only native checkpoint-family overlap and truncation audit.

This deliberately accepts an explicit released checkpoint/profile/lambda tuple.
It does not add diagnostic overlap points to the canonical operating-point map.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_h5, read_ply_ascii, write_ply_ascii
from scalable_attribute.canonical.prefix import FrozenUnicornPrefix
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import (
    aggregate_models, average_models, sample_identity)
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sparse_max_difference)


ENDPOINTS = ("B_native", "Official_Full")


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file-list", help="RWTT manifest, used together with --data-root")
    source.add_argument("--input-ply", help="Single external PLY")
    parser.add_argument("--data-root")
    parser.add_argument("--sequence")
    parser.add_argument("--frame")
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--checkpoint-profile", required=True)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--diagnostic-id", default="DIAGNOSTIC_OVERLAP")
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(args):
    if args.file_list:
        if not args.data_root:
            raise ValueError("--data-root is required with --file-list")
        with open(args.file_list, encoding="utf-8") as handle:
            entries = [line.strip() for line in handle if line.strip()]
        files = h5_files(args.data_root, args.file_list)
        if len(entries) != len(files):
            raise RuntimeError("Manifest entry/file count mismatch")
        inputs = []
        for entry, path in zip(entries, files):
            model_id, partition_id = sample_identity(entry)
            inputs.append((entry, path, model_id, partition_id, "h5"))
    else:
        if args.data_root:
            raise ValueError("--data-root is only valid with --file-list")
        sequence = args.sequence or Path(args.input_ply).stem
        entry = "{}:{}".format(sequence, args.frame or "single")
        inputs = [(entry, args.input_ply, sequence, 0, "ply")]
    return inputs[:args.max_samples] if args.max_samples else inputs


def sparse_input(path, kind):
    coords, rgb = read_h5(path) if kind == "h5" else read_ply_ascii(path)
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    batch_coords, batch_feats = ME.utils.sparse_collate([coords], [yuv])
    attribute = ME.SparseTensor(
        features=batch_feats, coordinates=batch_coords,
        tensor_stride=1, device="cuda")
    return coords, rgb, attribute


def main():
    args = parse_args()
    for name in ("released_checkpoint", "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    for name in ("data_root", "file_list", "input_ply"):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(os.path.expandvars(value)))
    if args.conditioning_lambda <= 0:
        raise ValueError("--conditioning-lambda must be positive")
    if not os.path.isfile(args.released_checkpoint):
        raise FileNotFoundError(args.released_checkpoint)
    actual_profile = os.path.basename(os.path.dirname(args.released_checkpoint))
    if actual_profile != args.checkpoint_profile:
        raise ValueError(
            "--checkpoint-profile {} does not match checkpoint directory {}"
            .format(args.checkpoint_profile, actual_profile))
    if not os.path.isfile(args.gpcc_binary):
        raise FileNotFoundError(args.gpcc_binary)

    protected = ("per_sample.csv", "per_model.csv", "endpoint_summary.csv")
    existing = [name for name in protected
                if os.path.exists(os.path.join(args.output_dir, name))]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w",
              encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.exists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    inputs = load_inputs(args)
    if not inputs:
        raise ValueError("No input samples")

    prefix = FrozenUnicornPrefix(args.released_checkpoint).cuda().eval()
    model = prefix.model
    rows = []
    metric_dir = os.path.join(args.output_dir, "metric_tmp")
    os.makedirs(metric_dir, exist_ok=True)
    os.chdir(args.output_dir)

    with torch.no_grad():
        for index, (entry, path, model_id, partition_id, kind) in enumerate(inputs):
            started = time.perf_counter()
            coords, rgb, attribute = sparse_input(path, kind)
            encoded, x_low, x_low_bits = model(
                attribute, training=False, lmb=args.conditioning_lambda,
                encode=True)
            if len(encoded) != 5:
                raise RuntimeError("Expected exactly five native residual streams")
            stream_bits = [int(len(item["strings"]) * 8) for item in encoded]
            prefix_bits = int(x_low_bits + sum(stream_bits[:4]))
            full_bits = int(prefix_bits + stream_bits[4])

            x0 = ME.SparseTensor(
                features=torch.zeros_like(attribute.F),
                coordinate_map_key=attribute.coordinate_map_key,
                coordinate_manager=attribute.coordinate_manager,
                device=attribute.device)
            x4, f4, d4 = model.decode(
                x0=x0, x_low=x_low, enc_set_list=encoded,
                lmb=args.conditioning_lambda, max_residual_stages=4,
                return_state=True)
            state = prefix._complete_state(x4, f4, d4)
            zero = type(state.f5p)(
                features=torch.zeros_like(state.f5p.F),
                coordinate_map_key=state.f5p.coordinate_map_key,
                coordinate_manager=state.f5p.coordinate_manager)
            _, native_delta = prefix.synthesize(state.f5p, zero)
            native = state.x5p + native_delta
            full = model.decode(
                x0=x0, x_low=x_low, enc_set_list=encoded,
                lmb=args.conditioning_lambda)
            hard_roundtrip = sparse_max_difference(
                encoded[-1]["x_out"], full, "Official Full hard round-trip")

            gt_path = os.path.join(metric_dir, "gt.ply")
            rec_path = os.path.join(metric_dir, "rec.ply")
            write_ply_ascii(gt_path, coords, rgb)
            common = {
                "diagnostic_id": args.diagnostic_id,
                "checkpoint_profile": args.checkpoint_profile,
                "base_lambda": args.conditioning_lambda,
                "model_id": model_id, "partition_id": partition_id,
                "sample": entry, "points": len(attribute),
                "x_low_bits": int(x_low_bits),
                "r1_bits": stream_bits[0], "r2_bits": stream_bits[1],
                "r3_bits": stream_bits[2], "r4_bits": stream_bits[3],
                "r5_bits": stream_bits[4], "prefix_bits": prefix_bits,
                "full_bits": full_bits,
                "num_native_residual_streams": 5,
            }
            for endpoint, reconstruction, bits in (
                    ("B_native", native, prefix_bits),
                    ("Official_Full", full, full_bits)):
                write_ply_ascii(
                    rec_path, reconstruction.C[:, 1:].cpu().numpy(),
                    reconstruction_rgb(reconstruction))
                quality = metric(gt_path, rec_path)
                rows.append({
                    "endpoint": endpoint, **common,
                    "physical_bits": bits,
                    "physical_bpp": bits / len(attribute),
                    **quality,
                    "hard_roundtrip_max_abs_difference": (
                        hard_roundtrip if endpoint == "Official_Full" else ""),
                    "seconds": time.perf_counter() - started,
                })
            write_csv(os.path.join(args.output_dir, "per_sample.csv"), rows)
            print("[{}/{}] {} prefix/full={:.6f}/{:.6f} bpp".format(
                index + 1, len(inputs), entry,
                prefix_bits / len(attribute), full_bits / len(attribute)),
                flush=True)

    per_model = []
    summaries = []
    for endpoint in ENDPOINTS:
        endpoint_rows = [row for row in rows if row["endpoint"] == endpoint]
        prepared = [{**row, "rate_id": args.diagnostic_id}
                    for row in endpoint_rows]
        aggregated = aggregate_models(
            prepared, bits_field="physical_bits", metric_prefix="")
        per_model.extend({"endpoint": endpoint, **row} for row in aggregated)
        summary = average_models(aggregated)
        stream_totals = {
            "r{}_bits".format(i): sum(
                int(row["r{}_bits".format(i)]) for row in endpoint_rows)
            for i in range(1, 6)
        }
        residual_total = sum(stream_totals.values())
        summaries.append({
            "endpoint": endpoint, "diagnostic_id": args.diagnostic_id,
            **summary,
            "x_low_bits": sum(int(row["x_low_bits"]) for row in endpoint_rows),
            **stream_totals,
            "residual_bits_r1_to_r5": residual_total,
            **{"r{}_residual_share".format(i): (
                stream_totals["r{}_bits".format(i)] / residual_total)
               for i in range(1, 6)},
            "prefix_bits": sum(int(row["prefix_bits"])
                               for row in endpoint_rows),
            "full_bits": sum(int(row["full_bits"])
                             for row in endpoint_rows),
            "hard_roundtrip_max_abs_difference": max(
                float(row["hard_roundtrip_max_abs_difference"] or 0.0)
                for row in endpoint_rows),
        })
    write_csv(os.path.join(args.output_dir, "per_model.csv"), per_model)
    write_csv(os.path.join(args.output_dir, "endpoint_summary.csv"), summaries)
    with open(os.path.join(args.output_dir, "endpoint_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "status": "PASS",
            "contract": "explicit diagnostic checkpoint profile + lambda",
            "official_mapping_modified": False,
            "bit_accounting": {
                "B_native": "x_low+r1+r2+r3+r4",
                "Official_Full": "x_low+r1+r2+r3+r4+r5",
            },
            "endpoints": summaries,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
