#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv, yuv2rgb
from data_utils.attribute.inout import read_h5, write_ply_ascii
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import (
    aggregate_models, average_models, sample_identity)
from scalable_attribute.reference_points import reference_point
from scalable_attribute.unicorn_reference import ReleasedUnicornAttribute
from third_party.pc_error_attr import pc_error


H5_FIELDS = (
    "rate_id", "checkpoint_profile", "base_lambda", "model_id",
    "partition_id", "sample", "points", "base_bits", "bpp",
    "y_mse", "u_mse", "v_mse", "y_psnr", "u_psnr", "v_psnr",
    "yuv_psnr_611", "seconds",
)
MODEL_FIELDS = (
    "rate_id", "checkpoint_profile", "base_lambda", "model_id", "num_h5",
    "total_points", "total_bits", "bpp", "y_mse", "u_mse", "v_mse",
    "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_611",
)
CURVE_FIELDS = (
    "rate_id", "checkpoint_profile", "base_lambda", "num_models", "num_h5",
    "total_points", "mean_model_bpp", "mean_model_y_psnr",
    "mean_model_yuv_psnr_611",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rate-id", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    return parser.parse_args()


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric(gt_path, rec_path):
    values = pc_error(gt_path, rec_path, res=1, show=False)
    required = (
        "  c[0],    F", "  c[1],    F", "  c[2],    F",
        "  c[0],PSNRF", "  c[1],PSNRF", "  c[2],PSNRF",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise RuntimeError("pc_error missing metrics: " + ", ".join(missing))
    psnrs = [float(values[key]) for key in required[3:]]
    return {
        "y_mse": float(values[required[0]]),
        "u_mse": float(values[required[1]]),
        "v_mse": float(values[required[2]]),
        "y_psnr": psnrs[0], "u_psnr": psnrs[1], "v_psnr": psnrs[2],
        "yuv_psnr_611": (6.0 * psnrs[0] + psnrs[1] + psnrs[2]) / 8.0,
    }


def main():
    args = parse_args()
    args.data_root = os.path.abspath(os.path.expandvars(args.data_root))
    args.file_list = os.path.abspath(os.path.expandvars(args.file_list))
    args.base_checkpoint = os.path.abspath(os.path.expandvars(args.base_checkpoint))
    args.gpcc_binary = os.path.abspath(os.path.expandvars(args.gpcc_binary))
    args.output_dir = os.path.abspath(os.path.expandvars(args.output_dir))
    rate_id, profile, base_lambda = reference_point(args.rate_id)
    expected = "/{}/epoch_last.pth".format(profile)
    if not args.base_checkpoint.replace("\\", "/").endswith(expected):
        raise ValueError("{} requires checkpoint profile {}".format(rate_id, profile))

    protected = ("per_h5.csv", "per_model.csv", "reference_curve.csv")
    existing = [name for name in protected
                if os.path.exists(os.path.join(args.output_dir, name))]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite completed reference artifacts: "
            + ", ".join(existing))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.exists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    os.chdir(args.output_dir)

    model = ReleasedUnicornAttribute(
        args.base_checkpoint, args.base_scale, args.base_stage,
        args.base_vmode).cuda().eval()
    with open(args.file_list, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    files = h5_files(args.data_root, args.file_list)
    if args.max_samples:
        entries, files = entries[:args.max_samples], files[:args.max_samples]
    if len(entries) != len(files):
        raise RuntimeError("Manifest entry/file count mismatch")

    metric_dir = os.path.join(args.output_dir, "metric_tmp")
    os.makedirs(metric_dir, exist_ok=True)
    rows = []
    for index, (entry, path) in enumerate(zip(entries, files)):
        started = time.perf_counter()
        model_id, partition_id = sample_identity(entry)
        coords, rgb = read_h5(path)
        yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
        batched_coords, batched_feats = ME.utils.sparse_collate([coords], [yuv])
        A = ME.SparseTensor(
            features=batched_feats, coordinates=batched_coords,
            tensor_stride=1, device="cuda")
        B, bits = model.hard_reconstruct(A, base_lambda)
        rec_rgb = yuv2rgb(torch.clamp(B.F.detach().cpu(), 0, 1), out_range=255)
        rec_rgb = np.clip(rec_rgb.round().int().numpy(), 0, 255)
        gt_path = os.path.join(metric_dir, "gt.ply")
        rec_path = os.path.join(metric_dir, "rec.ply")
        write_ply_ascii(gt_path, coords, rgb)
        write_ply_ascii(rec_path, B.C[:, 1:].detach().cpu().numpy(), rec_rgb)
        quality = metric(gt_path, rec_path)
        row = {
            "rate_id": rate_id, "checkpoint_profile": profile,
            "base_lambda": base_lambda, "model_id": model_id,
            "partition_id": partition_id, "sample": entry,
            "points": len(A), "base_bits": bits, "bpp": bits / len(A),
            "seconds": time.perf_counter() - started,
            **quality,
        }
        rows.append(row)
        write_csv("per_h5.csv", H5_FIELDS, rows)
        print("[{}/{}] {} bpp={:.6f} YUV-PSNR={:.4f}".format(
            index + 1, len(files), entry, row["bpp"], row["yuv_psnr_611"]),
            flush=True)

    models = aggregate_models(rows)
    write_csv("per_model.csv", MODEL_FIELDS, models)
    write_csv("reference_curve.csv", CURVE_FIELDS, [average_models(models)])


if __name__ == "__main__":
    main()
