#!/usr/bin/env python3
"""Formal physical-rate evaluation for the canonical Base/Full endpoints."""

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
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_finetuned_scalable, load_frozen_base)
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import (
    aggregate_models, average_models, sample_identity)
from third_party.pc_error_attr import pc_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--enhancement-checkpoint")
    checkpoint.add_argument("--scalable-checkpoint")
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(gt_path, rec_path):
    values = pc_error(gt_path, rec_path, res=1, show=False)
    keys = (
        "  c[0],    F", "  c[1],    F", "  c[2],    F",
        "  c[0],PSNRF", "  c[1],PSNRF", "  c[2],PSNRF",
    )
    missing = [key for key in keys if key not in values]
    if missing:
        raise RuntimeError("pc_error missing metrics: " + ", ".join(missing))
    psnrs = [float(values[key]) for key in keys[3:]]
    return {
        "y_mse": float(values[keys[0]]),
        "u_mse": float(values[keys[1]]),
        "v_mse": float(values[keys[2]]),
        "y_psnr": psnrs[0],
        "u_psnr": psnrs[1],
        "v_psnr": psnrs[2],
        "yuv_psnr_611": (6.0 * psnrs[0] + psnrs[1] + psnrs[2]) / 8.0,
    }


def sparse_max_difference(left, right, label):
    if list(left.tensor_stride) != list(right.tensor_stride):
        raise RuntimeError(label + " tensor strides differ")
    if not torch.equal(left.C, right.C):
        raise RuntimeError(label + " coordinates differ")
    return float((left.F - right.F).abs().max().item())


def reconstruction_rgb(sparse):
    rgb = yuv2rgb(torch.clamp(sparse.F.detach().cpu(), 0, 1), out_range=255)
    return np.clip(rgb.round().int().numpy(), 0, 255)


def endpoint_models(rows, endpoint):
    prepared = []
    for row in rows:
        item = dict(row)
        item["rate_id"] = "Canonical_" + endpoint
        item["checkpoint_profile"] = "32k8k"
        item["base_lambda"] = int(row["conditioning_lambda"])
        prepared.append(item)
    return aggregate_models(
        prepared, bits_field=endpoint.lower() + "_bits",
        metric_prefix=endpoint.lower() + "_")


def endpoint_summary(models, endpoint):
    result = average_models(models)
    return {
        "endpoint": endpoint,
        "num_models": result["num_models"],
        "num_h5": result["num_h5"],
        "total_points": result["total_points"],
        "mean_model_bpp": result["mean_model_bpp"],
        "mean_model_y_psnr": result["mean_model_y_psnr"],
        "mean_model_yuv_psnr_611": result["mean_model_yuv_psnr_611"],
    }


def main():
    args = parse_args()
    for name in (
            "data_root", "file_list", "released_checkpoint", "gpcc_binary",
            "base_synthesis_checkpoint", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    for name in ("enhancement_checkpoint", "scalable_checkpoint"):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(os.path.expandvars(value)))
    protected = ("per_h5.csv", "per_model.csv", "endpoint_summary.csv")
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
    os.chdir(args.output_dir)

    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base_config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(args.released_checkpoint, base_config).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)
    model = CanonicalScalableModel(base, args.conditioning_lambda).cuda().eval()
    if args.scalable_checkpoint:
        load_finetuned_scalable(
            model, args.scalable_checkpoint, args.conditioning_lambda)
    else:
        enhancement_state = torch.load(
            args.enhancement_checkpoint, map_location="cpu")
        if enhancement_state.get(
                "architecture") != "canonical_independent_enhancement":
            raise ValueError("Enhancement checkpoint architecture mismatch")
        if int(enhancement_state.get(
                "conditioning_lambda", -1)) != args.conditioning_lambda:
            raise ValueError("Enhancement checkpoint lambda mismatch")
        model.enhancement.vae.load_state_dict(
            enhancement_state["enhancement_vae"], strict=True)
    model.requires_grad_(False)

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
        attribute = ME.SparseTensor(
            features=batched_feats, coordinates=batched_coords,
            tensor_stride=1, device="cuda")
        hard = model.hard_reconstruct(attribute)
        details = hard["prefix_rate"]
        if details["num_residual_streams"] != 4:
            raise RuntimeError("Canonical Base must contain exactly four residual streams")
        residual_bits = list(details["residual_bits"])
        if len(residual_bits) != 4:
            raise RuntimeError("Residual bit breakdown must contain r1-r4")
        expected_base = int(details["bits_xlow"] + sum(residual_bits))
        if hard["base_bits"] != expected_base:
            raise RuntimeError("Base bit identity failed")
        if hard["full_bits"] != hard["base_bits"] + hard["enhancement_bits"]:
            raise RuntimeError("Full bit identity failed")
        full_difference = sparse_max_difference(
            hard["encoded_Full"], hard["Full"], "encoded/decoded Full")
        if args.require_exact and full_difference != 0.0:
            raise RuntimeError("Hard Full mismatch: {}".format(full_difference))

        gt_path = os.path.join(metric_dir, "gt.ply")
        base_path = os.path.join(metric_dir, "base.ply")
        full_path = os.path.join(metric_dir, "full.ply")
        write_ply_ascii(gt_path, coords, rgb)
        write_ply_ascii(base_path, hard["Base"].C[:, 1:].cpu().numpy(),
                        reconstruction_rgb(hard["Base"]))
        write_ply_ascii(full_path, hard["Full"].C[:, 1:].cpu().numpy(),
                        reconstruction_rgb(hard["Full"]))
        base_quality = metric(gt_path, base_path)
        full_quality = metric(gt_path, full_path)
        row = {
            "model_id": model_id,
            "partition_id": partition_id,
            "sample": entry,
            "points": len(attribute),
            "conditioning_lambda": args.conditioning_lambda,
            "x_low_bits": int(details["bits_xlow"]),
            "r1_bits": residual_bits[0],
            "r2_bits": residual_bits[1],
            "r3_bits": residual_bits[2],
            "r4_bits": residual_bits[3],
            "num_base_residual_streams": 4,
            "num_native_r5_streams": 0,
            "base_bits": hard["base_bits"],
            "enhancement_bits": hard["enhancement_bits"],
            "full_bits": hard["full_bits"],
            "base_bpp": hard["base_bits"] / len(attribute),
            "enhancement_bpp": hard["enhancement_bits"] / len(attribute),
            "full_bpp": hard["full_bits"] / len(attribute),
            **{"base_" + key: value for key, value in base_quality.items()},
            **{"full_" + key: value for key, value in full_quality.items()},
            "hard_full_max_abs_difference": full_difference,
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        write_csv(os.path.join(args.output_dir, "per_h5.csv"), rows)
        print("[{}/{}] {} Base={:.6f} Full={:.6f} exact={}".format(
            index + 1, len(files), entry, row["base_bpp"], row["full_bpp"],
            full_difference), flush=True)

    base_models = endpoint_models(rows, "Base")
    full_models = endpoint_models(rows, "Full")
    combined_models = []
    for endpoint, models in (("Base", base_models), ("Full", full_models)):
        for row in models:
            combined_models.append({"endpoint": endpoint, **row})
    write_csv(os.path.join(args.output_dir, "per_model.csv"), combined_models)
    summaries = [endpoint_summary(base_models, "Base"),
                 endpoint_summary(full_models, "Full")]
    write_csv(os.path.join(args.output_dir, "endpoint_summary.csv"), summaries)
    summary = {
        "status": "PASS",
        "rate_accounting": "x_low+r1+r2+r3+r4; min_v/max_v excluded",
        "native_r5_streams": 0,
        "hard_full_max_abs_difference": max(
            row["hard_full_max_abs_difference"] for row in rows),
        "endpoints": summaries,
    }
    with open(os.path.join(args.output_dir, "endpoint_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
