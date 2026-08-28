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
from scalable_attribute.coder import ScalableAttributeCoder
from scalable_attribute.config import EnhancementConfig
from scalable_attribute.c2_native import (
    C2ScalableAttributeCoder, C2ScalableAttributeModel)
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import (
    aggregate_models, average_models, sample_identity)
from scalable_attribute.model import ScalableAttributeModel
from third_party.pc_error_attr import pc_error


H5_FIELDS = (
    "experiment", "checkpoint_step", "base_profile", "base_lambda",
    "model_id", "partition_id", "sample", "points", "base_bits",
    "el_bits", "full_bits", "base_bpp", "el_bpp", "full_bpp",
    "base_y_mse", "base_u_mse", "base_v_mse", "base_y_psnr",
    "base_u_psnr", "base_v_psnr", "base_yuv_psnr_611",
    "full_y_mse", "full_u_mse", "full_v_mse", "full_y_psnr",
    "full_u_psnr", "full_v_psnr", "full_yuv_psnr_611",
    "el_symbol_count", "el_symbol_nonzero_count",
    "el_symbol_nonzero_fraction", "el_symbol_mean_abs",
    "el_symbol_min", "el_symbol_max", "el_active_channel_count",
    "el_active_channel_indices", "seconds",
)
MODEL_FIELDS = (
    "experiment", "checkpoint_step", "base_profile", "base_lambda",
    "model_id", "num_h5", "total_points", "base_bits", "el_bits",
    "full_bits", "base_bpp", "el_bpp", "full_bpp",
    "base_yuv_psnr_611", "full_yuv_psnr_611",
)
SUMMARY_FIELDS = (
    "experiment", "checkpoint_step", "base_profile", "base_lambda",
    "rd_lambda", "lr", "seed", "enhancement_config", "num_models",
    "num_h5", "total_points", "base_mean_bpp", "base_mean_yuv_psnr_611",
    "full_mean_bpp", "full_mean_yuv_psnr_611", "el_mean_bpp",
    "el_symbol_nonzero_fraction", "el_symbol_min", "el_symbol_max",
    "el_active_channel_count", "el_active_channel_indices",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--enhancement-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--base-profile-label", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--rd-lambda", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument(
        "--model-type", choices=("external", "c2_native"), default="external")
    return parser.parse_args()


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields}
                         for row in rows)


def author_metric(gt_path, rec_path, prefix):
    values = pc_error(gt_path, rec_path, res=1, show=False)
    mse_keys = ("  c[0],    F", "  c[1],    F", "  c[2],    F")
    psnr_keys = ("  c[0],PSNRF", "  c[1],PSNRF", "  c[2],PSNRF")
    missing = [key for key in mse_keys + psnr_keys if key not in values]
    if missing:
        raise RuntimeError("pc_error missing metrics: " + ", ".join(missing))
    psnrs = [float(values[key]) for key in psnr_keys]
    result = {}
    for channel, mse_key, psnr_value in zip("yuv", mse_keys, psnrs):
        result[prefix + channel + "_mse"] = float(values[mse_key])
        result[prefix + channel + "_psnr"] = psnr_value
    result[prefix + "yuv_psnr_611"] = (
        6.0 * psnrs[0] + psnrs[1] + psnrs[2]) / 8.0
    return result


def decoded_rgb(tensor):
    rgb = yuv2rgb(torch.clamp(tensor.F.detach().cpu(), 0, 1), out_range=255)
    return np.clip(rgb.round().int().numpy(), 0, 255)


def main():
    args = parse_args()
    for name in (
            "data_root", "file_list", "base_checkpoint",
            "enhancement_checkpoint", "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))

    protected = ("per_h5.csv", "per_model.csv", "endpoint_summary.csv")
    existing = [name for name in protected
                if os.path.exists(os.path.join(args.output_dir, name))]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite completed evaluation artifacts: "
            + ", ".join(existing))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w",
              encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")

    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.lexists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    os.chdir(args.output_dir)

    state = torch.load(args.enhancement_checkpoint, map_location="cpu")
    checkpoint_step = int(state["step"])
    if args.model_type == "c2_native":
        if state.get("architecture") != "c2_native":
            raise RuntimeError("Checkpoint is not a c2_native checkpoint")
        config_dict = state["config"]
        model = C2ScalableAttributeModel(
            args.base_checkpoint, base_scale=args.base_scale,
            base_stage=args.base_stage, base_vmode=args.base_vmode).cuda().eval()
        model.enhancement.load_state_dict(state["enhancement"])
        coder = C2ScalableAttributeCoder(model)
    else:
        config = EnhancementConfig(**state["config"])
        config_dict = config.to_dict()
        model = ScalableAttributeModel(
            args.base_checkpoint, config, base_scale=args.base_scale,
            base_stage=args.base_stage, base_vmode=args.base_vmode).cuda().eval()
        model.enhancement.load_state_dict(state["enhancement"])
        coder = ScalableAttributeCoder(model)

    with open(args.file_list, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    files = h5_files(args.data_root, args.file_list)
    if args.max_samples:
        entries, files = entries[:args.max_samples], files[:args.max_samples]
    if len(entries) != len(files):
        raise RuntimeError("Manifest entry/file count mismatch")

    metric_dir = os.path.join(args.output_dir, "metric_tmp")
    os.makedirs(metric_dir, exist_ok=True)
    gt_path = os.path.join(metric_dir, "gt.ply")
    base_path = os.path.join(metric_dir, "base.ply")
    full_path = os.path.join(metric_dir, "full.ply")
    rows = []
    active_channel_union = set()
    for index, (entry, file_path) in enumerate(zip(entries, files)):
        started = time.perf_counter()
        model_id, partition_id = sample_identity(entry)
        coords, rgb = read_h5(file_path)
        yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
        batched_coords, batched_feats = ME.utils.sparse_collate([coords], [yuv])
        A = ME.SparseTensor(
            features=batched_feats, coordinates=batched_coords,
            tensor_stride=1, device="cuda")
        result = coder.test(A, args.base_lambda)
        active_channel_union.update(result.get("el_active_channel_indices", ()))
        if not torch.equal(A.C, result["B"].C) or not torch.equal(A.C, result["Full"].C):
            raise RuntimeError("A/Base/Full coordinate mismatch for " + entry)

        write_ply_ascii(gt_path, coords, rgb)
        write_ply_ascii(base_path, result["B"].C[:, 1:].cpu().numpy(),
                        decoded_rgb(result["B"]))
        write_ply_ascii(full_path, result["Full"].C[:, 1:].cpu().numpy(),
                        decoded_rgb(result["Full"]))
        base_quality = author_metric(gt_path, base_path, "base_")
        full_quality = author_metric(gt_path, full_path, "full_")
        points = len(A)
        row = {
            "experiment": args.experiment_name,
            "checkpoint_step": checkpoint_step,
            "base_profile": args.base_profile_label,
            "base_lambda": args.base_lambda,
            "rate_id": args.experiment_name,
            "checkpoint_profile": args.base_profile_label,
            "model_id": model_id,
            "partition_id": partition_id,
            "sample": entry,
            "points": points,
            "base_bits": result["base_bits"],
            "el_bits": result["el_bits"],
            "full_bits": result["full_bits"],
            "el_symbol_count": result.get("el_symbol_count"),
            "el_symbol_nonzero_count": result.get("el_symbol_nonzero_count"),
            "el_symbol_nonzero_fraction": result.get("el_symbol_nonzero_fraction"),
            "el_symbol_mean_abs": result.get("el_symbol_mean_abs"),
            "el_symbol_min": result.get("el_symbol_min"),
            "el_symbol_max": result.get("el_symbol_max"),
            "el_active_channel_count": result.get("el_active_channel_count"),
            "el_active_channel_indices": ";".join(
                str(value) for value in result.get(
                    "el_active_channel_indices", ())),
            "base_bpp": result["base_bits"] / points,
            "el_bpp": result["el_bits"] / points,
            "full_bpp": result["full_bits"] / points,
            "seconds": time.perf_counter() - started,
            **base_quality,
            **full_quality,
        }
        rows.append(row)
        write_csv("per_h5.csv", H5_FIELDS, rows)
        print("[{}/{}] {} Base={:.4f}dB Full={:.4f}dB".format(
            index + 1, len(files), entry, row["base_yuv_psnr_611"],
            row["full_yuv_psnr_611"]), flush=True)

    base_models = aggregate_models(rows, "base_bits", "base_")
    full_models = aggregate_models(rows, "full_bits", "full_")
    model_rows = []
    for base, full in zip(base_models, full_models):
        if base["model_id"] != full["model_id"]:
            raise RuntimeError("Base/Full model aggregation mismatch")
        model_rows.append({
            "experiment": args.experiment_name,
            "checkpoint_step": checkpoint_step,
            "base_profile": args.base_profile_label,
            "base_lambda": args.base_lambda,
            "model_id": base["model_id"],
            "num_h5": base["num_h5"],
            "total_points": base["total_points"],
            "base_bits": base["total_bits"],
            "el_bits": full["total_bits"] - base["total_bits"],
            "full_bits": full["total_bits"],
            "base_bpp": base["bpp"],
            "el_bpp": full["bpp"] - base["bpp"],
            "full_bpp": full["bpp"],
            "base_yuv_psnr_611": base["yuv_psnr_611"],
            "full_yuv_psnr_611": full["yuv_psnr_611"],
        })
    write_csv("per_model.csv", MODEL_FIELDS, model_rows)

    base_average = average_models(base_models)
    full_average = average_models(full_models)
    symbol_count = sum(int(row.get("el_symbol_count") or 0) for row in rows)
    symbol_nonzero = sum(
        int(row.get("el_symbol_nonzero_count") or 0) for row in rows)
    symbol_mins = [int(row["el_symbol_min"]) for row in rows
                   if row.get("el_symbol_min") is not None]
    symbol_maxs = [int(row["el_symbol_max"]) for row in rows
                   if row.get("el_symbol_max") is not None]
    summary = {
        "experiment": args.experiment_name,
        "checkpoint_step": checkpoint_step,
        "base_profile": args.base_profile_label,
        "base_lambda": args.base_lambda,
        "rd_lambda": args.rd_lambda,
        "lr": args.lr,
        "seed": args.seed,
        "enhancement_config": json.dumps(config_dict, sort_keys=True),
        "num_models": base_average["num_models"],
        "num_h5": base_average["num_h5"],
        "total_points": base_average["total_points"],
        "base_mean_bpp": base_average["mean_model_bpp"],
        "base_mean_yuv_psnr_611": base_average["mean_model_yuv_psnr_611"],
        "full_mean_bpp": full_average["mean_model_bpp"],
        "full_mean_yuv_psnr_611": full_average["mean_model_yuv_psnr_611"],
        "el_mean_bpp": (full_average["mean_model_bpp"]
                         - base_average["mean_model_bpp"]),
        "el_symbol_nonzero_fraction": (
            symbol_nonzero / symbol_count if symbol_count else ""),
        "el_symbol_min": min(symbol_mins) if symbol_mins else "",
        "el_symbol_max": max(symbol_maxs) if symbol_maxs else "",
        "el_active_channel_count": len(active_channel_union),
        "el_active_channel_indices": ";".join(
            str(value) for value in sorted(active_channel_union)),
    }
    write_csv("endpoint_summary.csv", SUMMARY_FIELDS, [summary])


if __name__ == "__main__":
    main()
