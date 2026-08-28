#!/usr/bin/env python3
"""Generic B1 fixed step=4/8 Full28 shard evaluator; no training/search."""
import argparse
import csv
import os
import time

import numpy as np
import torch

import complete_b1_physical_rd as b1
import screen_native_and_successive as common
from data_utils.attribute.color_format import yuv2rgb
from data_utils.attribute.inout import write_ply_ascii
from scalable_attribute.base_adapter import BaseAdapter
from scalable_attribute.evaluation import sample_identity
from scripts.scalable_attribute.evaluate_unicorn_reference import metric


FIELDS = (
    "source_rate_id", "source_lambda", "sample", "model_id",
    "partition_id", "points", "step", "prefix_bits", "qB_bits",
    "qE_bits", "base_bits", "full_bits", "original_bits", "base_bpp",
    "full_bpp", "original_bpp", "base_y_mse", "base_u_mse",
    "base_v_mse", "base_y_psnr", "base_u_psnr", "base_v_psnr",
    "base_yuv_psnr_611", "full_y_mse", "full_u_mse", "full_v_mse",
    "full_y_psnr", "full_u_psnr", "full_v_psnr",
    "full_yuv_psnr_611", "full_minus_base_bpp",
    "full_minus_base_yuv_psnr_611", "layered_over_original_ratio",
    "qB_roundtrip_exact", "qB_plus_qE_exact_q",
    "full_max_abs_difference", "native_capture_max_abs_difference",
    "seconds",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--source-rate-id", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-lambda", type=int, required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    return parser.parse_args()


def quality(output_dir, label, coords, rgb, reconstruction):
    directory = os.path.join(output_dir, "metric_tmp", label)
    os.makedirs(directory, exist_ok=True)
    gt_path, rec_path = os.path.join(directory, "gt.ply"), os.path.join(directory, "rec.ply")
    write_ply_ascii(gt_path, coords, rgb)
    rec_rgb = yuv2rgb(torch.clamp(
        reconstruction.F.detach().cpu(), 0, 1), out_range=255)
    rec_rgb = np.clip(rec_rgb.round().int().numpy(), 0, 255)
    write_ply_ascii(rec_path, reconstruction.C[:, 1:].cpu().numpy(), rec_rgb)
    return metric(gt_path, rec_path)


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    for name in ("data_root", "file_list", "source_checkpoint",
                 "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    output_csv = os.path.join(args.output_dir, "per_h5.csv")
    if os.path.exists(output_csv):
        raise FileExistsError("Refusing to overwrite: " + output_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.lexists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    with open(args.file_list, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    end = args.end_index or len(entries)
    entries = entries[args.start_index:end]
    if not entries:
        raise ValueError("Empty manifest shard")
    os.chdir(args.output_dir)
    adapter = BaseAdapter(
        args.source_checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    rows = []
    for local_index, entry in enumerate(entries):
        started = time.perf_counter()
        global_index = args.start_index + local_index
        model_id, partition_id = sample_identity(entry)
        coords, rgb, A = common.load_sample(args.data_root, entry)
        encoded, x_low, gpcc_bits = adapter.base(
            A, training=False, lmb=args.source_lambda, encode=True)
        prefix_bits = int(gpcc_bits + sum(
            len(stream["strings"]) * 8 for stream in encoded[:4]))
        original_bits = prefix_bits + len(encoded[4]["strings"]) * 8
        context = b1.decoded_r5_context(
            adapter.base, A, x_low, encoded, args.source_lambda)
        official_full = adapter.base.decode(
            context["x0"], x_low, encoded, lmb=args.source_lambda)
        captured = common.synthesize(adapter.base, context["symbols"], context)
        native_diff = float((captured.F - official_full.F).abs().max().item())
        if native_diff != 0:
            raise RuntimeError("Captured native Full mismatch")
        full_quality = quality(
            args.output_dir, "full_{:04d}".format(global_index),
            coords, rgb, official_full)
        fine, loc = context["fine_sorted"], context["loc"].F
        scale = context["scale"].F.abs().clamp(min=1e-8)
        for step in (4, 8):
            coarse, recovered, qB_string, qE_string = b1.nested_roundtrip(
                adapter.base.VAE.entropy_fn, fine, loc, scale, step)
            mapping = context["sorted_to_prior_index"]
            coarse, recovered = coarse[mapping], recovered[mapping]
            base_rec = common.synthesize(
                adapter.base, common.sparse(coarse, context["symbols"]), context)
            full_rec = common.synthesize(
                adapter.base, common.sparse(recovered, context["symbols"]), context)
            full_diff = float((full_rec.F - official_full.F).abs().max().item())
            if full_diff != 0:
                raise RuntimeError("Layered Full mismatch")
            base_quality = quality(
                args.output_dir, "base_{:04d}_s{}".format(global_index, step),
                coords, rgb, base_rec)
            qB_bits, qE_bits = len(qB_string) * 8, len(qE_string) * 8
            base_bits, full_bits = prefix_bits + qB_bits, prefix_bits + qB_bits + qE_bits
            rows.append({
                "source_rate_id": args.source_rate_id,
                "source_lambda": args.source_lambda, "sample": entry,
                "model_id": model_id, "partition_id": partition_id,
                "points": len(A), "step": step, "prefix_bits": prefix_bits,
                "qB_bits": qB_bits, "qE_bits": qE_bits,
                "base_bits": base_bits, "full_bits": full_bits,
                "original_bits": original_bits, "base_bpp": base_bits / len(A),
                "full_bpp": full_bits / len(A),
                "original_bpp": original_bits / len(A),
                **{"base_" + key: value for key, value in base_quality.items()},
                **{"full_" + key: value for key, value in full_quality.items()},
                "full_minus_base_bpp": qE_bits / len(A),
                "full_minus_base_yuv_psnr_611": (
                    full_quality["yuv_psnr_611"] - base_quality["yuv_psnr_611"]),
                "layered_over_original_ratio": full_bits / original_bits,
                "qB_roundtrip_exact": 1, "qB_plus_qE_exact_q": 1,
                "full_max_abs_difference": full_diff,
                "native_capture_max_abs_difference": native_diff,
                "seconds": time.perf_counter() - started,
            })
        write_rows(output_csv, rows)
        print("[{}/{}] {}".format(local_index + 1, len(entries), entry), flush=True)
        del A, encoded, x_low, context, official_full, captured
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
