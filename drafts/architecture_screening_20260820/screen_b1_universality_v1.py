#!/usr/bin/env python3
"""Fixed-4 B1 universality screen; experiment-only, no training/search."""
import argparse
import csv
import os

import torch

import complete_b1_physical_rd as b1
import screen_native_and_successive as common
from scalable_attribute.base_adapter import BaseAdapter


SAMPLES = (
    "RWT115/model_mesh_P0.h5",
    "RWT182/572883_P15.h5",
    "RWT380/ujety_svah_ske_P15.h5",
    "RWT541/marco_cat_mesh_P9.h5",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--derived-rate-id", required=True)
    parser.add_argument("--derived-checkpoint", required=True)
    parser.add_argument("--derived-lambda", type=int, required=True)
    parser.add_argument("--neighbor", nargs=3, action="append", required=True,
                        metavar=("RATE_ID", "CHECKPOINT", "LAMBDA"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpcc-binary", required=True)
    return parser.parse_args()


def write_csv(path, rows):
    if os.path.exists(path):
        raise FileExistsError("Refusing to overwrite: " + path)
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


@torch.no_grad()
def derived_b1(args):
    adapter = BaseAdapter(
        args.derived_checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    rows, loaded = [], []
    for sample_index, entry in enumerate(SAMPLES):
        coords, rgb, A = common.load_sample(args.data_root, entry)
        encoded, x_low, gpcc_bits = adapter.base(
            A, training=False, lmb=args.derived_lambda, encode=True)
        prefix_bits = int(gpcc_bits + sum(
            len(stream["strings"]) * 8 for stream in encoded[:4]))
        original_bits = prefix_bits + len(encoded[4]["strings"]) * 8
        context = b1.decoded_r5_context(
            adapter.base, A, x_low, encoded, args.derived_lambda)
        official_full = adapter.base.decode(
            context["x0"], x_low, encoded, lmb=args.derived_lambda)
        captured_full = common.synthesize(
            adapter.base, context["symbols"], context)
        native_difference = float(
            (captured_full.F - official_full.F).abs().max().item())
        if native_difference != 0:
            raise RuntimeError("Captured Full differs from official decode")
        full_psnr = common.author_psnr(
            args.output_dir, "{}_full_{}".format(
                args.derived_rate_id, sample_index),
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
            full_difference = float(
                (full_rec.F - official_full.F).abs().max().item())
            if full_difference != 0:
                raise RuntimeError("Layered Full differs from official Full")
            qB_bits, qE_bits = len(qB_string) * 8, len(qE_string) * 8
            base_bits, full_bits = prefix_bits + qB_bits, prefix_bits + qB_bits + qE_bits
            base_psnr = common.author_psnr(
                args.output_dir,
                "{}_base_{}_s{}".format(
                    args.derived_rate_id, sample_index, step),
                coords, rgb, base_rec)
            rows.append({
                "derived_rate_id": args.derived_rate_id,
                "derived_lambda": args.derived_lambda,
                "sample": entry, "points": len(A), "step": step,
                "prefix_bits": prefix_bits, "qB_bits": qB_bits,
                "qE_bits": qE_bits, "base_bits": base_bits,
                "full_bits": full_bits, "original_bits": original_bits,
                "base_bpp": base_bits / len(A),
                "full_bpp": full_bits / len(A),
                "original_bpp": original_bits / len(A),
                "base_yuv_psnr_611": base_psnr,
                "full_yuv_psnr_611": full_psnr,
                "full_minus_base_bpp": (full_bits - base_bits) / len(A),
                "full_minus_base_yuv_psnr_611": full_psnr - base_psnr,
                "layered_over_original_ratio": full_bits / original_bits,
                "full_max_abs_difference": full_difference,
                "native_capture_max_abs_difference": native_difference,
            })
        loaded.append((entry, coords, rgb, A))
    return adapter, rows, loaded


@torch.no_grad()
def official_neighbors(args, derived_adapter, loaded):
    rows = []
    for rate_id, checkpoint, lambda_text in args.neighbor:
        lmb = int(lambda_text)
        same_checkpoint = os.path.abspath(checkpoint) == os.path.abspath(
            args.derived_checkpoint)
        adapter = derived_adapter if same_checkpoint else BaseAdapter(
            checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
        for sample_index, (entry, coords, rgb, A) in enumerate(loaded):
            reconstruction, _, bits = adapter.hard_reconstruct(A, lmb)
            rows.append({
                "rate_id": rate_id, "base_lambda": lmb, "sample": entry,
                "points": len(A), "physical_bits": bits,
                "physical_bpp": bits / len(A),
                "direct_yuv_psnr_611": common.author_psnr(
                    args.output_dir,
                    "neighbor_{}_{}".format(rate_id, sample_index),
                    coords, rgb, reconstruction),
            })
        if not same_checkpoint:
            del adapter
            torch.cuda.empty_cache()
    return rows


def main():
    args = parse_args()
    for name in ("data_root", "derived_checkpoint", "output_dir", "gpcc_binary"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    args.neighbor = [
        (rate, os.path.abspath(os.path.expandvars(checkpoint)), lmb)
        for rate, checkpoint, lmb in args.neighbor]
    os.makedirs(args.output_dir, exist_ok=True)
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.lexists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    os.chdir(args.output_dir)
    adapter, b1_rows, loaded = derived_b1(args)
    official_rows = official_neighbors(args, adapter, loaded)
    summaries = []
    for step in (4, 8):
        rows = [row for row in b1_rows if row["step"] == step]
        summaries.append({
            "derived_rate_id": args.derived_rate_id,
            "derived_lambda": args.derived_lambda, "step": step,
            "num_h5": len(rows),
            "mean_base_bpp": mean(rows, "base_bpp"),
            "mean_base_yuv_psnr_611": mean(rows, "base_yuv_psnr_611"),
            "mean_full_bpp": mean(rows, "full_bpp"),
            "mean_full_yuv_psnr_611": mean(rows, "full_yuv_psnr_611"),
            "full_minus_base_bpp": mean(rows, "full_minus_base_bpp"),
            "full_minus_base_yuv_psnr_611": mean(
                rows, "full_minus_base_yuv_psnr_611"),
            "mean_layered_over_original_ratio": mean(
                rows, "layered_over_original_ratio"),
            "max_full_abs_difference": max(
                float(row["full_max_abs_difference"]) for row in rows),
        })
    official_summary = []
    for rate_id, _, _ in args.neighbor:
        rows = [row for row in official_rows if row["rate_id"] == rate_id]
        official_summary.append({
            "rate_id": rate_id, "num_h5": len(rows),
            "mean_physical_bpp": mean(rows, "physical_bpp"),
            "mean_direct_yuv_psnr_611": mean(rows, "direct_yuv_psnr_611"),
        })
    write_csv("b1_per_h5.csv", b1_rows)
    write_csv("b1_summary.csv", summaries)
    write_csv("official_same4_per_h5.csv", official_rows)
    write_csv("official_same4_summary.csv", official_summary)


if __name__ == "__main__":
    main()
