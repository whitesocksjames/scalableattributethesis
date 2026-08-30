#!/usr/bin/env python3
"""Physical R01-R09 and canonical Base/Full RD on Longdress frame 1300."""

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

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_ply_ascii, write_ply_ascii
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_frozen_base)
from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS
from scalable_attribute.unicorn_reference import ReleasedUnicornAttribute
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sparse_max_difference, write_csv)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--released-checkpoint-root", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--enhancement-checkpoint", required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--conditioning-lambda", type=int, default=32768)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    for name in (
            "input_ply", "released_checkpoint_root",
            "base_synthesis_checkpoint", "enhancement_checkpoint",
            "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    output_csv = os.path.join(args.output_dir, "longdress_physical_rd.csv")
    if os.path.exists(output_csv):
        raise FileExistsError("Refusing to overwrite " + output_csv)
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

    coords, rgb = read_ply_ascii(args.input_ply)
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    batched_coords, batched_feats = ME.utils.sparse_collate([coords], [yuv])
    attribute = ME.SparseTensor(
        features=batched_feats, coordinates=batched_coords,
        tensor_stride=1, device="cuda")
    gt_path = os.path.join(args.output_dir, "longdress_vox10_1300_gt.ply")
    write_ply_ascii(gt_path, coords, rgb)
    rows = []

    def record(endpoint, reconstruction, bits, source, started, extra=None):
        rec_path = os.path.join(args.output_dir, endpoint + ".ply")
        write_ply_ascii(
            rec_path, reconstruction.C[:, 1:].detach().cpu().numpy(),
            reconstruction_rgb(reconstruction))
        quality = metric(gt_path, rec_path)
        row = {
            "endpoint": endpoint,
            "source": source,
            "points": len(attribute),
            "physical_bits": int(bits),
            "physical_bpp": int(bits) / len(attribute),
            **quality,
            "seconds": time.perf_counter() - started,
        }
        if extra:
            row.update(extra)
        rows.append(row)
        write_csv(output_csv, rows)
        print("{} bpp={:.6f} YUV-PSNR={:.4f}".format(
            endpoint, row["physical_bpp"], row["yuv_psnr_611"]), flush=True)

    for rate_id, profile, lmb in OFFICIAL_RWTT_REFERENCE_POINTS:
        started = time.perf_counter()
        checkpoint = os.path.join(
            args.released_checkpoint_root, profile, "epoch_last.pth")
        model = ReleasedUnicornAttribute(checkpoint).cuda().eval()
        reconstruction, bits = model.hard_reconstruct(attribute, lmb)
        record(rate_id, reconstruction, bits, "OFFICIAL_RELEASED", started, {
            "checkpoint_profile": profile,
            "lambda": lmb,
            "base_bits": "",
            "enhancement_bits": "",
            "hard_full_max_abs_difference": "",
        })
        del reconstruction, model
        torch.cuda.empty_cache()

    released_r01 = os.path.join(
        args.released_checkpoint_root, "32k8k", "epoch_last.pth")
    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base = CanonicalBaseModel(
        released_r01, BaseSynthesisConfig(**base_state["config"])).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, released_r01,
        args.conditioning_lambda)
    model = CanonicalScalableModel(base, args.conditioning_lambda).cuda().eval()
    enhancement_state = torch.load(args.enhancement_checkpoint, map_location="cpu")
    model.enhancement.vae.load_state_dict(
        enhancement_state["enhancement_vae"], strict=True)
    model.requires_grad_(False)
    started = time.perf_counter()
    hard = model.hard_reconstruct(attribute)
    difference = sparse_max_difference(
        hard["encoded_Full"], hard["Full"], "encoded/decoded Full")
    if difference != 0.0:
        raise RuntimeError("Canonical Full hard mismatch: {}".format(difference))
    if hard["prefix_rate"]["num_residual_streams"] != 4:
        raise RuntimeError("Canonical Base does not contain four residual streams")
    record("Canonical_Base", hard["Base"], hard["base_bits"],
           "CURRENT_CANONICAL", started, {
               "checkpoint_profile": "32k8k", "lambda": args.conditioning_lambda,
               "base_bits": hard["base_bits"], "enhancement_bits": 0,
               "hard_full_max_abs_difference": "",
           })
    record("Canonical_Full", hard["Full"], hard["full_bits"],
           "CURRENT_CANONICAL", started, {
               "checkpoint_profile": "32k8k", "lambda": args.conditioning_lambda,
               "base_bits": hard["base_bits"],
               "enhancement_bits": hard["enhancement_bits"],
               "hard_full_max_abs_difference": difference,
           })
    with open(os.path.join(args.output_dir, "longdress_physical_rd.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"status": "PASS", "rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
