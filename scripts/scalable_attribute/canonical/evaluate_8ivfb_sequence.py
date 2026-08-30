#!/usr/bin/env python3
"""External 8iVFB physical RD with one shared canonical hard prefix."""

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
from scalable_attribute.canonical.enhancement import EnhancementVAE
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import load_frozen_base
from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS
from scalable_attribute.unicorn_reference import ReleasedUnicornAttribute
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sparse_max_difference)


FIELDS = (
    "sequence", "frame", "endpoint", "source", "checkpoint_profile",
    "lambda", "checkpoint_step", "points", "physical_bits", "physical_bpp",
    "base_bits", "enhancement_bits", "x_low_bits", "r1_bits", "r2_bits",
    "r3_bits", "r4_bits", "num_base_residual_streams",
    "num_native_r5_streams", "y_mse", "u_mse", "v_mse", "y_psnr",
    "u_psnr", "v_psnr", "yuv_psnr_611", "hard_max_abs_difference",
    "seconds",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--released-checkpoint-root", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--enhancement-step1763", required=True)
    parser.add_argument("--enhancement-step3525")
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--conditioning-lambda", type=int, default=32768)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_step(path):
    state = torch.load(path, map_location="cpu")
    if state.get("architecture") != "canonical_independent_enhancement":
        raise ValueError("Enhancement checkpoint architecture mismatch")
    return state, int(state["step"])


def main():
    args = parse_args()
    for name in (
            "input_ply", "released_checkpoint_root",
            "base_synthesis_checkpoint", "enhancement_step1763",
            "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    if args.enhancement_step3525:
        args.enhancement_step3525 = os.path.abspath(
            os.path.expandvars(args.enhancement_step3525))
    output_csv = os.path.join(args.output_dir, "physical_rd.csv")
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
    batch_coords, batch_feats = ME.utils.sparse_collate([coords], [yuv])
    attribute = ME.SparseTensor(
        features=batch_feats, coordinates=batch_coords,
        tensor_stride=1, device="cuda")
    gt_path = os.path.join(args.output_dir, "metric_gt.ply")
    rec_path = os.path.join(args.output_dir, "metric_rec.ply")
    write_ply_ascii(gt_path, coords, rgb)
    rows = []

    def record(endpoint, reconstruction, bits, source, started, **extra):
        write_ply_ascii(
            rec_path, reconstruction.C[:, 1:].detach().cpu().numpy(),
            reconstruction_rgb(reconstruction))
        quality = metric(gt_path, rec_path)
        row = {key: "" for key in FIELDS}
        row.update({
            "sequence": args.sequence,
            "frame": args.frame,
            "endpoint": endpoint,
            "source": source,
            "points": len(attribute),
            "physical_bits": int(bits),
            "physical_bpp": int(bits) / len(attribute),
            **quality,
            "seconds": time.perf_counter() - started,
            **extra,
        })
        rows.append(row)
        write_rows(output_csv, rows)
        print("{} bpp={:.6f} YUV611={:.4f}".format(
            endpoint, row["physical_bpp"], row["yuv_psnr_611"]), flush=True)

    if args.run_official:
        for rate_id, profile, lmb in OFFICIAL_RWTT_REFERENCE_POINTS:
            started = time.perf_counter()
            checkpoint = os.path.join(
                args.released_checkpoint_root, profile, "epoch_last.pth")
            released = ReleasedUnicornAttribute(checkpoint).cuda().eval()
            reconstruction, bits = released.hard_reconstruct(attribute, lmb)
            record(rate_id, reconstruction, bits, "OFFICIAL_RELEASED", started,
                   checkpoint_profile=profile, **{"lambda": lmb},
                   num_native_r5_streams=1)
            del reconstruction, released
            torch.cuda.empty_cache()

    released_r01 = os.path.join(
        args.released_checkpoint_root, "32k8k", "epoch_last.pth")
    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base = CanonicalBaseModel(
        released_r01, BaseSynthesisConfig(**base_state["config"])).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, released_r01,
        args.conditioning_lambda)

    # This is the only canonical prefix hard invocation for the sequence.
    prefix_state, prefix_rate = base.prefix.hard_forward(
        attribute, args.conditioning_lambda, return_details=True)
    if prefix_rate["num_residual_streams"] != 4:
        raise RuntimeError("Canonical prefix must contain exactly r1-r4")
    residual_bits = prefix_rate["residual_bits"]
    if len(residual_bits) != 4:
        raise RuntimeError("Canonical prefix bit breakdown is not r1-r4")
    base_output = base.reconstruct_from_state(prefix_state)
    native = base.native_baselines(prefix_state)["B_native"]
    common = {
        "checkpoint_profile": "32k8k",
        "lambda": args.conditioning_lambda,
        "base_bits": prefix_rate["base_bits"],
        "x_low_bits": prefix_rate["bits_xlow"],
        "r1_bits": residual_bits[0], "r2_bits": residual_bits[1],
        "r3_bits": residual_bits[2], "r4_bits": residual_bits[3],
        "num_base_residual_streams": 4,
        "num_native_r5_streams": 0,
    }
    started = time.perf_counter()
    record("B_native", native, prefix_rate["base_bits"],
           "CURRENT_CANONICAL_DIAGNOSTIC", started,
           checkpoint_step=0, enhancement_bits=0, **common)
    record("Canonical_Base", base_output["Base"], prefix_rate["base_bits"],
           "CURRENT_CANONICAL", started,
           checkpoint_step=5525, enhancement_bits=0, **common)

    embedding = base.prefix.lambda_embedding(
        args.conditioning_lambda, attribute.device)
    enhancement_paths = [("Canonical_Full_step1763", args.enhancement_step1763)]
    if args.enhancement_step3525:
        enhancement_paths.append(
            ("Canonical_Full_step3525", args.enhancement_step3525))
    for endpoint, path in enhancement_paths:
        started = time.perf_counter()
        state, step = checkpoint_step(path)
        if int(state["conditioning_lambda"]) != args.conditioning_lambda:
            raise ValueError("Enhancement conditioning lambda mismatch")
        enhancement = EnhancementVAE(base.prefix.model.VAE).cuda().eval()
        enhancement.vae.load_state_dict(state["enhancement_vae"], strict=True)
        enhancement.requires_grad_(False)
        encoded = enhancement.encode(
            base_output["Base"], attribute, base_output["F_B"],
            base_output["d5p"], embedding)
        payload = {key: encoded[key] for key in ("strings", "min_v", "max_v")}
        decoded = enhancement.decode(
            payload, base_output["Base"], base_output["F_B"],
            base_output["d5p"], embedding)
        difference = sparse_max_difference(
            encoded["x_out"], decoded["x_out"], endpoint + " hard")
        if difference != 0.0:
            raise RuntimeError(endpoint + " hard round-trip mismatch")
        enhancement_bits = len(payload["strings"]) * 8
        record(endpoint, decoded["x_out"],
               prefix_rate["base_bits"] + enhancement_bits,
               "CURRENT_CANONICAL", started, checkpoint_step=step,
               enhancement_bits=enhancement_bits,
               hard_max_abs_difference=difference, **common)
        del enhancement, encoded, decoded
        torch.cuda.empty_cache()

    summary = {
        "status": "PASS",
        "sequence": args.sequence,
        "frame": args.frame,
        "canonical_prefix_hard_invocations": 1,
        "canonical_endpoints_share_prefix_state": True,
        "num_base_residual_streams": 4,
        "num_native_r5_streams": 0,
        "rows": rows,
    }
    with open(os.path.join(args.output_dir, "physical_rd.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
