#!/usr/bin/env python3
"""Hard RD trajectory for multiple Enhancement checkpoints on a fixed subset."""

import argparse
import csv
import json
import os
import shlex
import sys

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_h5, write_ply_ascii
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.enhancement import EnhancementVAE
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import load_frozen_base
from scalable_attribute.data import h5_files
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sample_identity, sparse_max_difference)


FIELDS = (
    "label", "checkpoint_step", "model_id", "sample", "points",
    "base_bits", "enhancement_bits", "full_bits", "base_bpp",
    "enhancement_bpp", "full_bpp", "y_mse", "u_mse", "v_mse",
    "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_611",
    "hard_full_max_abs_difference",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="LABEL=PATH; repeat for each Enhancement checkpoint")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def checkpoint_specs(values, released, base_checkpoint, conditioning_lambda):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("checkpoint must be LABEL=PATH")
        label, path = value.split("=", 1)
        path = os.path.abspath(os.path.expandvars(path))
        state = torch.load(path, map_location="cpu")
        if state.get("architecture") != "canonical_independent_enhancement":
            raise ValueError(label + " architecture mismatch")
        if int(state.get("conditioning_lambda", -1)) != conditioning_lambda:
            raise ValueError(label + " conditioning lambda mismatch")
        if os.path.realpath(state.get("released_checkpoint", "")) != os.path.realpath(released):
            raise ValueError(label + " released checkpoint mismatch")
        if os.path.realpath(state.get("base_synthesis_checkpoint", "")) != os.path.realpath(base_checkpoint):
            raise ValueError(label + " Base checkpoint mismatch")
        specs.append((label, path, int(state["step"]), state["enhancement_vae"]))
    if len({label for label, _, _, _ in specs}) != len(specs):
        raise ValueError("checkpoint labels must be unique")
    return specs


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    for name in ("data_root", "file_list", "released_checkpoint",
                 "base_synthesis_checkpoint", "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    output_csv = os.path.join(args.output_dir, "yaware_subset_trajectory.csv")
    if os.path.exists(output_csv):
        raise FileExistsError("Refusing to overwrite " + output_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    specs = checkpoint_specs(
        args.checkpoint, args.released_checkpoint,
        args.base_synthesis_checkpoint, args.conditioning_lambda)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w",
              encoding="utf-8") as handle:
        json.dump({**vars(args), "resolved_checkpoints": [
            {"label": label, "path": path, "step": step}
            for label, path, step, _ in specs]}, handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.exists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    os.chdir(args.output_dir)

    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base = CanonicalBaseModel(
        args.released_checkpoint,
        BaseSynthesisConfig(**base_state["config"])).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)
    enhancement = EnhancementVAE(base.prefix.model.VAE).cuda().eval()
    enhancement.requires_grad_(False)

    with open(args.file_list, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    files = h5_files(args.data_root, args.file_list)
    if len(entries) != len(files):
        raise RuntimeError("Manifest entry/file count mismatch")
    model_ids = [sample_identity(entry)[0] for entry in entries]
    if len(set(model_ids)) != len(model_ids):
        raise RuntimeError("Subset must contain exactly one H5 per model")

    metric_dir = os.path.join(args.output_dir, "metric_tmp")
    os.makedirs(metric_dir, exist_ok=True)
    rows = []
    for index, (entry, path) in enumerate(zip(entries, files)):
        model_id, _ = sample_identity(entry)
        coords, rgb = read_h5(path)
        yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
        batch_coords, batch_feats = ME.utils.sparse_collate([coords], [yuv])
        attribute = ME.SparseTensor(
            features=batch_feats, coordinates=batch_coords,
            tensor_stride=1, device="cuda")
        state, rate = base.prefix.hard_forward(
            attribute, args.conditioning_lambda, return_details=True)
        if rate["num_residual_streams"] != 4:
            raise RuntimeError("Canonical prefix must contain r1-r4")
        base_output = base.reconstruct_from_state(state)
        embedding = base.prefix.lambda_embedding(
            args.conditioning_lambda, attribute.device)
        gt_path = os.path.join(metric_dir, "gt.ply")
        rec_path = os.path.join(metric_dir, "rec.ply")
        write_ply_ascii(gt_path, coords, rgb)
        for label, _, step, weights in specs:
            enhancement.vae.load_state_dict(weights, strict=True)
            encoded = enhancement.encode(
                base_output["Base"], attribute, base_output["F_B"],
                base_output["d5p"], embedding)
            payload = {key: encoded[key] for key in ("strings", "min_v", "max_v")}
            decoded = enhancement.decode(
                payload, base_output["Base"], base_output["F_B"],
                base_output["d5p"], embedding)
            difference = sparse_max_difference(
                encoded["x_out"], decoded["x_out"], label + " hard")
            if difference != 0.0:
                raise RuntimeError(label + " hard round-trip mismatch")
            write_ply_ascii(
                rec_path, decoded["x_out"].C[:, 1:].cpu().numpy(),
                reconstruction_rgb(decoded["x_out"]))
            quality = metric(gt_path, rec_path)
            enhancement_bits = len(payload["strings"]) * 8
            base_bits = int(rate["base_bits"])
            rows.append({
                "label": label, "checkpoint_step": step,
                "model_id": model_id, "sample": entry,
                "points": len(attribute), "base_bits": base_bits,
                "enhancement_bits": enhancement_bits,
                "full_bits": base_bits + enhancement_bits,
                "base_bpp": base_bits / len(attribute),
                "enhancement_bpp": enhancement_bits / len(attribute),
                "full_bpp": (base_bits + enhancement_bits) / len(attribute),
                **quality, "hard_full_max_abs_difference": difference,
            })
            write_rows(output_csv, rows)
            del encoded, decoded
        print("[{}/{}] {} checkpoints={}".format(
            index + 1, len(files), entry, len(specs)), flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "status": "PASS", "num_h5": len(files),
            "num_models": len(set(model_ids)), "num_checkpoints": len(specs),
            "prefix_hard_invocations": len(files),
            "all_hard_exact": all(
                float(row["hard_full_max_abs_difference"]) == 0.0
                for row in rows),
        }, handle, indent=2)


if __name__ == "__main__":
    main()
