#!/usr/bin/env python3
"""Formal physical-rate evaluation for canonical Base checkpoints."""

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
from data_utils.attribute.inout import read_h5, write_ply_ascii
from scalable_attribute.canonical.base_synthesis import BaseSynthesis
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.operating_points import (
    DEFAULT_CONFIG, point_for_lambda, resolve_operating_point)
from scalable_attribute.canonical.scalable_model import load_frozen_base
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import (
    aggregate_models, average_models, sample_identity)
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sparse_max_difference)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--point")
    parser.add_argument("--operating-points-config", default=DEFAULT_CONFIG)
    parser.add_argument("--released-root")
    parser.add_argument("--released-checkpoint")
    parser.add_argument("--prefix-base-checkpoint")
    parser.add_argument(
        "--base-candidate", action="append", required=True,
        metavar="LABEL=PATH")
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--conditioning-lambda", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def labeled_checkpoint(value):
    if "=" not in value:
        raise ValueError("base-candidate must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError("base-candidate must be LABEL=PATH")
    return label, os.path.abspath(os.path.expandvars(path))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    candidates = [labeled_checkpoint(value) for value in args.base_candidate]
    labels = [label for label, _ in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("Base candidate labels must be unique")
    operating_point = None
    if args.point:
        operating_point = resolve_operating_point(
            args.point, args.released_root, args.operating_points_config)
        if (args.conditioning_lambda is not None and
                args.conditioning_lambda != operating_point["conditioning_lambda"]):
            raise ValueError("--conditioning-lambda conflicts with --point")
        if (args.released_checkpoint is not None and
                os.path.realpath(os.path.expandvars(args.released_checkpoint)) !=
                os.path.realpath(operating_point["released_checkpoint"])):
            raise ValueError("--released-checkpoint conflicts with --point")
        if args.released_checkpoint is None:
            args.released_checkpoint = operating_point["released_checkpoint"]
        if args.conditioning_lambda is None:
            args.conditioning_lambda = operating_point["conditioning_lambda"]
        profile = operating_point["released_profile"]
        rate_id = operating_point["rate_id"]
    else:
        if args.released_checkpoint is None or args.conditioning_lambda is None:
            raise ValueError(
                "Explicit mode requires --released-checkpoint and "
                "--conditioning-lambda")
        profile = os.path.basename(os.path.dirname(args.released_checkpoint))
        point_id, configured = point_for_lambda(
            args.conditioning_lambda, profile, args.operating_points_config)
        rate_id = configured["rate_id"]
    if args.prefix_base_checkpoint is None:
        args.prefix_base_checkpoint = candidates[0][1]
    for name in (
            "data_root", "file_list", "released_checkpoint",
            "prefix_base_checkpoint", "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    protected = ("per_h5.csv", "per_model.csv", "endpoint_summary.csv")
    existing = [name for name in protected
                if os.path.exists(os.path.join(args.output_dir, name))]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w",
              encoding="utf-8") as handle:
        json.dump({**vars(args), "base_candidates": candidates}, handle, indent=2)
    if operating_point is not None:
        with open(os.path.join(args.output_dir, "operating_point.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(operating_point, handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.exists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    os.chdir(args.output_dir)

    prefix_state = torch.load(args.prefix_base_checkpoint, map_location="cpu")
    prefix_model = CanonicalBaseModel(
        args.released_checkpoint,
        BaseSynthesisConfig(**prefix_state["config"])).cuda().eval()
    load_frozen_base(
        prefix_model, args.prefix_base_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)

    synthesis_modules = []
    steps = {}
    for label, path in candidates:
        state = torch.load(path, map_location="cpu")
        if state.get("architecture") != "canonical_base_predict_correct":
            raise ValueError(label + " architecture mismatch")
        if int(state.get("base_lambda", -1)) != args.conditioning_lambda:
            raise ValueError(label + " Base lambda mismatch")
        if os.path.realpath(state.get("base_checkpoint", "")) != os.path.realpath(
                args.released_checkpoint):
            raise ValueError(label + " released checkpoint mismatch")
        synthesis = BaseSynthesis(
            BaseSynthesisConfig(**state["config"])).cuda().eval()
        synthesis.load_state_dict(state["base_synthesis"], strict=True)
        synthesis.requires_grad_(False)
        synthesis_modules.append((label, synthesis))
        steps[label] = int(state["step"])

    def reconstruct(state, synthesis):
        compensation = synthesis(state)
        feature, correction = prefix_model.prefix.synthesize(
            state.f5p, compensation)
        return state.x5p + correction

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
        batch_coords, batch_feats = ME.utils.sparse_collate([coords], [yuv])
        attribute = ME.SparseTensor(
            features=batch_feats, coordinates=batch_coords,
            tensor_stride=1, device="cuda")
        soft_state = prefix_model.prefix(attribute, args.conditioning_lambda)
        hard_state, rate = prefix_model.prefix.hard_forward(
            attribute, args.conditioning_lambda, return_details=True)
        residual_bits = list(rate["residual_bits"])
        if rate["num_residual_streams"] != 4 or len(residual_bits) != 4:
            raise RuntimeError("Canonical Base must contain exactly r1-r4")
        base_bits = int(rate["bits_xlow"] + sum(residual_bits))
        if base_bits != int(rate["base_bits"]):
            raise RuntimeError("Base bit identity failed")

        gt_path = os.path.join(metric_dir, "gt.ply")
        rec_path = os.path.join(metric_dir, "base.ply")
        write_ply_ascii(gt_path, coords, rgb)
        for label, synthesis in synthesis_modules:
            soft = reconstruct(soft_state, synthesis)
            hard = reconstruct(hard_state, synthesis)
            difference = sparse_max_difference(soft, hard, label + " soft/hard")
            if args.require_exact and difference != 0.0:
                raise RuntimeError(label + " hard mismatch: {}".format(difference))
            write_ply_ascii(
                rec_path, hard.C[:, 1:].cpu().numpy(), reconstruction_rgb(hard))
            quality = metric(gt_path, rec_path)
            rows.append({
                "candidate": label, "checkpoint_step": steps[label],
                "rate_id": rate_id, "checkpoint_profile": profile,
                "base_lambda": args.conditioning_lambda,
                "model_id": model_id, "partition_id": partition_id,
                "sample": entry, "points": len(attribute),
                "conditioning_lambda": args.conditioning_lambda,
                "x_low_bits": int(rate["bits_xlow"]),
                "r1_bits": residual_bits[0], "r2_bits": residual_bits[1],
                "r3_bits": residual_bits[2], "r4_bits": residual_bits[3],
                "num_base_residual_streams": 4,
                "num_native_r5_streams": 0,
                "base_bits": base_bits, "base_bpp": base_bits / len(attribute),
                **{"base_" + key: value for key, value in quality.items()},
                "hard_base_max_abs_difference": difference,
                "seconds": time.perf_counter() - started,
            })
        write_csv(os.path.join(args.output_dir, "per_h5.csv"), rows)
        print("[{}/{}] {} Base bpp={:.6f}".format(
            index + 1, len(files), entry, base_bits / len(attribute)), flush=True)

    per_model = []
    summaries = []
    for label in labels:
        prepared = []
        for row in rows:
            if row["candidate"] != label:
                continue
            prepared.append({
                **row, "rate_id": rate_id,
                "checkpoint_profile": profile,
            })
        aggregated = aggregate_models(
            prepared, bits_field="base_bits", metric_prefix="base_")
        per_model.extend({"candidate": label, **row} for row in aggregated)
        average = average_models(aggregated)
        summaries.append({
            "candidate": label, "point": args.point or point_id,
            "checkpoint_step": steps[label],
            **average,
            "hard_base_max_abs_difference": max(
                row["hard_base_max_abs_difference"] for row in prepared),
        })
    write_csv(os.path.join(args.output_dir, "per_model.csv"), per_model)
    write_csv(os.path.join(args.output_dir, "endpoint_summary.csv"), summaries)
    with open(os.path.join(args.output_dir, "endpoint_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "status": "PASS",
            "rate_accounting": "x_low+r1+r2+r3+r4",
            "all_candidates_share_physical_prefix": True,
            "endpoints": summaries,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
