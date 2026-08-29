#!/usr/bin/env python3
"""Evaluate canonical Base against its two no-additional-bit baselines."""

import argparse
import csv
import json
import os
import shlex
import sys

import MinkowskiEngine as ME
import torch

from data_utils.dataloaders.attribute_dataloader import PCDataset
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.evaluation import evaluate_base
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--base-synthesis-checkpoint")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--hard-gate", action="store_true")
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    return parser.parse_args()


def _comparison(left, right, label):
    support = (
        list(left.tensor_stride) == list(right.tensor_stride)
        and torch.equal(left.C, right.C))
    if not support:
        raise RuntimeError(label + " support/stride mismatch")
    return float((left.F - right.F).abs().max().item())


def _quality(reference, reconstruction):
    if not torch.equal(reference.C, reconstruction.C):
        raise RuntimeError("Quality coordinates differ")
    mse = float(torch.mean((reference.F - reconstruction.F) ** 2).item())
    return mse, psnr(mse)


@torch.no_grad()
def run_hard_gate(model, files, entries, base_lambda, output_csv):
    fields = (
        "file", "model_id", "points", "bits_xlow", "bits_r1", "bits_r2",
        "bits_r3", "bits_r4", "base_bits", "base_bpp",
        "mse_unpool", "psnr_unpool", "mse_native", "psnr_native",
        "mse_learned_soft", "psnr_learned_soft", "mse_learned_hard",
        "psnr_learned_hard", "x4_max_abs", "f4_max_abs", "d4_max_abs",
        "x5p_max_abs", "f5p_max_abs", "d5p_max_abs", "base_max_abs",
        "num_residual_streams",
    )
    dataset = PCDataset(files, color_format="yuv", normalize=True)
    rows = []
    for index in range(len(dataset)):
        coords, feats = dataset[index]
        batched_coords, batched_feats = ME.utils.sparse_collate([coords], [feats])
        attribute = ME.SparseTensor(
            features=batched_feats, coordinates=batched_coords,
            tensor_stride=1, device="cuda")
        soft_state = model.prefix(attribute, base_lambda)
        hard_state, rate = model.prefix.hard_forward(
            attribute, base_lambda, return_details=True)
        if rate["num_residual_streams"] != 4:
            raise RuntimeError("Hard prefix did not produce exactly four streams")
        soft = model.reconstruct_from_state(soft_state)
        hard = model.reconstruct_from_state(hard_state)
        baselines = model.native_baselines(soft_state)
        differences = {
            name + "_max_abs": _comparison(
                getattr(soft_state, name), getattr(hard_state, name), name)
            for name in ("x4", "f4", "d4", "x5p", "f5p", "d5p")
        }
        differences["base_max_abs"] = _comparison(
            soft["Base"], hard["Base"], "Base")
        qualities = {}
        for name, value in (
                ("unpool", baselines["B_unpool"]),
                ("native", baselines["B_native"]),
                ("learned_soft", soft["Base"]),
                ("learned_hard", hard["Base"])):
            qualities["mse_" + name], qualities["psnr_" + name] = _quality(
                attribute, value)
        residual = rate["residual_bits"]
        row = {
            "file": entries[index], "model_id": entries[index].split("/")[0],
            "points": len(attribute), "bits_xlow": rate["bits_xlow"],
            "bits_r1": residual[0], "bits_r2": residual[1],
            "bits_r3": residual[2], "bits_r4": residual[3],
            "base_bits": rate["base_bits"],
            "base_bpp": rate["base_bits"] / len(attribute),
            "num_residual_streams": rate["num_residual_streams"],
            **qualities, **differences,
        }
        rows.append(row)
        with open(output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print("hard gate {}/{} {} bpp={:.6f} max_abs={}".format(
            index + 1, len(dataset), entries[index], row["base_bpp"],
            row["base_max_abs"]), flush=True)
    total_points = sum(row["points"] for row in rows)
    bit_names = ("bits_xlow", "bits_r1", "bits_r2", "bits_r3", "bits_r4")
    summary = {
        "status": "PASS",
        "num_h5": len(rows),
        "num_models": len(set(row["model_id"] for row in rows)),
        "num_residual_streams": 4,
        "total_points": total_points,
        **{name: sum(row[name] for row in rows) for name in bit_names},
        "base_bits": sum(row["base_bits"] for row in rows),
        "base_bpp": sum(row["base_bits"] for row in rows) / total_points,
        **{name: max(row[name] for row in rows) for name in (
            "x4_max_abs", "f4_max_abs", "d4_max_abs", "x5p_max_abs",
            "f5p_max_abs", "d5p_max_abs", "base_max_abs")},
    }
    for name in ("unpool", "native", "learned_soft", "learned_hard"):
        summary["mean_psnr_" + name] = sum(
            row["psnr_" + name] for row in rows) / len(rows)
    return summary


def main():
    args = parse_args()
    for path in (args.output_csv, args.summary_json):
        if os.path.exists(path):
            raise FileExistsError("Refusing to overwrite " + path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    checkpoint = None
    if args.base_synthesis_checkpoint:
        checkpoint = torch.load(
            args.base_synthesis_checkpoint, map_location="cpu")
        config = BaseSynthesisConfig(**checkpoint["config"])
    else:
        config = BaseSynthesisConfig()
    model = CanonicalBaseModel(
        args.base_checkpoint, config, scale=args.base_scale,
        stage=args.base_stage, vmode=args.base_vmode).cuda().eval()
    if checkpoint is not None:
        model.base_synthesis.load_state_dict(
            checkpoint["base_synthesis"], strict=True)

    with open(args.file_list, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    files = h5_files(args.data_root, args.file_list)
    if args.hard_gate:
        output_dir = os.path.dirname(os.path.abspath(args.output_csv))
        with open(os.path.join(output_dir, "resolved_args.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)
        with open(os.path.join(output_dir, "command.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(shlex.join([sys.executable] + sys.argv) + "\n")
        os.chdir(output_dir)
        summary = run_hard_gate(
            model, files, entries, args.base_lambda, args.output_csv)
    else:
        summary = evaluate_base(
            model, files, entries, args.base_lambda, args.output_csv)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
