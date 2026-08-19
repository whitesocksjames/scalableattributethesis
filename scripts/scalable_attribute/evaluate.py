#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shlex
import sys

import torch
import MinkowskiEngine as ME

from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.coder import ScalableAttributeCoder
from scalable_attribute.config import EnhancementConfig
from scalable_attribute.data import UncachedPCDataset, h5_files
from scalable_attribute.model import ScalableAttributeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--enhancement-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    parser.add_argument("--gpcc-binary", required=True)
    return parser.parse_args()


def mse(lhs, rhs):
    if not torch.equal(lhs.C, rhs.C):
        raise RuntimeError("Metric coordinate mismatch")
    return float(torch.mean((lhs.F - rhs.F) ** 2).item())


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")

    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.exists(gpcc_link):
        os.symlink(os.path.abspath(args.gpcc_binary), gpcc_link)
    os.chdir(args.output_dir)

    state = torch.load(args.enhancement_checkpoint, map_location="cpu")
    config = EnhancementConfig(**state["config"])
    model = ScalableAttributeModel(
        args.base_checkpoint,
        config,
        base_scale=args.base_scale,
        base_stage=args.base_stage,
        base_vmode=args.base_vmode,
    ).cuda().eval()
    model.enhancement.load_state_dict(state["enhancement"])
    coder = ScalableAttributeCoder(model)

    files = h5_files(args.data_root, args.file_list)
    if args.max_samples:
        files = files[:args.max_samples]
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    data_loader = make_data_loader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    fields = [
        "sample", "points", "base_bits", "el_bits", "full_bits",
        "R_base", "R_E", "R_full", "base_mse", "full_mse",
        "base_psnr", "full_psnr",
    ]
    with open("metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for file_path, (coords, feats) in zip(files, data_loader):
            A = ME.SparseTensor(
                features=feats, coordinates=coords, tensor_stride=1, device="cuda")
            result = coder.test(A, args.base_lambda)
            base_mse = mse(A, result["B"])
            full_mse = mse(A, result["Full"])
            writer.writerow({
                "sample": file_path,
                "points": len(A),
                "base_bits": result["base_bits"],
                "el_bits": result["el_bits"],
                "full_bits": result["full_bits"],
                "R_base": result["R_base"],
                "R_E": result["R_E"],
                "R_full": result["R_full"],
                "base_mse": base_mse,
                "full_mse": full_mse,
                "base_psnr": -10 * math.log10(max(base_mse, 1e-12)),
                "full_psnr": -10 * math.log10(max(full_mse, 1e-12)),
            })


if __name__ == "__main__":
    main()
