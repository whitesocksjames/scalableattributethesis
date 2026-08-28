#!/usr/bin/env python3
"""No-training residual spatial/quantization oracle for Family C."""
import argparse
import csv
import math
import os

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_h5
from scalable_attribute.base_adapter import BaseAdapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample", action="append", required=True)
    p.add_argument("--lambda-value", type=int, default=128)
    return p.parse_args()


def load_sample(root, entry):
    coords, rgb = read_h5(os.path.join(root, entry))
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    c, f = ME.utils.sparse_collate([coords], [yuv])
    A = ME.SparseTensor(features=f, coordinates=c,
                        tensor_stride=1, device="cuda")
    return coords.astype(np.int64), A


def spatial_representation(coords, residual, factor):
    if factor == 1:
        inverse = np.arange(len(residual))
        return residual.copy(), residual.copy(), inverse
    parent = np.floor_divide(coords, factor)
    _, inverse = np.unique(parent, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    means = np.stack([
        np.bincount(inverse, weights=residual[:, channel]) / counts
        for channel in range(residual.shape[1])], axis=1)
    return means[inverse], means, inverse


def entropy_bpp(symbols, full_points):
    bits = 0.0
    for channel in range(symbols.shape[1]):
        _, counts = np.unique(symbols[:, channel], return_counts=True)
        probability = counts.astype(np.float64) / counts.sum()
        bits += float((-counts * np.log2(probability)).sum())
    return bits / full_points


def psnr(mse):
    return -10.0 * math.log10(max(mse, 1e-12))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = BaseAdapter(args.checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    rows = []
    for entry in args.sample:
        coords, A = load_sample(args.data_root, entry)
        B, _ = adapter(A, args.lambda_value)
        residual = (A.F - B.F).detach().cpu().numpy().astype(np.float64)
        base_mse = float(np.mean(residual ** 2))
        for factor in (1, 2, 4):
            spatial, parent_values, inverse = spatial_representation(
                coords, residual, factor)
            unquantized_mse = float(np.mean((residual - spatial) ** 2))
            rows.append({
                "sample": entry, "spatial_factor": factor,
                "quant_step_255": 0, "points": len(A),
                "spatial_points": len(parent_values),
                "ideal_entropy_bpp": "",
                "base_mse": base_mse, "oracle_mse": unquantized_mse,
                "base_tensor_psnr": psnr(base_mse),
                "oracle_tensor_psnr": psnr(unquantized_mse),
                "delta_psnr": psnr(unquantized_mse) - psnr(base_mse),
                "distortion_removed_fraction": 1.0 - unquantized_mse / base_mse,
            })
            for q255 in (1, 2, 4, 8):
                q = q255 / 255.0
                symbols = np.rint(parent_values / q).astype(np.int64)
                quantized = (symbols.astype(np.float64) * q)[inverse]
                mse = float(np.mean((residual - quantized) ** 2))
                rows.append({
                    "sample": entry, "spatial_factor": factor,
                    "quant_step_255": q255, "points": len(A),
                    "spatial_points": len(parent_values),
                    "ideal_entropy_bpp": entropy_bpp(symbols, len(A)),
                    "base_mse": base_mse, "oracle_mse": mse,
                    "base_tensor_psnr": psnr(base_mse),
                    "oracle_tensor_psnr": psnr(mse),
                    "delta_psnr": psnr(mse) - psnr(base_mse),
                    "distortion_removed_fraction": 1.0 - mse / base_mse,
                })
        del A, B
        torch.cuda.empty_cache()
    path = os.path.join(args.output_dir, "family_c_residual_oracle.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
