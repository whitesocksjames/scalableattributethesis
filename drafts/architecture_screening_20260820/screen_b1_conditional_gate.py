#!/usr/bin/env python3
"""B1 conditional-rate legality gate; experiment-only, no codec changes."""
import argparse
import csv
import os

import torch

import screen_native_and_successive as common
from scalable_attribute.base_adapter import BaseAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--lambda-value", type=int, default=16384)
    return parser.parse_args()


def nested_symbols(fine, step):
    # Nearest coarse lattice with one deterministic half-open cell:
    # e in [-step/2, ..., step/2-1]. This avoids round-to-even ambiguity.
    coarse_index = torch.div(
        fine.to(torch.int64) + step // 2, step, rounding_mode="floor")
    coarse = coarse_index * step
    refinement = fine.to(torch.int64) - coarse
    return coarse.to(fine.dtype), refinement.to(fine.dtype)


def conditional_rates(entropy, fine, loc, scale, step, points):
    scale = scale.abs().clamp(min=1e-8)
    native_probability = entropy._likelihood(fine, loc, scale).clamp(min=1e-9)
    coarse, refinement = nested_symbols(fine, step)
    offsets = torch.arange(
        -step // 2, step // 2, device=fine.device, dtype=fine.dtype)
    cell_values = coarse.unsqueeze(-1) + offsets
    cell_probability = entropy._likelihood(
        cell_values, loc.unsqueeze(-1), scale.unsqueeze(-1)).sum(-1)
    cell_probability = cell_probability.clamp(min=1e-9)
    refinement_probability = (
        native_probability / cell_probability).clamp(min=1e-9, max=1.0)
    native_bpp = float((-torch.log2(native_probability)).sum().item() / points)
    coarse_bpp = float((-torch.log2(cell_probability)).sum().item() / points)
    refinement_bpp = float(
        (-torch.log2(refinement_probability)).sum().item() / points)
    return coarse, refinement, native_bpp, coarse_bpp, refinement_bpp


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = BaseAdapter(args.checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    base = adapter.base
    rows = []
    for sample_index, entry in enumerate(args.sample):
        coords, rgb, A = common.load_sample(args.data_root, entry)
        context = common.stage5_context(base, A, args.lambda_value)
        fine = context["symbols"].F
        loc = context["loc"].F
        scale = context["scale"].F
        full = common.synthesize(base, context["symbols"], context)
        for step in (2, 4, 8):
            coarse, refinement, native_bpp, coarse_bpp, refinement_bpp = (
                conditional_rates(
                    base.VAE.entropy_fn, fine, loc, scale, step, len(A)))
            coarse_sparse = common.sparse(coarse, context["symbols"])
            reconstruction = common.synthesize(base, coarse_sparse, context)
            rows.append({
                "sample": entry,
                "step": step,
                "points": len(A),
                "native_conditional_bpp": native_bpp,
                "coarse_conditional_bpp": coarse_bpp,
                "refinement_conditional_bpp": refinement_bpp,
                "layered_conditional_bpp": coarse_bpp + refinement_bpp,
                "layered_over_native_ratio": (
                    (coarse_bpp + refinement_bpp) / max(native_bpp, 1e-12)),
                "coarse_nonzero_fraction": float((coarse != 0).float().mean()),
                "refinement_nonzero_fraction": float(
                    (refinement != 0).float().mean()),
                "refinement_mean_abs": float(refinement.abs().float().mean()),
                "refinement_min": int(refinement.min().item()),
                "refinement_max": int(refinement.max().item()),
                "coarse_tensor_psnr": common.tensor_psnr(A, reconstruction),
                "full_tensor_psnr": common.tensor_psnr(A, full),
                "coarse_author_yuv_psnr_611": common.author_psnr(
                    args.output_dir,
                    "{}_coarse_s{}".format(sample_index, step),
                    coords, rgb, reconstruction),
                "full_author_yuv_psnr_611": common.author_psnr(
                    args.output_dir,
                    "{}_full".format(sample_index), coords, rgb, full),
            })
        del A, context, full
        torch.cuda.empty_cache()
    output = os.path.join(args.output_dir, "b1_conditional_rate_gate.csv")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
