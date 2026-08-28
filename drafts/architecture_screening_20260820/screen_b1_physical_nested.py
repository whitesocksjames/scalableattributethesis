#!/usr/bin/env python3
"""Isolated actual-torchac B1 smoke over captured R02/r5 symbols."""
import argparse
import csv
import os

import torch
import torchac

import screen_native_and_successive as common
from scalable_attribute.base_adapter import BaseAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--lambda-value", type=int, default=16384)
    parser.add_argument("--step", type=int, default=2)
    return parser.parse_args()


def cdf(pmf):
    pmf = pmf.clamp(min=1e-9)
    cumulative = pmf.cumsum(dim=-1).clamp(max=1.0)
    return torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative], dim=-1)


def nested_encode(entropy, fine, loc, scale, step):
    fine = fine.to(torch.int64)
    half = step // 2
    coarse_index = torch.div(fine + half, step, rounding_mode="floor")
    refinement = fine - coarse_index * step
    c_min = int(coarse_index.min().item())
    c_max = int(coarse_index.max().item())
    candidates = torch.arange(
        c_min, c_max + 1, device=fine.device, dtype=torch.float32)
    offsets = torch.arange(-half, half, device=fine.device, dtype=torch.float32)
    fine_candidates = candidates[None, None, :, None] * step + offsets[None, None, None, :]
    coarse_pmf = entropy._likelihood(
        fine_candidates, loc[..., None, None], scale[..., None, None]).sum(-1)
    coarse_cdf = cdf(coarse_pmf).cpu()
    coarse_values = (coarse_index - c_min).to(torch.int16).cpu()
    coarse_string = torchac.encode_float_cdf(
        coarse_cdf, coarse_values, check_input_bounds=True)
    coarse_decoded = torchac.decode_float_cdf(
        coarse_cdf, coarse_string).to(fine.device).to(torch.int64) + c_min
    if not torch.equal(coarse_decoded, coarse_index):
        raise RuntimeError("coarse torchac round-trip mismatch")

    decoded_coarse = coarse_decoded * step
    refinement_candidates = (
        decoded_coarse[..., None].float() + offsets[None, None, :])
    refinement_pmf = entropy._likelihood(
        refinement_candidates, loc[..., None], scale[..., None])
    refinement_pmf = refinement_pmf / refinement_pmf.sum(-1, keepdim=True).clamp(min=1e-9)
    refinement_cdf = cdf(refinement_pmf).cpu()
    refinement_values = (refinement + half).to(torch.int16).cpu()
    refinement_string = torchac.encode_float_cdf(
        refinement_cdf, refinement_values, check_input_bounds=True)
    refinement_decoded = torchac.decode_float_cdf(
        refinement_cdf, refinement_string).to(fine.device).to(torch.int64) - half
    recovered = decoded_coarse + refinement_decoded
    if not torch.equal(recovered, fine):
        raise RuntimeError("nested fine-symbol round-trip mismatch")
    return coarse_string, refinement_string


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = BaseAdapter(args.checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    base = adapter.base
    rows = []
    for entry in args.sample:
        _, _, A = common.load_sample(args.data_root, entry)
        context = common.stage5_context(base, A, args.lambda_value)
        fine = context["symbols"].F
        loc = context["loc"].F
        scale = context["scale"].F.abs().clamp(min=1e-8)
        original, _, _ = base.VAE.entropy_fn.compress(fine, loc, scale)
        coarse, refinement = nested_encode(
            base.VAE.entropy_fn, fine, loc, scale, args.step)
        original_bits = len(original) * 8
        coarse_bits = len(coarse) * 8
        refinement_bits = len(refinement) * 8
        rows.append({
            "sample": entry,
            "points": len(A),
            "step": args.step,
            "original_bits": original_bits,
            "coarse_bits": coarse_bits,
            "refinement_bits": refinement_bits,
            "layered_bits": coarse_bits + refinement_bits,
            "original_bpp": original_bits / len(A),
            "coarse_bpp": coarse_bits / len(A),
            "refinement_bpp": refinement_bits / len(A),
            "layered_bpp": (coarse_bits + refinement_bits) / len(A),
            "layered_over_original_ratio": (
                (coarse_bits + refinement_bits) / original_bits),
        })
    path = os.path.join(args.output_dir, "b1_physical_nested.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
