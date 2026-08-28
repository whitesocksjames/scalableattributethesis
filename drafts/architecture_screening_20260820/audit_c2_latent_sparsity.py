#!/usr/bin/env python3
"""Read-only per-H5/per-channel audit of a C2 probe checkpoint."""
import argparse
import csv
import json
import os

import numpy as np
import torch

import screen_native_and_successive as common
from scalable_attribute.base_adapter import BaseAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--c2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--lambda-value", type=int, default=128)
    return parser.parse_args()


def statistics(values):
    nonzero = values != 0
    absolute = values.abs()
    return {
        "symbol_count": values.numel(),
        "nonzero_count": int(nonzero.sum().item()),
        "nonzero_fraction": float(nonzero.float().mean().item()),
        "mean_abs": float(absolute.float().mean().item()),
        "mean_abs_nonzero": float(absolute[nonzero].float().mean().item())
        if nonzero.any() else 0.0,
        "abs_eq_1": int((absolute == 1).sum().item()),
        "abs_eq_2": int((absolute == 2).sum().item()),
        "abs_ge_3": int((absolute >= 3).sum().item()),
        "min": int(values.min().item()),
        "max": int(values.max().item()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = BaseAdapter(args.base_checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    vae = adapter.base.VAE.__class__(stride=[2, 2, 2]).cuda().eval()
    state = torch.load(args.c2_checkpoint, map_location="cpu")
    vae.load_state_dict(state["model"])
    per_h5 = []
    channel_values = {"micro_train": [], "holdout": [], "all": []}
    with torch.no_grad():
        for index, entry in enumerate(args.sample):
            split = "micro_train" if index < 2 else "holdout"
            _, _, A = common.load_sample(args.data_root, entry)
            B, _ = adapter(A, args.lambda_value)
            emb = adapter.embedding(args.lambda_value, A.device)
            latent = vae.EQlayer(vae.encoder(A - B), emb)
            symbols = torch.round(latent.F).to(torch.int64).cpu()
            per_h5.append({"sample": entry, "split": split, **statistics(symbols)})
            channel_values[split].append(symbols)
            channel_values["all"].append(symbols)
    per_channel = []
    for split, tensors in channel_values.items():
        values = torch.cat(tensors, dim=0)
        for channel in range(values.shape[1]):
            per_channel.append({
                "split": split,
                "channel": channel,
                **statistics(values[:, channel]),
            })
    for name, rows in (("c2_sparsity_per_h5.csv", per_h5),
                       ("c2_sparsity_per_channel.csv", per_channel)):
        with open(os.path.join(args.output_dir, name), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {}
    for split, tensors in channel_values.items():
        values = torch.cat(tensors, dim=0)
        active_channels = sum(
            bool((values[:, channel] != 0).any())
            for channel in range(values.shape[1]))
        summary[split] = {
            **statistics(values),
            "active_channels": active_channels,
            "channels": values.shape[1],
        }
    with open(os.path.join(args.output_dir, "c2_sparsity_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
