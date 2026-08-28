#!/usr/bin/env python3
"""Experiment-only C3 stride-4 capacity probe; not production codec code."""
import argparse
import csv
import json
import math
import os

import MinkowskiEngine as ME
import numpy as np
import torch

import screen_native_and_successive as common
from basic_models.backbone import Backbone
from data_utils.sparse_tensor import sort_sparse_tensor
from scalable_attribute.base_adapter import BaseAdapter
from scalable_attribute.entropy import EnhancementEntropy


SAMPLES = (
    "RWT115/model_mesh_P0.h5",
    "RWT182/572883_P15.h5",
    "RWT380/ujety_svah_ske_P15.h5",
    "RWT541/marco_cat_mesh_P9.h5",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-value", type=int, default=256)
    parser.add_argument("--rd-lambda", type=float, default=6500.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sparse(features, target):
    return ME.SparseTensor(
        features=features,
        coordinate_map_key=target.coordinate_map_key,
        coordinate_manager=target.coordinate_manager,
        device=target.device,
    )


def align(source, target, label):
    source = sort_sparse_tensor(source, target=target)
    if len(source) != len(target) or not torch.equal(source.C, target.C):
        raise RuntimeError(label + " coordinate support mismatch")
    return source


class Stride4Probe(torch.nn.Module):
    """One coarse latent, with native Base state available only as condition."""

    def __init__(self):
        super().__init__()
        self.analysis = Backbone(
            scale=2, in_channels=3, channels=128, out_channels=64,
            block_type="resnet", block_layers=2)
        self.prior = Backbone(
            scale=2, in_channels=259, channels=128, out_channels=128,
            block_type="resnet", block_layers=2)
        self.mu = ME.MinkowskiLinear(128, 64)
        self.sigma = ME.MinkowskiLinear(128, 64)
        self.entropy = EnhancementEntropy()
        self.synthesis = Backbone(
            scale=-2, in_channels=64, channels=128, out_channels=128,
            block_type="resnet", block_layers=2)
        self.fusion = Backbone(
            scale=0, in_channels=387, channels=128, out_channels=128,
            block_type="resnet", block_layers=2)
        self.out = Backbone(
            scale=0, in_channels=128, channels=128, out_channels=3,
            block_type="linear", block_layers=2)

    def condition(self, B, F_U, D_U, target=None):
        prior = self.prior(ME.cat([B, F_U, D_U]))
        mu, sigma = self.mu(prior), self.sigma(prior)
        if target is not None:
            mu = align(mu, target, "C3 mu")
            sigma = align(sigma, target, "C3 sigma")
        return mu, sigma

    def correction(self, y_hat, B, F_U, D_U):
        decoded = align(self.synthesis(y_hat), B, "C3 synthesis")
        return self.out(self.fusion(ME.cat([decoded, B, F_U, D_U])))

    def reconstruct(self, y_hat, B, F_U, D_U):
        delta_y = self.correction(y_hat, B, F_U, D_U)
        zero = sparse(torch.zeros_like(y_hat.F), y_hat)
        delta_0 = self.correction(zero, B, F_U, D_U)
        return B + delta_y - delta_0

    def forward(self, A, B, F_U, D_U):
        y = self.analysis(A - B)
        mu, sigma = self.condition(B, F_U, D_U, y)
        y_hat, likelihood = self.entropy(y, mu, sigma)
        return y, y_hat, likelihood, self.reconstruct(y_hat, B, F_U, D_U)

    @torch.no_grad()
    def hard(self, A, B, F_U, D_U):
        y = self.analysis(A - B)
        mu, sigma = self.condition(B, F_U, D_U, y)
        y_hat, encoded = self.entropy.encode(y, mu, sigma)
        decoded = self.entropy.decode(encoded, mu, sigma)
        if not torch.equal(y_hat.F, decoded.F):
            raise RuntimeError("C3 torchac symbol round-trip mismatch")
        full = self.reconstruct(decoded, B, F_U, D_U)
        full_zero = self.reconstruct(
            sparse(torch.zeros_like(decoded.F), decoded), B, F_U, D_U)
        return y, decoded, full, full_zero, len(encoded.strings) * 8


def psnr(A, reconstruction):
    mse = torch.mean((A.F - reconstruction.F).square()).item()
    return -10.0 * math.log10(max(mse, 1e-12))


def load_state(adapter, root, entry, lmb):
    _, _, A = common.load_sample(root, entry)
    with torch.no_grad():
        B, F_U, D_U = adapter.forward_state(A, lmb)
    return A, B, F_U, D_U


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = BaseAdapter(args.checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    adapter.requires_grad_(False)
    probe = Stride4Probe().cuda().train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    trajectory = []
    record_steps = {1, 20, 50, 100, args.steps}
    for step in range(args.steps):
        A, B, F_U, D_U = load_state(
            adapter, args.data_root, SAMPLES[step % 2], args.lambda_value)
        optimizer.zero_grad(set_to_none=True)
        y, _, likelihood, full = probe(A, B, F_U, D_U)
        rate = -torch.log2(likelihood.clamp(min=1e-9)).sum() / len(A)
        distortion = torch.mean((A.F - full.F).square())
        loss = rate + args.rd_lambda * distortion
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite C3 loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in probe.parameters()):
            raise RuntimeError("non-finite C3 gradient")
        optimizer.step()
        if step + 1 in record_steps:
            trajectory.append({
                "step": step + 1,
                "loss": float(loss.item()),
                "estimated_el_bpp": float(rate.item()),
                "distortion": float(distortion.item()),
                "raw_y_mean_abs": float(y.F.abs().mean().item()),
            })
        del A, B, F_U, D_U, y, likelihood, full, loss

    probe.eval()
    rows = []
    for index, entry in enumerate(SAMPLES):
        A, B, F_U, D_U = load_state(
            adapter, args.data_root, entry, args.lambda_value)
        y, q, full, full_zero, bits = probe.hard(A, B, F_U, D_U)
        nonzero = int((q.F != 0).sum().item())
        rows.append({
            "sample": entry,
            "split": "micro_train" if index < 2 else "holdout",
            "points": len(A),
            "latent_points": len(q),
            "symbol_count": q.F.numel(),
            "nonzero_count": nonzero,
            "nonzero_fraction": nonzero / q.F.numel(),
            "symbol_mean_abs": float(q.F.abs().mean().item()),
            "symbol_min": int(q.F.min().item()),
            "symbol_max": int(q.F.max().item()),
            "hard_el_bits": bits,
            "hard_el_bpp": bits / len(A),
            "base_tensor_psnr": psnr(A, B),
            "full_tensor_psnr": psnr(A, full),
            "full_zero_tensor_psnr": psnr(A, full_zero),
            "full_minus_base_db": psnr(A, full) - psnr(A, B),
        })
        del A, B, F_U, D_U, y, q, full, full_zero
    with open(os.path.join(args.output_dir, "c3_hard_diagnostic.csv"), "w",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.output_dir, "c3_training.json"), "w",
              encoding="utf-8") as handle:
        json.dump(trajectory, handle, indent=2)
        handle.write("\n")
    torch.save({"model": probe.state_dict(), "args": vars(args)},
               os.path.join(args.output_dir, "c3_stride4_probe.pth"))
    print(args.output_dir)


if __name__ == "__main__":
    main()
