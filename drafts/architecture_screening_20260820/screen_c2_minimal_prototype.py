#!/usr/bin/env python3
"""Matched C1-like/C2 appended ResidualVAE capacity probe; experiment-only."""
import argparse
import csv
import json
import math
import os
import time

import MinkowskiEngine as ME
import numpy as np
import torch

import screen_native_and_successive as common
from data_utils.sparse_tensor import sort_sparse_tensor
from scalable_attribute.base_adapter import BaseAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--lambda-value", type=int, default=128)
    parser.add_argument("--rd-lambda", type=float, default=6500.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant", action="append", choices=("b_only", "native_state"))
    return parser.parse_args()


def sparse_like(features, target):
    return ME.SparseTensor(
        features=features,
        coordinate_map_key=target.coordinate_map_key,
        coordinate_manager=target.coordinate_manager,
        device=target.device)


@torch.no_grad()
def base_states(adapter, A, lmb):
    captured = []

    def capture_last_dec(_module, _inputs, output):
        captured.append(output["dec"])

    handle = adapter.base.VAE.register_forward_hook(capture_last_dec)
    try:
        B, F_U = adapter(A, lmb)
    finally:
        handle.remove()
    if len(captured) != 5:
        raise RuntimeError("Expected five native VAE calls, got {}".format(len(captured)))
    D_U = captured[-1]
    if not torch.equal(B.C, D_U.C):
        D_U = sort_sparse_tensor(D_U, target=B)
    if not torch.equal(B.C, F_U.C) or not torch.equal(B.C, D_U.C):
        raise RuntimeError("B/F_U/D_U support mismatch")
    return B, F_U, D_U


def condition_states(variant, B, F_U, D_U):
    if variant == "b_only":
        return sparse_like(torch.zeros_like(F_U.F), F_U), sparse_like(
            torch.zeros_like(D_U.F), D_U)
    if variant == "native_state":
        return F_U, D_U
    raise ValueError(variant)


def raw_symbols(vae, A, B, emb):
    latent = vae.encoder(A - B)
    latent = vae.EQlayer(latent, emb)
    symbols = torch.round(latent.F)
    return latent, symbols


def zero_reconstruction(vae, latent, B, f_in, emb):
    q = sparse_like(torch.zeros_like(latent.F), latent)
    q = vae.DQlayer(q, emb)
    dec = vae.decoder(q)
    dec = sort_sparse_tensor(dec, target=f_in)
    f_out = vae.fuseNet(f_in + dec)
    return vae.outNet(f_out) + B


@torch.no_grad()
def deterministic_diagnostic(adapter, vae, variant, args, entry, hard=False):
    _, _, A = common.load_sample(args.data_root, entry)
    B, F_U, D_U = base_states(adapter, A, args.lambda_value)
    f_in, prior_dec = condition_states(variant, B, F_U, D_U)
    emb = adapter.base.embedder(args.lambda_value, device=A.device)
    latent, symbols = raw_symbols(vae, A, B, emb)
    output = vae(
        x_in=B, x_gt=A, f_in=f_in, prior_dec=prior_dec,
        training=False, emb=emb, real_coding=False)
    estimated_bpp = float(
        (-torch.log2(output["likelihood"].clamp(min=1e-9))).sum().item()
        / len(A))
    full_zero = zero_reconstruction(vae, latent, B, f_in, emb)
    row = {
        "variant": variant,
        "sample": entry,
        "points": len(A),
        "estimated_bpp": estimated_bpp,
        "symbol_nonzero_fraction": float((symbols != 0).float().mean()),
        "symbol_nonzero_count": int((symbols != 0).sum().item()),
        "symbol_mean_abs": float(symbols.abs().mean().item()),
        "symbol_min": int(symbols.min().item()),
        "symbol_max": int(symbols.max().item()),
        "base_tensor_psnr": common.tensor_psnr(A, B),
        "full_tensor_psnr": common.tensor_psnr(A, output["x_out"]),
        "full_zero_tensor_psnr": common.tensor_psnr(A, full_zero),
        "hard_bits": "",
        "hard_bpp": "",
        "hard_decode_max_abs_diff": "",
    }
    if hard:
        encoded = vae.encode(B, A, f_in, prior_dec, emb)
        decoded = vae.decode(
            encoded["strings"], encoded["min_v"], encoded["max_v"],
            B, f_in, prior_dec, emb)
        row["hard_bits"] = len(encoded["strings"]) * 8
        row["hard_bpp"] = row["hard_bits"] / len(A)
        row["hard_decode_max_abs_diff"] = float(
            (decoded["x_out"].F - encoded["x_out"].F).abs().max().item())
        row["full_tensor_psnr"] = common.tensor_psnr(A, decoded["x_out"])
    return row


def train_variant(adapter, initial_vae, variant, args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    vae = initial_vae.__class__(stride=[2, 2, 2])
    vae.load_state_dict(initial_vae.state_dict())
    vae = vae.cuda().train()
    vae.requires_grad_(True)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    trace = []
    milestones = {0, 20, 50, 100, args.steps}
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for step in range(args.steps + 1):
        if step in milestones:
            row = deterministic_diagnostic(
                adapter, vae.eval(), variant, args, args.sample[0], hard=False)
            row["step"] = step
            row["distortion"] = 10.0 ** (-row["full_tensor_psnr"] / 10.0)
            row["objective"] = (
                row["estimated_bpp"] + args.rd_lambda * row["distortion"])
            trace.append(row)
            vae.train()
        if step == args.steps:
            break
        entry = args.sample[step % 2]
        _, _, A = common.load_sample(args.data_root, entry)
        with torch.no_grad():
            B, F_U, D_U = base_states(adapter, A, args.lambda_value)
            f_in, prior_dec = condition_states(variant, B, F_U, D_U)
            emb = adapter.base.embedder(args.lambda_value, device=A.device)
        optimizer.zero_grad(set_to_none=True)
        output = vae(
            x_in=B, x_gt=A, f_in=f_in, prior_dec=prior_dec,
            training=True, emb=emb, real_coding=False)
        rate = (-torch.log2(output["likelihood"].clamp(min=1e-9))).sum() / len(A)
        distortion = torch.mean((A.F - output["x_out"].F).square())
        loss = rate + args.rd_lambda * distortion
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss at {} step {}".format(variant, step))
        loss.backward()
        for name, parameter in vae.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    "non-finite gradient {} at {} step {}".format(name, variant, step))
        optimizer.step()
        del A, B, F_U, D_U, f_in, prior_dec, output, rate, distortion, loss
    elapsed = time.time() - start
    final_rows = [
        deterministic_diagnostic(
            adapter, vae.eval(), variant, args, entry, hard=True)
        for entry in args.sample
    ]
    stats = {
        "variant": variant,
        "trainable_parameters": sum(p.numel() for p in vae.parameters()),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_update": elapsed / max(args.steps, 1),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    return vae, trace, final_rows, stats


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = BaseAdapter(args.checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    initial_vae = adapter.base.VAE
    all_trace = []
    all_final = []
    all_stats = []
    for variant in (args.variant or ("b_only", "native_state")):
        vae, trace, final_rows, stats = train_variant(
            adapter, initial_vae, variant, args)
        all_trace.extend(trace)
        all_final.extend(final_rows)
        all_stats.append(stats)
        torch.save(
            {"model": vae.state_dict(), "variant": variant, "args": vars(args)},
            os.path.join(args.output_dir, "c2_{}.pth".format(variant)))
        del vae
        torch.cuda.empty_cache()
    write_csv(os.path.join(args.output_dir, "c2_training_trace.csv"), all_trace)
    write_csv(os.path.join(args.output_dir, "c2_hard_diagnostic.csv"), all_final)
    with open(os.path.join(args.output_dir, "c2_runtime.json"), "w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, indent=2)
        handle.write("\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
