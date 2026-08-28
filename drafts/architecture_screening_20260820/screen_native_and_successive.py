#!/usr/bin/env python3
"""Cheap Family-A/Familiy-B screening; experiment-only, not codec code."""
import argparse
import copy
import csv
import json
import math
import os

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv, yuv2rgb
from data_utils.attribute.inout import read_h5, write_ply_ascii
from data_utils.sparse_tensor import sort_sparse_tensor
from scalable_attribute.base_adapter import BaseAdapter
from scripts.scalable_attribute.evaluate_unicorn_reference import metric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample", action="append", required=True)
    p.add_argument("--lambda-value", type=int, default=16384)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def sparse(features, target):
    return ME.SparseTensor(features=features,
                           coordinate_map_key=target.coordinate_map_key,
                           coordinate_manager=target.coordinate_manager,
                           device=target.device)


def load_sample(root, entry):
    coords, rgb = read_h5(os.path.join(root, entry))
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    c, f = ME.utils.sparse_collate([coords], [yuv])
    return coords, rgb, ME.SparseTensor(
        features=f, coordinates=c, tensor_stride=1, device="cuda")


@torch.no_grad()
def stage5_context(base, A, lmb):
    emb = base.embedder(lmb, device=A.device)
    x_set = {str(A.tensor_stride): A}
    x_low = A
    for pooling in base.pooling_list:
        x_low = pooling(x_low)
        x_set[str(x_low.tensor_stride)] = x_low
    x_low = sparse(torch.round(x_low.F * 255.0) / 255.0, x_low)
    curr_x = x_low
    curr_f = base.linear_in(curr_x)
    curr_dec = curr_f - curr_f
    for idx, unpooling in enumerate(base.unpooling_list):
        curr_f = base.upscaler(ME.cat([curr_f, curr_x]))
        curr_x = unpooling(curr_x)
        curr_dec = unpooling(curr_dec)
        x_gt = x_set[str(curr_x.tensor_stride)]
        prior = base.VAE.block_prior(ME.cat([curr_x, curr_f, curr_dec]))
        if idx == 4:
            loc = base.VAE.loc_net(prior)
            scale = base.VAE.scale_net(prior)
            residual = x_gt - curr_x
            latent = base.VAE.EQlayer(base.VAE.encoder(residual), emb)
            q = sparse(torch.round(latent.F), latent)
            q = sort_sparse_tensor(q, target=prior)
            return {"A": x_gt, "x_in": curr_x, "f_in": curr_f,
                    "prior_dec": curr_dec, "prior": prior, "loc": loc,
                    "scale": scale, "symbols": q, "emb": emb}
        residual = x_gt - curr_x
        latent = base.VAE.EQlayer(base.VAE.encoder(residual), emb)
        q = sparse(torch.round(latent.F), latent)
        q = sort_sparse_tensor(q, target=prior)
        curr_dec = base.VAE.decoder(base.VAE.DQlayer(q, emb))
        curr_dec = sort_sparse_tensor(curr_dec, target=curr_f)
        curr_f = base.VAE.fuseNet(curr_f + curr_dec)
        curr_x = base.VAE.outNet(curr_f) + curr_x
    raise AssertionError("missing Stage 5")


def synthesize(base, q, context):
    dec = base.VAE.decoder(base.VAE.DQlayer(q, context["emb"]))
    dec = sort_sparse_tensor(dec, target=context["f_in"])
    f_out = base.VAE.fuseNet(context["f_in"] + dec)
    return base.VAE.outNet(f_out) + context["x_in"]


def tensor_psnr(A, B):
    mse = torch.mean((A.F - B.F).square()).item()
    return -10.0 * math.log10(max(mse, 1e-12))


def author_psnr(output_dir, label, coords, rgb, reconstruction):
    directory = os.path.join(output_dir, "metric_tmp", label)
    os.makedirs(directory, exist_ok=True)
    gt = os.path.join(directory, "gt.ply")
    rec = os.path.join(directory, "rec.ply")
    write_ply_ascii(gt, coords, rgb)
    rec_rgb = yuv2rgb(torch.clamp(
        reconstruction.F.detach().cpu(), 0, 1), out_range=255)
    rec_rgb = np.clip(rec_rgb.round().int().numpy(), 0, 255)
    write_ply_ascii(rec, reconstruction.C[:, 1:].cpu().numpy(), rec_rgb)
    return metric(gt, rec)["yuv_psnr_611"]


def channel_entropy_bits(symbols):
    values = symbols.detach().cpu().numpy().astype(np.int64)
    bits = 0.0
    for channel in range(values.shape[1]):
        _, counts = np.unique(values[:, channel], return_counts=True)
        probabilities = counts.astype(np.float64) / counts.sum()
        bits += float((-counts * np.log2(probabilities)).sum())
    return bits


def continuation(base, context, mode, predictor=None):
    if mode == "zero":
        features = torch.zeros_like(context["loc"].F)
    elif mode == "round_loc":
        features = torch.round(context["loc"].F)
    elif mode == "continuous_loc":
        features = context["loc"].F
    elif mode == "learned":
        features = predictor(context["prior"]).F
    else:
        raise ValueError(mode)
    return synthesize(base, sparse(features, context["prior"]), context)


def evaluate_modes(base, predictor, args):
    rows = []
    for index, entry in enumerate(args.sample):
        coords, rgb, A = load_sample(args.data_root, entry)
        context = stage5_context(base, A, args.lambda_value)
        full = synthesize(base, context["symbols"], context)
        for mode in ("zero", "round_loc", "continuous_loc", "learned"):
            reconstruction = continuation(base, context, mode, predictor)
            rows.append({
                "sample": entry, "split": "micro_train" if index < 2 else "holdout",
                "mode": mode, "points": len(A),
                "tensor_psnr": tensor_psnr(A, reconstruction),
                "author_yuv_psnr_611": author_psnr(
                    args.output_dir, "{}_{}".format(index, mode),
                    coords, rgb, reconstruction),
                "full_tensor_psnr": tensor_psnr(A, full),
                "full_author_yuv_psnr_611": author_psnr(
                    args.output_dir, "{}_full".format(index), coords, rgb, full),
            })
        del A, context, full
        torch.cuda.empty_cache()
    return rows


def successive_oracle(base, args):
    rows = []
    for entry in args.sample:
        _, _, A = load_sample(args.data_root, entry)
        context = stage5_context(base, A, args.lambda_value)
        original = context["symbols"].F
        original_bits = channel_entropy_bits(original)
        full = synthesize(base, context["symbols"], context)
        for step in (2, 4, 8):
            coarse = torch.round(original / step) * step
            refinement = original - coarse
            reconstruction = synthesize(
                base, sparse(coarse, context["symbols"]), context)
            coarse_bits = channel_entropy_bits(coarse)
            refinement_bits = channel_entropy_bits(refinement)
            rows.append({
                "sample": entry, "step": step, "points": len(A),
                "original_ideal_bpp": original_bits / len(A),
                "coarse_ideal_bpp": coarse_bits / len(A),
                "refinement_ideal_bpp": refinement_bits / len(A),
                "layered_ideal_bpp": (coarse_bits + refinement_bits) / len(A),
                "layered_over_original_ratio": (
                    (coarse_bits + refinement_bits) / max(original_bits, 1e-12)),
                "coarse_nonzero_fraction": float((coarse != 0).float().mean()),
                "refinement_nonzero_fraction": float((refinement != 0).float().mean()),
                "coarse_tensor_psnr": tensor_psnr(A, reconstruction),
                "full_tensor_psnr": tensor_psnr(A, full),
                "remaining_gap_db": tensor_psnr(A, full) - tensor_psnr(A, reconstruction),
            })
        del A, context, full
        torch.cuda.empty_cache()
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = BaseAdapter(args.checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    base = adapter.base
    predictor = copy.deepcopy(base.VAE.loc_net).cuda().train()
    predictor.requires_grad_(True)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    training = []
    for step in range(args.steps + 1):
        if step in (0, 20, 50, 100):
            training.append({"step": step, "last_loss": None if not training else training[-1].get("last_loss")})
        if step == args.steps:
            break
        entry = args.sample[step % 2]
        _, _, A = load_sample(args.data_root, entry)
        with torch.no_grad():
            context = stage5_context(base, A, args.lambda_value)
        optimizer.zero_grad(set_to_none=True)
        reconstruction = continuation(base, context, "learned", predictor)
        loss = torch.mean((A.F - reconstruction.F).square())
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite learned continuation loss")
        training[-1]["last_loss"] = float(loss.item())
        del A, context, reconstruction, loss
    predictor.eval()
    continuation_rows = evaluate_modes(base, predictor, args)
    successive_rows = successive_oracle(base, args)
    write_csv(os.path.join(args.output_dir, "family_a_continuations.csv"), continuation_rows)
    write_csv(os.path.join(args.output_dir, "family_b_successive_oracle.csv"), successive_rows)
    torch.save({"predictor": predictor.state_dict(), "steps": args.steps,
                "lr": args.lr, "lambda": args.lambda_value},
               os.path.join(args.output_dir, "family_a_capacity_predictor.pth"))
    with open(os.path.join(args.output_dir, "family_a_training.json"), "w", encoding="utf-8") as handle:
        json.dump(training, handle, indent=2)
        handle.write("\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
