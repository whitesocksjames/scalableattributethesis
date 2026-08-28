#!/usr/bin/env python3
"""Experiment-only B1 physical Base/Full RD completion; no training."""
import argparse
import csv
import os

import MinkowskiEngine as ME
import torch
import torchac

import screen_native_and_successive as common
from data_utils.sparse_tensor import array2vector, sort_sparse_tensor
from scalable_attribute.base_adapter import BaseAdapter


SAMPLES = (
    "RWT115/model_mesh_P0.h5",
    "RWT182/572883_P15.h5",
    "RWT380/ujety_svah_ske_P15.h5",
    "RWT541/marco_cat_mesh_P9.h5",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--r02-checkpoint", required=True)
    parser.add_argument("--r04-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def cdf(pmf):
    pmf = pmf.clamp(min=1e-9)
    cumulative = pmf.cumsum(dim=-1).clamp(max=1.0)
    return torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative], dim=-1)


def nested_roundtrip(entropy, fine, loc, scale, step):
    fine = fine.to(torch.int64)
    half = step // 2
    coarse_index = torch.div(fine + half, step, rounding_mode="floor")
    refinement = fine - coarse_index * step
    coarse_min = int(coarse_index.min().item())
    coarse_max = int(coarse_index.max().item())
    candidates = torch.arange(
        coarse_min, coarse_max + 1, device=fine.device, dtype=torch.float32)
    offsets = torch.arange(-half, half, device=fine.device, dtype=torch.float32)
    fine_candidates = (
        candidates[None, None, :, None] * step
        + offsets[None, None, None, :])
    coarse_pmf = entropy._likelihood(
        fine_candidates, loc[..., None, None],
        scale[..., None, None]).sum(-1)
    coarse_cdf = cdf(coarse_pmf).cpu()
    coarse_values = (coarse_index - coarse_min).to(torch.int16).cpu()
    coarse_string = torchac.encode_float_cdf(
        coarse_cdf, coarse_values, check_input_bounds=True)
    decoded_index = torchac.decode_float_cdf(
        coarse_cdf, coarse_string).to(fine.device).to(torch.int64) + coarse_min
    if not torch.equal(decoded_index, coarse_index):
        raise RuntimeError("B1 coarse torchac round-trip mismatch")

    decoded_coarse = decoded_index * step
    refinement_candidates = decoded_coarse[..., None].float() + offsets[None, None, :]
    refinement_pmf = entropy._likelihood(
        refinement_candidates, loc[..., None], scale[..., None])
    refinement_pmf = refinement_pmf / refinement_pmf.sum(
        -1, keepdim=True).clamp(min=1e-9)
    refinement_cdf = cdf(refinement_pmf).cpu()
    refinement_values = (refinement + half).to(torch.int16).cpu()
    refinement_string = torchac.encode_float_cdf(
        refinement_cdf, refinement_values, check_input_bounds=True)
    decoded_refinement = torchac.decode_float_cdf(
        refinement_cdf, refinement_string).to(fine.device).to(torch.int64) - half
    recovered = decoded_coarse + decoded_refinement
    if not torch.equal(recovered, fine):
        raise RuntimeError("B1 exact fine-symbol recovery mismatch")
    return decoded_coarse.float(), recovered.float(), coarse_string, refinement_string


@torch.no_grad()
def decoded_r5_context(base, A, x_low, encoded, lmb):
    x0 = ME.SparseTensor(
        features=torch.zeros_like(A.F),
        coordinate_map_key=A.coordinate_map_key,
        coordinate_manager=A.coordinate_manager,
        device=A.device)
    x0_low = x0
    for pooling in base.pooling_list:
        x0_low = pooling(x0_low)
    curr_x = ME.SparseTensor(
        features=x_low.F,
        coordinate_map_key=x0_low.coordinate_map_key,
        coordinate_manager=x0_low.coordinate_manager,
        device=x0_low.device)
    curr_f = base.linear_in(curr_x)
    curr_dec = curr_f - curr_f
    emb = base.embedder(lmb, device=A.device)

    for index in range(4):
        curr_f = base.upscaler(ME.cat([curr_f, curr_x]))
        curr_x = base.unpooling_list[index](curr_x)
        curr_dec = base.unpooling_list[index](curr_dec)
        stream = encoded[index]
        decoded = base.VAE.decode(
            stream["strings"], stream["min_v"], stream["max_v"],
            curr_x, curr_f, curr_dec, emb)
        curr_x, curr_f, curr_dec = (
            decoded["x_out"], decoded["f_out"], decoded["dec"])

    curr_f = base.upscaler(ME.cat([curr_f, curr_x]))
    curr_x = base.unpooling_list[4](curr_x)
    curr_dec = base.unpooling_list[4](curr_dec)
    prior = base.VAE.block_prior(ME.cat([curr_x, curr_f, curr_dec]))
    loc = sort_sparse_tensor(base.VAE.loc_net(prior))
    scale = sort_sparse_tensor(base.VAE.scale_net(prior))
    stream = encoded[4]
    fine_sorted = base.VAE.entropy_fn.decompress(
        stream["strings"], loc.F, scale.F.abs().clamp(min=1e-8),
        stream["min_v"], stream["max_v"], channels=loc.F.shape[1])
    fine_sorted = fine_sorted.to(loc.F.device)
    index = array2vector(prior.C, step=prior.C.max() + 1).argsort().argsort()
    fine = fine_sorted[index.to(fine_sorted.device)]
    q_fine = common.sparse(fine, prior)
    return {
        "x_in": curr_x, "f_in": curr_f, "prior_dec": curr_dec,
        "prior": prior, "loc": loc, "scale": scale, "symbols": q_fine,
        "emb": emb, "x0": x0, "fine_sorted": fine_sorted,
        "sorted_to_prior_index": index.to(fine_sorted.device),
    }


def write_csv(path, rows):
    if os.path.exists(path):
        raise FileExistsError("Refusing to overwrite B1 result: " + path)
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def main():
    args = parse_args()
    args.data_root = os.path.abspath(args.data_root)
    args.r02_checkpoint = os.path.abspath(args.r02_checkpoint)
    args.r04_checkpoint = os.path.abspath(args.r04_checkpoint)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    r02 = BaseAdapter(args.r02_checkpoint, scale=5, stage=1, vmode=1).cuda().eval()
    b1_rows = []
    official_rows = []
    loaded = []
    for sample_index, entry in enumerate(SAMPLES):
        coords, rgb, A = common.load_sample(args.data_root, entry)
        encoded, x_low, gpcc_bits = r02.base(
            A, training=False, lmb=16384, encode=True)
        prefix_bits = int(gpcc_bits + sum(
            len(stream["strings"]) * 8 for stream in encoded[:4]))
        native_r5_bits = len(encoded[4]["strings"]) * 8
        context = decoded_r5_context(r02.base, A, x_low, encoded, 16384)
        official_full = r02.base.decode(
            context["x0"], x_low, encoded, lmb=16384)
        native_from_q = common.synthesize(r02.base, context["symbols"], context)
        native_max_difference = float(
            (native_from_q.F - official_full.F).abs().max().item())
        if native_max_difference != 0:
            raise RuntimeError("Captured native r5 reconstruction differs from official decode")
        official_psnr = common.author_psnr(
            args.output_dir, "official_R02_{}".format(sample_index),
            coords, rgb, official_full)
        official_bits = prefix_bits + native_r5_bits
        official_rows.append({
            "rate_id": "R02", "sample": entry, "points": len(A),
            "physical_bits": official_bits,
            "physical_bpp": official_bits / len(A),
            "direct_yuv_psnr_611": official_psnr,
        })

        fine = context["fine_sorted"]
        loc = context["loc"].F
        scale = context["scale"].F.abs().clamp(min=1e-8)
        for step in (2, 4, 8):
            coarse, recovered, coarse_string, refinement_string = nested_roundtrip(
                r02.base.VAE.entropy_fn, fine, loc, scale, step)
            mapping = context["sorted_to_prior_index"]
            coarse = coarse[mapping]
            recovered = recovered[mapping]
            base_reconstruction = common.synthesize(
                r02.base, common.sparse(coarse, context["symbols"]), context)
            full_reconstruction = common.synthesize(
                r02.base, common.sparse(recovered, context["symbols"]), context)
            full_max_difference = float(
                (full_reconstruction.F - official_full.F).abs().max().item())
            if full_max_difference != 0:
                raise RuntimeError("B1 layered Full differs from official R02 Full")
            coarse_bits = len(coarse_string) * 8
            refinement_bits = len(refinement_string) * 8
            base_bits = prefix_bits + coarse_bits
            full_bits = base_bits + refinement_bits
            b1_rows.append({
                "sample": entry, "points": len(A), "step": step,
                "prefix_bits": prefix_bits, "qB_bits": coarse_bits,
                "qE_bits": refinement_bits,
                "base_bits": base_bits, "full_bits": full_bits,
                "official_r02_bits": official_bits,
                "base_bpp": base_bits / len(A),
                "full_bpp": full_bits / len(A),
                "official_r02_bpp": official_bits / len(A),
                "base_direct_yuv_psnr_611": common.author_psnr(
                    args.output_dir,
                    "b1_base_{}_s{}".format(sample_index, step),
                    coords, rgb, base_reconstruction),
                "full_direct_yuv_psnr_611": common.author_psnr(
                    args.output_dir,
                    "b1_full_{}_s{}".format(sample_index, step),
                    coords, rgb, full_reconstruction),
                "layered_over_original_ratio": full_bits / official_bits,
                "full_max_abs_difference": full_max_difference,
                "native_capture_max_abs_difference": native_max_difference,
            })
        loaded.append((entry, coords, rgb, A))

    for rate_id, adapter, lmb in (
            ("R03", r02, 8192),
            ("R04", BaseAdapter(
                args.r04_checkpoint, scale=5, stage=1, vmode=1).cuda().eval(), 4096)):
        for sample_index, (entry, coords, rgb, A) in enumerate(loaded):
            reconstruction, _, bits = adapter.hard_reconstruct(A, lmb)
            official_rows.append({
                "rate_id": rate_id, "sample": entry, "points": len(A),
                "physical_bits": bits, "physical_bpp": bits / len(A),
                "direct_yuv_psnr_611": common.author_psnr(
                    args.output_dir,
                    "official_{}_{}".format(rate_id, sample_index),
                    coords, rgb, reconstruction),
            })

    b1_summary = []
    for step in (2, 4, 8):
        rows = [row for row in b1_rows if row["step"] == step]
        b1_summary.append({
            "step": step, "num_models": len(rows), "num_h5": len(rows),
            "mean_base_bpp": mean_rows(rows, "base_bpp"),
            "mean_full_bpp": mean_rows(rows, "full_bpp"),
            "mean_base_yuv_psnr_611": mean_rows(
                rows, "base_direct_yuv_psnr_611"),
            "mean_full_yuv_psnr_611": mean_rows(
                rows, "full_direct_yuv_psnr_611"),
            "mean_layered_over_original_ratio": mean_rows(
                rows, "layered_over_original_ratio"),
            "max_full_abs_difference": max(
                float(row["full_max_abs_difference"]) for row in rows),
        })
    official_summary = []
    for rate_id in ("R02", "R03", "R04"):
        rows = [row for row in official_rows if row["rate_id"] == rate_id]
        official_summary.append({
            "rate_id": rate_id, "num_models": len(rows), "num_h5": len(rows),
            "mean_physical_bpp": mean_rows(rows, "physical_bpp"),
            "mean_direct_yuv_psnr_611": mean_rows(
                rows, "direct_yuv_psnr_611"),
        })
    write_csv("b1_physical_rd_per_h5.csv", b1_rows)
    write_csv("b1_physical_rd_summary.csv", b1_summary)
    write_csv("official_physical_rd_per_h5.csv", official_rows)
    write_csv("official_physical_rd_summary.csv", official_summary)
    print(args.output_dir)


if __name__ == "__main__":
    main()
