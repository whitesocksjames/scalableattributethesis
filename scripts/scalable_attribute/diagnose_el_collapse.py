#!/usr/bin/env python3
"""Small checkpoint diagnostic for EL symbols, zero-latent gain and gradients."""

import argparse
import json
import os

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_h5
from scalable_attribute.config import EnhancementConfig
from scalable_attribute.data import h5_files
from scalable_attribute.losses import rate_distortion_loss
from scalable_attribute.model import ScalableAttributeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH; may be repeated")
    parser.add_argument("--rd-lambda", action="append", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    return parser.parse_args()


def sparse_sample(path):
    coords, rgb = read_h5(path)
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    coords, feats = ME.utils.sparse_collate([coords], [yuv])
    return ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1, device="cuda")


def tensor_psnr(A, reconstruction):
    if not torch.equal(A.C, reconstruction.C):
        raise RuntimeError("Diagnostic coordinate mismatch")
    mse = torch.mean((A.F - reconstruction.F) ** 2).item()
    return -10.0 * np.log10(max(mse, 1e-12))


def norm_by_group(named_parameters, gradients):
    sums = {"analysis": 0.0, "prior": 0.0, "synthesis": 0.0}
    used = {name: 0 for name in sums}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        if name.startswith(("residual_stem", "analysis_fusion", "analysis_transform")):
            group = "analysis"
        elif name.startswith(("prior_transform", "mu_head", "sigma_head")):
            group = "prior"
        else:
            group = "synthesis"
        sums[group] += gradient.detach().float().pow(2).sum().item()
        used[group] += 1
    result = {name: sums[name] ** 0.5 for name in sums}
    result["total"] = sum(sums.values()) ** 0.5
    result["parameter_tensors_used"] = used
    return result


def gradient_diagnostic(model, A, B, F_U, rd_lambda, seed):
    model.enhancement.train()
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = model.enhancement(A, B, F_U)
    loss, rate, distortion = rate_distortion_loss(
        A, output["Full"], output["likelihood"], rd_lambda)
    named = [(name, parameter) for name, parameter
             in model.enhancement.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    rate_grad = torch.autograd.grad(
        rate, parameters, retain_graph=True, allow_unused=True)
    distortion_grad = torch.autograd.grad(
        distortion, parameters, retain_graph=True, allow_unused=True)
    weighted_distortion_grad = torch.autograd.grad(
        rd_lambda * distortion, parameters, retain_graph=True, allow_unused=True)
    total_grad = torch.autograd.grad(
        loss, parameters, allow_unused=True)
    return {
        "rate": float(rate.item()),
        "distortion": float(distortion.item()),
        "lambda_distortion": float((rd_lambda * distortion).item()),
        "loss": float(loss.item()),
        "rate_gradient": norm_by_group(named, rate_grad),
        "distortion_gradient": norm_by_group(named, distortion_grad),
        "weighted_distortion_gradient": norm_by_group(
            named, weighted_distortion_grad),
        "total_gradient": norm_by_group(named, total_grad),
    }


@torch.no_grad()
def symbol_diagnostic(model, A, B, F_U):
    model.enhancement.eval()
    layer = model.enhancement
    y_E = layer._analysis(A, B, F_U)
    mu, sigma = layer._prior(B, F_U, target=y_E)
    symbols = layer.entropy.conditional._quantize(y_E.F, mode="symbols")
    y_hat = ME.SparseTensor(
        features=symbols,
        coordinate_map_key=y_E.coordinate_map_key,
        coordinate_manager=y_E.coordinate_manager,
        device=y_E.device,
    )
    zero_hat = ME.SparseTensor(
        features=torch.zeros_like(symbols),
        coordinate_map_key=y_E.coordinate_map_key,
        coordinate_manager=y_E.coordinate_manager,
        device=y_E.device,
    )
    _, Full = layer._synthesis(y_hat, B, F_U)
    _, Full_zero = layer._synthesis(zero_hat, B, F_U)
    _, encoded = layer.entropy.encode(y_E, mu, sigma)
    _, symbol_likelihood = layer.entropy.conditional(
        symbols, mu.F, sigma.F.abs().clamp(min=1e-8), quantize_mode=None)
    flat = symbols.detach().cpu().reshape(-1)
    values, counts = torch.unique(flat, return_counts=True)
    histogram = {str(int(value.item())): int(count.item())
                 for value, count in zip(values, counts)}
    return {
        "num_latent_points": len(y_E),
        "num_symbols": flat.numel(),
        "zero_fraction": float((flat == 0).float().mean().item()),
        "nonzero_count": int((flat != 0).sum().item()),
        "y_E_mean_abs": float(y_E.F.abs().mean().item()),
        "y_E_min": float(y_E.F.min().item()),
        "y_E_max": float(y_E.F.max().item()),
        "symbol_min": int(flat.min().item()),
        "symbol_max": int(flat.max().item()),
        "symbol_histogram": histogram,
        "actual_el_bits": len(encoded.strings) * 8,
        "hard_el_bpp": len(encoded.strings) * 8 / len(A),
        "symbol_estimated_el_bpp": float(
            (-torch.log2(symbol_likelihood).sum() / len(A)).item()),
        "base_tensor_psnr": tensor_psnr(A, B),
        "full_tensor_psnr": tensor_psnr(A, Full),
        "full_zero_tensor_psnr": tensor_psnr(A, Full_zero),
        "full_minus_base_psnr": tensor_psnr(A, Full) - tensor_psnr(A, B),
        "full_minus_zero_psnr": (
            tensor_psnr(A, Full) - tensor_psnr(A, Full_zero)),
        "full_vs_zero_max_abs": float(
            (Full.F - Full_zero.F).abs().max().item()),
    }


def main():
    args = parse_args()
    if len(args.checkpoint) != len(args.rd_lambda):
        raise ValueError("Provide one --rd-lambda for each --checkpoint")
    checkpoints = []
    for value, rd_lambda in zip(args.checkpoint, args.rd_lambda):
        if "=" not in value:
            raise ValueError("--checkpoint must be LABEL=PATH")
        label, path = value.split("=", 1)
        checkpoints.append((label, os.path.expandvars(path), rd_lambda))

    files = h5_files(
        os.path.expandvars(args.data_root), os.path.expandvars(args.file_list))
    if args.num_samples < 1 or args.num_samples > len(files):
        raise ValueError("Invalid --num-samples")
    indices = np.linspace(0, len(files) - 1, args.num_samples, dtype=int)
    selected = [files[index] for index in indices]
    result = {"samples": selected, "checkpoints": {}}

    for label, checkpoint, rd_lambda in checkpoints:
        state = torch.load(checkpoint, map_location="cpu")
        config = EnhancementConfig(**state["config"])
        model = ScalableAttributeModel(
            os.path.expandvars(args.base_checkpoint), config,
            base_scale=args.base_scale, base_stage=args.base_stage,
            base_vmode=args.base_vmode).cuda().eval()
        model.enhancement.load_state_dict(state["enhancement"])
        samples = []
        gradient = None
        for index, path in enumerate(selected):
            A = sparse_sample(path)
            B, F_U = model.base_adapter(A, args.base_lambda)
            samples.append(symbol_diagnostic(model, A, B, F_U))
            if index == 0:
                gradient = gradient_diagnostic(
                    model, A, B, F_U, rd_lambda, args.seed)
        result["checkpoints"][label] = {
            "checkpoint": checkpoint,
            "checkpoint_step": int(state["step"]),
            "rd_lambda": rd_lambda,
            "samples": samples,
            "gradient": gradient,
        }
        del model
        torch.cuda.empty_cache()

    output = os.path.abspath(os.path.expandvars(args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
