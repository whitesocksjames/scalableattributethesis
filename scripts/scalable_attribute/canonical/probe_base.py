#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.canonical.base_synthesis import BaseSynthesis
from scalable_attribute.canonical.config import (
    BaseSynthesisConfig, add_base_architecture_arguments)
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.data import UncachedPCDataset, h5_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    return add_base_architecture_arguments(parser).parse_args()


def sparse_difference(left, right, label):
    if list(left.tensor_stride) != list(right.tensor_stride):
        raise RuntimeError(label + " tensor strides differ")
    if not torch.equal(left.C, right.C):
        raise RuntimeError(label + " coordinates differ")
    return float((left.F - right.F).abs().max().item())


def full_regression(prefix, attribute, base_lambda):
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output)

    handle = prefix.model.VAE.register_forward_hook(capture)
    try:
        default = prefix.model(
            attribute, training=False, lmb=base_lambda,
            real_coding=False)
    finally:
        handle.remove()
    if len(captured) != 5:
        raise RuntimeError(
            "Released full path invoked ResidualVAE {} times, expected 5".format(
                len(captured)))

    explicit = prefix.model(
        attribute, training=False, lmb=base_lambda, real_coding=False,
        max_residual_stages=5)
    differences = []
    for index, (left, right) in enumerate(zip(
            default["out_list"], explicit["out_list"])):
        differences.append(sparse_difference(
            left, right, "released output {}".format(index)))
    differences.append(sparse_difference(
        default["curr_f"], explicit["curr_f"], "released curr_f"))
    if max(differences) != 0.0:
        raise RuntimeError("Explicit full traversal changed released output")
    return captured[3], max(differences)


def gradient_norm(parameters):
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("BaseSynthesis has a non-finite gradient")
        value = parameter.grad.detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    if squared is None:
        raise RuntimeError("BaseSynthesis received no gradients")
    value = squared.sqrt()
    if not torch.isfinite(value) or value.item() == 0:
        raise RuntimeError("BaseSynthesis gradient norm is not finite/nonzero")
    return float(value.item())


def main():
    args = parse_args()
    if args.max_steps < 1 or args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("max_steps, max_samples and batch_size must be positive")
    if os.path.exists(args.output_dir):
        raise FileExistsError("E0 output directory already exists")
    os.makedirs(args.output_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = BaseSynthesisConfig.from_args(args)
    files = h5_files(args.data_root, args.file_list)[:args.max_samples]
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    loader = make_data_loader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    with open(os.path.join(args.output_dir, "selected_h5.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(files) + "\n")
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w",
              encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.output_dir, "base_config.json"), "w",
              encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")

    model = CanonicalBaseModel(
        args.base_checkpoint, config, scale=args.base_scale,
        stage=args.base_stage, vmode=args.base_vmode).cuda()
    optimizer = torch.optim.Adam(
        model.base_synthesis.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=0.0)
    if model.prefix.training or any(
            parameter.requires_grad for parameter in model.prefix.parameters()):
        raise RuntimeError("Unicorn prefix is not frozen and in eval mode")

    torch.cuda.reset_peak_memory_stats()
    results = {
        "status": "RUNNING",
        "selected_h5": files,
        "steps": [],
    }
    step = 0
    first_batch_checked = False
    while step < args.max_steps:
        for coords, feats in loader:
            attribute = ME.SparseTensor(
                features=feats, coordinates=coords, tensor_stride=1,
                device="cuda")
            if not first_batch_checked:
                captured_r4, released_difference = full_regression(
                    model.prefix, attribute, args.base_lambda)
                prefix_state = model.prefix(attribute, args.base_lambda)
                r4_differences = {
                    "x4": sparse_difference(
                        captured_r4["x_out"], prefix_state.x4, "captured/prefix x4"),
                    "f4": sparse_difference(
                        captured_r4["f_out"], prefix_state.f4, "captured/prefix f4"),
                    "d4": sparse_difference(
                        captured_r4["dec"], prefix_state.d4, "captured/prefix d4"),
                }
                calls = []
                handle = model.prefix.model.VAE.register_forward_hook(
                    lambda _module, _inputs, _output: calls.append(1))
                try:
                    initial_output = model(attribute, args.base_lambda)
                finally:
                    handle.remove()
                if len(calls) != 4:
                    raise RuntimeError(
                        "Canonical Base invoked ResidualVAE {} times; r5 was not excluded".format(
                            len(calls)))
                results["released_full_max_abs_difference"] = released_difference
                results["r4_max_abs_difference"] = r4_differences
                results["canonical_residual_invocations"] = len(calls)
                results["initial_c_B_max_abs"] = float(
                    initial_output["c_B"].F.abs().max().item())
                results["states"] = {
                    name: {
                        "shape": list(value.F.shape),
                        "tensor_stride": list(value.tensor_stride),
                    }
                    for name, value in {
                        "x4": prefix_state.x4, "f4": prefix_state.f4,
                        "d4": prefix_state.d4, "x5p": prefix_state.x5p,
                        "f5p": prefix_state.f5p, "d5p": prefix_state.d5p,
                        "c_B": initial_output["c_B"],
                        "F_B": initial_output["F_B"],
                        "Base": initial_output["Base"],
                    }.items()
                }
                first_batch_checked = True

            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(attribute, args.base_lambda)
            loss = torch.mean((attribute.F - output["Base"].F) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Base distortion")
            loss.backward()
            if any(parameter.grad is not None
                   for parameter in model.prefix.parameters()):
                raise RuntimeError("Frozen Unicorn parameter received a gradient")
            norm = gradient_norm(model.base_synthesis.parameters())
            optimizer.step()
            step += 1
            results["steps"].append({
                "step": step,
                "loss": float(loss.item()),
                "gradient_norm": norm,
                "points": len(attribute),
            })
            if step >= args.max_steps:
                break

    checkpoint_path = os.path.join(args.output_dir, "e0_base_synthesis.pth")
    torch.save({
        "architecture": "canonical_base_predict_correct",
        "base_synthesis": model.base_synthesis.state_dict(),
        "config": config.to_dict(),
        "step": step,
    }, checkpoint_path)
    reloaded = BaseSynthesis(config).cuda()
    saved = torch.load(checkpoint_path, map_location="cpu")
    reloaded.load_state_dict(saved["base_synthesis"], strict=True)
    for name, value in model.base_synthesis.state_dict().items():
        if not torch.equal(value.detach().cpu(), reloaded.state_dict()[name].cpu()):
            raise RuntimeError("E0 checkpoint reload differs at " + name)

    results.update({
        "status": "PASS",
        "checkpoint_reload": True,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024 ** 3),
    })
    with open(os.path.join(args.output_dir, "e0_results.json"), "w",
              encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("CANONICAL E0 PASS")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
