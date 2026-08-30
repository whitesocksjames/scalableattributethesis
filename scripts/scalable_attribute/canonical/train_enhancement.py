#!/usr/bin/env python3
"""Train the canonical independent EnhancementVAE for a fixed short run."""

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch

from basic_models.loss import get_bits
from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_frozen_base)
from scalable_attribute.data import UncachedPCDataset, h5_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--rd-lambda", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[100, 250])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gradient_norm(parameters):
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("EnhancementVAE has a non-finite gradient")
        value = parameter.grad.detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    if squared is None:
        raise RuntimeError("EnhancementVAE received no gradients")
    value = squared.sqrt()
    if not torch.isfinite(value):
        raise RuntimeError("EnhancementVAE gradient norm is non-finite")
    return float(value.item())


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_steps < 1:
        raise ValueError("batch-size and max-steps must be positive")
    if args.lr <= 0 or args.rd_lambda <= 0:
        raise ValueError("lr and rd-lambda must be positive")
    if sorted(set(args.save_steps)) != sorted(args.save_steps):
        raise ValueError("save-steps must be unique")
    if any(step < 1 or step > args.max_steps for step in args.save_steps):
        raise ValueError("save-steps must lie within the training run")
    if os.path.exists(args.output_dir):
        existing = set(os.listdir(args.output_dir))
        if existing - {"slurm"}:
            raise FileExistsError("Output directory already has run artifacts")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base_config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(args.released_checkpoint, base_config).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)
    model = CanonicalScalableModel(
        base, conditioning_lambda=args.conditioning_lambda).cuda()
    model.train()
    trainable = list(model.enhancement.parameters())
    if not trainable or any(not parameter.requires_grad for parameter in trainable):
        raise RuntimeError("Enhancement trainable-parameter contract failed")
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("Canonical Base is not frozen")
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    files = h5_files(args.data_root, args.train_file_list)
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    loader = make_data_loader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)
    command = shlex.join([sys.executable] + sys.argv)
    metadata = dict(vars(args))
    metadata.update({
        "architecture": "canonical_independent_enhancement",
        "base_config": base_config.to_dict(),
        "optimizer": {"name": "Adam", "lr": args.lr,
                      "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "scheduler": None},
        "objective": "R_noise + rd_lambda * mean((GT.F - Full.F) ** 2)",
        "rate": "-sum(log2(likelihood_E)) / N_full",
        "num_train_h5": len(files),
        "drop_last": False,
        "shuffle": True,
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "command": command,
    })
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    write_json(os.path.join(args.output_dir, "enhancement_config.json"), {
        "source": "released ResidualVAE exact initialization",
        "conditioning_lambda": args.conditioning_lambda,
        "rd_lambda": args.rd_lambda,
    })
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(command + "\n")

    def save_checkpoint(step):
        path = os.path.join(checkpoint_dir, "step_{}.pth".format(step))
        torch.save({
            "architecture": "canonical_independent_enhancement",
            "enhancement_vae": model.enhancement.vae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "conditioning_lambda": args.conditioning_lambda,
            "rd_lambda": args.rd_lambda,
            "released_checkpoint": args.released_checkpoint,
            "base_synthesis_checkpoint": args.base_synthesis_checkpoint,
            "base_config": base_config.to_dict(),
            "resolved_args": metadata,
        }, path)
        return path

    fields = [
        "step", "R_noise", "D_F", "lambda_D", "loss", "gradient_norm",
        "lr", "points", "step_seconds", "peak_gpu_memory_gib", "finite",
    ]
    metrics_path = os.path.join(args.output_dir, "training_metrics.csv")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    step = 0
    saved = {}
    final_row = None
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while step < args.max_steps:
            for coords, feats in loader:
                step_started = time.monotonic()
                attribute = ME.SparseTensor(
                    features=feats, coordinates=coords, tensor_stride=1,
                    device="cuda")
                optimizer.zero_grad(set_to_none=True)
                output = model(attribute)
                likelihood = output["likelihood_E"]
                rate = get_bits(likelihood) / len(attribute)
                distortion = torch.mean((attribute.F - output["Full"].F) ** 2)
                lambda_distortion = args.rd_lambda * distortion
                loss = rate + lambda_distortion
                finite = all(torch.isfinite(value).all().item() for value in (
                    likelihood, rate, distortion, lambda_distortion, loss))
                if not finite:
                    raise RuntimeError(
                        "Non-finite Enhancement objective at step {}".format(
                            step + 1))
                loss.backward()
                if any(parameter.grad is not None
                       for parameter in model.base.parameters()):
                    raise RuntimeError("Frozen Base received a gradient")
                norm = gradient_norm(trainable)
                optimizer.step()
                torch.cuda.synchronize()
                step += 1
                final_row = {
                    "step": step,
                    "R_noise": float(rate.item()),
                    "D_F": float(distortion.item()),
                    "lambda_D": float(lambda_distortion.item()),
                    "loss": float(loss.item()),
                    "gradient_norm": norm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "points": len(attribute),
                    "step_seconds": time.monotonic() - step_started,
                    "peak_gpu_memory_gib": (
                        torch.cuda.max_memory_allocated() / 1024 ** 3),
                    "finite": True,
                }
                writer.writerow(final_row)
                handle.flush()
                if step == 1 or step % 20 == 0 or step == args.max_steps:
                    print(json.dumps(final_row), flush=True)
                if step in args.save_steps:
                    saved[str(step)] = save_checkpoint(step)
                if step >= args.max_steps:
                    break

    if str(args.max_steps) not in saved:
        saved[str(args.max_steps)] = save_checkpoint(args.max_steps)
    reloaded = torch.load(saved[str(args.max_steps)], map_location="cpu")
    if int(reloaded["step"]) != args.max_steps:
        raise RuntimeError("Saved Enhancement checkpoint step mismatch")
    summary = {
        "status": "PASS",
        "steps": step,
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "final_metrics": final_row,
        "checkpoints": saved,
        "checkpoint_reload": True,
        "base_frozen": True,
        "only_enhancement_trainable": True,
    }
    write_json(os.path.join(args.output_dir, "training_summary.json"), summary)
    print("CANONICAL ENHANCEMENT E1 TRAINING PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
