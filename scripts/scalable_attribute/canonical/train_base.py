#!/usr/bin/env python3
"""Train the canonical full-resolution Base synthesis module."""

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

from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.canonical.base_synthesis import BaseSynthesis
from scalable_attribute.canonical.config import (
    BaseSynthesisConfig, add_base_architecture_arguments)
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.data import UncachedPCDataset, h5_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    return add_base_architecture_arguments(parser).parse_args()


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
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError("BaseSynthesis gradient norm is non-finite")
    return float(norm.item())


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_steps < 1:
        raise ValueError("batch-size and max-steps must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive when provided")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if os.path.exists(args.output_dir):
        raise FileExistsError("Output directory already exists: " + args.output_dir)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = BaseSynthesisConfig.from_args(args)
    files = h5_files(args.data_root, args.train_file_list)
    if args.max_samples is not None:
        files = files[:args.max_samples]
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    loader = make_data_loader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)

    command = shlex.join([sys.executable] + sys.argv)
    metadata = dict(vars(args))
    metadata.update({
        "architecture": config.to_dict(),
        "optimizer": {
            "name": "Adam", "lr": args.lr,
            "betas": [0.9, 0.999], "weight_decay": 0.0,
            "scheduler": None,
        },
        "objective": "mean((A.F - Base.F) ** 2)",
        "selected_h5_count": len(files),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "command": command,
    })
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    write_json(os.path.join(args.output_dir, "base_config.json"), config.to_dict())
    with open(os.path.join(args.output_dir, "selected_h5.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(files) + "\n")
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(command + "\n")

    model = CanonicalBaseModel(
        args.base_checkpoint, config, scale=args.base_scale,
        stage=args.base_stage, vmode=args.base_vmode).cuda()
    model.train()
    trainable = list(model.base_synthesis.parameters())
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    if model.prefix.training or any(
            parameter.requires_grad for parameter in model.prefix.parameters()):
        raise RuntimeError("Unicorn prefix is not frozen in eval mode")
    if not trainable or any(not parameter.requires_grad for parameter in trainable):
        raise RuntimeError("BaseSynthesis trainable-parameter contract failed")

    trajectory_path = os.path.join(args.output_dir, "training_metrics.csv")
    fields = [
        "step", "mse", "gradient_norm", "points", "step_seconds",
        "peak_gpu_memory_gib", "lr",
    ]
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    losses = []
    step = 0
    with open(trajectory_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while step < args.max_steps:
            for coords, feats in loader:
                step_start = time.monotonic()
                attribute = ME.SparseTensor(
                    features=feats, coordinates=coords, tensor_stride=1,
                    device="cuda")
                optimizer.zero_grad(set_to_none=True)
                output = model(attribute, args.base_lambda)
                mse = torch.mean((attribute.F - output["Base"].F) ** 2)
                if not torch.isfinite(mse):
                    raise RuntimeError("Non-finite Base MSE at step {}".format(step + 1))
                mse.backward()
                if any(parameter.grad is not None
                       for parameter in model.prefix.parameters()):
                    raise RuntimeError("Frozen Unicorn parameter received a gradient")
                norm = gradient_norm(trainable)
                optimizer.step()
                torch.cuda.synchronize()
                step += 1
                loss_value = float(mse.item())
                losses.append(loss_value)
                row = {
                    "step": step,
                    "mse": loss_value,
                    "gradient_norm": norm,
                    "points": len(attribute),
                    "step_seconds": time.monotonic() - step_start,
                    "peak_gpu_memory_gib": (
                        torch.cuda.max_memory_allocated() / 1024 ** 3),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                writer.writerow(row)
                handle.flush()
                if step == 1 or step % 25 == 0 or step == args.max_steps:
                    print(json.dumps(row), flush=True)
                if step >= args.max_steps:
                    break

    runtime = time.monotonic() - start
    checkpoint_path = os.path.join(
        args.output_dir, "checkpoints", "step_{}.pth".format(step))
    torch.save({
        "architecture": "canonical_base_predict_correct",
        "base_synthesis": model.base_synthesis.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "base_checkpoint": args.base_checkpoint,
        "base_lambda": args.base_lambda,
        "resolved_args": metadata,
    }, checkpoint_path)
    reloaded = BaseSynthesis(config).cuda()
    saved = torch.load(checkpoint_path, map_location="cpu")
    reloaded.load_state_dict(saved["base_synthesis"], strict=True)
    for name, value in model.base_synthesis.state_dict().items():
        if not torch.equal(value.detach().cpu(), reloaded.state_dict()[name].cpu()):
            raise RuntimeError("Checkpoint reload differs at " + name)

    summary = {
        "status": "PASS",
        "steps": step,
        "initial_mse": losses[0],
        "final_mse": losses[-1],
        "best_mse": min(losses),
        "runtime_seconds": runtime,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "checkpoint": checkpoint_path,
        "checkpoint_reload": True,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("CANONICAL BASE TRAINING PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
