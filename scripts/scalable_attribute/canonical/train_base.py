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
from scalable_attribute.canonical.evaluation import evaluate_base
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.data import UncachedPCDataset, h5_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validation-file-list")
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--validate-at-start", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
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
    if args.validate_every < 0 or args.save_every < 0:
        raise ValueError("validate-every and save-every cannot be negative")
    if ((args.validate_every or args.validate_at_start)
            and not args.validation_file_list):
        raise ValueError("Validation schedule requires validation-file-list")
    if os.path.exists(args.output_dir):
        existing = set(os.listdir(args.output_dir))
        if existing - {"slurm"}:
            raise FileExistsError(
                "Output directory already contains run artifacts: "
                + args.output_dir)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = BaseSynthesisConfig.from_args(args)
    resume = None
    starting_step = 0
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu")
        if resume.get("architecture") != "canonical_base_predict_correct":
            raise ValueError("Resume checkpoint architecture mismatch")
        if resume.get("config") != config.to_dict():
            raise ValueError("Resume checkpoint BaseSynthesis config mismatch")
        if os.path.realpath(resume.get("base_checkpoint", "")) != os.path.realpath(
                args.base_checkpoint):
            raise ValueError("Resume checkpoint released Base mismatch")
        if int(resume.get("base_lambda", -1)) != args.base_lambda:
            raise ValueError("Resume checkpoint Base lambda mismatch")
        if "base_synthesis" not in resume or "optimizer" not in resume:
            raise ValueError("Resume checkpoint lacks model or optimizer state")
        starting_step = int(resume.get("step", -1))
        if starting_step < 0:
            raise ValueError("Resume checkpoint has invalid global step")
        if args.max_steps <= starting_step:
            raise ValueError("max-steps must exceed resumed global step")
    files = h5_files(args.data_root, args.train_file_list)
    if args.max_samples is not None:
        files = files[:args.max_samples]
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    loader = make_data_loader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)
    validation_files = []
    validation_entries = []
    if args.validation_file_list:
        validation_files = h5_files(args.data_root, args.validation_file_list)
        with open(args.validation_file_list, encoding="utf-8") as handle:
            validation_entries = [line.strip() for line in handle if line.strip()]

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
        "validation_h5_count": len(validation_files),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "command": command,
        "resumed_from": args.resume_checkpoint,
        "starting_step": starting_step,
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
    if resume is not None:
        model.base_synthesis.load_state_dict(
            resume["base_synthesis"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        group = optimizer.param_groups[0]
        if (group["lr"] != args.lr or tuple(group["betas"]) != (0.9, 0.999)
                or group["weight_decay"] != 0.0):
            raise ValueError("Resume checkpoint optimizer config mismatch")
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
    step = starting_step
    validation_summaries = {}

    def save_checkpoint(current_step):
        path = os.path.join(
            args.output_dir, "checkpoints",
            "step_{}.pth".format(current_step))
        torch.save({
            "architecture": "canonical_base_predict_correct",
            "base_synthesis": model.base_synthesis.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "step": current_step,
            "base_checkpoint": args.base_checkpoint,
            "base_lambda": args.base_lambda,
            "resolved_args": metadata,
        }, path)
        return path

    def validate(current_step):
        name = "step{:03d}".format(current_step)
        summary = evaluate_base(
            model, validation_files, validation_entries, args.base_lambda,
            os.path.join(args.output_dir, "validation_{}.csv".format(name)))
        if current_step == 0 and summary["max_abs_learned_native"] > 1e-7:
            raise RuntimeError(
                "Zero-init learned/native regression failed: {}".format(
                    summary["max_abs_learned_native"]))
        validation_summaries[name] = summary
        write_json(
            os.path.join(args.output_dir, "validation_summary.json"),
            validation_summaries)

    if args.validate_at_start:
        validate(0)

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
                if args.save_every and step % args.save_every == 0:
                    save_checkpoint(step)
                if args.validate_every and step % args.validate_every == 0:
                    validate(step)
                if step >= args.max_steps:
                    break

    runtime = time.monotonic() - start
    checkpoint_path = os.path.join(
        args.output_dir, "checkpoints", "step_{}.pth".format(step))
    if not os.path.exists(checkpoint_path):
        checkpoint_path = save_checkpoint(step)
    reloaded = BaseSynthesis(config).cuda()
    saved = torch.load(checkpoint_path, map_location="cpu")
    reloaded.load_state_dict(saved["base_synthesis"], strict=True)
    for name, value in model.base_synthesis.state_dict().items():
        if not torch.equal(value.detach().cpu(), reloaded.state_dict()[name].cpu()):
            raise RuntimeError("Checkpoint reload differs at " + name)

    summary = {
        "status": "PASS",
        "steps": step,
        "starting_step": starting_step,
        "resumed_from": args.resume_checkpoint,
        "initial_mse": losses[0],
        "final_mse": losses[-1],
        "best_mse": min(losses),
        "runtime_seconds": runtime,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "checkpoint": checkpoint_path,
        "checkpoint_reload": True,
        "validation_steps": validation_summaries,
        "training_mse_at_updates": {
            "step{}_pre_update".format(starting_step + index): value
            for index, value in enumerate(losses, start=1)
            if starting_step + index in (starting_step + 1, 1000, 2000)
        },
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("CANONICAL BASE TRAINING PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
