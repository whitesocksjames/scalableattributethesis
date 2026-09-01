#!/usr/bin/env python3
"""Train canonical Base/Full with random single-endpoint objectives."""

import argparse
import csv
import json
import os
import random
import shlex
import socket
import subprocess
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.dataloaders.attribute_dataloader import collate_pointcloud_fn
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.data_schedule import ContinuationBatchSampler
from scalable_attribute.canonical.joint_endpoint import (
    joint_endpoint_objective, sample_endpoint)
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    FINE_TUNE_ARCHITECTURE, CanonicalScalableModel, load_frozen_base)
from scalable_attribute.data import UncachedPCDataset, h5_files


JOINT_TRAINING_MODE = "tafa_random_single_endpoint_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--lambda-base", type=float, required=True)
    parser.add_argument("--lambda-full", type=float, required=True)
    parser.add_argument("--p-full", type=float, default=0.5)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, nargs="+",
                        default=[250, 500, 1000])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-only", action="store_true")
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


def assert_optimizer_contract(model, optimizer):
    required = {id(value) for value in model.parameters()
                if value.requires_grad}
    actual = {id(value) for group in optimizer.param_groups
              for value in group["params"]}
    if required != actual:
        raise RuntimeError("optimizer params != requires_grad params")


def gradient_summary(module):
    gradients = [parameter.grad for parameter in module.parameters()
                 if parameter.grad is not None]
    if any(not torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("non-finite module gradient")
    norm = 0.0
    if gradients:
        norm = float(torch.stack([
            value.detach().float().pow(2).sum()
            for value in gradients]).sum().sqrt().item())
    return {"present": bool(gradients), "tensors": len(gradients), "norm": norm}


def gradient_groups(model):
    return {
        "linear_in": model.base.prefix.model.linear_in,
        "upscaler": model.base.prefix.model.upscaler,
        "prefix_vae": model.base.prefix.model.VAE,
        "embedder": model.base.prefix.model.embedder,
        "base_synthesis": model.base.base_synthesis,
        "enhancement_vae": model.enhancement,
    }


def assert_gradient_contract(model, endpoint):
    result = {name: gradient_summary(module)
              for name, module in gradient_groups(model).items()}
    for name in ("linear_in", "upscaler", "prefix_vae", "embedder",
                 "base_synthesis"):
        if not result[name]["present"]:
            raise RuntimeError("{} missing {} gradient".format(endpoint, name))
    expected_enhancement = endpoint == "Full"
    if result["enhancement_vae"]["present"] != expected_enhancement:
        raise RuntimeError("{} Enhancement gradient contract failed".format(
            endpoint))
    return result


def total_gradient_norm(parameters):
    gradients = [parameter.grad for parameter in parameters
                 if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(value).all()
                            for value in gradients):
        raise RuntimeError("missing or non-finite joint gradient")
    return float(torch.stack([
        value.detach().float().pow(2).sum()
        for value in gradients]).sum().sqrt().item())


def make_attribute(coords, feats):
    return ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1, device="cuda")


def checkpoint(model, optimizer, args, base_config, metadata, step,
               endpoint_counts, data_schedule):
    return {
        "architecture": FINE_TUNE_ARCHITECTURE,
        "training_mode": JOINT_TRAINING_MODE,
        "scalable_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "conditioning_lambda": args.conditioning_lambda,
        "lambda_base": args.lambda_base,
        "lambda_full": args.lambda_full,
        "p_full": args.p_full,
        "distortion_weights": [1.0, 1.0, 1.0],
        "base_config": base_config.to_dict(),
        "released_checkpoint": args.released_checkpoint,
        "base_synthesis_initialization": args.base_checkpoint,
        "enhancement_initialization": "released ResidualVAE exact clone",
        "endpoint_counts": dict(endpoint_counts),
        "data_schedule": data_schedule,
        "resolved_args": metadata,
    }


def main():
    args = parse_args()
    source_commit = git_commit()
    for name in ("data_root", "train_file_list", "released_checkpoint",
                 "base_checkpoint", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(
            os.path.expanduser(getattr(args, name)))))
    if args.conditioning_lambda <= 0 or args.lr <= 0:
        raise ValueError("conditioning-lambda and lr must be positive")
    if args.lambda_base != args.conditioning_lambda:
        raise ValueError("V1 requires lambda-base == conditioning-lambda")
    if args.lambda_full != args.conditioning_lambda:
        raise ValueError("V1 requires lambda-full == conditioning-lambda")
    if not 0.0 <= args.p_full <= 1.0:
        raise ValueError("p-full must lie in [0, 1]")
    if args.batch_size < 1 or args.max_steps < 1:
        raise ValueError("batch-size and max-steps must be positive")
    if len(set(args.save_steps)) != len(args.save_steps):
        raise ValueError("save-steps must be unique")
    if (not args.smoke_only and
            any(step < 1 or step > args.max_steps for step in args.save_steps)):
        raise ValueError("save-steps must lie within the run")
    if os.path.exists(args.output_dir):
        existing = set(os.listdir(args.output_dir))
        if existing - {"slurm"}:
            raise FileExistsError("output directory already has run artifacts")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_state = torch.load(args.base_checkpoint, map_location="cpu")
    base_config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(args.released_checkpoint, base_config).cuda()
    load_frozen_base(base, args.base_checkpoint, args.released_checkpoint,
                     args.conditioning_lambda)
    model = CanonicalScalableModel(base, args.conditioning_lambda).cuda()
    model.set_trainable_scope("full")
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    assert_optimizer_contract(model, optimizer)

    files = h5_files(args.data_root, args.train_file_list)
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    sampler = ContinuationBatchSampler(
        len(dataset), args.batch_size, 0,
        1 if args.smoke_only else args.max_steps, args.seed)
    manifest = os.path.realpath(args.train_file_list)
    data_schedule = sampler.metadata(manifest, args.data_root)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers,
        collate_fn=collate_pointcloud_fn, pin_memory=True)

    command = shlex.join([sys.executable] + sys.argv)
    metadata = dict(vars(args))
    metadata.update({
        "architecture": FINE_TUNE_ARCHITECTURE,
        "training_mode": JOINT_TRAINING_MODE,
        "trainable_scope": model.trainable_scope,
        "trainable_parameter_count": sum(value.numel() for value in trainable),
        "initialization": {
            "prefix": "released checkpoint",
            "base_synthesis": args.base_checkpoint,
            "enhancement": "released ResidualVAE exact clone",
            "optimizer": "fresh Adam",
        },
        "optimizer": {"name": "Adam", "lr": args.lr,
                      "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "scheduler": None},
        "objective_base": "R_Base + lambda_base * D111(Base)",
        "objective_full": "R_Base + R_Enh + lambda_full * D111(Full)",
        "rate_normalization": "full-resolution N0",
        "data_schedule": data_schedule,
        "num_train_h5": len(files),
        "git_commit": source_commit,
        "hostname": socket.gethostname(),
        "command": command,
    })
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(command + "\n")

    if args.smoke_only:
        coords, feats = next(iter(loader))
        smoke = {}
        for endpoint in ("Base", "Full"):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = joint_endpoint_objective(
                model, make_attribute(coords, feats), endpoint,
                args.lambda_base, args.lambda_full)
            if not all(torch.isfinite(value).all() for value in (
                    result.loss, result.rate_base, result.rate_enhancement,
                    result.distortion)):
                raise RuntimeError(endpoint + " smoke objective is non-finite")
            result.loss.backward()
            smoke[endpoint] = {
                "loss": float(result.loss.item()),
                "R_B_est": float(result.rate_base.item()),
                "R_E_est": float(result.rate_enhancement.item()),
                "D111": float(result.distortion.item()),
                "gradients": assert_gradient_contract(model, endpoint),
            }
        write_json(os.path.join(args.output_dir, "smoke_summary.json"), {
            "status": "PASS", "endpoints": smoke,
            "optimizer_contract": True,
            "conditioning_lambda": args.conditioning_lambda,
            "lambda_base": args.lambda_base,
            "lambda_full": args.lambda_full,
        })
        print(json.dumps(smoke, indent=2))
        return

    fields = ["step", "endpoint", "R_B_est", "R_E_est", "D111",
              "D_Y", "D_U", "D_V", "loss", "gradient_norm", "lr",
              "points", "step_seconds", "peak_gpu_memory_gib"]
    metrics_path = os.path.join(args.output_dir, "training_metrics.csv")
    endpoint_rng = random.Random(args.seed)
    endpoint_counts = {"Base": 0, "Full": 0}
    saved = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    step = 0
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for coords, feats in loader:
            step_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            endpoint = sample_endpoint(endpoint_rng, args.p_full)
            attribute = make_attribute(coords, feats)
            result = joint_endpoint_objective(
                model, attribute, endpoint,
                args.lambda_base, args.lambda_full)
            if not all(torch.isfinite(value).all() for value in (
                    result.loss, result.rate_base, result.rate_enhancement,
                    result.distortion)):
                raise RuntimeError("non-finite joint objective at step {}".format(
                    step + 1))
            result.loss.backward()
            gradients = assert_gradient_contract(model, endpoint)
            norm = total_gradient_norm(trainable)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            endpoint_counts[endpoint] += 1
            row = {
                "step": step, "endpoint": endpoint,
                "R_B_est": float(result.rate_base.item()),
                "R_E_est": float(result.rate_enhancement.item()),
                "D111": float(result.distortion.item()),
                "D_Y": float(result.channel_mse[0].item()),
                "D_U": float(result.channel_mse[1].item()),
                "D_V": float(result.channel_mse[2].item()),
                "loss": float(result.loss.item()),
                "gradient_norm": norm,
                "lr": optimizer.param_groups[0]["lr"],
                "points": len(attribute),
                "step_seconds": time.monotonic() - step_started,
                "peak_gpu_memory_gib": (
                    torch.cuda.max_memory_allocated() / 1024 ** 3),
            }
            writer.writerow(row)
            handle.flush()
            if step == 1 or step % 20 == 0 or step == args.max_steps:
                print(json.dumps({**row, "gradient_groups": gradients}),
                      flush=True)
            if step in args.save_steps or step == args.max_steps:
                path = os.path.join(
                    checkpoint_dir, "step_{}.pth".format(step))
                torch.save(checkpoint(
                    model, optimizer, args, base_config, metadata, step,
                    endpoint_counts, data_schedule), path)
                saved[str(step)] = path

    summary = {
        "status": "PASS", "steps": step,
        "runtime_seconds": time.monotonic() - started,
        "endpoint_counts": endpoint_counts,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024 ** 3),
        "checkpoints": saved,
    }
    write_json(os.path.join(args.output_dir, "training_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
