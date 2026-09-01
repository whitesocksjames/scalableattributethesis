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
from data_utils.dataloaders.attribute_dataloader import collate_pointcloud_fn
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.data_schedule import (
    ContinuationBatchSampler, require_compatible_schedule)
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.operating_points import (
    DEFAULT_CONFIG, OperatingPointConfig)
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_frozen_base)
from scalable_attribute.data import UncachedPCDataset, h5_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--point")
    parser.add_argument("--operating-points-config", default=DEFAULT_CONFIG)
    parser.add_argument("--released-root")
    parser.add_argument("--canonical-experiment-root")
    parser.add_argument("--enhancement-stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--released-checkpoint")
    parser.add_argument("--base-synthesis-checkpoint")
    parser.add_argument("--conditioning-lambda", type=int)
    parser.add_argument("--rd-lambda", type=float)
    parser.add_argument("--distortion-weights")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-steps", type=int, nargs="+")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--allow-resume-distortion-weight-change",
                        action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_training_args(args):
    """Resolve one canonical point or retain the explicit legacy CLI."""
    point = None
    if args.point:
        point = OperatingPointConfig.resolve(
            args.point, args.released_root, args.canonical_experiment_root,
            args.operating_points_config)
        fixed = {
            "released_checkpoint": point.released_checkpoint,
            "base_synthesis_checkpoint": point.selected_base_checkpoint,
            "conditioning_lambda": point.conditioning_lambda,
            "rd_lambda": float(point.conditioning_lambda),
        }
        for name, expected in fixed.items():
            actual = getattr(args, name)
            if actual is not None:
                if name.endswith("checkpoint"):
                    matches = (
                        os.path.realpath(os.path.expandvars(actual))
                        == os.path.realpath(expected))
                else:
                    matches = float(actual) == float(expected)
                if not matches:
                    raise ValueError(
                        "--{} conflicts with canonical --point mapping".format(
                            name.replace("_", "-")))
            setattr(args, name, expected)

        stage = point.enhancement["stage{}".format(args.enhancement_stage)]
        defaults = {
            "distortion_weights": stage["distortion_weights"],
            "lr": stage["lr"],
            "batch_size": stage["batch_size"],
            "max_steps": stage["max_steps"],
            "save_steps": stage["save_steps"],
            "seed": stage["seed"],
        }
        for name, value in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        if args.enhancement_stage == 2:
            if not stage.get("manager_trigger_required"):
                raise ValueError(
                    "Canonical Enhancement Stage 2 trigger contract is invalid")
            if not args.resume_checkpoint:
                raise ValueError(
                    "Enhancement Stage 2 requires explicit --resume-checkpoint")
            args.allow_resume_distortion_weight_change = True
    else:
        required = (
            "released_checkpoint", "base_synthesis_checkpoint",
            "conditioning_lambda", "rd_lambda", "lr", "max_steps")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "Explicit mode requires --{}".format(
                    ", --".join(name.replace("_", "-") for name in missing)))
        if args.distortion_weights is None:
            args.distortion_weights = "1,1,1"
        if args.batch_size is None:
            args.batch_size = 4
        if args.save_steps is None:
            args.save_steps = [100, 250]
        if args.seed is None:
            args.seed = 0
        if args.enhancement_stage == 2 and not args.resume_checkpoint:
            raise ValueError(
                "Enhancement Stage 2 requires explicit --resume-checkpoint")
    return point


def parse_distortion_weights(value):
    try:
        weights = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("distortion-weights must be wY,wU,wV") from exc
    if len(weights) != 3 or any(weight <= 0 for weight in weights):
        raise ValueError("distortion-weights must contain three positive values")
    return weights


def weighted_distortion(reference, reconstruction, weights):
    channel_mse = torch.mean((reference - reconstruction) ** 2, dim=0)
    weight = channel_mse.new_tensor(weights)
    return torch.sum(channel_mse * weight) / torch.sum(weight), channel_mse


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
    operating_point = resolve_training_args(args)
    for name in (
            "data_root", "train_file_list", "released_checkpoint",
            "base_synthesis_checkpoint", "output_dir",
            "operating_points_config"):
        value = getattr(args, name)
        setattr(args, name, os.path.abspath(os.path.expandvars(value)))
    for name in (
            "resume_checkpoint", "released_root",
            "canonical_experiment_root"):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(os.path.expandvars(value)))
    distortion_weights = parse_distortion_weights(args.distortion_weights)
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
    starting_step = 0
    resume = None
    resumed_distortion_weights = None
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu")
        if resume.get("architecture") != "canonical_independent_enhancement":
            raise ValueError("Enhancement resume architecture mismatch")
        if int(resume.get("conditioning_lambda", -1)) != args.conditioning_lambda:
            raise ValueError("Enhancement resume conditioning lambda mismatch")
        if float(resume.get("rd_lambda", -1)) != args.rd_lambda:
            raise ValueError("Enhancement resume RD lambda mismatch")
        resume_weights = tuple(float(value) for value in resume.get(
            "distortion_weights", [1.0, 1.0, 1.0]))
        resumed_distortion_weights = resume_weights
        if (resume_weights != distortion_weights and
                not args.allow_resume_distortion_weight_change):
            raise ValueError("Enhancement resume distortion weights mismatch")
        if os.path.realpath(resume.get("released_checkpoint", "")) != os.path.realpath(
                args.released_checkpoint):
            raise ValueError("Enhancement resume released checkpoint mismatch")
        if os.path.realpath(resume.get("base_synthesis_checkpoint", "")) != os.path.realpath(
                args.base_synthesis_checkpoint):
            raise ValueError("Enhancement resume Base checkpoint mismatch")
        if resume.get("base_config") != base_config.to_dict():
            raise ValueError("Enhancement resume Base config mismatch")
        if "enhancement_vae" not in resume or "optimizer" not in resume:
            raise ValueError("Enhancement resume lacks model or optimizer state")
        starting_step = int(resume.get("step", -1))
        if starting_step < 0 or args.max_steps <= starting_step:
            raise ValueError("max-steps must exceed resumed global step")
        if operating_point is not None and args.enhancement_stage == 2:
            stage1_stop = int(
                operating_point.enhancement["stage1"]["max_steps"])
            if starting_step < stage1_stop:
                raise ValueError(
                    "Enhancement Stage 2 checkpoint predates the Stage-1 gate")
        model.enhancement.vae.load_state_dict(
            resume["enhancement_vae"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    files = h5_files(args.data_root, args.train_file_list)
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    manifest_path = os.path.realpath(os.path.expandvars(
        os.path.expanduser(args.train_file_list)))
    batch_sampler = ContinuationBatchSampler(
        num_samples=len(dataset), batch_size=args.batch_size,
        start_step=starting_step, stop_step=args.max_steps, seed=args.seed)
    data_schedule = batch_sampler.metadata(manifest_path, args.data_root)
    if resume is not None:
        require_compatible_schedule(resume.get("data_schedule"), data_schedule)
    start_epoch, start_batch_in_epoch = batch_sampler.position(starting_step)
    stop_epoch, stop_batch_in_epoch = batch_sampler.position(args.max_steps)
    data_schedule_run = {
        **data_schedule,
        "start_step": starting_step,
        "stop_step_exclusive": args.max_steps,
        "start_epoch": start_epoch,
        "start_batch_in_epoch": start_batch_in_epoch,
        "next_epoch_after_run": stop_epoch,
        "next_batch_in_epoch_after_run": stop_batch_in_epoch,
    }
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler,
        num_workers=args.num_workers, collate_fn=collate_pointcloud_fn,
        pin_memory=True)
    command = shlex.join([sys.executable] + sys.argv)
    metadata = dict(vars(args))
    metadata.update({
        "architecture": "canonical_independent_enhancement",
        "operating_point": (
            operating_point.to_dict() if operating_point is not None else None),
        "enhancement_stage": args.enhancement_stage,
        "base_config": base_config.to_dict(),
        "optimizer": {"name": "Adam", "lr": args.lr,
                      "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "scheduler": None},
        "objective": "R_noise + rd_lambda * sum_c(w_c*MSE_c)/sum_c(w_c)",
        "distortion_weights": list(distortion_weights),
        "distortion_definition": "sum_c(w_c*MSE_c)/sum_c(w_c)",
        "rate": "-sum(log2(likelihood_E)) / N_full",
        "num_train_h5": len(files),
        "drop_last": False,
        "shuffle": True,
        "data_schedule": data_schedule_run,
        "next_data_step": starting_step,
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "command": command,
        "resumed_from": args.resume_checkpoint,
        "starting_step": starting_step,
        "resumed_distortion_weights": (
            list(resumed_distortion_weights)
            if resumed_distortion_weights is not None else None),
    })
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    write_json(os.path.join(args.output_dir, "enhancement_config.json"), {
        "source": "released ResidualVAE exact initialization",
        "operating_point": (
            operating_point.to_dict() if operating_point is not None else None),
        "enhancement_stage": args.enhancement_stage,
        "conditioning_lambda": args.conditioning_lambda,
        "rd_lambda": args.rd_lambda,
    })
    write_json(os.path.join(args.output_dir, "data_schedule.json"),
               data_schedule_run)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(command + "\n")

    def save_checkpoint(step):
        path = os.path.join(checkpoint_dir, "step_{}.pth".format(step))
        torch.save({
            "architecture": "canonical_independent_enhancement",
            "operating_point": (
                operating_point.to_dict()
                if operating_point is not None else None),
            "enhancement_stage": args.enhancement_stage,
            "enhancement_vae": model.enhancement.vae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "conditioning_lambda": args.conditioning_lambda,
            "rd_lambda": args.rd_lambda,
            "distortion_weights": list(distortion_weights),
            "released_checkpoint": args.released_checkpoint,
            "base_synthesis_checkpoint": args.base_synthesis_checkpoint,
            "base_config": base_config.to_dict(),
            "data_schedule": data_schedule,
            "resolved_args": metadata,
        }, path)
        return path

    fields = [
        "step", "data_epoch", "batch_in_epoch", "R_noise", "D_F",
        "D_Y", "D_U", "D_V", "lambda_D", "loss", "gradient_norm",
        "lr", "points", "step_seconds", "peak_gpu_memory_gib", "finite",
    ]
    metrics_path = os.path.join(args.output_dir, "training_metrics.csv")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    step = starting_step
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
                distortion, channel_mse = weighted_distortion(
                    attribute.F, output["Full"].F, distortion_weights)
                if step == starting_step and distortion_weights == (1.0, 1.0, 1.0):
                    original = torch.mean((attribute.F - output["Full"].F) ** 2)
                    if not torch.allclose(distortion, original, rtol=1e-6, atol=1e-10):
                        raise RuntimeError("D111 weighted distortion equivalence failed")
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
                data_epoch, batch_in_epoch = batch_sampler.position(step)
                step += 1
                final_row = {
                    "step": step,
                    "data_epoch": data_epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "R_noise": float(rate.item()),
                    "D_F": float(distortion.item()),
                    "D_Y": float(channel_mse[0].item()),
                    "D_U": float(channel_mse[1].item()),
                    "D_V": float(channel_mse[2].item()),
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
        "starting_step": starting_step,
        "data_schedule": data_schedule,
        "next_data_step": step,
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
