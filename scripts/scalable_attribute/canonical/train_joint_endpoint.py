#!/usr/bin/env python3
"""Train canonical Base/Full with random single-endpoint objectives."""

import argparse
import csv
import hashlib
import json
import math
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
    parser.add_argument("--checkpoint-profile")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--initial-enhancement-checkpoint")
    parser.add_argument("--source-hard-reference-json")
    parser.add_argument("--source-conditioning-lambda", type=int)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--lambda-base", type=float, required=True)
    parser.add_argument("--lambda-full", type=float, required=True)
    parser.add_argument("--p-full", type=float, default=0.5)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--one-pass", action="store_true")
    parser.add_argument("--save-steps", type=int, nargs="+",
                        default=[250, 500, 1000])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--empty-cache-every", type=int, default=0)
    return parser.parse_args()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_tensor_sha256(value):
    digest = hashlib.sha256()
    for tensor in (value.C, value.F):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


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


def hard_contract(model, attribute, conditioning_lambda):
    previous = model.conditioning_lambda
    try:
        model.conditioning_lambda = conditioning_lambda
        hard = model.hard_reconstruct(attribute)
    finally:
        model.conditioning_lambda = previous
    rate = hard["prefix_rate"]
    if rate["num_residual_streams"] != 4:
        raise RuntimeError("Hard Base is not exactly r1-r4")
    if hard["base_bits"] != rate["bits_xlow"] + sum(rate["residual_bits"]):
        raise RuntimeError("Hard Base bit identity failed")
    if hard["full_bits"] != hard["base_bits"] + hard["enhancement_bits"]:
        raise RuntimeError("Hard Full bit identity failed")
    if (list(hard["encoded_Full"].tensor_stride) !=
            list(hard["Full"].tensor_stride) or
            not torch.equal(hard["encoded_Full"].C, hard["Full"].C)):
        raise RuntimeError("Hard Enhancement support mismatch")
    difference = float((hard["encoded_Full"].F - hard["Full"].F).abs().max())
    if difference != 0.0:
        raise RuntimeError("Hard Enhancement round-trip mismatch")
    return {
        "conditioning_lambda": conditioning_lambda,
        "base_bits": int(hard["base_bits"]),
        "enhancement_bits": int(hard["enhancement_bits"]),
        "full_bits": int(hard["full_bits"]),
        "num_base_residual_streams": 4,
        "native_r5_used": False,
        "hard_roundtrip_max_abs_difference": difference,
        "base_reconstruction_sha256": sparse_tensor_sha256(hard["Base"]),
        "full_reconstruction_sha256": sparse_tensor_sha256(hard["Full"]),
    }


def verify_source_hard_reference(observed, path):
    with open(path, encoding="utf-8") as handle:
        expected = json.load(handle)
    fields = ("conditioning_lambda", "base_bits", "enhancement_bits",
              "full_bits", "base_reconstruction_sha256",
              "full_reconstruction_sha256")
    missing = [name for name in fields if name not in expected]
    if missing:
        raise ValueError("Source hard reference lacks: " + ", ".join(missing))
    mismatches = {name: {"expected": expected[name], "observed": observed[name]}
                  for name in fields if expected[name] != observed[name]}
    if mismatches:
        raise RuntimeError("Historical 8K hard reproduction failed: " +
                           json.dumps(mismatches, sort_keys=True))
    return {"status": "PASS", "reference": path,
            "matched_fields": list(fields)}


def checkpoint(model, optimizer, args, base_config, metadata, step,
               endpoint_counts, data_schedule):
    return {
        "architecture": FINE_TUNE_ARCHITECTURE,
        "training_mode": JOINT_TRAINING_MODE,
        "scalable_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "source_conditioning_lambda": args.source_conditioning_lambda,
        "conditioning_lambda": args.conditioning_lambda,
        "lambda_base": args.lambda_base,
        "lambda_full": args.lambda_full,
        "p_full": args.p_full,
        "distortion_weights": [1.0, 1.0, 1.0],
        "base_config": base_config.to_dict(),
        "released_checkpoint": args.released_checkpoint,
        "base_synthesis_initialization": args.base_checkpoint,
        "enhancement_initialization": (
            args.initial_enhancement_checkpoint or
            "released ResidualVAE exact clone"),
        "endpoint_counts": dict(endpoint_counts),
        "data_schedule": data_schedule,
        "resolved_args": metadata,
    }


def verify_checkpoint_reload(path, model, optimizer, expected_step):
    state = torch.load(path, map_location="cpu")
    if int(state.get("step", -1)) != expected_step:
        raise RuntimeError("Checkpoint reload step mismatch")
    current = model.state_dict()
    saved = state.get("scalable_model") or {}
    if set(current) != set(saved):
        raise RuntimeError("Checkpoint reload model keys mismatch")
    for name, value in current.items():
        if not torch.equal(value.detach().cpu(), saved[name]):
            raise RuntimeError("Checkpoint reload tensor mismatch: " + name)
    if "optimizer" not in state or not state["optimizer"].get("param_groups"):
        raise RuntimeError("Checkpoint reload optimizer state missing")
    if len(state["optimizer"]["param_groups"]) != len(optimizer.param_groups):
        raise RuntimeError("Checkpoint reload optimizer groups mismatch")
    return True


def main():
    args = parse_args()
    source_commit = git_commit()
    for name in ("data_root", "train_file_list", "released_checkpoint",
                 "base_checkpoint", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(
            os.path.expanduser(getattr(args, name)))))
    if args.initial_enhancement_checkpoint:
        args.initial_enhancement_checkpoint = os.path.abspath(
            os.path.expandvars(os.path.expanduser(
                args.initial_enhancement_checkpoint)))
    if args.source_hard_reference_json:
        args.source_hard_reference_json = os.path.abspath(
            os.path.expandvars(os.path.expanduser(
                args.source_hard_reference_json)))
    if args.source_conditioning_lambda is None:
        args.source_conditioning_lambda = args.conditioning_lambda
    if args.checkpoint_profile is None:
        args.checkpoint_profile = os.path.basename(
            os.path.dirname(args.released_checkpoint))
    if os.path.basename(os.path.dirname(
            args.released_checkpoint)) != args.checkpoint_profile:
        raise ValueError("released checkpoint/profile mismatch")
    for label, path in (("train file list", args.train_file_list),
                        ("released checkpoint", args.released_checkpoint),
                        ("Base checkpoint", args.base_checkpoint),
                        ("initial Enhancement checkpoint",
                         args.initial_enhancement_checkpoint),
                        ("source hard reference",
                         args.source_hard_reference_json)):
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError("{} not found: {}".format(label, path))
    if args.conditioning_lambda <= 0 or args.lr <= 0:
        raise ValueError("conditioning-lambda and lr must be positive")
    if args.lambda_base != args.conditioning_lambda:
        raise ValueError("V1 requires lambda-base == conditioning-lambda")
    if args.lambda_full != args.conditioning_lambda:
        raise ValueError("V1 requires lambda-full == conditioning-lambda")
    if not 0.0 <= args.p_full <= 1.0:
        raise ValueError("p-full must lie in [0, 1]")
    if args.batch_size < 1 or args.smoke_steps < 1:
        raise ValueError("batch-size and smoke-steps must be positive")
    if args.empty_cache_every < 0:
        raise ValueError("empty-cache-every must be nonnegative")
    files = h5_files(args.data_root, args.train_file_list)
    one_pass_steps = int(math.ceil(len(files) / args.batch_size))
    if args.one_pass:
        if args.max_steps is not None and args.max_steps != one_pass_steps:
            raise ValueError("max-steps conflicts with calculated one-pass length")
        args.max_steps = one_pass_steps
    if args.max_steps is None or args.max_steps < 1:
        raise ValueError("max-steps is required unless --one-pass is used")
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
                     args.source_conditioning_lambda)
    model = CanonicalScalableModel(
        base, args.source_conditioning_lambda).cuda()
    enhancement_state = None
    if args.initial_enhancement_checkpoint:
        enhancement_state = torch.load(
            args.initial_enhancement_checkpoint, map_location="cpu")
        if enhancement_state.get(
                "architecture") != "canonical_independent_enhancement":
            raise ValueError("Initial Enhancement checkpoint architecture mismatch")
        if int(enhancement_state.get("conditioning_lambda", -1)) != int(
                args.source_conditioning_lambda):
            raise ValueError("Initial Enhancement source lambda mismatch")
        if float(enhancement_state.get("rd_lambda", -1)) != float(
                args.source_conditioning_lambda):
            raise ValueError("Initial Enhancement RD lambda mismatch")
        if tuple(float(value) for value in enhancement_state.get(
                "distortion_weights", ())) != (1.0, 1.0, 1.0):
            raise ValueError("Initial Enhancement is not D111")
        if os.path.realpath(enhancement_state.get(
                "base_synthesis_checkpoint", "")) != os.path.realpath(
                    args.base_checkpoint):
            raise ValueError("Initial Enhancement Base checkpoint mismatch")
        model.enhancement.vae.load_state_dict(
            enhancement_state["enhancement_vae"], strict=True)
    model.conditioning_lambda = args.conditioning_lambda
    model.set_trainable_scope("full")
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    assert_optimizer_contract(model, optimizer)

    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    sampler = ContinuationBatchSampler(
        len(dataset), args.batch_size, 0,
        args.smoke_steps if args.smoke_only else args.max_steps, args.seed)
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
            "source_conditioning_lambda": args.source_conditioning_lambda,
            "target_conditioning_lambda": args.conditioning_lambda,
            "prefix": args.base_checkpoint,
            "base_synthesis": args.base_checkpoint,
            "enhancement": (args.initial_enhancement_checkpoint or
                            "released ResidualVAE exact clone"),
            "optimizer": "fresh Adam",
        },
        "initialization_sha256": {
            "released_checkpoint": sha256(args.released_checkpoint),
            "base_checkpoint": sha256(args.base_checkpoint),
            "enhancement_checkpoint": (
                sha256(args.initial_enhancement_checkpoint)
                if args.initial_enhancement_checkpoint else None),
        },
        "optimizer": {"name": "Adam", "lr": args.lr,
                      "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "scheduler": None},
        "objective_base": "R_Base + lambda_base * D111(Base)",
        "objective_full": "R_Base + R_Enh + lambda_full * D111(Full)",
        "rate_normalization": "full-resolution N0",
        "data_schedule": data_schedule,
        "num_train_h5": len(files),
        "calculated_one_pass_steps": one_pass_steps,
        "git_commit": source_commit,
        "checkpoint_profile": args.checkpoint_profile,
        "experiment_role": (
            "cross_lambda_joint_diagnostic"
            if args.source_conditioning_lambda != args.conditioning_lambda
            else "same_lambda_joint_diagnostic"),
        "hostname": socket.gethostname(),
        "command": command,
    })
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(command + "\n")

    if args.smoke_only:
        # Physical coding is defined per H5. Never pass a collated multi-sample
        # tensor to the author G-PCC helper because it drops the batch column.
        hard_index = next(iter(ContinuationBatchSampler(
            len(dataset), 1, 0, 1, args.seed)))[0]
        hard_coords, hard_feats = collate_pointcloud_fn([dataset[hard_index]])
        hard_attribute = make_attribute(hard_coords, hard_feats)
        coords, feats = next(iter(loader))
        attribute = make_attribute(coords, feats)
        hard = {
            "source": hard_contract(
                model, hard_attribute, args.source_conditioning_lambda),
            "target": hard_contract(
                model, hard_attribute, args.conditioning_lambda),
        }
        if (args.source_conditioning_lambda != args.conditioning_lambda and
                args.source_hard_reference_json is None):
            raise ValueError(
                "Cross-lambda smoke requires --source-hard-reference-json")
        source_reproduction = (
            verify_source_hard_reference(
                hard["source"], args.source_hard_reference_json)
            if args.source_hard_reference_json else {"status": "NOT_REQUESTED"})
        smoke = {}
        call_count = {"value": 0}
        hook = model.enhancement.register_forward_hook(
            lambda *_: call_count.__setitem__("value", call_count["value"] + 1))
        for endpoint in ("Base", "Full"):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            before = call_count["value"]
            result = joint_endpoint_objective(
                model, attribute, endpoint,
                args.lambda_base, args.lambda_full)
            if not all(torch.isfinite(value).all() for value in (
                    result.loss, result.rate_base, result.rate_enhancement,
                    result.distortion)):
                raise RuntimeError(endpoint + " smoke objective is non-finite")
            result.loss.backward()
            calls = call_count["value"] - before
            if calls != (1 if endpoint == "Full" else 0):
                raise RuntimeError(endpoint + " Enhancement call contract failed")
            smoke[endpoint] = {
                "loss": float(result.loss.item()),
                "R_B_est": float(result.rate_base.item()),
                "R_E_est": float(result.rate_enhancement.item()),
                "D111": float(result.distortion.item()),
                "enhancement_forward_calls": calls,
                "gradients": assert_gradient_contract(model, endpoint),
            }
        hook.remove()
        endpoint_rng = random.Random(args.seed)
        endpoint_counts = {"Base": 0, "Full": 0}
        torch.cuda.reset_peak_memory_stats()
        last = None
        for step, (coords, feats) in enumerate(loader, 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            endpoint = sample_endpoint(endpoint_rng, args.p_full)
            attribute = make_attribute(coords, feats)
            result = joint_endpoint_objective(
                model, attribute, endpoint,
                args.lambda_base, args.lambda_full)
            if not all(torch.isfinite(value).all() for value in (
                    result.loss, result.rate_base, result.rate_enhancement,
                    result.distortion)):
                raise RuntimeError("Non-finite smoke step")
            result.loss.backward(); assert_gradient_contract(model, endpoint)
            optimizer.step(); torch.cuda.synchronize()
            endpoint_counts[endpoint] += 1
            last = {"step": step, "endpoint": endpoint,
                    "loss": float(result.loss.item())}
            del result, attribute, coords, feats
            if args.empty_cache_every and step % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
        if not all(endpoint_counts.values()):
            raise RuntimeError(
                "Smoke did not exercise both random endpoints: {}".format(
                    endpoint_counts))
        smoke_checkpoint = os.path.join(
            checkpoint_dir, "smoke_step_{:04d}.pth".format(args.smoke_steps))
        torch.save(checkpoint(
            model, optimizer, args, base_config, metadata, args.smoke_steps,
            endpoint_counts, data_schedule), smoke_checkpoint)
        reload_pass = verify_checkpoint_reload(
            smoke_checkpoint, model, optimizer, args.smoke_steps)
        write_json(os.path.join(args.output_dir, "smoke_summary.json"), {
            "status": "PASS", "endpoints": smoke,
            "hard_contract": hard,
            "hard_reference_h5": files[hard_index],
            "optimizer_smoke_physical_batch_size": args.batch_size,
            "historical_source_reproduction": source_reproduction,
            "smoke_steps": args.smoke_steps,
            "endpoint_counts": endpoint_counts,
            "last_step": last,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024 ** 3),
            "optimizer_contract": True,
            "checkpoint": smoke_checkpoint,
            "checkpoint_reload_pass": reload_pass,
            "source_conditioning_lambda": args.source_conditioning_lambda,
            "conditioning_lambda": args.conditioning_lambda,
            "lambda_base": args.lambda_base,
            "lambda_full": args.lambda_full,
        })
        print(json.dumps(smoke, indent=2))
        return

    fields = ["step", "endpoint", "R_B_est", "R_E_est", "D111",
              "D_Y", "D_U", "D_V", "loss", "gradient_norm", "lr",
              "points", "step_seconds", "memory_allocated_gib",
              "memory_reserved_gib", "max_memory_allocated_gib",
              "max_memory_reserved_gib", "memory_free_gib",
              "memory_total_gib"]
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
            memory_free, memory_total = torch.cuda.mem_get_info()
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
                "memory_allocated_gib": (
                    torch.cuda.memory_allocated() / 1024 ** 3),
                "memory_reserved_gib": (
                    torch.cuda.memory_reserved() / 1024 ** 3),
                "max_memory_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 1024 ** 3),
                "max_memory_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 1024 ** 3),
                "memory_free_gib": memory_free / 1024 ** 3,
                "memory_total_gib": memory_total / 1024 ** 3,
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
            del result, attribute, coords, feats
            if args.empty_cache_every and step % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    final_checkpoint = saved.get(str(step))
    if final_checkpoint is None:
        raise RuntimeError("Final checkpoint was not saved")
    reload_pass = verify_checkpoint_reload(
        final_checkpoint, model, optimizer, step)
    summary = {
        "status": "PASS", "steps": step,
        "runtime_seconds": time.monotonic() - started,
        "endpoint_counts": endpoint_counts,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024 ** 3),
        "checkpoints": saved,
        "checkpoint_reload_pass": reload_pass,
    }
    write_json(os.path.join(args.output_dir, "training_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
