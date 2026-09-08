#!/usr/bin/env python3
"""Fine-tune canonical scalable Attribute endpoints on processed MVUB H5."""

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

from basic_models.loss import get_bits
from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    FINE_TUNE_ARCHITECTURE, CanonicalScalableModel, load_finetuned_scalable,
    load_frozen_base)
from scalable_attribute.data import UncachedPCDataset, h5_files


ARMS = ("enhancement_only", "full_only", "random_base_full")
ARM_SCOPE = {
    "enhancement_only": "enhancement_only",
    "full_only": "full",
    "random_base_full": "full",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--validation-file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--enhancement-checkpoint", required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--conditioning-lambda", type=int, default=32768)
    parser.add_argument("--rd-lambda", type=float, default=32768)
    parser.add_argument("--distortion-weights", default="6,1,1")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[250, 500])
    parser.add_argument("--validation-steps", type=int, nargs="+",
                        default=[0, 250, 500])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_weights(value):
    weights = tuple(float(item) for item in value.split(","))
    if len(weights) != 3 or any(weight <= 0 for weight in weights):
        raise ValueError("distortion-weights must be three positive values")
    return weights


def weighted_distortion(reference, reconstruction, weights):
    channel_mse = torch.mean((reference - reconstruction) ** 2, dim=0)
    weight = channel_mse.new_tensor(weights)
    return torch.sum(weight * channel_mse) / torch.sum(weight), channel_mse


def estimated_rate(likelihoods, points):
    if likelihoods is None:
        return None
    if len(likelihoods) == 0:
        raise RuntimeError("Estimated rate received an empty likelihood list")
    return sum(get_bits(value) for value in likelihoods) / points


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty CSV")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path, row):
    """Append one metrics row without rewriting the complete trajectory."""
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sparse_snapshot(value):
    return {
        "coordinates": value.C.detach().cpu().clone(),
        "features": value.F.detach().cpu().clone(),
        "tensor_stride": tuple(value.tensor_stride),
    }


def assert_sparse_snapshot(snapshot, value, label):
    if tuple(value.tensor_stride) != snapshot["tensor_stride"]:
        raise RuntimeError(label + " tensor stride changed")
    if not torch.equal(value.C.detach().cpu(), snapshot["coordinates"]):
        raise RuntimeError(label + " coordinates changed")
    if not torch.equal(value.F.detach().cpu(), snapshot["features"]):
        difference = (value.F.detach().cpu() - snapshot["features"]).abs().max()
        raise RuntimeError("{} features changed; max_abs={}".format(
            label, float(difference)))


def assert_optimizer_contract(model, optimizer):
    expected = {id(parameter) for parameter in model.parameters()
                if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups
              for parameter in group["params"]}
    if expected != actual:
        raise RuntimeError("Optimizer parameters != requires_grad parameters")


def assert_group_gradient(module, label, required):
    gradients = [parameter.grad for parameter in module.parameters()]
    present = [gradient for gradient in gradients if gradient is not None]
    if required and not present:
        raise RuntimeError(label + " received no gradient")
    if not required and present:
        raise RuntimeError(label + " unexpectedly received a gradient")
    if any(not torch.isfinite(gradient).all() for gradient in present):
        raise RuntimeError(label + " received a non-finite gradient")


def gradient_statistics(module):
    parameters = [parameter for parameter in module.parameters()
                  if parameter.requires_grad]
    gradients = [parameter.grad for parameter in parameters
                 if parameter.grad is not None]
    if gradients:
        squared_norm = torch.stack([
            gradient.detach().float().pow(2).sum()
            for gradient in gradients]).sum()
        norm = float(squared_norm.sqrt().item())
        elements = sum(gradient.numel() for gradient in gradients)
    else:
        norm = 0.0
        elements = 0
    return {
        "present": bool(gradients),
        "parameter_tensors": len(parameters),
        "gradient_tensors": len(gradients),
        "gradient_elements": elements,
        "norm": norm,
    }


def total_gradient_norm(parameters):
    values = [parameter.grad.detach().float().pow(2).sum()
              for parameter in parameters if parameter.grad is not None]
    if not values:
        raise RuntimeError("No trainable gradients")
    result = torch.stack(values).sum().sqrt()
    if not torch.isfinite(result):
        raise RuntimeError("Gradient norm is non-finite")
    return float(result.item())


def make_attribute(coords, feats):
    return ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1, device="cuda")


@torch.no_grad()
def validate(model, loader, weights, step, output_dir):
    model.eval()
    rows = []
    for index, (coords, feats) in enumerate(loader):
        attribute = make_attribute(coords, feats)
        output = model.deterministic_forward(attribute)
        base_distortion, _ = weighted_distortion(
            attribute.F, output["Base"].F, weights)
        full_distortion, _ = weighted_distortion(
            attribute.F, output["Full"].F, weights)
        rows.append({
            "step": step,
            "batch": index,
            "points": len(attribute),
            "D_B611": float(base_distortion.item()),
            "D_F611": float(full_distortion.item()),
        })
    path = os.path.join(output_dir, "validation_step_{:04d}.csv".format(step))
    write_csv(path, rows)
    return {
        "step": step,
        "batches": len(rows),
        "mean_D_B611": sum(row["D_B611"] for row in rows) / len(rows),
        "mean_D_F611": sum(row["D_F611"] for row in rows) / len(rows),
        "csv": path,
    }


def complete_checkpoint(model, optimizer, args, base_config, metadata, step):
    return {
        "architecture": FINE_TUNE_ARCHITECTURE,
        "scalable_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "arm": args.arm,
        "trainable_scope": model.trainable_scope,
        "conditioning_lambda": args.conditioning_lambda,
        "rd_lambda": args.rd_lambda,
        "distortion_weights": list(parse_weights(args.distortion_weights)),
        "base_config": base_config.to_dict(),
        "released_checkpoint": args.released_checkpoint,
        "base_synthesis_initialization": args.base_synthesis_checkpoint,
        "enhancement_initialization": args.enhancement_checkpoint,
        "resolved_args": metadata,
    }


def main():
    args = parse_args()
    source_git_commit = git_commit()
    for name in (
            "data_root", "train_file_list", "validation_file_list",
            "released_checkpoint", "base_synthesis_checkpoint",
            "enhancement_checkpoint", "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(
            os.path.expandvars(os.path.expanduser(getattr(args, name)))))
    weights = parse_weights(args.distortion_weights)
    if args.batch_size < 1 or args.max_steps < 1 or args.lr <= 0:
        raise ValueError("batch-size, max-steps and lr must be positive")
    if args.conditioning_lambda != 32768 or args.rd_lambda != 32768:
        raise ValueError("MVUB V1 is locked to conditioning/rd lambda 32768")
    if weights != (6.0, 1.0, 1.0):
        raise ValueError("MVUB V1 is locked to D611")
    if any(step < 1 or step > args.max_steps for step in args.save_steps):
        raise ValueError("save-steps must lie within the run")
    if any(step < 0 or step > args.max_steps for step in args.validation_steps):
        raise ValueError("validation-steps must lie within the run")
    if len(set(args.save_steps)) != len(args.save_steps):
        raise ValueError("save-steps must be unique")
    if len(set(args.validation_steps)) != len(args.validation_steps):
        raise ValueError("validation-steps must be unique")
    if os.path.exists(args.output_dir):
        existing = set(os.listdir(args.output_dir))
        if existing - {"slurm"}:
            raise FileExistsError("Output directory already has run artifacts")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.exists(gpcc_link):
        os.symlink(os.path.abspath(args.gpcc_binary), gpcc_link)
    os.chdir(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base_config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(args.released_checkpoint, base_config).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)
    model = CanonicalScalableModel(base, args.conditioning_lambda).cuda()
    enhancement_state = torch.load(args.enhancement_checkpoint, map_location="cpu")
    if enhancement_state.get("architecture") != "canonical_independent_enhancement":
        raise ValueError("D611 initialization architecture mismatch")
    if int(enhancement_state.get("conditioning_lambda", -1)) != 32768:
        raise ValueError("D611 initialization conditioning lambda mismatch")
    if float(enhancement_state.get("rd_lambda", -1)) != 32768:
        raise ValueError("D611 initialization RD lambda mismatch")
    if tuple(float(value) for value in enhancement_state.get(
            "distortion_weights", ())) != weights:
        raise ValueError("D611 initialization distortion weights mismatch")
    model.enhancement.vae.load_state_dict(
        enhancement_state["enhancement_vae"], strict=True)

    train_files = h5_files(args.data_root, args.train_file_list)
    validation_files = h5_files(args.data_root, args.validation_file_list)
    if set(train_files) & set(validation_files):
        raise RuntimeError("MVUB train/validation overlap")
    train_loader = make_data_loader(
        UncachedPCDataset(train_files, color_format="yuv", normalize=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = make_data_loader(
        UncachedPCDataset(validation_files, color_format="yuv", normalize=True),
        batch_size=1, shuffle=False, num_workers=args.num_workers)

    correctness_coords, correctness_feats = next(iter(validation_loader))
    correctness_attribute = make_attribute(
        correctness_coords, correctness_feats)
    model.eval()
    legacy_deterministic = model.deterministic_forward(correctness_attribute)
    legacy_hard = model.hard_reconstruct(correctness_attribute)
    legacy_full = sparse_snapshot(legacy_deterministic["Full"])
    legacy_hard_full = sparse_snapshot(legacy_hard["Full"])
    legacy_base = sparse_snapshot(legacy_deterministic["Base"])
    legacy_bits = (legacy_hard["base_bits"], legacy_hard["enhancement_bits"],
                   legacy_hard["full_bits"])
    if legacy_bits[2] != legacy_bits[0] + legacy_bits[1]:
        raise RuntimeError("Initial Full bits identity failed")

    model.set_trainable_scope(ARM_SCOPE[args.arm])
    model.eval()
    scoped_deterministic = model.deterministic_forward(correctness_attribute)
    scoped_hard = model.hard_reconstruct(correctness_attribute)
    assert_sparse_snapshot(legacy_base, scoped_deterministic["Base"],
                           "step0 D611 Base")
    assert_sparse_snapshot(legacy_full, scoped_deterministic["Full"],
                           "step0 D611 deterministic Full")
    assert_sparse_snapshot(legacy_hard_full, scoped_hard["Full"],
                           "step0 D611 hard Full")
    scoped_bits = (scoped_hard["base_bits"], scoped_hard["enhancement_bits"],
                   scoped_hard["full_bits"])
    if scoped_bits != legacy_bits:
        raise RuntimeError("step0 D611 physical bits changed")

    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    assert_optimizer_contract(model, optimizer)
    command = shlex.join([sys.executable] + sys.argv)
    named_trainable = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    metadata = dict(vars(args))
    metadata.update({
        "architecture": FINE_TUNE_ARCHITECTURE,
        "trainable_scope": model.trainable_scope,
        "trainable_parameter_count": sum(parameter.numel()
                                         for parameter in trainable),
        "trainable_parameter_names": named_trainable,
        "optimizer": {"name": "Adam", "fresh": True, "lr": args.lr,
                      "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "scheduler": None},
        "num_train_h5": len(train_files),
        "num_validation_h5": len(validation_files),
        "shuffle": True,
        "drop_last": False,
        "git_commit": source_git_commit,
        "hostname": socket.gethostname(),
        "command": command,
        "step0_legacy_d611_exact": True,
    })
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(command + "\n")

    validation_summaries = []
    if 0 in args.validation_steps:
        validation_summaries.append(validate(
            model, validation_loader, weights, 0, args.output_dir))

    metrics_path = os.path.join(args.output_dir, "training_metrics.csv")
    endpoint_counts = {"Base": 0, "Full": 0}
    saved = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    step = 0
    enhancement_calls = {"count": 0}

    def count_enhancement(_module, _inputs, _output):
        enhancement_calls["count"] += 1

    hook = model.enhancement.register_forward_hook(count_enhancement)
    last_gradient_groups = None
    try:
        while step < args.max_steps:
            for coords, feats in train_loader:
                model.train()
                attribute = make_attribute(coords, feats)
                optimizer.zero_grad(set_to_none=True)
                enhancement_calls["count"] = 0
                if args.arm == "random_base_full":
                    endpoint = "Base" if random.random() < 0.5 else "Full"
                else:
                    endpoint = "Full"

                if endpoint == "Base":
                    output = model.forward_base_train(attribute)
                    rate_base = estimated_rate(
                        output["prefix_likelihoods"], len(attribute))
                    rate_enhancement = rate_base.new_zeros(())
                    distortion, channels = weighted_distortion(
                        attribute.F, output["Base"].F, weights)
                    if enhancement_calls["count"] != 0:
                        raise RuntimeError("Base batch invoked EnhancementVAE")
                else:
                    output = model.forward_full_train(attribute)
                    if args.arm == "enhancement_only":
                        rate_base = output["likelihood_E"].new_zeros(())
                    else:
                        rate_base = estimated_rate(
                            output["prefix_likelihoods"], len(attribute))
                    rate_enhancement = get_bits(
                        output["likelihood_E"]) / len(attribute)
                    distortion, channels = weighted_distortion(
                        attribute.F, output["Full"].F, weights)
                    if enhancement_calls["count"] != 1:
                        raise RuntimeError("Full batch did not invoke EnhancementVAE once")

                rate = rate_base + rate_enhancement
                lambda_distortion = args.rd_lambda * distortion
                loss = rate + lambda_distortion
                if not all(torch.isfinite(value).all() for value in (
                        rate_base, rate_enhancement, distortion,
                        lambda_distortion, loss)):
                    raise RuntimeError("Non-finite MVUB objective at step {}".format(
                        step + 1))
                loss.backward()

                full_scope = model.trainable_scope == "full"
                gradient_modules = {
                    "linear_in": model.base.prefix.model.linear_in,
                    "upscaler": model.base.prefix.model.upscaler,
                    "prefix_vae": model.base.prefix.model.VAE,
                    "embedder": model.base.prefix.model.embedder,
                    "base_synthesis": model.base.base_synthesis,
                    "enhancement_vae": model.enhancement,
                }
                gradient_required = {
                    "linear_in": full_scope,
                    "upscaler": full_scope,
                    "prefix_vae": full_scope,
                    "embedder": full_scope,
                    "base_synthesis": full_scope,
                    "enhancement_vae": endpoint == "Full",
                }
                gradient_groups = {}
                for name, module in gradient_modules.items():
                    assert_group_gradient(
                        module, name, gradient_required[name])
                    gradient_groups[name] = gradient_statistics(module)
                if args.arm == "enhancement_only":
                    assert_group_gradient(model.base, "Frozen Base", False)
                norm = total_gradient_norm(trainable)
                optimizer.step()
                step += 1
                endpoint_counts[endpoint] += 1
                row = {
                    "step": step,
                    "endpoint": endpoint,
                    "R_B_est": float(rate_base.item()),
                    "R_E_est": float(rate_enhancement.item()),
                    "D_611": float(distortion.item()),
                    "D_Y": float(channels[0].item()),
                    "D_U": float(channels[1].item()),
                    "D_V": float(channels[2].item()),
                    "lambda_D": float(lambda_distortion.item()),
                    "loss": float(loss.item()),
                    "gradient_norm": norm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "points": len(attribute),
                    "peak_gpu_memory_gib": (
                        torch.cuda.max_memory_allocated() / 1024 ** 3),
                }
                for name, statistics in gradient_groups.items():
                    for field, value in statistics.items():
                        row["grad_{}_{}".format(name, field)] = value
                append_csv_row(metrics_path, row)
                last_gradient_groups = gradient_groups
                if step in args.save_steps or step == args.max_steps:
                    path = os.path.join(
                        checkpoint_dir, "step_{}.pth".format(step))
                    torch.save(complete_checkpoint(
                        model, optimizer, args, base_config, metadata, step), path)
                    saved[str(step)] = path
                if step in args.validation_steps:
                    validation_summaries.append(validate(
                        model, validation_loader, weights, step,
                        args.output_dir))
                if step >= args.max_steps:
                    break
    finally:
        hook.remove()

    final_path = saved.get(str(args.max_steps))
    if final_path is None:
        final_path = os.path.join(
            checkpoint_dir, "step_{}.pth".format(args.max_steps))
        torch.save(complete_checkpoint(
            model, optimizer, args, base_config, metadata, step), final_path)
        saved[str(args.max_steps)] = final_path

    model.eval()
    pre_reload_deterministic = model.deterministic_forward(
        correctness_attribute)
    pre_reload_hard = model.hard_reconstruct(correctness_attribute)
    deterministic_snapshot = sparse_snapshot(pre_reload_deterministic["Full"])
    hard_snapshot = sparse_snapshot(pre_reload_hard["Full"])
    hard_bits = (pre_reload_hard["base_bits"],
                 pre_reload_hard["enhancement_bits"],
                 pre_reload_hard["full_bits"])
    if hard_bits[2] != hard_bits[0] + hard_bits[1]:
        raise RuntimeError("Final Full bits identity failed")
    loaded = load_finetuned_scalable(model, final_path, args.conditioning_lambda)
    if int(loaded["step"]) != args.max_steps:
        raise RuntimeError("Reloaded checkpoint step mismatch")
    post_reload_deterministic = model.deterministic_forward(
        correctness_attribute)
    post_reload_hard = model.hard_reconstruct(correctness_attribute)
    assert_sparse_snapshot(
        deterministic_snapshot, post_reload_deterministic["Full"],
        "checkpoint deterministic reload")
    assert_sparse_snapshot(
        hard_snapshot, post_reload_hard["Full"], "checkpoint hard reload")
    post_bits = (post_reload_hard["base_bits"],
                 post_reload_hard["enhancement_bits"],
                 post_reload_hard["full_bits"])
    if post_bits != hard_bits or post_bits[2] != post_bits[0] + post_bits[1]:
        raise RuntimeError("Checkpoint reload physical bits changed")

    summary = {
        "status": "PASS",
        "arm": args.arm,
        "steps": step,
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "endpoint_counts": endpoint_counts,
        "trainable_parameter_count": metadata["trainable_parameter_count"],
        "step0_legacy_d611_exact": True,
        "checkpoint_reload_deterministic_exact": True,
        "checkpoint_reload_hard_exact": True,
        "full_bits_identity": True,
        "hard_bits": {"base": post_bits[0], "enhancement": post_bits[1],
                      "full": post_bits[2]},
        "last_gradient_groups": last_gradient_groups,
        "checkpoints": saved,
        "validation": validation_summaries,
    }
    write_json(os.path.join(args.output_dir, "training_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
