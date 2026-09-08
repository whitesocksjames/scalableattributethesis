#!/usr/bin/env python3
"""Train a clean-initialized EnhancementVAE on a fixed RWTT/MVUB schedule."""

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
from collections import Counter

import MinkowskiEngine as ME
import h5py
import numpy as np
import torch

from basic_models.loss import get_bits
from data_utils.dataloaders.attribute_dataloader import collate_pointcloud_fn
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_frozen_base)
from scalable_attribute.data import UncachedPCDataset, h5_files


DOMAINS = ("RWTT", "MVUB")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--rwtt-root", required=True)
    parser.add_argument("--rwtt-file-list", required=True)
    parser.add_argument("--mvub-root", required=True)
    parser.add_argument("--mvub-file-list", required=True)
    parser.add_argument("--andrew-file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--conditioning-lambda", type=int, default=32768)
    parser.add_argument("--rd-lambda", type=float, default=32768)
    parser.add_argument("--distortion-weights", default="6,1,1")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--physical-batch-size", type=int, choices=(2, 4),
                        default=4)
    parser.add_argument("--effective-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1762)
    parser.add_argument("--mvub-updates", type=int, required=True)
    parser.add_argument("--save-steps", type=int, nargs="+",
                        default=(250, 500, 1000, 1762))
    parser.add_argument("--validation-steps", type=int, nargs="*",
                        default=(0, 500, 1000, 1762))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--largest-mvub-samples", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_weights(value):
    weights = tuple(float(item) for item in value.split(","))
    if len(weights) != 3 or any(weight <= 0 for weight in weights):
        raise ValueError("distortion-weights must be wY,wU,wV")
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
    values = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("EnhancementVAE has a non-finite gradient")
        values.append(parameter.grad.detach().float().pow(2).sum())
    if not values:
        raise RuntimeError("EnhancementVAE received no gradients")
    result = torch.stack(values).sum().sqrt()
    if not torch.isfinite(result):
        raise RuntimeError("EnhancementVAE gradient norm is non-finite")
    return float(result.item())


def relative(root, paths):
    return [os.path.relpath(path, root) for path in paths]


def largest_paths(paths, count):
    shortlist = sorted(paths, key=os.path.getsize, reverse=True)[:max(count * 4, 64)]
    sizes = []
    for path in shortlist:
        with h5py.File(path, "r") as handle:
            sizes.append((int(handle["coords"].shape[0]), path))
    return [path for _, path in sorted(sizes, reverse=True)[:count]]


def build_schedule(rwtt_files, mvub_files, args, starting_step=0):
    continuation_updates = args.max_steps - starting_step
    rwtt_updates = continuation_updates - args.mvub_updates
    if rwtt_updates < 0:
        raise ValueError("mvub-updates exceeds max-steps")
    required = {
        "RWTT": rwtt_updates * args.effective_batch_size,
        "MVUB": args.mvub_updates * args.effective_batch_size,
    }
    if required["RWTT"] > len(rwtt_files) or required["MVUB"] > len(mvub_files):
        raise ValueError("Fixed schedule would repeat a training H5")
    rwtt = list(rwtt_files)
    mvub = list(mvub_files)
    random.Random(args.seed + 1001).shuffle(rwtt)
    random.Random(args.seed + 2001).shuffle(mvub)
    if args.largest_mvub_samples:
        if rwtt_updates:
            raise ValueError("largest-mvub-samples is only for an all-MVUB gate")
        mvub = largest_paths(mvub_files, required["MVUB"])
    selected = {"RWTT": rwtt[:required["RWTT"]],
                "MVUB": mvub[:required["MVUB"]]}
    domains = ["RWTT"] * rwtt_updates + ["MVUB"] * args.mvub_updates
    random.Random(args.seed + 3001).shuffle(domains)
    cursors = {domain: 0 for domain in DOMAINS}
    schedule = []
    for step, domain in enumerate(domains, starting_step + 1):
        begin = cursors[domain]
        end = begin + args.effective_batch_size
        batch = selected[domain][begin:end]
        cursors[domain] = end
        schedule.append({"step": step, "domain": domain, "files": batch})
    actual_counts = Counter(item["domain"] for item in schedule)
    if (actual_counts["RWTT"] != rwtt_updates or
            actual_counts["MVUB"] != args.mvub_updates):
        raise RuntimeError("Domain schedule count mismatch")
    return schedule


def make_attribute(items):
    coords, feats = collate_pointcloud_fn(items)
    return ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1, device="cuda")


@torch.no_grad()
def validate(model, dataset, weights, step, output_dir):
    model.eval()
    rows = []
    for index in range(len(dataset)):
        attribute = make_attribute([dataset[index]])
        output = model.deterministic_forward(attribute)
        distortion, channels = weighted_distortion(
            attribute.F, output["Full"].F, weights)
        rows.append({
            "step": step, "index": index, "points": len(attribute),
            "D_F611": float(distortion.item()),
            "D_Y": float(channels[0].item()),
            "D_U": float(channels[1].item()),
            "D_V": float(channels[2].item()),
        })
    path = os.path.join(output_dir, "validation_step_{:04d}.csv".format(step))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model.train()
    return {"step": step, "num_h5": len(rows),
            "mean_D_F611": sum(row["D_F611"] for row in rows) / len(rows),
            "csv": path}


def main():
    args = parse_args()
    source_commit = git_commit()
    weights = parse_weights(args.distortion_weights)
    if args.conditioning_lambda != 32768 or args.rd_lambda != 32768:
        raise ValueError("Direct-D611 experiment is locked to lambda 32768")
    if weights != (6.0, 1.0, 1.0):
        raise ValueError("Direct-D611 experiment is locked to D611")
    if args.lr != 5e-5:
        raise ValueError("Direct-D611 experiment is locked to lr=5e-5")
    if args.effective_batch_size != 4:
        raise ValueError("effective-batch-size must be 4")
    if args.effective_batch_size % args.physical_batch_size:
        raise ValueError("physical batch size must divide effective batch size")
    if any(step < 1 or step > args.max_steps for step in args.save_steps):
        raise ValueError("save-steps must lie within the run")
    if any(step < 0 or step > args.max_steps for step in args.validation_steps):
        raise ValueError("validation-steps must lie within the run")
    if os.path.exists(args.output_dir) and set(os.listdir(args.output_dir)) - {"slurm"}:
        raise FileExistsError("Output directory already has run artifacts")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base_config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(args.released_checkpoint, base_config).cuda()
    load_frozen_base(base, args.base_synthesis_checkpoint,
                     args.released_checkpoint, args.conditioning_lambda)
    model = CanonicalScalableModel(base, args.conditioning_lambda).cuda()
    model.set_trainable_scope("enhancement_only")
    model.train()
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad
                            for parameter in model.base.parameters()):
        raise RuntimeError("Enhancement-only trainable scope failed")

    released_parameters = dict(model.base.prefix.model.VAE.named_parameters())
    enhancement_parameters = dict(model.enhancement.vae.named_parameters())
    if released_parameters.keys() != enhancement_parameters.keys():
        raise RuntimeError("Released/Enhancement parameter keys differ")
    for name, value in enhancement_parameters.items():
        source = released_parameters[name]
        if not torch.equal(value.detach(), source.detach()):
            raise RuntimeError("Clean clone value mismatch: " + name)
        if value.data_ptr() == source.data_ptr():
            raise RuntimeError("Enhancement still shares storage: " + name)

    base_snapshot = {name: value.detach().cpu().clone()
                     for name, value in model.base.state_dict().items()}
    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    starting_step = 0
    resume_lineage = None
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu")
        if resume.get("architecture") != "canonical_independent_enhancement":
            raise ValueError("Enhancement resume architecture mismatch")
        if int(resume.get("conditioning_lambda", -1)) != args.conditioning_lambda:
            raise ValueError("Enhancement resume conditioning lambda mismatch")
        if float(resume.get("rd_lambda", -1)) != args.rd_lambda:
            raise ValueError("Enhancement resume RD lambda mismatch")
        if tuple(float(value) for value in resume.get(
                "distortion_weights", ())) != weights:
            raise ValueError("Enhancement resume distortion weights mismatch")
        if os.path.realpath(resume.get("released_checkpoint", "")) != os.path.realpath(
                args.released_checkpoint):
            raise ValueError("Enhancement resume released checkpoint mismatch")
        if os.path.realpath(resume.get(
                "base_synthesis_checkpoint", "")) != os.path.realpath(
                    args.base_synthesis_checkpoint):
            raise ValueError("Enhancement resume Base checkpoint mismatch")
        if resume.get("base_config") != base_config.to_dict():
            raise ValueError("Enhancement resume Base config mismatch")
        if (int(resume.get("physical_batch_size", -1)) !=
                args.physical_batch_size or
                int(resume.get("effective_batch_size", -1)) !=
                args.effective_batch_size):
            raise ValueError("Enhancement resume batch-size mismatch")
        if "enhancement_vae" not in resume or "optimizer" not in resume:
            raise ValueError("Enhancement resume lacks model or optimizer state")
        starting_step = int(resume.get("step", -1))
        if starting_step < 0 or args.max_steps <= starting_step:
            raise ValueError("max-steps must exceed resumed global step")
        model.enhancement.vae.load_state_dict(
            resume["enhancement_vae"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        resume_lineage = resume.get("initialization_lineage")

    if any(step <= starting_step for step in args.save_steps):
        raise ValueError("save-steps must be after the starting step")
    if any(step < starting_step for step in args.validation_steps):
        raise ValueError("validation-steps must not precede the starting step")

    rwtt_files = h5_files(args.rwtt_root, args.rwtt_file_list)
    mvub_files = (h5_files(args.mvub_root, args.mvub_file_list)
                  if args.mvub_updates else [])
    andrew_files = (h5_files(args.mvub_root, args.andrew_file_list)
                    if args.validation_steps else [])
    if set(mvub_files) & set(andrew_files):
        raise RuntimeError("MVUB train/Andrew validation overlap")
    schedule = build_schedule(rwtt_files, mvub_files, args, starting_step)
    schedule_record = []
    for item in schedule:
        root = args.rwtt_root if item["domain"] == "RWTT" else args.mvub_root
        schedule_record.append({"step": item["step"], "domain": item["domain"],
                                "files": relative(root, item["files"])})
    schedule_path = os.path.join(args.output_dir, "domain_schedule.json")
    write_json(schedule_path, schedule_record)
    counted_domains = Counter(item["domain"] for item in schedule)
    domain_counts = {domain: counted_domains[domain] for domain in DOMAINS}
    subject_counts = Counter()
    frame_counts = Counter()
    for item in schedule_record:
        if item["domain"] != "MVUB":
            continue
        for path in item["files"]:
            subject_counts[path.split(os.sep)[0]] += 1
            name = os.path.basename(path)
            frame = name.split("_P", 1)[0]
            frame_counts[frame] += 1

    datasets = {
        "RWTT": UncachedPCDataset(rwtt_files, color_format="yuv", normalize=True),
        "MVUB": UncachedPCDataset(mvub_files, color_format="yuv", normalize=True),
    }
    indices = {domain: {path: i for i, path in enumerate(files)}
               for domain, files in (("RWTT", rwtt_files), ("MVUB", mvub_files))}
    andrew = (UncachedPCDataset(andrew_files, color_format="yuv", normalize=True)
              if andrew_files else None)
    command = shlex.join([sys.executable] + sys.argv)
    metadata = dict(vars(args))
    metadata.update({
        "architecture": "canonical_independent_enhancement",
        "initialization_lineage": resume_lineage or {
            "released": args.released_checkpoint,
            "base": args.base_synthesis_checkpoint,
            "enhancement": "exact independent clone of released ResidualVAE",
            "old_enhancement_checkpoint_loaded": False,
            "optimizer": "fresh Adam",
        },
        "resumed_from": args.resume_checkpoint,
        "starting_step": starting_step,
        "additional_updates": args.max_steps - starting_step,
        "optimizer_state_restored": bool(args.resume_checkpoint),
        "stage_boundary_dataloader_reconstructed": bool(args.resume_checkpoint),
        "base_config": base_config.to_dict(),
        "optimizer": {"name": "Adam", "lr": args.lr,
                      "betas": [0.9, 0.999], "weight_decay": 0.0,
                      "scheduler": None},
        "objective": "R_E + rd_lambda*(6*MSE_Y+MSE_U+MSE_V)/8",
        "rate": "-sum(log2(likelihood_E))/N_full",
        "gradient_accumulation": args.effective_batch_size // args.physical_batch_size,
        "domain_updates": domain_counts,
        "domain_samples": {key: value * args.effective_batch_size
                           for key, value in domain_counts.items()},
        "mvub_subject_samples": dict(subject_counts),
        "mvub_unique_frames": len(frame_counts),
        "domain_schedule": schedule_path,
        "git_commit": source_commit,
        "hostname": socket.gethostname(),
        "command": command,
    })
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(command + "\n")

    def save_checkpoint(step):
        path = os.path.join(checkpoint_dir, "step_{}.pth".format(step))
        torch.save({
            "architecture": "canonical_independent_enhancement",
            "enhancement_vae": model.enhancement.vae.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step,
            "conditioning_lambda": args.conditioning_lambda,
            "rd_lambda": args.rd_lambda,
            "distortion_weights": list(weights),
            "released_checkpoint": args.released_checkpoint,
            "base_synthesis_checkpoint": args.base_synthesis_checkpoint,
            "base_config": base_config.to_dict(),
            "initialization_lineage": metadata["initialization_lineage"],
            "arm_name": args.arm_name,
            "physical_batch_size": args.physical_batch_size,
            "gradient_accumulation": metadata["gradient_accumulation"],
            "effective_batch_size": args.effective_batch_size,
            "domain_updates": metadata["domain_updates"],
            "domain_samples": metadata["domain_samples"],
            "domain_schedule": schedule_record,
            "resolved_args": metadata,
        }, path)
        return path

    validation = []
    if starting_step in args.validation_steps:
        validation.append(validate(
            model, andrew, weights, starting_step, args.output_dir))
    metrics_path = os.path.join(args.output_dir, "training_metrics.csv")
    fields = ("step", "domain", "R_E", "D611", "D_Y", "D_U", "D_V",
              "lambda_D", "loss", "gradient_norm", "points", "lr",
              "step_seconds", "peak_allocated_gib", "peak_reserved_gib", "finite")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    saved = {}
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in schedule:
            step_started = time.monotonic()
            domain = item["domain"]
            dataset = datasets[domain]
            batch_indices = [indices[domain][path] for path in item["files"]]
            micro = args.physical_batch_size
            cpu_batches = [
                [dataset[index] for index in batch_indices[begin:begin + micro]]
                for begin in range(0, len(batch_indices), micro)]
            total_points = sum(sum(len(sample[0]) for sample in batch)
                               for batch in cpu_batches)
            optimizer.zero_grad(set_to_none=True)
            aggregate = {key: 0.0 for key in
                         ("R_E", "D611", "D_Y", "D_U", "D_V", "loss")}
            for batch in cpu_batches:
                attribute = make_attribute(batch)
                output = model(attribute)
                likelihood = output["likelihood_E"]
                rate = get_bits(likelihood) / len(attribute)
                distortion, channels = weighted_distortion(
                    attribute.F, output["Full"].F, weights)
                loss = rate + args.rd_lambda * distortion
                if not all(torch.isfinite(value).all() for value in
                           (likelihood, rate, distortion, loss)):
                    raise RuntimeError("Non-finite objective at step {}".format(
                        item["step"]))
                fraction = len(attribute) / float(total_points)
                (loss * fraction).backward()
                aggregate["R_E"] += float(rate.item()) * fraction
                aggregate["D611"] += float(distortion.item()) * fraction
                aggregate["D_Y"] += float(channels[0].item()) * fraction
                aggregate["D_U"] += float(channels[1].item()) * fraction
                aggregate["D_V"] += float(channels[2].item()) * fraction
                aggregate["loss"] += float(loss.item()) * fraction
            if any(parameter.grad is not None for parameter in model.base.parameters()):
                raise RuntimeError("Frozen Base received a gradient")
            norm = gradient_norm(trainable)
            optimizer.step()
            torch.cuda.synchronize()
            row = {
                "step": item["step"], "domain": domain,
                **aggregate,
                "lambda_D": args.rd_lambda * aggregate["D611"],
                "gradient_norm": norm, "points": total_points,
                "lr": optimizer.param_groups[0]["lr"],
                "step_seconds": time.monotonic() - step_started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
                "finite": True,
            }
            writer.writerow(row)
            handle.flush()
            if item["step"] == 1 or item["step"] % 20 == 0:
                print(json.dumps(row), flush=True)
            if item["step"] in args.save_steps:
                saved[str(item["step"])] = save_checkpoint(item["step"])
            if item["step"] in args.validation_steps:
                validation.append(validate(
                    model, andrew, weights, item["step"], args.output_dir))

    for name, value in model.base.state_dict().items():
        if not torch.equal(value.detach().cpu(), base_snapshot[name]):
            raise RuntimeError("Frozen Base parameter changed: " + name)
    final_path = saved.get(str(args.max_steps)) or save_checkpoint(args.max_steps)
    reloaded = torch.load(final_path, map_location="cpu")
    if int(reloaded["step"]) != args.max_steps:
        raise RuntimeError("Final checkpoint reload step mismatch")
    summary = {
        "status": "PASS", "steps": args.max_steps,
        "starting_step": starting_step,
        "additional_updates": args.max_steps - starting_step,
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "base_bitwise_unchanged": True, "base_gradients_none": True,
        "clean_released_clone_exact_before_resume": True,
        "only_enhancement_trainable": True,
        "domain_updates": domain_counts,
        "checkpoints": saved, "validation": validation,
        "checkpoint_reload": True,
    }
    write_json(os.path.join(args.output_dir, "training_summary.json"), summary)
    print("DIRECT D611 MIXED TRAINING PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
