#!/usr/bin/env python3
"""Train one Base-only arm in the 4K/2K distribution rescue screen."""

import argparse
import csv
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import platform

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.dataloaders.attribute_dataloader import collate_pointcloud_fn
from scalable_attribute.canonical.base_rescue import (
    DeterministicWeightedBatchSampler, base_rescue_objective,
    classify_difficulty, load_difficulty_scores, sample_key)
from scalable_attribute.canonical.config import (
    BaseSynthesisConfig, add_base_architecture_arguments)
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.data import UncachedPCDataset, h5_files


ARCHITECTURE = "canonical_base_rescue_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-file-list", required=True)
    p.add_argument("--content-statistics", required=True)
    p.add_argument("--released-checkpoint", required=True)
    p.add_argument("--initial-base-checkpoint")
    p.add_argument("--checkpoint-profile", required=True)
    p.add_argument("--conditioning-lambda", type=int, required=True)
    p.add_argument("--trainable-scope", choices=("base_synthesis_only", "base_path"), required=True)
    p.add_argument("--sampling", choices=("uniform", "high_energy"), required=True)
    p.add_argument("--prefix-lr", type=float, default=1e-5)
    p.add_argument("--base-synthesis-lr", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--save-steps", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expected-source-commit", required=True)
    p.add_argument("--smoke-only", action="store_true")
    return add_base_architecture_arguments(p).parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def assert_source_state(expected):
    head = git_commit()
    if head != expected:
        raise RuntimeError(
            "Runtime source HEAD mismatch: expected {}, got {}".format(expected, head))
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot verify runtime source cleanliness") from error
    if dirty:
        raise RuntimeError("Runtime source worktree is dirty:\n" + dirty)
    return head


def assert_inputs_exist(args):
    required_files = {
        "train_file_list": args.train_file_list,
        "content_statistics": args.content_statistics,
        "released_checkpoint": args.released_checkpoint,
    }
    if args.initial_base_checkpoint:
        required_files["initial_base_checkpoint"] = args.initial_base_checkpoint
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError("data_root does not exist: " + args.data_root)
    for name, path in required_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("{} does not exist: {}".format(name, path))


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def environment_fingerprint():
    gpu_count = torch.cuda.device_count()
    return {
        "cluster": "N30R3",
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpu_models": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
    }


def module_gradient(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Non-finite module gradient")
    norm = 0.0 if not gradients else float(torch.stack([
        g.detach().float().pow(2).sum() for g in gradients
    ]).sum().sqrt().item())
    return {"present": bool(gradients), "tensors": len(gradients), "norm": norm}


def gradient_groups(model):
    return {
        "linear_in": model.prefix.model.linear_in,
        "upscaler": model.prefix.model.upscaler,
        "prefix_vae_shared_r1_r4": model.prefix.model.VAE,
        "embedder": model.prefix.model.embedder,
        "base_synthesis": model.base_synthesis,
    }


def assert_gradient_contract(model, scope):
    summary = {name: module_gradient(module)
               for name, module in gradient_groups(model).items()}
    prefix_expected = scope == "base_path"
    for name in ("linear_in", "upscaler", "prefix_vae_shared_r1_r4", "embedder"):
        if summary[name]["present"] != prefix_expected:
            raise RuntimeError("Gradient scope mismatch for " + name)
    if not summary["base_synthesis"]["present"]:
        raise RuntimeError("BaseSynthesis received no gradient")
    return summary


def assert_optimizer_contract(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if expected != actual:
        raise RuntimeError("optimizer params != requires_grad params")


def make_optimizer(model, args):
    groups = []
    if args.trainable_scope == "base_path":
        groups.append({"name": "native_prefix", "params": list(model.prefix.parameters()),
                       "lr": args.prefix_lr})
    groups.append({"name": "base_synthesis", "params": list(model.base_synthesis.parameters()),
                   "lr": args.base_synthesis_lr})
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), weight_decay=0.0)
    assert_optimizer_contract(model, optimizer)
    return optimizer


def load_initial_base(model, path, released_checkpoint, lmb):
    if path is None:
        if not model.config.zero_init:
            raise ValueError("Fresh Base rescue initialization must be zero-output")
        return {"kind": "fresh_zero_output", "sha256": None, "step": 0}
    state = torch.load(path, map_location="cpu")
    if state.get("architecture") != "canonical_base_predict_correct":
        raise ValueError("Initial Base architecture mismatch")
    if state.get("config") != model.config.to_dict():
        raise ValueError("Initial Base config mismatch")
    if int(state.get("base_lambda", -1)) != int(lmb):
        raise ValueError("Initial Base lambda mismatch")
    if os.path.realpath(state.get("base_checkpoint", "")) != os.path.realpath(released_checkpoint):
        raise ValueError("Initial Base released checkpoint mismatch")
    model.base_synthesis.load_state_dict(state["base_synthesis"], strict=True)
    return {"kind": "selected_base", "sha256": sha256(path), "step": int(state["step"])}


def verify_saved_checkpoint(path, model, optimizer, expected_step):
    """Fail closed if the persistent training state cannot be reloaded exactly."""
    state = torch.load(path, map_location="cpu")
    if state.get("architecture") != ARCHITECTURE or int(state.get("step", -1)) != expected_step:
        raise RuntimeError("Saved rescue checkpoint metadata mismatch")
    current = model.state_dict()
    saved = state["base_model"]
    if current.keys() != saved.keys():
        raise RuntimeError("Saved rescue checkpoint state keys mismatch")
    for name, value in current.items():
        if not torch.equal(value.detach().cpu(), saved[name].detach().cpu()):
            raise RuntimeError("Saved rescue checkpoint tensor mismatch: " + name)
    probe = make_optimizer(model, argparse.Namespace(
        trainable_scope=state["trainable_scope"],
        prefix_lr=state["optimizer"]["param_groups"][0]["lr"]
        if state["trainable_scope"] == "base_path" else 1e-5,
        base_synthesis_lr=state["optimizer"]["param_groups"][-1]["lr"]))
    probe.load_state_dict(state["optimizer"])
    if len(probe.state) != len(optimizer.state):
        raise RuntimeError("Saved optimizer state did not reload completely")
    return True


def main():
    args = parse_args()
    for name in ("data_root", "train_file_list", "content_statistics",
                 "released_checkpoint", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(os.path.expanduser(getattr(args, name)))))
    if args.initial_base_checkpoint:
        args.initial_base_checkpoint = os.path.abspath(os.path.expandvars(
            os.path.expanduser(args.initial_base_checkpoint)))
    source_commit = assert_source_state(args.expected_source_commit)
    assert_inputs_exist(args)
    if args.batch_size < 1 or args.max_steps < 1:
        raise ValueError("batch-size/max-steps must be positive")
    if args.conditioning_lambda <= 0 or args.prefix_lr <= 0 or args.base_synthesis_lr <= 0:
        raise ValueError("lambda/LRs must be positive")
    if len(set(args.save_steps)) != len(args.save_steps):
        raise ValueError("save-steps must be unique")
    if not args.smoke_only and any(x < 1 or x > args.max_steps for x in args.save_steps):
        raise ValueError("save-steps outside training range")
    # Complete data/difficulty validation must precede any output creation.
    files = h5_files(args.data_root, args.train_file_list)
    scores = load_difficulty_scores(args.content_statistics)
    labels, thresholds = classify_difficulty(files, scores)
    if os.path.exists(args.output_dir) and set(os.listdir(args.output_dir)) - {"slurm"}:
        raise FileExistsError("Output directory already contains run artifacts")
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    random_seed = args.seed
    np.random.seed(random_seed); torch.manual_seed(random_seed); torch.cuda.manual_seed_all(random_seed)
    high_weight = 3.0 if args.sampling == "high_energy" else 1.0
    steps = 2 if args.smoke_only else args.max_steps
    sampler = DeterministicWeightedBatchSampler(
        labels, args.batch_size, steps, args.seed, high_weight)
    sample_meta = sampler.metadata(files)

    config = BaseSynthesisConfig.from_args(args)
    model = CanonicalBaseModel(args.released_checkpoint, config).cuda()
    initialization = load_initial_base(
        model, args.initial_base_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)
    model.set_trainable_scope(args.trainable_scope)
    optimizer = make_optimizer(model, args)
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers,
        collate_fn=collate_pointcloud_fn, pin_memory=True)

    command = shlex.join([sys.executable] + sys.argv)
    metadata = {
        **vars(args), "architecture": ARCHITECTURE,
        "objective": "R_B_est(r1-r4)/N_full + lambda*D111(Base)",
        "quantization_path": "training=True Uniform-noise for both scopes",
        "historical_continuation_equivalent": False,
        "released_checkpoint_sha256": sha256(args.released_checkpoint),
        "initialization": initialization,
        "git_commit": source_commit, "hostname": socket.gethostname(),
        "sampling": {**thresholds, **sample_meta}, "command": command,
    }
    fingerprint = environment_fingerprint()
    metadata["environment"] = fingerprint
    write_json(os.path.join(args.output_dir, "resolved_args.json"), metadata)
    write_json(os.path.join(args.output_dir, "environment_fingerprint.json"), fingerprint)
    write_json(os.path.join(args.output_dir, "sampling.json"), {**thresholds, **sample_meta})
    with open(os.path.join(args.output_dir, "selected_h5.tsv"), "w", encoding="utf-8") as h:
        h.write("class\tscore\tpath\n")
        for label, path in zip(labels, files):
            h.write("{}\t{:.17g}\t{}\n".format(label, scores[sample_key(path)], path))
    with open(os.path.join(args.output_dir, "sampled_h5.tsv"), "w", encoding="utf-8") as h:
        h.write("step\tslot\tclass\tpath\n")
        for draw, index in enumerate(sampler.draws):
            h.write("{}\t{}\t{}\t{}\n".format(
                draw // args.batch_size + 1, draw % args.batch_size,
                labels[index], files[index]))
    with open(os.path.join(args.output_dir, "command.txt"), "w", encoding="utf-8") as h:
        h.write(command + "\n")

    fields = ["step","loss","R_B_est","D111","D_Y","D_U","D_V",
              "gradient_norm","prefix_lr","base_synthesis_lr","points",
              "step_seconds","peak_gpu_memory_gib"]
    metrics = os.path.join(args.output_dir, "training_metrics.csv")
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic(); saved = {}
    with open(metrics, "w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader()
        for step, (coords, feats) in enumerate(loader, start=1):
            tick=time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
            attribute=ME.SparseTensor(features=feats, coordinates=coords,
                                      tensor_stride=1, device="cuda")
            output=model.forward_base_training(attribute, args.conditioning_lambda)
            if output["completed_residual_stages"] != 4:
                raise RuntimeError("Base path accessed an unexpected residual stage")
            loss,rate,distortion,channel=base_rescue_objective(
                output,attribute,args.conditioning_lambda)
            if not all(torch.isfinite(x).all() for x in (loss,rate,distortion,channel)):
                raise RuntimeError("Non-finite objective at step {}".format(step))
            loss.backward(); gradients=assert_gradient_contract(model,args.trainable_scope)
            trainable=[p for p in model.parameters() if p.requires_grad]
            norm=float(torch.stack([p.grad.detach().float().pow(2).sum()
                       for p in trainable if p.grad is not None]).sum().sqrt().item())
            optimizer.step(); torch.cuda.synchronize()
            lrs={g["name"]:g["lr"] for g in optimizer.param_groups}
            row={"step":step,"loss":float(loss.item()),"R_B_est":float(rate.item()),
                 "D111":float(distortion.item()),"D_Y":float(channel[0].item()),
                 "D_U":float(channel[1].item()),"D_V":float(channel[2].item()),
                 "gradient_norm":norm,"prefix_lr":lrs.get("native_prefix"),
                 "base_synthesis_lr":lrs["base_synthesis"],"points":len(attribute),
                 "step_seconds":time.monotonic()-tick,
                 "peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/1024**3}
            writer.writerow(row); h.flush()
            if step == 1 or step % 25 == 0 or step == steps:
                print(json.dumps({**row,"gradient_groups":gradients}),flush=True)
            should_save = (args.smoke_only and step == steps) or (
                not args.smoke_only and (step in args.save_steps or step == args.max_steps))
            if should_save:
                path=os.path.join(args.output_dir,"checkpoints","step_{}.pth".format(step))
                torch.save({"architecture":ARCHITECTURE,"base_model":model.state_dict(),
                            "optimizer":optimizer.state_dict(),"step":step,
                            "config":config.to_dict(),"conditioning_lambda":args.conditioning_lambda,
                            "checkpoint_profile":args.checkpoint_profile,
                            "released_checkpoint":args.released_checkpoint,
                            "released_checkpoint_sha256":metadata["released_checkpoint_sha256"],
                            "initial_base_checkpoint":args.initial_base_checkpoint,
                            "initialization":initialization,"trainable_scope":args.trainable_scope,
                            "sampling":metadata["sampling"],"resolved_args":metadata},path)
                saved[str(step)]=path
    final_step = steps if args.smoke_only else args.max_steps
    final_path = saved[str(final_step)]
    reload_pass = verify_saved_checkpoint(
        final_path, model, optimizer, final_step)
    runtime_seconds = time.monotonic()-started
    write_json(os.path.join(args.output_dir,"training_summary.json"),{
        "status":"PASS","steps":steps,"runtime_seconds":time.monotonic()-started,
        "peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/1024**3,
        "gpu_hours":runtime_seconds / 3600.0,
        "checkpoints":saved,"sampling":sample_meta,
        "last_gradient_groups":gradients,"optimizer_contract":True,
        "checkpoint_reload_pass":reload_pass,
        "num_residual_stages":4,"native_r5_used":False})


if __name__ == "__main__":
    main()
