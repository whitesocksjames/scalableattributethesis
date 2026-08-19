#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import sys
import time

import torch
import MinkowskiEngine as ME

from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.config import EnhancementConfig, add_architecture_arguments
from scalable_attribute.data import UncachedPCDataset, h5_files
from scalable_attribute.losses import rate_distortion_loss
from scalable_attribute.model import ScalableAttributeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list")
    parser.add_argument("--val-data-root")
    parser.add_argument("--val-file-list")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--rd-lambda", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-scale", type=int, default=5)
    parser.add_argument("--base-stage", type=int, default=1)
    parser.add_argument("--base-vmode", type=int, default=1)
    return add_architecture_arguments(parser).parse_args()


def loader(root, file_list, args, shuffle):
    dataset = UncachedPCDataset(
        h5_files(root, file_list), color_format="yuv", normalize=True)
    return make_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def sparse_tensor(batch):
    coords, feats = batch
    return ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1, device="cuda")


def save_checkpoint(path, model, optimizer, config, step):
    torch.save({
        "enhancement": model.enhancement.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": step,
    }, path)


METRIC_FIELDS = (
    "step", "split", "loss", "R_E", "distortion", "lambda_distortion",
    "grad_norm", "lr", "points", "step_seconds", "max_memory_gib",
    "loss_ema", "R_E_ema", "distortion_ema",
)


def append_metric(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(model, data_loader, args, step, metrics_path):
    model.eval()
    totals = {"loss": 0.0, "R_E": 0.0, "distortion": 0.0}
    count = 0
    for batch in data_loader:
        A = sparse_tensor(batch)
        output = model(A, args.base_lambda)
        loss, rate, distortion = rate_distortion_loss(
            A, output["Full"], output["likelihood"], args.rd_lambda)
        totals["loss"] += loss.item()
        totals["R_E"] += rate.item()
        totals["distortion"] += distortion.item()
        count += 1
    append_metric(metrics_path, {
        "step": step,
        "split": "val",
        **{name: value / count for name, value in totals.items()},
    })


def gradient_norm(parameters, step, batch_index, points):
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise RuntimeError(
                "Non-finite gradient at step={} batch={} points={}".format(
                    step, batch_index, points))
        value = gradient.float().pow(2).sum()
        squared = value if squared is None else squared + value
    if squared is None:
        raise RuntimeError("No EL gradients at step={}".format(step))
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError("Non-finite gradient norm at step={}".format(step))
    return norm.item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = EnhancementConfig.from_args(args)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.output_dir, "enhancement_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")

    model = ScalableAttributeModel(
        args.base_checkpoint,
        config,
        base_scale=args.base_scale,
        base_stage=args.base_stage,
        base_vmode=args.base_vmode,
    ).cuda()
    optimizer = torch.optim.Adam(
        model.enhancement.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=0.0)
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if state["config"] != config.to_dict():
            raise RuntimeError("Resume checkpoint EnhancementConfig differs")
        model.enhancement.load_state_dict(state["enhancement"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])

    train_loader = loader(
        args.data_root, args.train_file_list, args, shuffle=True)
    val_root = args.val_data_root or args.data_root
    val_loader = loader(
        val_root, args.val_file_list, args, shuffle=False) if args.val_file_list else None
    if args.val_data_root and not args.val_file_list:
        val_loader = loader(args.val_data_root, None, args, shuffle=False)
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    torch.cuda.reset_peak_memory_stats()
    ema = None
    stop = False
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            started = time.perf_counter()
            model.train()
            if model.base_adapter.training or any(
                    parameter.requires_grad
                    for parameter in model.base_adapter.parameters()):
                raise RuntimeError("Frozen Unicorn Base entered trainable mode")
            optimizer.zero_grad(set_to_none=True)
            A = sparse_tensor(batch)
            output = model(A, args.base_lambda)
            loss, rate, distortion = rate_distortion_loss(
                A, output["Full"], output["likelihood"], args.rd_lambda)
            loss.backward()
            if any(parameter.grad is not None
                   for parameter in model.base_adapter.parameters()):
                raise RuntimeError("Frozen Unicorn Base received gradients")
            next_step = step + 1
            grad_norm = gradient_norm(
                model.enhancement.parameters(), next_step, batch_index, len(A))
            optimizer.step()
            step = next_step
            raw = (loss.item(), rate.item(), distortion.item())
            if ema is None:
                ema = raw
            else:
                ema = tuple(0.05 * value + 0.95 * average
                            for value, average in zip(raw, ema))
            append_metric(metrics_path, {
                "step": step,
                "split": "train",
                "loss": raw[0],
                "R_E": raw[1],
                "distortion": raw[2],
                "lambda_distortion": args.rd_lambda * raw[2],
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "points": len(A),
                "step_seconds": time.perf_counter() - started,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                "loss_ema": ema[0],
                "R_E_ema": ema[1],
                "distortion_ema": ema[2],
            })

            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    os.path.join(checkpoint_dir, "step_{}.pth".format(step)),
                    model, optimizer, config, step)
            if val_loader and args.val_every and step % args.val_every == 0:
                validate(model, val_loader, args, step, metrics_path)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    final_checkpoint = os.path.join(checkpoint_dir, "step_{}.pth".format(step))
    if not os.path.exists(final_checkpoint):
        save_checkpoint(final_checkpoint, model, optimizer, config, step)
    save_checkpoint(
        os.path.join(checkpoint_dir, "last.pth"), model, optimizer, config, step)


if __name__ == "__main__":
    main()
