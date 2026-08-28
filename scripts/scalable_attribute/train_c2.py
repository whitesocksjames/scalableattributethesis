#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np
import torch

from scalable_attribute.c2_native import C2ScalableAttributeModel
from scalable_attribute.losses import rate_distortion_loss
from scripts.scalable_attribute.train import (
    append_metric, gradient_norm, loader, sparse_tensor)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--val-file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--rd-lambda", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=705)
    parser.add_argument("--val-every", type=int, default=705)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, args, step):
    torch.save({
        "architecture": "c2_native",
        "enhancement": model.enhancement.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {"architecture": "c2_native"},
        "step": step,
        "base_lambda": args.base_lambda,
        "rd_lambda": args.rd_lambda,
    }, path)


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


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")

    model = C2ScalableAttributeModel(args.base_checkpoint).cuda()
    optimizer = torch.optim.Adam(
        model.enhancement.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=0.0)
    train_loader = loader(args.data_root, args.train_file_list, args, shuffle=True)
    val_loader = loader(args.data_root, args.val_file_list, args, shuffle=False)
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    torch.cuda.reset_peak_memory_stats()
    ema = None
    step = 0
    stop = False
    for _epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            started = time.perf_counter()
            model.train()
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
            ema = raw if ema is None else tuple(
                0.05 * value + 0.95 * average
                for value, average in zip(raw, ema))
            append_metric(metrics_path, {
                "step": step, "split": "train", "loss": raw[0],
                "R_E": raw[1], "distortion": raw[2],
                "lambda_distortion": args.rd_lambda * raw[2],
                "grad_norm": grad_norm, "lr": args.lr, "points": len(A),
                "step_seconds": time.perf_counter() - started,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                "loss_ema": ema[0], "R_E_ema": ema[1],
                "distortion_ema": ema[2],
            })
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(os.path.join(
                    checkpoint_dir, "step_{}.pth".format(step)),
                    model, optimizer, args, step)
            if args.val_every and step % args.val_every == 0:
                validate(model, val_loader, args, step, metrics_path)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break
    final = os.path.join(checkpoint_dir, "step_{}.pth".format(step))
    if not os.path.exists(final):
        save_checkpoint(final, model, optimizer, args, step)
    save_checkpoint(os.path.join(checkpoint_dir, "last.pth"), model, optimizer, args, step)


if __name__ == "__main__":
    main()
