#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.config import EnhancementConfig
from scalable_attribute.data import UncachedPCDataset, h5_files
from scalable_attribute.model import ScalableAttributeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-args", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--failed-step", type=int, required=True,
                        help="First step that failed; completed metrics end at failed-step - 1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worst-manifest", required=True)
    return parser.parse_args()


def point_count(path):
    with h5py.File(path, "r") as handle:
        return int(handle["coords"].shape[0])


def main():
    args = parse_args()
    with open(args.resolved_args, encoding="utf-8") as handle:
        resolved = json.load(handle)
    torch.manual_seed(resolved["seed"])
    config = EnhancementConfig.from_args(argparse.Namespace(**resolved))
    model = ScalableAttributeModel(
        resolved["base_checkpoint"], config,
        base_scale=resolved["base_scale"],
        base_stage=resolved["base_stage"],
        base_vmode=resolved["base_vmode"],
    )
    torch.optim.Adam(model.enhancement.parameters(), lr=resolved["lr"],
                     betas=(0.9, 0.999), weight_decay=0.0)

    files = h5_files(resolved["data_root"], resolved["train_file_list"])
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    loader = make_data_loader(
        dataset, batch_size=resolved["batch_size"], shuffle=True,
        num_workers=resolved["num_workers"])
    iterator = iter(loader)
    batches = list(iterator._sampler_iter)
    failed_indices = batches[args.failed_step - 1]

    counts = [point_count(path) for path in files]
    with open(args.metrics, newline="", encoding="utf-8") as handle:
        metric_points = [int(row["points"]) for row in csv.DictReader(handle)
                         if row["split"] == "train"]
    reproduced = [sum(counts[index] for index in batch)
                  for batch in batches[:len(metric_points)]]
    if reproduced != metric_points:
        mismatch = next(index for index, values in enumerate(
                        zip(reproduced, metric_points)) if values[0] != values[1])
        raise RuntimeError("Sampler reproduction mismatch at step {}".format(mismatch + 1))

    offending = [{"file": files[index], "points": counts[index]}
                 for index in failed_indices]
    largest = sorted(range(len(files)), key=lambda index: counts[index], reverse=True)[:4]
    worst_relative = [os.path.relpath(files[index], resolved["data_root"])
                      for index in largest]
    Path(args.worst_manifest).write_text(
        "".join(path + "\n" for path in worst_relative), encoding="utf-8")
    values = np.asarray(counts)
    result = {
        "samples": len(files),
        "distribution": {
            "min": int(values.min()),
            "median": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": int(values.max()),
        },
        "sampler_verified_steps": len(metric_points),
        "offending_step": args.failed_step,
        "offending_batch": offending,
        "offending_total_points": sum(item["points"] for item in offending),
        "worst4": [{"file": files[index], "points": counts[index]}
                   for index in largest],
        "worst4_total_points": sum(counts[index] for index in largest),
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
