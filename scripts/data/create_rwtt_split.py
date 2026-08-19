#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_lines(path, values):
    path.write_text("".join(value + "\n" for value in values), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.h5_root).resolve()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Split output already exists and is not empty: {}".format(output))
    model_files = {}
    for path in sorted(root.rglob("*.h5")):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise RuntimeError("Expected model directory above HDF5: {}".format(path))
        model_files.setdefault(relative.parts[0], []).append(str(relative))
    if not model_files:
        raise RuntimeError("No HDF5 files found under {}".format(root))

    models = sorted(model_files)
    shuffled = list(models)
    random.Random(args.seed).shuffle(shuffled)
    val_count = int(math.ceil(len(models) * args.val_fraction))
    val_models = sorted(shuffled[:val_count])
    train_models = sorted(set(models) - set(val_models))
    train_h5 = sorted(path for model in train_models for path in model_files[model])
    val_h5 = sorted(path for model in val_models for path in model_files[model])
    if set(train_models) & set(val_models):
        raise RuntimeError("RWTT model overlap")
    if set(train_h5) & set(val_h5):
        raise RuntimeError("RWTT HDF5 overlap")

    output.mkdir(parents=True, exist_ok=True)
    write_lines(output / "train_models.txt", train_models)
    write_lines(output / "val_models.txt", val_models)
    write_lines(output / "train_h5.txt", train_h5)
    write_lines(output / "val_h5.txt", val_h5)
    metadata = {
        "h5_root": str(root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_models": len(train_models),
        "val_models": len(val_models),
        "train_h5": len(train_h5),
        "val_h5": len(val_h5),
        "model_overlap": 0,
        "h5_overlap": 0,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
