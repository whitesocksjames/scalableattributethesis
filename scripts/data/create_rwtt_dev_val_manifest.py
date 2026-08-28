#!/usr/bin/env python3
import argparse
import math
import os
import re


MODEL = re.compile(r"RWT[0-9]+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-models", type=int, default=14)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("Refusing to overwrite fixed dev manifest: " + args.output)
    with open(args.validation_manifest, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    model_by_entry = [entry.split("/", 1)[0] for entry in entries]
    if any(not MODEL.fullmatch(model) for model in model_by_entry):
        raise ValueError("Validation manifest contains an invalid RWTT model path")
    models = sorted(set(model_by_entry), key=lambda value: int(value[3:]))
    if not 1 < args.num_models <= len(models):
        raise ValueError("num-models must be in [2, {}]".format(len(models)))
    indices = [math.floor(
        index * (len(models) - 1) / (args.num_models - 1) + 0.5)
        for index in range(args.num_models)]
    selected = [models[index] for index in indices]
    selected_set = set(selected)
    dev_entries = [entry for entry, model in zip(entries, model_by_entry)
                   if model in selected_set]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        handle.write("\n".join(dev_entries) + "\n")
    print("selected_models={}".format(",".join(selected)))
    print("num_models={}".format(len(selected)))
    print("num_h5={}".format(len(dev_entries)))


if __name__ == "__main__":
    main()
