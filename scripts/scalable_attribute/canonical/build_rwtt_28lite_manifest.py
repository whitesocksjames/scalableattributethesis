#!/usr/bin/env python3
"""Select one permanent, deterministic middle H5 from each RWTT model."""

import argparse
import json
import os
import subprocess

from scalable_attribute.evaluation import sample_identity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--expected-models", type=int, default=28)
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_key(model_id):
    return int(model_id[3:])


def main():
    args = parse_args()
    for path in (args.output_manifest, args.metadata):
        if os.path.exists(path):
            raise FileExistsError("Refusing to overwrite frozen manifest: " + path)
    with open(args.source_manifest, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    grouped = {}
    for entry in entries:
        model_id, partition_id = sample_identity(entry)
        if not model_id.startswith("RWT"):
            raise ValueError("RWTT-28Lite cannot contain " + entry)
        grouped.setdefault(model_id, []).append((partition_id, entry))
    if len(grouped) != args.expected_models:
        raise ValueError(
            "Expected {} RWTT models, found {}".format(
                args.expected_models, len(grouped)))

    selections = []
    for model_id in sorted(grouped, key=model_key):
        candidates = sorted(grouped[model_id], key=lambda item: (item[0], item[1]))
        selected_index = (len(candidates) - 1) // 2
        partition_id, entry = candidates[selected_index]
        selections.append({
            "model_id": model_id,
            "available_h5": len(candidates),
            "selected_index_zero_based": selected_index,
            "selected_partition": partition_id,
            "selected_h5": entry,
        })
    selected_entries = [item["selected_h5"] for item in selections]
    if len(set(selected_entries)) != args.expected_models:
        raise RuntimeError("RWTT-28Lite selection is not one H5 per model")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_manifest)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.metadata)), exist_ok=True)
    with open(args.output_manifest, "x", encoding="utf-8") as handle:
        handle.write("\n".join(selected_entries) + "\n")
    with open(args.metadata, "x", encoding="utf-8") as handle:
        json.dump({
            "benchmark": "RWTT-28Lite",
            "selection_rule": (
                "sort each model by (partition_id, manifest_entry), then choose "
                "lower-middle index floor((n-1)/2)"),
            "source_manifest": os.path.abspath(args.source_manifest),
            "num_models": len(selections),
            "num_h5": len(selected_entries),
            "creation_git_commit": git_commit(),
            "selected": selections,
        }, handle, indent=2)
        handle.write("\n")
    print("RWTT-28Lite: {} models / {} H5".format(
        len(selections), len(selected_entries)))


if __name__ == "__main__":
    main()
