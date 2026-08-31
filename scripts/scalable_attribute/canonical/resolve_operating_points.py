#!/usr/bin/env python3
"""Resolve canonical point mappings against a released-checkpoint root."""

import argparse
import json
import os
import subprocess

import torch

from scalable_attribute.canonical.operating_points import (
    DEFAULT_CONFIG, load_operating_points, resolve_operating_point)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--released-root", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--load-test", action="store_true",
        help="Instantiate each released model and strictly load its state")
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_metadata(path):
    if not os.path.isfile(path):
        return {"checkpoint_exists": False, "load_test": False}
    state = torch.load(path, map_location="cpu")
    model_state = state.get("model")
    metadata = {
        "checkpoint_exists": True,
        "checkpoint_size_bytes": os.path.getsize(path),
        "checkpoint_top_level_keys": sorted(state),
        "model_parameter_tensor_count": (
            len(model_state) if isinstance(model_state, dict) else None),
        "saved_epoch": state.get("epoch"),
        "saved_args": state.get("args"),
        "load_test": False,
    }
    return metadata


def main():
    args = parse_args()
    config, config_path = load_operating_points(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(
        args.output_dir, "MULTIRATE_OPERATING_POINT_MAPPING.json")
    md_path = os.path.join(
        args.output_dir, "MULTIRATE_OPERATING_POINT_MAPPING.md")
    for path in (json_path, md_path):
        if os.path.exists(path):
            raise FileExistsError("Refusing to overwrite " + path)

    rows = []
    loaded_profiles = {}
    for point_id in config["points"]:
        point = resolve_operating_point(
            point_id, args.released_root, config_path)
        metadata = checkpoint_metadata(point["released_checkpoint"])
        if args.load_test and metadata["checkpoint_exists"]:
            profile = point["released_profile"]
            if profile not in loaded_profiles:
                from scalable_attribute.unicorn_reference import (
                    ReleasedUnicornAttribute)
                ReleasedUnicornAttribute(point["released_checkpoint"])
                loaded_profiles[profile] = True
            metadata["load_test"] = True
        rows.append({
            "point_id": point_id,
            "rate_id": point["rate_id"],
            "conditioning_lambda": point["conditioning_lambda"],
            "released_profile": point["released_profile"],
            "released_checkpoint_path": point["released_checkpoint"],
            "checkpoint_metadata": metadata,
        })

    report = {
        "status": "PASS" if all(
            row["checkpoint_metadata"]["checkpoint_exists"] for row in rows
        ) and (not args.load_test or all(
            row["checkpoint_metadata"]["load_test"] for row in rows)) else "FAIL",
        "git_commit": git_commit(),
        "config_path": config_path,
        "released_root": os.path.abspath(os.path.expandvars(args.released_root)),
        "load_test_requested": args.load_test,
        "points": rows,
    }
    with open(json_path, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    with open(md_path, "x", encoding="utf-8") as handle:
        handle.write("# Canonical multi-rate operating-point mapping\n\n")
        handle.write("| Point | Official ID | Lambda | Released profile | Exists | Load test |\n")
        handle.write("|---|---:|---:|---|---:|---:|\n")
        for row in rows:
            metadata = row["checkpoint_metadata"]
            handle.write(
                "| {point_id} | {rate_id} | {conditioning_lambda} | "
                "{released_profile} | {exists} | {loaded} |\n".format(
                    **row, exists=metadata["checkpoint_exists"],
                    loaded=metadata["load_test"]))
        handle.write("\nExact checkpoint paths and retained metadata are in the JSON file.\n")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
