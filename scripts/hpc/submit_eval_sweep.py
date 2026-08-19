#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

from sweep_utils import (
    experiment_cli, load_spec, resolve_base_checkpoint, resolve_checkpoint,
    submit_job,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--afterok")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def train_args(experiment_root):
    path = Path(experiment_root) / "train" / "resolved_args.json"
    if not path.exists():
        raise FileNotFoundError(
            "Training resolved args not found: {}".format(path))
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def inherit_train_values(experiment, namespace, spec, experiment_root, afterok):
    try:
        trained = train_args(experiment_root)
    except FileNotFoundError:
        if not afterok:
            raise
        _, train_experiments = load_spec(spec, "train")
        matches = [item for item in train_experiments
                   if item["name"] == experiment["train_experiment"]]
        if not matches:
            raise ValueError("No matching train experiment in spec")
        trained = matches[0]
    inherited = dict(experiment)
    for key in namespace.get("EVAL_INHERIT_TRAIN_ARGS", ()):
        if key in trained:
            inherited.setdefault(key, trained[key])
    return inherited


def main():
    args = parse_args()
    namespace, experiments = load_spec(args.spec, "eval")
    failures = []
    selected = [item for item in experiments
                if not args.only or item["name"] in args.only
                or item.get("train_experiment") in args.only]
    if not selected:
        raise ValueError("No experiments selected")
    for experiment in selected:
        name = experiment["train_experiment"]
        experiment_root = os.path.join(args.experiments_root, args.study, name)
        try:
            experiment = inherit_train_values(
                experiment, namespace, args.spec, experiment_root, args.afterok)
            if args.afterok and experiment["checkpoint_tag"] == "latest":
                raise ValueError(
                    "afterok eval requires a stable stepN tag; latest cannot be "
                    "resolved before training finishes")
            checkpoint, stable_tag = resolve_checkpoint(
                os.path.join(experiment_root, "train", "checkpoints"),
                experiment["checkpoint_tag"],
                require_exists=not (args.dry_run or args.afterok))
            run_dir = os.path.join(experiment_root, "eval", stable_tag)
            if not args.dry_run and os.path.exists(
                    os.path.join(run_dir, "metrics.csv")):
                raise RuntimeError(
                    "Completed eval output already exists: {}".format(run_dir))
            base_checkpoint = resolve_base_checkpoint(
                experiment, namespace, args.checkpoints_root)
            command = [
                args.python,
                os.path.join(args.repo, "scripts/scalable_attribute/evaluate.py"),
                "--data-root", args.data_root,
                "--base-checkpoint", base_checkpoint,
                "--enhancement-checkpoint", checkpoint,
                "--output-dir", run_dir,
                "--gpcc-binary", args.gpcc_binary,
            ]
            command.extend(experiment_cli(experiment))
            submit_job(
                command, "eval_" + name, experiment_root, run_dir,
                args.repo, args.python, args.profile, "eval", args.dry_run,
                args.verbose, dependency=args.afterok, checkpoint_tag=stable_tag)
        except Exception as error:
            failures.append((name, str(error)))
            print("submission failed for {}: {}".format(name, error), file=sys.stderr)
    if failures:
        print("{} of {} submissions failed".format(len(failures), len(selected)),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
