#!/usr/bin/env python3
import argparse
import os
import sys

from sweep_utils import (
    experiment_cli, load_spec, resolve_base_checkpoint, submit_job,
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
    parser.add_argument("--val-data-root")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--afterok")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    namespace, experiments = load_spec(args.spec, "train")
    failures = []
    selected = [item for item in experiments
                if not args.only or item["name"] in args.only]
    if not selected:
        raise ValueError("No experiments selected")
    for experiment in selected:
        name = experiment["name"]
        experiment_root = os.path.join(args.experiments_root, args.study, name)
        run_dir = os.path.join(experiment_root, "train")
        try:
            base_checkpoint = resolve_base_checkpoint(
                experiment, namespace, args.checkpoints_root)
            command = [
                args.python,
                os.path.join(args.repo, "scripts/scalable_attribute/train.py"),
                "--data-root", args.data_root,
                "--base-checkpoint", base_checkpoint,
                "--output-dir", run_dir,
            ]
            if args.val_data_root:
                command.extend(["--val-data-root", args.val_data_root])
            command.extend(experiment_cli(experiment))
            submit_job(
                command, name, experiment_root, run_dir, args.repo, args.python,
                args.profile, "train", args.dry_run, args.verbose,
                dependency=args.afterok)
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
