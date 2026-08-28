#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from sweep_utils import experiment_cli, submit_job


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--dirty-worktree", action="store_true")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.repo)
    from scalable_attribute.evaluation import sample_identity
    from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS
    namespace = runpy.run_path(args.spec)
    common = namespace.get("REFERENCE_COMMON", {})
    common = dict(common)
    experiments = namespace.get("REFERENCE_EXPERIMENTS")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("REFERENCE_EXPERIMENTS must be a non-empty list")
    selected = [item for item in experiments
                if not args.only or item["name"] in args.only]
    manifest_path = os.path.expandvars(common["file_list"])
    with open(manifest_path, encoding="utf-8") as handle:
        manifest_entries = [line.strip() for line in handle if line.strip()]
    selected_models = sorted(
        {sample_identity(entry)[0] for entry in manifest_entries},
        key=lambda value: int(value[3:]))
    reference_root = Path(args.experiments_root) / args.study / "reference_full"
    info = {
        "git_commit": args.git_commit,
        "dirty_worktree": args.dirty_worktree,
        "validation_manifest": manifest_path,
        "num_models": len(selected_models),
        "num_h5": len(manifest_entries),
        "metric": "author pc_error YUV-PSNR 6:1:1",
        "aggregation": "original-model equal average",
        "operating_points": [
            {"rate_id": rate_id, "checkpoint_profile": profile,
             "base_lambda": base_lambda}
            for rate_id, profile, base_lambda in OFFICIAL_RWTT_REFERENCE_POINTS
        ],
    }
    info_path = reference_root / "reference_info.json"
    if not args.dry_run:
        if info_path.exists():
            with info_path.open(encoding="utf-8") as handle:
                if json.load(handle) != info:
                    raise RuntimeError("Existing reference_info.json differs: " + str(info_path))
        else:
            reference_root.mkdir(parents=True, exist_ok=True)
            with info_path.open("x", encoding="utf-8") as handle:
                json.dump(info, handle, indent=2)
                handle.write("\n")
    failures = []
    for item in selected:
        experiment = dict(common)
        experiment.update(item)
        name = experiment["name"]
        run_dir = os.path.join(
            args.experiments_root, args.study, "reference_full", name)
        experiment_root = os.path.join(args.experiments_root, args.study)
        completed = [artifact for artifact in (
            "per_h5.csv", "per_model.csv", "reference_curve.csv")
            if os.path.exists(os.path.join(run_dir, artifact))]
        if completed and not args.dry_run:
            failures.append((name, "completed artifacts already exist"))
            print("submission failed for {}: completed artifacts exist: {}".format(
                name, ", ".join(completed)), file=sys.stderr)
            continue
        checkpoint = experiment["base_checkpoint"]
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(args.checkpoints_root, checkpoint)
        command = [
            args.python,
            os.path.join(args.repo, "scripts/scalable_attribute/evaluate_unicorn_reference.py"),
            "--data-root", args.data_root,
            "--base-checkpoint", checkpoint,
            "--gpcc-binary", args.gpcc_binary,
            "--output-dir", run_dir,
        ]
        command.extend(experiment_cli(
            experiment, excluded=("checkpoint_profile", "base_lambda")))
        try:
            submit_job(
                command, "ref_" + name, experiment_root, run_dir,
                args.repo, args.python, args.profile, "reference",
                args.dry_run, args.verbose,
                checkpoint_tag="reference_full_" + name)
        except Exception as error:
            failures.append((name, str(error)))
            print("submission failed for {}: {}".format(name, error), file=sys.stderr)
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
