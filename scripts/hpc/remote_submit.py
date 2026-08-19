#!/usr/bin/env python3
"""Thin local-first entry point for the TinyGPU thesis workflow."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from sweep_utils import checkpoint_step, read_manifest, submit_job
from resource_profiles import DEFAULT_RESOURCE_PROFILES


SSH_ALIAS = "tinyx"
REMOTE_REPO = "$HOME/Scalable-Attribute-Thesis"
WORKSPACE = "$WORK/scalable_attribute_thesis"
PYTHON = "$WORK/software/private/conda/envs/unicorn-me-py38/bin/python"
DATA_ROOT = "$WORK/scalable_attribute_thesis/datasets/RWTT/processed/train_h5/h5/100000"
GPCC_BINARY = "$HOME/Scalable-Attribute-Thesis/third_party/tmc3_v21"
DEFAULT_SPEC = "scripts/hpc/sweep_spec.py"


def local_repo():
    return Path(__file__).resolve().parents[2]


def run(command, capture=False, check=True):
    result = subprocess.run(command, text=True, check=check,
                            capture_output=capture)
    return result.stdout if capture else ""


def ssh(command, capture=False, check=True):
    return run(["ssh", SSH_ALIAS, command], capture=capture, check=check)


def remote_prefix():
    return (
        'REPO="{}"; ROOT="{}"; PY="{}"; '
        'cd "$REPO"; '
    ).format(REMOTE_REPO, WORKSPACE, PYTHON)


def remote_value(value):
    if value.startswith("$WORK/"):
        return '"$WORK/{}"'.format(value[len("$WORK/"):])
    if value.startswith("$HOME/"):
        return '"$HOME/{}"'.format(value[len("$HOME/"):])
    return shlex.quote(value)


def validate_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Names may contain only letters, digits, dot, underscore, dash")
    return value


def sync_source():
    excludes = [
        ".git/", "__pycache__/", "*.pyc", "datasets/", "checkpoints/",
        "experiments/", "logs/", "ckpts/", "output/", "results/",
        "*.pdf", "*.pth", "*.h5", "*.obj", "*.mtl", "*.png", "*.jpg",
        "*.zip", "*.tar", "*.tar.gz", "*.tgz",
    ]
    command = ["rsync", "-az"]
    for pattern in excludes:
        command.extend(["--exclude", pattern])
    command.extend([str(local_repo()) + "/",
                    SSH_ALIAS + ":Scalable-Attribute-Thesis/"])
    run(command)


def common_submit_command(script, args):
    spec = args.spec
    if os.path.isabs(spec):
        raise ValueError("Spec must be a path relative to the local repository")
    command = [
        '"$PY"', '"$REPO/{}"'.format(script),
        "--spec", '"$REPO/{}"'.format(spec),
        "--repo", '"$REPO"',
        "--python", '"$PY"',
        "--data-root", remote_value(args.data_root),
        "--checkpoints-root", '"$ROOT/checkpoints"',
        "--experiments-root", '"$ROOT/experiments"',
        "--study", shlex.quote(validate_name(args.study)),
        "--profile", shlex.quote(args.profile),
    ]
    for name in args.only:
        command.extend(["--only", shlex.quote(validate_name(name))])
    if args.dry_run:
        command.append("--dry-run")
    if args.verbose:
        command.append("--verbose")
    return command


def submit_train(args):
    if not args.no_sync:
        sync_source()
    command = common_submit_command(
        "scripts/hpc/submit_train_sweep.py", args)
    if args.val_data_root:
        command.extend(["--val-data-root", remote_value(args.val_data_root)])
    if args.afterok:
        command.extend(["--afterok", shlex.quote(args.afterok)])
    result = subprocess.run(
        ["ssh", SSH_ALIAS, remote_prefix() + " ".join(command)],
        text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    output = result.stdout
    if args.submit_eval_after_train:
        if args.dry_run:
            names = re.findall(r"^train:\s+(\S+)\s+->", output,
                               flags=re.MULTILINE)
            submitted = [(name, "TRAIN_JOB_ID") for name in names]
        else:
            submitted = re.findall(r"^SUBMITTED\s+(\S+)\s+(\d+)\s*$",
                                   output, flags=re.MULTILINE)
        eval_failures = []
        for name, job_id in submitted:
            eval_args = argparse.Namespace(**vars(args))
            eval_args.only = [name]
            eval_args.afterok = job_id
            eval_args.no_sync = True
            eval_args.profile = DEFAULT_RESOURCE_PROFILES["eval"]
            try:
                submit_eval(eval_args)
            except subprocess.CalledProcessError as error:
                eval_failures.append(name)
                print("afterok eval submission failed for {}: {}".format(
                    name, error), file=sys.stderr)
        if eval_failures:
            raise RuntimeError(
                "afterok eval submission failed for: {}".format(
                    ", ".join(eval_failures)))
    result.check_returncode()


def submit_eval(args):
    if not args.no_sync:
        sync_source()
    command = common_submit_command(
        "scripts/hpc/submit_eval_sweep.py", args)
    command.extend(["--gpcc-binary", '"{}"'.format(GPCC_BINARY)])
    if getattr(args, "afterok", None):
        command.extend(["--afterok", shlex.quote(args.afterok)])
    result = subprocess.run(
        ["ssh", SSH_ALIAS, remote_prefix() + " ".join(command)],
        text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()


def experiment_ref(value):
    if "/" not in value:
        raise ValueError("Experiment must be STUDY/NAME")
    study, name = value.split("/", 1)
    return validate_name(study), validate_name(name)


def remote_internal(action, args, extra=()):
    study, name = experiment_ref(args.experiment)
    command = [
        '"$PY"', '"$REPO/scripts/hpc/remote_submit.py"', action,
        "--experiment-root", '"$ROOT/experiments/{}/{}"'.format(
            study, name),
        "--repo", '"$REPO"', "--python", '"$PY"',
    ]
    command.extend(extra)
    return remote_prefix() + " ".join(command)


def slurm_state(job_id):
    result = subprocess.run(
        ["sacct", "-M", "tinygpu", "-X", "-j", str(job_id),
         "--noheader", "--parsable2", "-o", "JobIDRaw,State,ExitCode"],
        text=True, capture_output=True)
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields and fields[0] == str(job_id):
            return fields[1], fields[2]
    return "UNKNOWN", ""


def latest_record(manifest, kind, tag=None):
    if kind == "train":
        attempts = manifest.get("train", {}).get("attempts", [])
    else:
        evaluations = manifest.get("evaluations", {})
        if not evaluations:
            raise RuntimeError("No evaluation recorded")
        tag = tag or max(
            evaluations,
            key=lambda value: checkpoint_step(Path(value + ".pth")))
        attempts = evaluations[tag].get("attempts", [])
    records = [item for item in attempts if item.get("job_id")]
    if not records:
        raise RuntimeError("No submitted {} job recorded".format(kind))
    return records[-1], tag


def latest_checkpoint(experiment_root):
    checkpoints = Path(experiment_root) / "train" / "checkpoints"
    paths = sorted(checkpoints.glob("step_*.pth"), key=checkpoint_step)
    return str(paths[-1]) if paths else None


def metric_summary(path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    summary = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def internal_result(args):
    root = Path(args.experiment_root)
    manifest = read_manifest(root)
    jobs = []
    for record in manifest.get("train", {}).get("attempts", []):
        if record.get("job_id"):
            jobs.append(("train", record["job_id"], *slurm_state(record["job_id"])))
    for tag, evaluation in manifest.get("evaluations", {}).items():
        for record in evaluation.get("attempts", []):
            if record.get("job_id"):
                jobs.append(("eval/" + tag, record["job_id"],
                             *slurm_state(record["job_id"])))
    result = {
        "experiment": manifest.get("experiment", root.name),
        "output_dir": str(root),
        "jobs": [{"kind": kind, "job_id": job, "state": state,
                  "exit_code": code} for kind, job, state, code in jobs],
        "latest_checkpoint": latest_checkpoint(root),
        "training_metrics": str(root / "train" / "metrics.csv")
        if (root / "train" / "metrics.csv").exists() else None,
        "evaluations": {},
    }
    for tag, evaluation in manifest.get("evaluations", {}).items():
        csv_path = Path(evaluation["output_dir"]) / "metrics.csv"
        result["evaluations"][tag] = {
            "hard_eval_csv": str(csv_path) if csv_path.exists() else None,
            "mean": metric_summary(csv_path),
        }
    print(json.dumps(result, indent=2))


def internal_log_path(args):
    manifest = read_manifest(args.experiment_root)
    record, _ = latest_record(manifest, args.kind, args.tag)
    print(record[args.stream])


def internal_cancel(args):
    manifest = read_manifest(args.experiment_root)
    record, _ = latest_record(manifest, args.kind, args.tag)
    subprocess.run(["scancel", "-M", "tinygpu", record["job_id"]], check=True)
    print("cancelled {}".format(record["job_id"]))


def replace_option(command, option, value):
    if option in command:
        index = command.index(option)
        command[index + 1] = value
    else:
        command.extend([option, value])


def internal_retry(args):
    root = Path(args.experiment_root)
    manifest = read_manifest(root)
    record, tag = latest_record(manifest, args.kind, args.tag)
    state, _ = slurm_state(record["job_id"])
    if state.split("+")[0] not in {"NODE_FAIL", "TIMEOUT"}:
        raise RuntimeError(
            "Retry is reserved for NODE_FAIL/TIMEOUT; job {} is {}".format(
                record["job_id"], state))
    command = shlex.split(Path(record["command"]).read_text(encoding="utf-8"))
    if args.kind == "train" and args.resume:
        checkpoint = latest_checkpoint(root)
        if checkpoint:
            replace_option(command, "--resume", checkpoint)
            print("resuming from {}".format(checkpoint))
        else:
            print("no numbered checkpoint found; retrying from the original start")
    run_dir = root / ("train" if args.kind == "train" else "eval/" + tag)
    submit_job(
        command,
        root.name if args.kind == "train" else "eval_" + root.name,
        root, run_dir, args.repo, args.python, record["profile"], args.kind,
        checkpoint_tag=tag, walltime=args.walltime, retry=True)


def add_submit_common(parser, profile):
    parser.add_argument("--study", required=True)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--profile", default=profile)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("submit-train")
    add_submit_common(train, DEFAULT_RESOURCE_PROFILES["train"])
    train.add_argument("--val-data-root")
    train.add_argument("--afterok")
    train.add_argument("--submit-eval-after-train", action="store_true")
    evaluation = sub.add_parser("submit-eval")
    add_submit_common(evaluation, DEFAULT_RESOURCE_PROFILES["eval"])
    evaluation.add_argument("--afterok")
    sub.add_parser("sync")
    sub.add_parser("queue")
    for name in ("status", "result"):
        item = sub.add_parser(name)
        item.add_argument("experiment")
    logs = sub.add_parser("logs")
    logs.add_argument("experiment")
    logs.add_argument("--kind", choices=("train", "eval"), default="train")
    logs.add_argument("--tag")
    logs.add_argument("--stream", choices=("stdout", "stderr"), default="stdout")
    logs.add_argument("--follow", action="store_true")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("experiment")
    cancel.add_argument("--kind", choices=("train", "eval"), default="train")
    cancel.add_argument("--tag")
    retry = sub.add_parser("retry")
    retry.add_argument("experiment")
    retry.add_argument("--kind", choices=("train", "eval"), default="train")
    retry.add_argument("--tag")
    retry.add_argument("--resume", action="store_true")
    retry.add_argument("--walltime")
    retry.add_argument("--no-sync", action="store_true")

    # Private commands run only on the HPC login node through SSH.
    for name in ("_result", "_log-path", "_cancel", "_retry"):
        item = sub.add_parser(name, help=argparse.SUPPRESS)
        item.add_argument("--experiment-root", required=True)
        item.add_argument("--repo", required=True)
        item.add_argument("--python", required=True)
        if name in ("_log-path", "_cancel", "_retry"):
            item.add_argument("--kind", choices=("train", "eval"), default="train")
            item.add_argument("--tag")
        if name == "_log-path":
            item.add_argument("--stream", choices=("stdout", "stderr"), default="stdout")
        if name == "_retry":
            item.add_argument("--resume", action="store_true")
            item.add_argument("--walltime")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "sync":
        sync_source()
    elif args.command == "submit-train":
        submit_train(args)
    elif args.command == "submit-eval":
        submit_eval(args)
    elif args.command == "queue":
        ssh("squeue.tinygpu -u \"$USER\" -o '%.18i %.24j %.10T %.10M %.9P %R'")
    elif args.command in ("status", "result"):
        ssh(remote_internal("_result", args))
    elif args.command == "logs":
        extra = ["--kind", args.kind, "--stream", args.stream]
        if args.tag:
            extra.extend(["--tag", shlex.quote(args.tag)])
        path = ssh(remote_internal("_log-path", args, extra), capture=True).strip()
        ssh("tail {} -n 80 {}".format("-f" if args.follow else "", shlex.quote(path)))
    elif args.command == "cancel":
        extra = ["--kind", args.kind]
        if args.tag:
            extra.extend(["--tag", shlex.quote(args.tag)])
        ssh(remote_internal("_cancel", args, extra))
    elif args.command == "retry":
        if not args.no_sync:
            sync_source()
        extra = ["--kind", args.kind]
        if args.tag:
            extra.extend(["--tag", shlex.quote(args.tag)])
        if args.resume:
            extra.append("--resume")
        if args.walltime:
            extra.extend(["--walltime", shlex.quote(args.walltime)])
        ssh(remote_internal("_retry", args, extra))
    elif args.command == "_result":
        internal_result(args)
    elif args.command == "_log-path":
        internal_log_path(args)
    elif args.command == "_cancel":
        internal_cancel(args)
    elif args.command == "_retry":
        internal_retry(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError,
            subprocess.CalledProcessError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
