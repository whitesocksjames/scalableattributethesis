#!/usr/bin/env python3
"""Fail-closed runner for one Base-rescue arm's RWTT-28Lite checkpoints."""

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--checkpoint-profile", required=True)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--gpcc-binary", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = os.path.join(args.train_dir, "training_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    required = {
        "status": "PASS",
        "checkpoint_reload_pass": True,
        "num_residual_stages": 4,
        "native_r5_used": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                "Training preflight {} mismatch: expected {!r}, got {!r}".format(
                    key, expected, summary.get(key)))
    checkpoints = {}
    for step in args.steps:
        path = os.path.join(args.train_dir, "checkpoints", "step_{}.pth".format(step))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        checkpoints[step] = path
    for path in (args.source_root, args.data_root, args.file_list,
                 args.released_checkpoint, args.gpcc_binary):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    evaluator = os.path.join(
        args.source_root, "scripts", "scalable_attribute", "canonical",
        "evaluate_base_rescue.py")
    for step in args.steps:
        output = os.path.join(args.eval_root, "step_{}".format(step))
        command = [
            sys.executable, evaluator,
            "--data-root", args.data_root,
            "--file-list", args.file_list,
            "--checkpoint", checkpoints[step],
            "--released-checkpoint", args.released_checkpoint,
            "--checkpoint-profile", args.checkpoint_profile,
            "--conditioning-lambda", str(args.conditioning_lambda),
            "--gpcc-binary", args.gpcc_binary,
            "--output-dir", output,
        ]
        subprocess.run(command, cwd=args.source_root, check=True)


if __name__ == "__main__":
    main()
