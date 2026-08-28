#!/usr/bin/env python3
"""Run the existing evaluator on fixed C2 checkpoints; no codec logic here."""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--rd-lambda", type=float, required=True)
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    evaluator = os.path.join(
        os.path.abspath(args.repo), "scripts/scalable_attribute/evaluate.py")
    for step in args.steps:
        checkpoint = os.path.join(
            args.train_dir, "checkpoints", "step_{}.pth".format(step))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        subprocess.run([
            sys.executable, evaluator,
            "--data-root", args.data_root,
            "--file-list", args.file_list,
            "--base-checkpoint", args.base_checkpoint,
            "--enhancement-checkpoint", checkpoint,
            "--output-dir", os.path.join(
                args.output_root, "step_{}".format(step)),
            "--experiment-name", args.experiment_name,
            "--base-profile-label", "official_R08_2k128_l256",
            "--base-lambda", "256",
            "--rd-lambda", str(args.rd_lambda),
            "--lr", "5e-5",
            "--seed", "0",
            "--gpcc-binary", args.gpcc_binary,
            "--model-type", "c2_native",
        ], check=True)


if __name__ == "__main__":
    main()
