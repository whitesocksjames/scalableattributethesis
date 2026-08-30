#!/usr/bin/env python3
"""Combine the four canonical Enhancement E1 arm artifacts."""

import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 100, 250])
    parser.add_argument("--step-width", type=int, default=3)
    parser.add_argument("--output-stem", default="combined_summary")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    rows = []
    arm_summaries = {}
    for arm in args.arms:
        arm_dir = os.path.join(args.experiment_root, arm)
        training = read_json(os.path.join(arm_dir, "training_summary.json"))
        resolved = read_json(os.path.join(arm_dir, "resolved_args.json"))
        arm_rows = []
        for step in args.steps:
            result = read_json(os.path.join(
                arm_dir, ("validation_step{:0" + str(args.step_width)
                          + "d}.json").format(step)))
            rates = result["model_equal"]
            symbols = result["hard_symbol_statistics"]
            row = {
                "arm": arm,
                "lr": resolved["lr"],
                "rd_lambda": resolved["rd_lambda"],
                "step": step,
                "R_noise_bpp": rates["enh"]["noise"],
                "R_symbol_bpp": rates["enh"]["symbol"],
                "R_hard_bpp": rates["enh"]["hard"],
                "noise_over_hard": rates["enh"]["noise_over_hard"],
                "symbol_over_hard": rates["enh"]["symbol_over_hard"],
                "base_physical_bpp": rates["base_hard_bpp"],
                "scalable_physical_bpp": rates[
                    "enhancement_total_hard_bpp"],
                "native_full_physical_bpp": rates[
                    "native_total_hard_bpp"],
                "base_psnr": rates["base_psnr"],
                "scalable_full_psnr": rates["enhancement_full_psnr"],
                "native_full_psnr": rates["native_full_psnr"],
                "full_minus_base_psnr": (
                    rates["enhancement_full_psnr"] - rates["base_psnr"]),
                "full_minus_native_psnr": (
                    rates["enhancement_full_psnr"]
                    - rates["native_full_psnr"]),
                "nonzero_fraction": symbols["nonzero_fraction"],
                "active_channels": symbols["active_channel_count"],
                "symbol_min": symbols["min"],
                "symbol_max": symbols["max"],
                "hard_exact_max_abs": result[
                    "enhancement_deterministic_hard_max_abs"],
                "evaluation_runtime_seconds": result["runtime_seconds"],
            }
            rows.append(row)
            arm_rows.append(row)
        arm_summaries[arm] = {
            "training": training,
            "trajectory": arm_rows,
            "final": arm_rows[-1],
        }

    csv_path = os.path.join(
        args.experiment_root, args.output_stem + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "status": "PASS",
        "arms": arm_summaries,
        "comparison_scope": "28-H5/28-model E0-C sanity manifest",
        "decision_note": (
            "Short screening evidence only; no formal RD or architecture "
            "decision is encoded automatically."),
    }
    with open(os.path.join(
            args.experiment_root, args.output_stem + ".json"), "w",
              encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
