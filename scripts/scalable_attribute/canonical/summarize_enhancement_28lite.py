#!/usr/bin/env python3
"""Summarize a four-checkpoint Enhancement RWTT-28Lite trajectory."""

import argparse
import csv
import json
import math
from pathlib import Path


OUTPUT_STEM = "enhancement_28lite_trajectory"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="STEP=DIR",
        help="One evaluate_scalable_formal output directory; repeat four times")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_checkpoints(values):
    checkpoints = []
    seen = set()
    for value in values:
        try:
            step_text, directory = value.split("=", 1)
            step = int(step_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "--checkpoint must have the form STEP=DIR") from exc
        if step < 1 or not directory:
            raise ValueError("checkpoint step and directory must be non-empty")
        if step in seen:
            raise ValueError("duplicate checkpoint step: {}".format(step))
        seen.add(step)
        checkpoints.append((step, Path(directory).expanduser().resolve()))
    if len(checkpoints) != 4:
        raise ValueError("RWTT-28Lite trajectory requires exactly four checkpoints")
    return sorted(checkpoints)


def as_int(row, name):
    return int(row[name])


def as_float(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite {}".format(name))
    return value


def model_equal(rows, endpoint):
    selected = [row for row in rows if row["endpoint"] == endpoint]
    if len(selected) != 28:
        raise ValueError(
            "{} endpoint must contain exactly 28 RWTT models, found {}".format(
                endpoint, len(selected)))
    if len({row["model_id"] for row in selected}) != 28:
        raise ValueError(endpoint + " endpoint contains duplicate models")
    fields = ("bpp", "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_611")
    result = {
        name: sum(as_float(row, name) for row in selected) / len(selected)
        for name in fields
    }
    result.update({
        "num_models": len(selected),
        "num_h5": sum(as_int(row, "num_h5") for row in selected),
        "total_points": sum(as_int(row, "total_points") for row in selected),
        "models": sorted(row["model_id"] for row in selected),
    })
    return result


def check_hard_correctness(rows, summary, resolved):
    if len(rows) != 28:
        raise ValueError(
            "RWTT-28Lite must contain exactly 28 H5, found {}".format(len(rows)))
    samples = [row["sample"] for row in rows]
    models = [row["model_id"] for row in rows]
    if len(set(samples)) != 28 or len(set(models)) != 28:
        raise ValueError("RWTT-28Lite must contain one unique H5 per model")
    if not resolved.get("require_exact"):
        raise ValueError("evaluation was not run with --require-exact")
    if summary.get("status") != "PASS":
        raise ValueError("evaluation summary status is not PASS")

    maximum_difference = as_float(
        {"value": summary.get("hard_full_max_abs_difference")}, "value")
    identities = {
        "four_base_residual_streams": all(
            as_int(row, "num_base_residual_streams") == 4 for row in rows),
        "no_native_r5_stream": all(
            as_int(row, "num_native_r5_streams") == 0 for row in rows),
        "base_bit_identity": all(
            as_int(row, "base_bits") == as_int(row, "x_low_bits")
            + sum(as_int(row, "r{}_bits".format(index))
                  for index in range(1, 5))
            for row in rows),
        "full_bit_identity": all(
            as_int(row, "full_bits")
            == as_int(row, "base_bits") + as_int(row, "enhancement_bits")
            for row in rows),
        "hard_full_exact": maximum_difference == 0.0 and all(
            as_float(row, "hard_full_max_abs_difference") == 0.0
            for row in rows),
    }
    failed = [name for name, passed in identities.items() if not passed]
    if failed:
        raise ValueError("hard correctness failed: " + ", ".join(failed))
    return {
        "status": "PASS",
        "require_exact": True,
        "max_abs_difference": maximum_difference,
        **identities,
        "samples": samples,
    }


def summarize_checkpoint(point, step, directory):
    required = {
        name: directory / name for name in (
            "resolved_args.json", "per_h5.csv", "per_model.csv",
            "endpoint_summary.json")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing evaluation artifacts: " + ", ".join(missing))

    resolved = read_json(required["resolved_args.json"])
    h5_rows = read_csv(required["per_h5.csv"])
    model_rows = read_csv(required["per_model.csv"])
    summary = read_json(required["endpoint_summary.json"])
    base = model_equal(model_rows, "Base")
    full = model_equal(model_rows, "Full")
    if base["models"] != full["models"]:
        raise ValueError("Base and Full model coverage differs")
    if base["num_h5"] != 28 or full["num_h5"] != 28:
        raise ValueError("each endpoint must aggregate exactly 28 H5")
    hard = check_hard_correctness(h5_rows, summary, resolved)

    return {
        "point": point,
        "checkpoint_step": step,
        "evaluation_dir": str(directory),
        "enhancement_checkpoint": resolved.get("enhancement_checkpoint"),
        "conditioning_lambda": int(resolved["conditioning_lambda"]),
        "num_models": base["num_models"],
        "num_h5": base["num_h5"],
        "total_points": base["total_points"],
        "base_bpp": base["bpp"],
        "enhancement_bpp": full["bpp"] - base["bpp"],
        "full_bpp": full["bpp"],
        "base_y_psnr": base["y_psnr"],
        "base_u_psnr": base["u_psnr"],
        "base_v_psnr": base["v_psnr"],
        "base_yuv_psnr_611": base["yuv_psnr_611"],
        "full_y_psnr": full["y_psnr"],
        "full_u_psnr": full["u_psnr"],
        "full_v_psnr": full["v_psnr"],
        "full_yuv_psnr_611": full["yuv_psnr_611"],
        "hard_correctness": hard,
    }


def dominates(left, right):
    """Return True when left is no worse in RD and strictly better in one."""
    rate_no_worse = left["full_bpp"] <= right["full_bpp"]
    quality_no_worse = (
        left["full_yuv_psnr_611"] >= right["full_yuv_psnr_611"])
    strictly_better = (
        left["full_bpp"] < right["full_bpp"]
        or left["full_yuv_psnr_611"] > right["full_yuv_psnr_611"])
    return rate_no_worse and quality_no_worse and strictly_better


def pareto_frontier(results):
    frontier = [
        candidate for candidate in results
        if not any(
            other is not candidate and dominates(other, candidate)
            for other in results)
    ]
    return sorted(
        frontier,
        key=lambda result: (
            result["full_bpp"], -result["full_yuv_psnr_611"],
            result["checkpoint_step"]))


def flat_row(result, pareto, unique):
    row = {
        key: value for key, value in result.items()
        if key != "hard_correctness"
    }
    row.update({
        "hard_status": result["hard_correctness"]["status"],
        "hard_max_abs_difference": result["hard_correctness"][
            "max_abs_difference"],
        "four_base_residual_streams": result["hard_correctness"][
            "four_base_residual_streams"],
        "no_native_r5_stream": result["hard_correctness"][
            "no_native_r5_stream"],
        "base_bit_identity": result["hard_correctness"]["base_bit_identity"],
        "full_bit_identity": result["hard_correctness"]["full_bit_identity"],
        "hard_full_exact": result["hard_correctness"]["hard_full_exact"],
        "pareto_candidate": pareto,
        "unique_candidate": unique,
    })
    return row


def write_outputs(output_dir, point, results, frontier):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {suffix: output_dir / (OUTPUT_STEM + "." + suffix)
             for suffix in ("csv", "json", "md")}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    frontier_steps = {result["checkpoint_step"] for result in frontier}
    unique = frontier[0] if len(frontier) == 1 else None
    rows = [flat_row(
        result, result["checkpoint_step"] in frontier_steps,
        unique is result) for result in results]
    with paths["csv"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rule = {
        "axes": "model-equal physical Full bpp and Full YUV611",
        "dominance": (
            "A dominates B iff A.bpp <= B.bpp and A.YUV611 >= B.YUV611, "
            "with at least one strict inequality"),
        "automatic_winner": "only when the Pareto frontier has one checkpoint",
    }
    payload = {
        "status": "PASS",
        "benchmark": "RWTT-28Lite",
        "point": point,
        "selection_rule": rule,
        "pareto_candidates": frontier,
        "unique_candidate": unique,
        "manager_review_required": len(frontier) != 1,
        "trajectory": results,
    }
    with paths["json"].open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    with paths["md"].open("x", encoding="utf-8") as handle:
        handle.write("# Enhancement {} RWTT-28Lite trajectory\n\n".format(point))
        handle.write(
            "Selection uses the `(physical Full bpp, Full YUV611)` Pareto "
            "frontier. A unique candidate is declared only when one checkpoint "
            "dominates every other checkpoint.\n\n")
        handle.write(
            "| Step | Base bpp | Enh. bpp | Full bpp | Full Y | Full U | "
            "Full V | Full YUV611 | Hard | Pareto | Unique |\n")
        handle.write(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n")
        for row in rows:
            handle.write(
                "| {checkpoint_step} | {base_bpp:.9f} | {enhancement_bpp:.9f} | "
                "{full_bpp:.9f} | {full_y_psnr:.6f} | {full_u_psnr:.6f} | "
                "{full_v_psnr:.6f} | {full_yuv_psnr_611:.6f} | "
                "{hard_status} | {pareto_candidate} | "
                "{unique_candidate} |\n".format(**row))
        handle.write("\nPareto checkpoints: {}.\n".format(
            ", ".join("step{}".format(item["checkpoint_step"])
                      for item in frontier)))
        if unique is None:
            handle.write("Manager review is required; no unique winner.\n")
        else:
            handle.write("Unique candidate: step `{}`.\n".format(
                unique["checkpoint_step"]))
    return paths


def main():
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    results = [summarize_checkpoint(args.point, step, directory)
               for step, directory in checkpoints]
    lambdas = {result["conditioning_lambda"] for result in results}
    if len(lambdas) != 1:
        raise ValueError("checkpoint evaluations use different lambdas")
    samples = [result["hard_correctness"]["samples"] for result in results]
    if any(current != samples[0] for current in samples[1:]):
        raise ValueError("checkpoint evaluations use different RWTT-28Lite samples/order")

    frontier = pareto_frontier(results)
    paths = write_outputs(
        Path(args.output_dir).expanduser().resolve(), args.point, results,
        frontier)
    print(json.dumps({
        "status": "PASS",
        "pareto_steps": [item["checkpoint_step"] for item in frontier],
        "unique_candidate_step": (
            frontier[0]["checkpoint_step"] if len(frontier) == 1 else None),
        "manager_review_required": len(frontier) != 1,
        "outputs": {name: str(path) for name, path in paths.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
