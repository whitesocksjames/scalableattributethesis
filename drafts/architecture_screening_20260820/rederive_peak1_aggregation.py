#!/usr/bin/env python3
"""Re-derive peak=1 aggregation without overwriting retained artifacts."""
import argparse
import csv
import json
import math
import os

from scalable_attribute.evaluation import aggregate_models, average_models
from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS


SHIFT_DB = 20.0 * math.log10(255.0)
MODEL_FIELDS = (
    "rate_id", "checkpoint_profile", "base_lambda", "model_id", "num_h5",
    "total_points", "total_bits", "bpp", "y_mse", "u_mse", "v_mse",
    "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_611",
)
CURVE_FIELDS = (
    "rate_id", "checkpoint_profile", "base_lambda", "num_models", "num_h5",
    "total_points", "mean_model_bpp", "mean_model_y_psnr",
    "mean_model_yuv_psnr_611",
)
SCALABLE_MODEL_FIELDS = (
    "experiment", "checkpoint_step", "base_profile", "base_lambda",
    "model_id", "num_h5", "total_points", "base_bits", "el_bits",
    "full_bits", "base_bpp", "el_bpp", "full_bpp",
    "base_yuv_psnr_611", "full_yuv_psnr_611",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-study-root", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--scalable-eval-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    if os.path.exists(path):
        raise FileExistsError("Refusing to overwrite corrected artifact: " + path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def close(a, b, tolerance=1e-9):
    return abs(float(a) - float(b)) <= tolerance


def check_reference(old_models, new_models, label, checks):
    if len(old_models) != len(new_models):
        raise RuntimeError(label + " model count changed")
    for old, new in zip(old_models, new_models):
        if old["model_id"] != new["model_id"]:
            raise RuntimeError(label + " model order changed")
        if not close(old["bpp"], new["bpp"], 0.0):
            raise RuntimeError(label + " bpp changed for " + new["model_id"])
        for field in ("y_psnr", "u_psnr", "v_psnr", "yuv_psnr_611"):
            shift = float(old[field]) - float(new[field])
            if not close(shift, SHIFT_DB, 1e-9):
                raise RuntimeError(
                    "{} {} shift is {}, expected {}".format(
                        label, field, shift, SHIFT_DB))
    checks[label] = {
        "num_models": len(new_models),
        "bpp_unchanged": True,
        "uniform_psnr_drop_db": SHIFT_DB,
    }


def scalable_models(rows):
    rows = [dict(row, rate_id=row["experiment"],
                 checkpoint_profile=row["base_profile"])
            for row in rows]
    base = aggregate_models(rows, "base_bits", "base_")
    full = aggregate_models(rows, "full_bits", "full_")
    output = []
    first = rows[0]
    for base_row, full_row in zip(base, full):
        if base_row["model_id"] != full_row["model_id"]:
            raise RuntimeError("Scalable Base/Full model alignment changed")
        output.append({
            "experiment": first["experiment"],
            "checkpoint_step": first["checkpoint_step"],
            "base_profile": first["base_profile"],
            "base_lambda": first["base_lambda"],
            "model_id": base_row["model_id"],
            "num_h5": base_row["num_h5"],
            "total_points": base_row["total_points"],
            "base_bits": base_row["total_bits"],
            "el_bits": full_row["total_bits"] - base_row["total_bits"],
            "full_bits": full_row["total_bits"],
            "base_bpp": base_row["bpp"],
            "el_bpp": full_row["bpp"] - base_row["bpp"],
            "full_bpp": full_row["bpp"],
            "base_yuv_psnr_611": base_row["yuv_psnr_611"],
            "full_yuv_psnr_611": full_row["yuv_psnr_611"],
        })
    return base, full, output


def main():
    args = parse_args()
    for name in vars(args):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    if os.path.exists(args.output_root):
        raise FileExistsError(
            "Corrected output root already exists: " + args.output_root)

    source_full = os.path.join(args.reference_study_root, "reference_full")
    source_dev = os.path.join(args.reference_study_root, "reference_dev")
    output_full = os.path.join(args.output_root, "reference_full")
    output_dev = os.path.join(args.output_root, "reference_dev")
    with open(args.dev_manifest, encoding="utf-8") as handle:
        dev_samples = {line.strip() for line in handle if line.strip()}

    checks = {}
    full_curve = []
    dev_curve = []
    for rate_id, _, _ in OFFICIAL_RWTT_REFERENCE_POINTS:
        rows = read_csv(os.path.join(source_full, rate_id, "per_h5.csv"))
        full_models = aggregate_models(rows)
        old_full_models = read_csv(
            os.path.join(source_full, rate_id, "per_model.csv"))
        check_reference(old_full_models, full_models, rate_id + "_full", checks)
        write_csv(os.path.join(output_full, rate_id, "per_model.csv"),
                  MODEL_FIELDS, full_models)
        full_curve.append(average_models(full_models))

        dev_rows = [row for row in rows if row["sample"] in dev_samples]
        if {row["sample"] for row in dev_rows} != dev_samples:
            raise RuntimeError(rate_id + " does not cover fixed Dev manifest")
        dev_models = aggregate_models(dev_rows)
        old_dev_models = read_csv(
            os.path.join(source_dev, rate_id, "per_model.csv"))
        check_reference(old_dev_models, dev_models, rate_id + "_dev", checks)
        write_csv(os.path.join(output_dev, rate_id, "per_model.csv"),
                  MODEL_FIELDS, dev_models)
        dev_curve.append(average_models(dev_models))

    write_csv(os.path.join(output_full, "reference_curve.csv"),
              CURVE_FIELDS, full_curve)
    write_csv(os.path.join(output_dev, "reference_dev_curve.csv"),
              CURVE_FIELDS, dev_curve)

    scalable_rows = read_csv(os.path.join(args.scalable_eval_root, "per_h5.csv"))
    base_models, full_models, model_rows = scalable_models(scalable_rows)
    write_csv(os.path.join(args.output_root, "c2_r08_smoke", "per_model.csv"),
              SCALABLE_MODEL_FIELDS, model_rows)
    old_summary = read_csv(
        os.path.join(args.scalable_eval_root, "endpoint_summary.csv"))[0]
    base_average = average_models(base_models)
    full_average = average_models(full_models)
    summary = dict(old_summary)
    summary.update({
        "base_mean_bpp": base_average["mean_model_bpp"],
        "base_mean_yuv_psnr_611": base_average["mean_model_yuv_psnr_611"],
        "full_mean_bpp": full_average["mean_model_bpp"],
        "full_mean_yuv_psnr_611": full_average["mean_model_yuv_psnr_611"],
        "el_mean_bpp": (full_average["mean_model_bpp"]
                         - base_average["mean_model_bpp"]),
    })
    write_csv(os.path.join(args.output_root, "c2_r08_smoke", "endpoint_summary.csv"),
              tuple(summary), [summary])

    direct = scalable_rows[0]
    for prefix, models in (("base_", base_models), ("full_", full_models)):
        aggregated = models[0]["yuv_psnr_611"]
        direct_psnr = float(direct[prefix + "yuv_psnr_611"])
        if not close(aggregated, direct_psnr, 1e-3):
            raise RuntimeError(
                "Single-H5 {} aggregation {} differs from direct pc_error {}".format(
                    prefix, aggregated, direct_psnr))
    old_base = float(old_summary["base_mean_yuv_psnr_611"])
    old_full = float(old_summary["full_mean_yuv_psnr_611"])
    new_base = float(summary["base_mean_yuv_psnr_611"])
    new_full = float(summary["full_mean_yuv_psnr_611"])
    if not close(old_base - new_base, SHIFT_DB, 1e-9):
        raise RuntimeError("Scalable Base PSNR does not have the expected shift")
    if not close(old_full - new_full, SHIFT_DB, 1e-9):
        raise RuntimeError("Scalable Full PSNR does not have the expected shift")
    if not close(old_full - old_base, new_full - new_base, 1e-9):
        raise RuntimeError("Scalable endpoint PSNR difference changed")
    checks["c2_r08_smoke"] = {
        "bpp_unchanged": (
            close(old_summary["base_mean_bpp"], summary["base_mean_bpp"], 0.0)
            and close(old_summary["full_mean_bpp"], summary["full_mean_bpp"], 0.0)),
        "uniform_psnr_drop_db": SHIFT_DB,
        "endpoint_difference_unchanged": True,
        "single_h5_matches_direct_pc_error_within_db": 1e-3,
    }
    if not checks["c2_r08_smoke"]["bpp_unchanged"]:
        raise RuntimeError("Scalable endpoint bpp changed")

    with open(os.path.join(args.output_root, "regression_checks.json"), "x",
              encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(args.output_root, "provenance.json"), "x",
              encoding="utf-8") as handle:
        json.dump({
            "source_reference_full": source_full,
            "source_reference_dev": source_dev,
            "source_scalable_eval": args.scalable_eval_root,
            "dev_manifest": args.dev_manifest,
            "aggregation": "point-weighted per-channel MSE within original model; PSNR peak=1; 6:1:1; equal average across models",
            "retained_per_h5_reused": True,
            "source_artifacts_overwritten": False,
        }, handle, indent=2)
        handle.write("\n")
    print(args.output_root)


if __name__ == "__main__":
    main()
