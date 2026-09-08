"""Evaluation shared by canonical Base training and standalone evaluation."""

import csv
import os
import statistics
import time

import MinkowskiEngine as ME
import torch

from scalable_attribute.evaluation import psnr, sample_identity
from scalable_attribute.data import UncachedPCDataset


FIELDS = (
    "file", "model_id", "point_count",
    "mse_unpool", "mse_native", "mse_learned",
    "psnr_unpool", "psnr_native", "psnr_learned",
    "delta_psnr_learned_native", "delta_psnr_learned_unpool",
    "max_abs_learned_native",
)


def _mse(reference, reconstruction):
    if list(reference.tensor_stride) != list(reconstruction.tensor_stride):
        raise RuntimeError("Validation tensor strides differ")
    if not torch.equal(reference.C, reconstruction.C):
        raise RuntimeError("Validation coordinates differ")
    value = torch.mean((reference.F - reconstruction.F) ** 2)
    if not torch.isfinite(value):
        raise RuntimeError("Validation MSE is non-finite")
    return float(value.item())


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_models(rows):
    grouped = {}
    for row in rows:
        current = grouped.setdefault(row["model_id"], {
            "points": 0,
            "weighted_mse_unpool": 0.0,
            "weighted_mse_native": 0.0,
            "weighted_mse_learned": 0.0,
        })
        points = int(row["point_count"])
        current["points"] += points
        for name in ("unpool", "native", "learned"):
            current["weighted_mse_" + name] += row["mse_" + name] * points

    models = []
    for model_id, values in grouped.items():
        result = {"model_id": model_id, "points": values["points"]}
        for name in ("unpool", "native", "learned"):
            mse = values["weighted_mse_" + name] / values["points"]
            result["mse_" + name] = mse
            result["psnr_" + name] = psnr(mse)
        result["delta_psnr_learned_native"] = (
            result["psnr_learned"] - result["psnr_native"])
        result["delta_psnr_learned_unpool"] = (
            result["psnr_learned"] - result["psnr_unpool"])
        models.append(result)
    return models


def summarize(rows, runtime_seconds):
    if not rows:
        raise ValueError("No validation rows")
    models = _aggregate_models(rows)
    summary = {
        "metric": "normalized YUV 1:1:1 full-resolution MSE; PSNR peak=1",
        "aggregation": (
            "point-weighted MSE within original model, then PSNR, "
            "then model-equal mean"),
        "num_h5": len(rows),
        "num_models": len(models),
        "runtime_seconds": runtime_seconds,
    }
    for level, values in (("h5", rows), ("model", models)):
        for name in ("unpool", "native", "learned"):
            summary[level + "_mean_mse_" + name] = statistics.fmean(
                item["mse_" + name] for item in values)
            summary[level + "_mean_psnr_" + name] = statistics.fmean(
                item["psnr_" + name] for item in values)
        for comparison in ("learned_native", "learned_unpool"):
            key = "delta_psnr_" + comparison
            summary[level + "_mean_" + key] = statistics.fmean(
                item[key] for item in values)
            summary[level + "_median_" + key] = statistics.median(
                item[key] for item in values)

    native_improved = sum(
        row["psnr_learned"] > row["psnr_native"] for row in rows)
    unpool_improved = sum(
        row["psnr_learned"] > row["psnr_unpool"] for row in rows)
    summary.update({
        "h5_improved_over_native": native_improved,
        "h5_improved_over_native_percent": 100.0 * native_improved / len(rows),
        "h5_improved_over_unpool": unpool_improved,
        "h5_improved_over_unpool_percent": 100.0 * unpool_improved / len(rows),
        "max_abs_learned_native": max(
            row["max_abs_learned_native"] for row in rows),
    })
    return summary


@torch.no_grad()
def evaluate_base(model, files, entries, base_lambda, output_csv):
    if len(files) != len(entries):
        raise ValueError("Validation file/manifest entry counts differ")
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    was_training = model.training
    model.eval()
    rows = []
    started = time.monotonic()
    for index in range(len(dataset)):
        coords, feats = dataset[index]
        batched_coords, batched_feats = ME.utils.sparse_collate(
            [coords], [feats])
        attribute = ME.SparseTensor(
            features=batched_feats, coordinates=batched_coords,
            tensor_stride=1, device="cuda")
        output = model(attribute, base_lambda)
        baselines = model.native_baselines(output["prefix_state"])
        values = {}
        for name, reconstruction in (
                ("unpool", baselines["B_unpool"]),
                ("native", baselines["B_native"]),
                ("learned", output["Base"])):
            values["mse_" + name] = _mse(attribute, reconstruction)
            values["psnr_" + name] = psnr(values["mse_" + name])
        model_id, _ = sample_identity(entries[index])
        row = {
            "file": entries[index], "model_id": model_id,
            "point_count": len(attribute), **values,
            "delta_psnr_learned_native": (
                values["psnr_learned"] - values["psnr_native"]),
            "delta_psnr_learned_unpool": (
                values["psnr_learned"] - values["psnr_unpool"]),
            "max_abs_learned_native": float((
                output["Base"].F - baselines["B_native"].F
            ).abs().max().item()),
        }
        rows.append(row)
        _write_csv(output_csv, rows)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print("validation {}/{} learned-native={:.6f} dB".format(
                index + 1, len(dataset), row["delta_psnr_learned_native"]),
                flush=True)
    summary = summarize(rows, time.monotonic() - started)
    if was_training:
        model.train()
    return summary
