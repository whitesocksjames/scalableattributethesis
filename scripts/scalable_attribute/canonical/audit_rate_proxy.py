#!/usr/bin/env python3
"""Compare noise, symbol and physical rate proxies without training."""

import argparse
import csv
import json
import os
import shlex
import statistics
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch

from basic_models.loss import get_bits
from data_utils.dataloaders.attribute_dataloader import PCDataset
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_frozen_base)
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import psnr


RATE_NAMES = ("noise", "symbol", "hard")
FAMILIES = ("native_r5", "enh")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--tmc3-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--enhancement-checkpoint")
    parser.add_argument("--artifact-prefix", default="e0c_rate_proxy")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def same_support(left, right, label):
    if list(left.tensor_stride) != list(right.tensor_stride):
        raise RuntimeError(label + " tensor strides differ")
    if not torch.equal(left.C, right.C):
        raise RuntimeError(label + " coordinates differ")


def quality(reference, reconstruction):
    same_support(reference, reconstruction, "quality")
    mse = float(torch.mean((reference.F - reconstruction.F) ** 2).item())
    if not np.isfinite(mse):
        raise RuntimeError("Non-finite reconstruction MSE")
    return mse, psnr(mse)


@torch.no_grad()
def enhancement_hard_with_symbols(model, attribute):
    """Capture the exact symbol tensor passed to torchac without changing it."""
    entropy = model.enhancement.vae.entropy_fn
    original_compress = entropy.compress
    captured = []

    def capture(symbols, *args, **kwargs):
        captured.append(symbols.detach().cpu())
        return original_compress(symbols, *args, **kwargs)

    entropy.compress = capture
    try:
        result = model.hard_reconstruct(attribute)
    finally:
        entropy.compress = original_compress
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one Enhancement symbol tensor")
    symbols = captured[0]
    if not torch.equal(symbols, torch.round(symbols)):
        raise RuntimeError("Captured Enhancement symbols are not integral")
    return result, symbols


@torch.no_grad()
def native_estimated_r5(prefix, attribute, conditioning_lambda, training):
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output)

    handle = prefix.model.VAE.register_forward_hook(capture)
    try:
        output = prefix.model(
            attribute, training=training, lmb=conditioning_lambda,
            real_coding=False)
    finally:
        handle.remove()
    if len(captured) != 5 or len(output["likelihood_list"]) != 5:
        raise RuntimeError("Native forward did not invoke five residual stages")
    returned_to_invocation = []
    for likelihood in output["likelihood_list"]:
        matches = [
            index for index, stage in enumerate(captured)
            if likelihood is stage["likelihood"]]
        if len(matches) != 1:
            raise RuntimeError("Could not identify returned likelihood stage")
        returned_to_invocation.append(matches[0])
    if returned_to_invocation != [4, 3, 2, 1, 0]:
        raise RuntimeError("Native likelihood reversal mapping changed")
    r5_likelihood = output["likelihood_list"][0]
    if not torch.isfinite(r5_likelihood).all():
        raise RuntimeError("Native r5 likelihood is non-finite")
    bits = float(get_bits(r5_likelihood).item())
    if not np.isfinite(bits):
        raise RuntimeError("Native r5 estimated bits are non-finite")
    return bits, returned_to_invocation


@torch.no_grad()
def native_hard(prefix, attribute, conditioning_lambda):
    entropy = prefix.model.VAE.entropy_fn
    original_compress = entropy.compress
    captured = []

    def capture(symbols, *args, **kwargs):
        captured.append(symbols.detach().cpu())
        return original_compress(symbols, *args, **kwargs)

    entropy.compress = capture
    try:
        encoded, x_low, gpcc_bits = prefix.model(
            attribute, training=False, lmb=conditioning_lambda, encode=True)
    finally:
        entropy.compress = original_compress
    if len(encoded) != 5:
        raise RuntimeError("Native hard path did not produce five streams")
    if len(captured) != 5:
        raise RuntimeError("Native hard path did not expose five symbol tensors")
    output_strides = [list(item["x_out"].tensor_stride) for item in encoded]
    expected = [[value, value, value] for value in (16, 8, 4, 2, 1)]
    if output_strides != expected:
        raise RuntimeError(
            "Native hard stream stride mapping changed: {}".format(
                output_strides))
    x0 = ME.SparseTensor(
        features=torch.zeros_like(attribute.F),
        coordinate_map_key=attribute.coordinate_map_key,
        coordinate_manager=attribute.coordinate_manager,
        device=attribute.device)
    reconstruction = prefix.model.decode(
        x0=x0, x_low=x_low, enc_set_list=encoded,
        lmb=conditioning_lambda)
    residual_bits = [int(len(item["strings"]) * 8) for item in encoded]
    if residual_bits[4] <= 0:
        raise RuntimeError("Native r5 hard stream is empty")
    _, value_psnr = quality(attribute, reconstruction)
    return {
        "r5_bits": residual_bits[4],
        "total_bits": int(gpcc_bits + sum(residual_bits)),
        "full_psnr": value_psnr,
        "stream_strides": output_strides,
        "r5_symbols": captured[4],
    }


def pearson(left, right):
    value = float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])
    if not np.isfinite(value):
        raise RuntimeError("Pearson correlation is non-finite")
    return value


def ratio_statistics(rows, family, numerator):
    values = [
        row[family + "_" + numerator + "_bits"]
        / row[family + "_hard_bits"] for row in rows]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(rows):
    total_points = sum(row["points"] for row in rows)
    summary = {
        "num_h5": len(rows),
        "num_models": len({row["model"] for row in rows}),
        "total_points": total_points,
        "point_weighted": {},
        "model_equal": {},
        "per_h5_ratio_statistics": {},
        "correlations": {},
    }
    for family in FAMILIES:
        point_values = {
            rate: sum(row[family + "_" + rate + "_bits"] for row in rows)
            / total_points for rate in RATE_NAMES}
        model_values = {
            rate: statistics.fmean(
                row[family + "_" + rate + "_bpp"] for row in rows)
            for rate in RATE_NAMES}
        for values in (point_values, model_values):
            values["symbol_over_hard"] = values["symbol"] / values["hard"]
            values["noise_over_hard"] = values["noise"] / values["hard"]
        summary["point_weighted"][family] = point_values
        summary["model_equal"][family] = model_values
        summary["per_h5_ratio_statistics"][family] = {
            "symbol_over_hard": ratio_statistics(rows, family, "symbol"),
            "noise_over_hard": ratio_statistics(rows, family, "noise"),
        }
        hard = [row[family + "_hard_bpp"] for row in rows]
        summary["correlations"][family] = {
            "pearson_symbol_hard": pearson(
                [row[family + "_symbol_bpp"] for row in rows], hard),
            "pearson_noise_hard": pearson(
                [row[family + "_noise_bpp"] for row in rows], hard),
        }
    for name in ("native_full_psnr", "base_psnr", "step0_full_psnr"):
        summary["model_equal"][name] = statistics.fmean(
            row[name] for row in rows)
    for name in ("base_hard_bpp", "native_total_hard_bpp",
                 "step0_total_hard_bpp"):
        summary["point_weighted"][name] = (
            sum(row[name.replace("_bpp", "_bits")] for row in rows)
            / total_points)
        summary["model_equal"][name] = statistics.fmean(
            row[name] for row in rows)
    summary["model_equal"].update({
        "step0_minus_base_psnr": (
            summary["model_equal"]["step0_full_psnr"]
            - summary["model_equal"]["base_psnr"]),
        "step0_minus_native_psnr": (
            summary["model_equal"]["step0_full_psnr"]
            - summary["model_equal"]["native_full_psnr"]),
    })
    total_symbols = sum(row["enh_symbol_count"] for row in rows)
    total_nonzero = sum(row["enh_symbol_nonzero_count"] for row in rows)
    summary["hard_symbol_statistics"] = {
        "count": total_symbols,
        "nonzero_count": total_nonzero,
        "nonzero_fraction": total_nonzero / total_symbols,
        "min": min(row["enh_symbol_min"] for row in rows),
        "max": max(row["enh_symbol_max"] for row in rows),
        "active_channels_union": sorted({
            channel for row in rows
            for channel in json.loads(row["enh_active_channels"])}),
    }
    summary["hard_symbol_statistics"]["active_channel_count"] = len(
        summary["hard_symbol_statistics"]["active_channels_union"])
    for label, count_key, nonzero_key, active_key, per_channel_key in (
            ("native_r5", "native_r5_symbol_count",
             "native_r5_symbol_nonzero_count", "native_r5_active_channels",
             "native_r5_per_channel_nonzero_fraction"),
            ("enhancement", "enh_symbol_count", "enh_symbol_nonzero_count",
             "enh_active_channels", "enh_per_channel_nonzero_fraction")):
        channel_lists = [json.loads(row[per_channel_key]) for row in rows]
        channels = len(channel_lists[0])
        if any(len(values) != channels for values in channel_lists):
            raise RuntimeError(label + " channel count changed across samples")
        sites = [row[count_key] // channels for row in rows]
        total_sites = sum(sites)
        summary[label + "_channel_statistics"] = {
            "active_channels_union": sorted({
                channel for row in rows
                for channel in json.loads(row[active_key])}),
            "nonzero_fraction": (
                sum(row[nonzero_key] for row in rows)
                / sum(row[count_key] for row in rows)),
            "per_channel_nonzero_fraction": [
                sum(values[channel] * sites[index]
                    for index, values in enumerate(channel_lists)) / total_sites
                for channel in range(channels)],
        }
        summary[label + "_channel_statistics"]["active_channel_count"] = len(
            summary[label + "_channel_statistics"]["active_channels_union"])
    return summary


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    started = time.monotonic()
    if os.path.exists(args.output_dir) and not args.allow_existing_output_dir:
        raise FileExistsError("E0-C output directory already exists")
    if not os.path.isfile(args.tmc3_path) or not os.access(
            args.tmc3_path, os.X_OK):
        raise FileNotFoundError("Executable tmc3 not found: " + args.tmc3_path)
    os.makedirs(args.output_dir, exist_ok=args.allow_existing_output_dir)
    tmc3_link = os.path.join(args.output_dir, "tmc3_v21")
    if os.path.lexists(tmc3_link):
        if os.path.realpath(tmc3_link) != os.path.realpath(args.tmc3_path):
            raise RuntimeError("Existing tmc3_v21 points to another executable")
    else:
        os.symlink(os.path.realpath(args.tmc3_path), tmc3_link)
    os.chdir(args.output_dir)
    csv_path = args.artifact_prefix + ".csv"
    summary_path = args.artifact_prefix + ".json"
    if os.path.exists(csv_path) or os.path.exists(summary_path):
        raise FileExistsError("Rate-audit artifact already exists")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.file_list, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    models = [entry.split("/")[0] for entry in entries]
    if len(entries) < 12 or len(set(models)) != len(entries):
        raise ValueError("E0-C requires at least 12 distinct original models")
    files = h5_files(args.data_root, args.file_list)
    with open(args.artifact_prefix + "_selected_h5.txt", "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(entries) + "\n")
    metadata = dict(vars(args))
    metadata["noise_seed_policy"] = (
        "reset numpy seed to seed + sample_index independently for native "
        "and Enhancement noise paths")
    with open(args.artifact_prefix + "_resolved_args.json", "w",
              encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(args.artifact_prefix + "_command.txt", "w",
              encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")

    checkpoint = torch.load(
        args.base_synthesis_checkpoint, map_location="cpu")
    config = BaseSynthesisConfig(**checkpoint["config"])
    base = CanonicalBaseModel(args.released_checkpoint, config).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)
    model = CanonicalScalableModel(
        base, conditioning_lambda=args.conditioning_lambda).cuda()
    checkpoint_step = 0
    if args.enhancement_checkpoint:
        enhancement_state = torch.load(
            args.enhancement_checkpoint, map_location="cpu")
        if enhancement_state.get("architecture") != (
                "canonical_independent_enhancement"):
            raise ValueError("Enhancement checkpoint architecture mismatch")
        if int(enhancement_state.get("conditioning_lambda", -1)) != int(
                args.conditioning_lambda):
            raise ValueError("Enhancement conditioning lambda mismatch")
        if os.path.realpath(enhancement_state.get(
                "released_checkpoint", "")) != os.path.realpath(
                    args.released_checkpoint):
            raise ValueError("Enhancement released checkpoint mismatch")
        if os.path.realpath(enhancement_state.get(
                "base_synthesis_checkpoint", "")) != os.path.realpath(
                    args.base_synthesis_checkpoint):
            raise ValueError("Enhancement Base checkpoint mismatch")
        model.enhancement.vae.load_state_dict(
            enhancement_state["enhancement_vae"], strict=True)
        checkpoint_step = int(enhancement_state.get("step", -1))
        if checkpoint_step < 0:
            raise ValueError("Enhancement checkpoint step is invalid")
    model.eval()
    dataset = PCDataset(files, color_format="yuv", normalize=True)
    rows = []
    likelihood_mapping = None
    hard_stream_mapping = None

    for index in range(len(dataset)):
        coords, feats = dataset[index]
        batch_coords, batch_feats = ME.utils.sparse_collate([coords], [feats])
        attribute = ME.SparseTensor(
            features=batch_feats, coordinates=batch_coords, tensor_stride=1,
            device="cuda")
        points = len(attribute)

        np.random.seed(args.seed + index)
        native_noise_bits, noise_mapping = native_estimated_r5(
            model.base.prefix, attribute, args.conditioning_lambda,
            training=True)
        native_symbol_bits, symbol_mapping = native_estimated_r5(
            model.base.prefix, attribute, args.conditioning_lambda,
            training=False)
        if noise_mapping != symbol_mapping:
            raise RuntimeError("Native noise/symbol stage mappings differ")
        native = native_hard(
            model.base.prefix, attribute, args.conditioning_lambda)
        native_symbols = native.pop("r5_symbols")
        native_nonzero = native_symbols.ne(0)
        native_active_channels = torch.nonzero(
            native_nonzero.any(dim=0), as_tuple=False).flatten().tolist()

        np.random.seed(args.seed + index)
        model.train()
        enhancement_noise = model(attribute)
        enh_noise_likelihood = enhancement_noise["likelihood_E"]
        if not torch.isfinite(enh_noise_likelihood).all():
            raise RuntimeError("Enhancement noise likelihood is non-finite")
        enh_noise_bits = float(get_bits(enh_noise_likelihood).item())
        model.eval()
        enhancement_symbol = model.deterministic_forward(attribute)
        enh_symbol_likelihood = enhancement_symbol["likelihood_E"]
        if not torch.isfinite(enh_symbol_likelihood).all():
            raise RuntimeError("Enhancement symbol likelihood is non-finite")
        enh_symbol_bits = float(get_bits(enh_symbol_likelihood).item())
        enhancement_hard, hard_symbols = enhancement_hard_with_symbols(
            model, attribute)
        enh_hard_bits = enhancement_hard["enhancement_bits"]
        if enh_hard_bits <= 0:
            raise RuntimeError("Enhancement hard stream is empty")
        same_support(
            enhancement_symbol["Full"], enhancement_hard["Full"],
            "Enhancement deterministic/hard Full")
        hard_difference = float((
            enhancement_symbol["Full"].F - enhancement_hard["Full"].F
        ).abs().max().item())
        if hard_difference != 0.0:
            raise RuntimeError("Enhancement deterministic/hard Full differs")
        _, base_psnr = quality(attribute, enhancement_hard["Base"])
        _, step0_psnr = quality(attribute, enhancement_hard["Full"])
        nonzero = hard_symbols.ne(0)
        active_channels = torch.nonzero(
            nonzero.any(dim=0), as_tuple=False).flatten().tolist()

        values = {
            "file": entries[index], "model": models[index],
            "points": points,
            "native_full_psnr": native["full_psnr"],
            "step0_full_psnr": step0_psnr,
            "enhancement_full_psnr": step0_psnr,
            "base_psnr": base_psnr,
            "native_total_hard_bpp": native["total_bits"] / points,
            "step0_total_hard_bpp": enhancement_hard["full_bits"] / points,
            "enhancement_total_hard_bpp": (
                enhancement_hard["full_bits"] / points),
            "native_total_hard_bits": native["total_bits"],
            "base_hard_bits": enhancement_hard["base_bits"],
            "base_hard_bpp": enhancement_hard["base_bits"] / points,
            "step0_total_hard_bits": enhancement_hard["full_bits"],
            "enhancement_total_hard_bits": enhancement_hard["full_bits"],
            "enh_deterministic_hard_max_abs": hard_difference,
            "enh_symbol_count": hard_symbols.numel(),
            "enh_symbol_nonzero_count": int(nonzero.sum().item()),
            "enh_symbol_nonzero_fraction": float(nonzero.float().mean().item()),
            "enh_symbol_min": float(hard_symbols.min().item()),
            "enh_symbol_max": float(hard_symbols.max().item()),
            "enh_active_channel_count": len(active_channels),
            "enh_active_channels": json.dumps(active_channels),
            "enh_per_channel_nonzero_fraction": json.dumps(
                nonzero.float().mean(dim=0).tolist()),
            "native_r5_symbol_count": native_symbols.numel(),
            "native_r5_symbol_nonzero_count": int(native_nonzero.sum().item()),
            "native_r5_symbol_nonzero_fraction": float(
                native_nonzero.float().mean().item()),
            "native_r5_active_channel_count": len(native_active_channels),
            "native_r5_active_channels": json.dumps(native_active_channels),
            "native_r5_per_channel_nonzero_fraction": json.dumps(
                native_nonzero.float().mean(dim=0).tolist()),
        }
        bits = {
            "native_r5_noise": native_noise_bits,
            "native_r5_symbol": native_symbol_bits,
            "native_r5_hard": native["r5_bits"],
            "enh_noise": enh_noise_bits,
            "enh_symbol": enh_symbol_bits,
            "enh_hard": enh_hard_bits,
        }
        for name, value in bits.items():
            if not np.isfinite(value) or value <= 0:
                raise RuntimeError(name + " bits are invalid")
            values[name + "_bits"] = value
            values[name + "_bpp"] = value / points
        for family in FAMILIES:
            values[family + "_symbol_over_hard"] = (
                values[family + "_symbol_bits"]
                / values[family + "_hard_bits"])
            values[family + "_noise_over_hard"] = (
                values[family + "_noise_bits"]
                / values[family + "_hard_bits"])
            values[family + "_symbol_minus_hard_bpp"] = (
                values[family + "_symbol_bpp"]
                - values[family + "_hard_bpp"])
        rows.append(values)
        write_csv(csv_path, rows)
        likelihood_mapping = noise_mapping
        hard_stream_mapping = native["stream_strides"]
        print("E0-C {}/{} {}".format(
            index + 1, len(dataset), entries[index]), flush=True)

    summary = aggregate(rows)
    summary.update({
        "status": "PASS",
        "conditioning_lambda": args.conditioning_lambda,
        "seed": args.seed,
        "native_returned_likelihood_to_invocation": likelihood_mapping,
        "native_returned_likelihood_index_0": "r5",
        "native_hard_stream_strides": hard_stream_mapping,
        "native_hard_stream_index_4": "r5",
        "enhancement_deterministic_hard_max_abs": max(
            row["enh_deterministic_hard_max_abs"] for row in rows),
        "optimizer_steps": 0,
        "parameters_updated": bool(args.enhancement_checkpoint),
        "min_max_excluded_from_rate": True,
        "enhancement_checkpoint": args.enhancement_checkpoint,
        "checkpoint_step": checkpoint_step,
        "runtime_seconds": time.monotonic() - started,
    })
    summary["model_equal"]["enhancement_full_psnr"] = summary[
        "model_equal"]["step0_full_psnr"]
    summary["model_equal"]["enhancement_total_hard_bpp"] = summary[
        "model_equal"]["step0_total_hard_bpp"]
    summary["point_weighted"]["enhancement_total_hard_bpp"] = summary[
        "point_weighted"]["step0_total_hard_bpp"]
    summary["optimizer_steps"] = checkpoint_step
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("CANONICAL ENHANCEMENT E0-C PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
