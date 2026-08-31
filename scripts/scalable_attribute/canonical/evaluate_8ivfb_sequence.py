#!/usr/bin/env python3
"""External 8iVFB physical RD with one shared canonical hard prefix."""

import argparse
import csv
import json
import os
import shlex
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_ply_ascii, write_ply_ascii
from scalable_attribute.canonical.base_synthesis import BaseSynthesis
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.enhancement import EnhancementVAE
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.operating_points import (
    DEFAULT_CONFIG, point_for_lambda, resolve_operating_point)
from scalable_attribute.canonical.scalable_model import load_frozen_base
from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS
from scalable_attribute.unicorn_reference import ReleasedUnicornAttribute
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sparse_max_difference)


FIELDS = (
    "sequence", "frame", "endpoint", "source", "checkpoint_profile",
    "lambda", "checkpoint_step", "points", "physical_bits", "physical_bpp",
    "base_bits", "enhancement_bits", "x_low_bits", "r1_bits", "r2_bits",
    "r3_bits", "r4_bits", "num_base_residual_streams",
    "num_native_r5_streams", "y_mse", "u_mse", "v_mse", "y_psnr",
    "u_psnr", "v_psnr", "yuv_psnr_611",
    "soft_hard_max_abs_difference", "hard_roundtrip_max_abs_difference",
    "seconds",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--point")
    parser.add_argument("--operating-points-config", default=DEFAULT_CONFIG)
    parser.add_argument("--released-checkpoint-root", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument(
        "--base-candidate", action="append", default=[],
        metavar="LABEL=PATH",
        help="Additional BaseSynthesis checkpoint; may be repeated")
    parser.add_argument("--enhancement-step1763")
    parser.add_argument("--enhancement-step3525")
    parser.add_argument(
        "--enhancement-checkpoint", action="append", default=[],
        metavar="LABEL=PATH",
        help="Independent EnhancementVAE checkpoint; may be repeated")
    parser.add_argument(
        "--scalable-checkpoint", action="append", default=[],
        metavar="LABEL=PATH",
        help="Complete fine-tuned scalable checkpoint; may be repeated")
    parser.add_argument("--gpcc-binary", required=True)
    parser.add_argument("--conditioning-lambda", type=int)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-rate-ids", nargs="+")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_step(path):
    state = torch.load(path, map_location="cpu")
    if state.get("architecture") != "canonical_independent_enhancement":
        raise ValueError("Enhancement checkpoint architecture mismatch")
    return state, int(state["step"])


def labeled_checkpoint(value):
    if "=" not in value:
        raise ValueError("checkpoint argument must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError("checkpoint argument must be LABEL=PATH")
    return label, os.path.abspath(os.path.expandvars(path))


def main():
    args = parse_args()
    for name in (
            "input_ply", "released_checkpoint_root",
            "base_synthesis_checkpoint",
            "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    operating_point = None
    if args.point:
        operating_point = resolve_operating_point(
            args.point, args.released_checkpoint_root,
            args.operating_points_config)
        if (args.conditioning_lambda is not None and
                args.conditioning_lambda != operating_point["conditioning_lambda"]):
            raise ValueError("--conditioning-lambda conflicts with --point")
        args.conditioning_lambda = operating_point["conditioning_lambda"]
        canonical_profile = operating_point["released_profile"]
    else:
        if args.conditioning_lambda is None:
            args.conditioning_lambda = 32768
        _, configured = point_for_lambda(
            args.conditioning_lambda, config_path=args.operating_points_config)
        canonical_profile = configured["released_profile"]
    for name in ("enhancement_step1763", "enhancement_step3525"):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(os.path.expandvars(value)))
    enhancement_checkpoints = [
        labeled_checkpoint(value) for value in args.enhancement_checkpoint]
    base_candidates = [
        labeled_checkpoint(value) for value in args.base_candidate]
    scalable_checkpoints = [
        labeled_checkpoint(value) for value in args.scalable_checkpoint]
    if not (args.enhancement_step1763 or args.enhancement_step3525 or
            base_candidates or enhancement_checkpoints or scalable_checkpoints or
            args.run_official):
        raise ValueError("No released/canonical/scalable endpoint requested")
    output_csv = os.path.join(args.output_dir, "physical_rd.csv")
    if os.path.exists(output_csv):
        raise FileExistsError("Refusing to overwrite " + output_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w",
              encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    if operating_point is not None:
        with open(os.path.join(args.output_dir, "operating_point.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(operating_point, handle, indent=2)
    with open(os.path.join(args.output_dir, "command.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if not os.path.exists(gpcc_link):
        os.symlink(args.gpcc_binary, gpcc_link)
    os.chdir(args.output_dir)

    coords, rgb = read_ply_ascii(args.input_ply)
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    batch_coords, batch_feats = ME.utils.sparse_collate([coords], [yuv])
    attribute = ME.SparseTensor(
        features=batch_feats, coordinates=batch_coords,
        tensor_stride=1, device="cuda")
    gt_path = os.path.join(args.output_dir, "metric_gt.ply")
    rec_path = os.path.join(args.output_dir, "metric_rec.ply")
    write_ply_ascii(gt_path, coords, rgb)
    rows = []

    def record(endpoint, reconstruction, bits, source, started, **extra):
        write_ply_ascii(
            rec_path, reconstruction.C[:, 1:].detach().cpu().numpy(),
            reconstruction_rgb(reconstruction))
        quality = metric(gt_path, rec_path)
        row = {key: "" for key in FIELDS}
        row.update({
            "sequence": args.sequence,
            "frame": args.frame,
            "endpoint": endpoint,
            "source": source,
            "points": len(attribute),
            "physical_bits": int(bits),
            "physical_bpp": int(bits) / len(attribute),
            **quality,
            "seconds": time.perf_counter() - started,
            **extra,
        })
        rows.append(row)
        write_rows(output_csv, rows)
        print("{} bpp={:.6f} YUV611={:.4f}".format(
            endpoint, row["physical_bpp"], row["yuv_psnr_611"]), flush=True)

    if args.run_official:
        for rate_id, profile, lmb in OFFICIAL_RWTT_REFERENCE_POINTS:
            if (args.official_rate_ids is not None and
                    rate_id not in args.official_rate_ids):
                continue
            started = time.perf_counter()
            checkpoint = os.path.join(
                args.released_checkpoint_root, profile, "epoch_last.pth")
            released = ReleasedUnicornAttribute(checkpoint).cuda().eval()
            reconstruction, bits = released.hard_reconstruct(attribute, lmb)
            record(rate_id, reconstruction, bits, "OFFICIAL_RELEASED", started,
                   checkpoint_profile=profile, **{"lambda": lmb},
                   num_native_r5_streams=1)
            del reconstruction, released
            torch.cuda.empty_cache()

    released_checkpoint = os.path.join(
        args.released_checkpoint_root, canonical_profile, "epoch_last.pth")
    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base = CanonicalBaseModel(
        released_checkpoint, BaseSynthesisConfig(**base_state["config"])).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, released_checkpoint,
        args.conditioning_lambda)

    # One deterministic prefix supports soft/hard equivalence; the one hard
    # invocation supplies the shared physical Base stream for every candidate.
    soft_prefix_state = base.prefix(attribute, args.conditioning_lambda)
    prefix_state, prefix_rate = base.prefix.hard_forward(
        attribute, args.conditioning_lambda, return_details=True)
    if prefix_rate["num_residual_streams"] != 4:
        raise RuntimeError("Canonical prefix must contain exactly r1-r4")
    residual_bits = prefix_rate["residual_bits"]
    if len(residual_bits) != 4:
        raise RuntimeError("Canonical prefix bit breakdown is not r1-r4")
    expected_base_bits = int(prefix_rate["bits_xlow"] + sum(residual_bits))
    if int(prefix_rate["base_bits"]) != expected_base_bits:
        raise RuntimeError("Canonical Base physical bit identity failed")
    base_output = base.reconstruct_from_state(prefix_state)
    soft_base_output = base.reconstruct_from_state(soft_prefix_state)
    base_difference = sparse_max_difference(
        soft_base_output["Base"], base_output["Base"], "Canonical Base soft/hard")
    native = base.native_baselines(prefix_state)["B_native"]
    common = {
        "checkpoint_profile": canonical_profile,
        "lambda": args.conditioning_lambda,
        "base_bits": prefix_rate["base_bits"],
        "x_low_bits": prefix_rate["bits_xlow"],
        "r1_bits": residual_bits[0], "r2_bits": residual_bits[1],
        "r3_bits": residual_bits[2], "r4_bits": residual_bits[3],
        "num_base_residual_streams": 4,
        "num_native_r5_streams": 0,
    }
    started = time.perf_counter()
    record("B_native", native, prefix_rate["base_bits"],
           "CURRENT_CANONICAL_DIAGNOSTIC", started,
           checkpoint_step=0, enhancement_bits=0, **common)
    record("Canonical_Base", base_output["Base"], prefix_rate["base_bits"],
           "CURRENT_CANONICAL", started,
           checkpoint_step=int(base_state["step"]), enhancement_bits=0,
           soft_hard_max_abs_difference=base_difference, **common)

    for label, path in base_candidates:
        started = time.perf_counter()
        state = torch.load(path, map_location="cpu")
        if state.get("architecture") != "canonical_base_predict_correct":
            raise ValueError(label + " Base checkpoint architecture mismatch")
        if int(state.get("base_lambda", -1)) != args.conditioning_lambda:
            raise ValueError(label + " Base lambda mismatch")
        if os.path.realpath(state.get("base_checkpoint", "")) != os.path.realpath(
                released_checkpoint):
            raise ValueError(label + " released checkpoint mismatch")
        synthesis = BaseSynthesis(
            BaseSynthesisConfig(**state["config"])).cuda().eval()
        synthesis.load_state_dict(state["base_synthesis"], strict=True)
        synthesis.requires_grad_(False)

        def reconstruct(state_value):
            compensation = synthesis(state_value)
            _, correction = base.prefix.synthesize(
                state_value.f5p, compensation)
            return state_value.x5p + correction

        candidate_output = reconstruct(prefix_state)
        candidate_soft = reconstruct(soft_prefix_state)
        difference = sparse_max_difference(
            candidate_soft, candidate_output,
            label + " Base soft/hard")
        record(label, candidate_output, prefix_rate["base_bits"],
               "CURRENT_CANONICAL_BASE_ABLATION", started,
               checkpoint_step=int(state["step"]), enhancement_bits=0,
               soft_hard_max_abs_difference=difference, **common)
        del synthesis, candidate_output, candidate_soft
        torch.cuda.empty_cache()

    embedding = base.prefix.lambda_embedding(
        args.conditioning_lambda, attribute.device)
    enhancement_paths = []
    if args.enhancement_step1763:
        enhancement_paths.append((None, args.enhancement_step1763))
    if args.enhancement_step3525:
        enhancement_paths.append((None, args.enhancement_step3525))
    enhancement_paths.extend(enhancement_checkpoints)
    for label, path in enhancement_paths:
        started = time.perf_counter()
        state, step = checkpoint_step(path)
        endpoint = label or "Canonical_Full_step{}".format(step)
        if int(state["conditioning_lambda"]) != args.conditioning_lambda:
            raise ValueError("Enhancement conditioning lambda mismatch")
        enhancement = EnhancementVAE(base.prefix.model.VAE).cuda().eval()
        enhancement.vae.load_state_dict(state["enhancement_vae"], strict=True)
        enhancement.requires_grad_(False)
        encoded = enhancement.encode(
            base_output["Base"], attribute, base_output["F_B"],
            base_output["d5p"], embedding)
        payload = {key: encoded[key] for key in ("strings", "min_v", "max_v")}
        decoded = enhancement.decode(
            payload, base_output["Base"], base_output["F_B"],
            base_output["d5p"], embedding)
        difference = sparse_max_difference(
            encoded["x_out"], decoded["x_out"], endpoint + " hard")
        if difference != 0.0:
            raise RuntimeError(endpoint + " hard round-trip mismatch")
        enhancement_bits = len(payload["strings"]) * 8
        record(endpoint, decoded["x_out"],
               prefix_rate["base_bits"] + enhancement_bits,
               "CURRENT_CANONICAL", started, checkpoint_step=step,
               enhancement_bits=enhancement_bits,
               hard_roundtrip_max_abs_difference=difference, **common)
        del enhancement, encoded, decoded
        torch.cuda.empty_cache()

    for label, path in scalable_checkpoints:
        from scalable_attribute.canonical.scalable_model import (
            CanonicalScalableModel, load_finetuned_scalable)

        started = time.perf_counter()
        candidate_base = CanonicalBaseModel(
            released_checkpoint,
            BaseSynthesisConfig(**base_state["config"])).cuda()
        load_frozen_base(
            candidate_base, args.base_synthesis_checkpoint,
            released_checkpoint,
            args.conditioning_lambda)
        candidate = CanonicalScalableModel(
            candidate_base, args.conditioning_lambda).cuda().eval()
        state = load_finetuned_scalable(
            candidate, path, args.conditioning_lambda)
        candidate.requires_grad_(False)
        hard = candidate.hard_reconstruct(attribute)
        if hard["full_bits"] != hard["base_bits"] + hard["enhancement_bits"]:
            raise RuntimeError(label + " Full bit identity failed")
        difference = sparse_max_difference(
            hard["encoded_Full"], hard["Full"], label + " hard")
        if difference != 0.0:
            raise RuntimeError(label + " hard round-trip mismatch")
        rate = hard["prefix_rate"]
        residual_bits = rate["residual_bits"]
        if rate["num_residual_streams"] != 4 or len(residual_bits) != 4:
            raise RuntimeError(label + " Base is not exactly r1-r4")
        candidate_common = {
            "checkpoint_profile": canonical_profile,
            "lambda": args.conditioning_lambda,
            "checkpoint_step": int(state["step"]),
            "base_bits": hard["base_bits"],
            "x_low_bits": rate["bits_xlow"],
            "r1_bits": residual_bits[0], "r2_bits": residual_bits[1],
            "r3_bits": residual_bits[2], "r4_bits": residual_bits[3],
            "num_base_residual_streams": 4,
            "num_native_r5_streams": 0,
        }
        record(label + "_Base", hard["Base"], hard["base_bits"],
               "CURRENT_MVUB_FINETUNED", started, enhancement_bits=0,
               **candidate_common)
        record(label + "_Full", hard["Full"], hard["full_bits"],
               "CURRENT_MVUB_FINETUNED", started,
               enhancement_bits=hard["enhancement_bits"],
               hard_roundtrip_max_abs_difference=difference,
               **candidate_common)
        del candidate, candidate_base, hard
        torch.cuda.empty_cache()

    summary = {
        "status": "PASS",
        "sequence": args.sequence,
        "frame": args.frame,
        "canonical_prefix_hard_invocations": 1,
        "canonical_endpoints_share_prefix_state": True,
        "num_base_residual_streams": 4,
        "num_native_r5_streams": 0,
        "base_soft_hard_difference_is_diagnostic_only": True,
        "full_hard_roundtrip_required_exact": True,
        "rows": rows,
    }
    with open(os.path.join(args.output_dir, "physical_rd.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
