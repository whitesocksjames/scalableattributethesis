#!/usr/bin/env python3
"""External 8iVFB physical RD with one shared canonical hard prefix."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
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
from scalable_attribute.runtime.operating_points import (
    DEFAULT_CONFIG, point_for_lambda, resolve_operating_point)
from scalable_attribute.canonical.scalable_model import load_frozen_base
from scalable_attribute.reference_points import OFFICIAL_RWTT_REFERENCE_POINTS
from scalable_attribute.unicorn_reference import ReleasedUnicornAttribute
from scripts.scalable_attribute.evaluation.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sparse_max_difference)


FIELDS = (
    "sequence", "frame", "endpoint", "source", "checkpoint_profile",
    "lambda", "checkpoint_step", "points", "physical_bits", "physical_bpp",
    "base_bits", "enhancement_bits", "x_low_bits", "r1_bits", "r2_bits",
    "r3_bits", "r4_bits", "num_base_residual_streams",
    "num_native_r5_streams", "y_mse", "u_mse", "v_mse", "y_psnr",
    "u_psnr", "v_psnr", "yuv_psnr_611",
    "soft_hard_max_abs_difference", "hard_roundtrip_max_abs_difference",
    "reconstruction_ply", "seconds",
)


# Formal Ours evidence deliberately has no generic ``seconds`` field.  The
# old schema above remains available to callers of the legacy CLI.
FORMAL_FIELDS = (
    "sequence", "frame", "operating_point", "endpoint", "source",
    "checkpoint_profile", "lambda", "checkpoint_step", "endpoint_status",
    "task_status", "gate_failures", "gates", "points", "physical_bits",
    "physical_bpp", "base_bits", "enhancement_bits", "full_bits",
    "x_low_bits", "r1_bits", "r2_bits", "r3_bits", "r4_bits",
    "num_base_residual_streams", "num_native_r5_streams", "y_mse",
    "u_mse", "v_mse", "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_611",
    "hard_roundtrip_max_abs_difference", "base_encode_feature_max_abs_difference",
    "base_encode_coordinate_max_abs_difference",
    "base_decode_feature_max_abs_difference",
    "base_decode_coordinate_max_abs_difference", "reconstruction_ply",
    "reconstruction_sha256", "reconstruction_bytes", "prefix_encode_seconds",
    "prefix_decode_seconds",
    "base_synthesis_seconds", "enhancement_encode_seconds",
    "enhancement_decode_seconds", "base_codec_seconds", "full_codec_seconds",
    "metric_seconds", "base_metric_seconds", "full_metric_seconds",
    "io_seconds", "base_io_seconds", "full_io_seconds",
    "model_load_seconds", "task_wall_seconds", "peak_vram_bytes",
)

FORMAL_REUSABLE = "FORMAL_REUSABLE"
ENDPOINT_FAILED = "FAILED"
ENDPOINT_NOT_ATTEMPTED = "NOT_ATTEMPTED"
TASK_PASS = "PASS"
TASK_PARTIAL = "PARTIAL"
TASK_FAILED = "FAIL"
SCRIPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
FORMAL_CHECKPOINT_MANIFEST = os.path.join(
    SCRIPT_ROOT, "configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json")


class _FormalStopAfterBase(RuntimeError):
    """Internal signal used only by the isolated-Base preflight mode."""


def derive_codec_timings(prefix_encode_seconds, prefix_decode_seconds,
                         base_synthesis_seconds,
                         enhancement_encode_seconds=0.0,
                         enhancement_decode_seconds=0.0):
    """Derive endpoint runtimes from CUDA-synchronised component durations."""
    values = {
        "prefix_encode_seconds": float(prefix_encode_seconds),
        "prefix_decode_seconds": float(prefix_decode_seconds),
        "base_synthesis_seconds": float(base_synthesis_seconds),
        "enhancement_encode_seconds": float(enhancement_encode_seconds),
        "enhancement_decode_seconds": float(enhancement_decode_seconds),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(name + " must be finite and non-negative")
    values["base_codec_seconds"] = (
        values["prefix_encode_seconds"] + values["prefix_decode_seconds"] +
        values["base_synthesis_seconds"])
    values["full_codec_seconds"] = (
        values["base_codec_seconds"] +
        values["enhancement_encode_seconds"] +
        values["enhancement_decode_seconds"])
    return values


def endpoint_status(gates):
    """Return FORMAL_REUSABLE only when every endpoint gate is true."""
    if not isinstance(gates, dict) or not gates:
        raise ValueError("endpoint gates must be a non-empty mapping")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("endpoint gates must contain booleans")
    return FORMAL_REUSABLE if all(gates.values()) else ENDPOINT_FAILED


def task_status(base_status, full_status):
    """Keep a valid Base reusable when a later Full attempt fails."""
    if base_status == FORMAL_REUSABLE and full_status == FORMAL_REUSABLE:
        return TASK_PASS
    if base_status == FORMAL_REUSABLE:
        return TASK_PARTIAL
    return TASK_FAILED


def validate_base_rate_details(rate):
    """Validate x_low+r1+r2+r3+r4 and return normalized components."""
    if not isinstance(rate, dict):
        raise ValueError("Base rate details must be a mapping")
    try:
        stream_count = int(rate["num_residual_streams"])
        residual_bits = [int(value) for value in rate["residual_bits"]]
        x_low_bits = int(rate["bits_xlow"])
        base_bits = int(rate["base_bits"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Base rate details are incomplete") from error
    if stream_count != 4 or len(residual_bits) != 4:
        raise ValueError("Base must contain exactly r1-r4")
    if x_low_bits < 0 or any(value < 0 for value in residual_bits):
        raise ValueError("Base bit components must be non-negative")
    if int(rate.get("num_native_r5_streams", 0)) != 0:
        raise ValueError("Base must not contain a native r5 stream")
    expected = x_low_bits + sum(residual_bits)
    if base_bits != expected:
        raise ValueError("Base bits identity failed")
    return {"base_bits": base_bits, "x_low_bits": x_low_bits,
            "residual_bits": residual_bits, "num_base_residual_streams": 4,
            "num_native_r5_streams": 0}


def validate_full_bit_identity(base_bits, enhancement_bits, full_bits):
    base_bits, enhancement_bits, full_bits = map(
        int, (base_bits, enhancement_bits, full_bits))
    if min(base_bits, enhancement_bits, full_bits) < 0:
        raise ValueError("Full bit components must be non-negative")
    if full_bits != base_bits + enhancement_bits:
        raise ValueError("Full bits identity failed")
    return {"base_bits": base_bits, "enhancement_bits": enhancement_bits,
            "full_bits": full_bits}


def validate_independent_enhancement_metadata(
        state, base_synthesis_checkpoint, released_checkpoint,
        conditioning_lambda, frozen_lineage=None):
    """Require an Enhancement checkpoint to name the passed frozen lineage."""
    if not isinstance(state, dict):
        raise ValueError("Enhancement checkpoint is not a mapping")
    if state.get("architecture") != "canonical_independent_enhancement":
        raise ValueError("Enhancement checkpoint architecture mismatch")
    for key, expected in (
            ("base_synthesis_checkpoint", base_synthesis_checkpoint),
            ("released_checkpoint", released_checkpoint)):
        actual = state.get(key)
        if not isinstance(actual, str) or not actual:
            raise ValueError("Enhancement checkpoint is missing " + key)
        expected_identity = (expected if frozen_lineage is None
                             else frozen_lineage[key])
        normalize = os.path.realpath if frozen_lineage is None else os.path.normpath
        if normalize(actual) != normalize(expected_identity):
            raise ValueError("Enhancement " + key + " mismatch")
    if int(state.get("conditioning_lambda", -1)) != int(conditioning_lambda):
        raise ValueError("Enhancement conditioning lambda mismatch")
    return True


def validate_joint_checkpoint_metadata(
        state, base_synthesis_checkpoint, released_checkpoint,
        conditioning_lambda, checkpoint_profile, frozen_lineage=None):
    """Require the complete joint checkpoint's target/bootstrap lineage."""
    if not isinstance(state, dict):
        raise ValueError("Joint checkpoint is not a mapping")
    if state.get("architecture") != "canonical_scalable_mvub_finetune_v1":
        raise ValueError("Joint checkpoint architecture mismatch")
    normalize = os.path.realpath if frozen_lineage is None else os.path.normpath
    expected_base = (base_synthesis_checkpoint if frozen_lineage is None else
                     frozen_lineage["base_synthesis_initialization"])
    expected_released = (released_checkpoint if frozen_lineage is None else
                         frozen_lineage["released_checkpoint"])
    if normalize(state.get("base_synthesis_initialization", "")) != \
            normalize(expected_base):
        raise ValueError("Joint base synthesis initialization mismatch")
    if normalize(state.get("released_checkpoint", "")) != \
            normalize(expected_released):
        raise ValueError("Joint released checkpoint mismatch")
    if int(state.get("conditioning_lambda", -1)) != int(conditioning_lambda):
        raise ValueError("Joint conditioning lambda mismatch")
    actual_profile = state.get("checkpoint_profile")
    if actual_profile is None:
        actual_profile = state.get("resolved_args", {}).get("checkpoint_profile")
    if actual_profile != checkpoint_profile:
        raise ValueError("Joint checkpoint profile mismatch")
    return True


def validate_relocated_artifact(path, descriptor, label="checkpoint"):
    """Match a local or relocated artifact to its frozen content identity."""
    if not isinstance(descriptor, dict):
        raise ValueError(label + " frozen descriptor is not a mapping")
    try:
        expected_size = int(descriptor["size_bytes"])
        expected_sha256 = descriptor["sha256"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(label + " frozen descriptor is incomplete") from error
    if not os.path.isfile(path):
        raise FileNotFoundError(label + " not found: " + path)
    if os.path.getsize(path) != expected_size:
        raise ValueError(label + " size mismatch")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(label + " SHA-256 mismatch")
    return True


def frozen_artifact_lineage(origin_root, descriptor, label="checkpoint"):
    """Return the exact lexical lineage identity declared by the frozen manifest."""
    if not isinstance(origin_root, str) or not os.path.isabs(origin_root):
        raise ValueError("frozen origin_root must be absolute")
    relative = descriptor.get("path") if isinstance(descriptor, dict) else None
    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise ValueError(label + " frozen path must be relative")
    normalized = os.path.normpath(relative)
    if normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError(label + " frozen path escapes origin_root")
    return os.path.normpath(os.path.join(origin_root, normalized))


def _formal_checkpoint_contract(point, manifest_path=FORMAL_CHECKPOINT_MANIFEST):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("contract_id") != "formal_static_rgb_v1_20260909":
        raise ValueError("Frozen checkpoint contract ID mismatch")
    matches = [item for item in manifest.get("operating_points", [])
               if item.get("point") == point]
    if len(matches) != 1:
        raise ValueError("Frozen checkpoint point mapping is not unique: " + str(point))
    return manifest, matches[0]


def _validate_formal_checkpoint_selection(
        point, mode, canonical_profile, conditioning_lambda,
        selected_base, enhancement_path, released_checkpoint):
    manifest, operation = _formal_checkpoint_contract(point)
    if int(operation.get("lambda", -1)) != int(conditioning_lambda):
        raise ValueError("Frozen checkpoint lambda mismatch")
    if operation.get("released_profile") != canonical_profile:
        raise ValueError("Frozen checkpoint profile mismatch")
    if mode == "joint":
        if (operation.get("base_loader") != "joint_scalable_base" or
                operation.get("full_loader") != "joint_scalable_full"):
            raise ValueError("Frozen joint loader mapping mismatch")
        selected_base_descriptor = operation.get("bootstrap_checkpoint")
    else:
        if operation.get("base_loader") not in ("base_synthesis", "rescued_base"):
            raise ValueError("Frozen Base loader mapping mismatch")
        if operation.get("full_loader") not in (
                "independent_enhancement", "sequential_enhancement"):
            raise ValueError("Frozen Enhancement loader mapping mismatch")
        selected_base_descriptor = operation.get("base_checkpoint")
    released_descriptor = manifest["released_checkpoints"][canonical_profile]
    full_descriptor = operation.get("full_checkpoint")
    validate_relocated_artifact(selected_base, selected_base_descriptor,
                                "selected Base checkpoint")
    validate_relocated_artifact(enhancement_path, full_descriptor,
                                "selected Full checkpoint")
    validate_relocated_artifact(released_checkpoint, released_descriptor,
                                "selected released checkpoint")
    origin_root = manifest["origin_root"]
    return {
        "base": frozen_artifact_lineage(
            origin_root, selected_base_descriptor, "selected Base checkpoint"),
        "released": frozen_artifact_lineage(
            origin_root, released_descriptor, "selected released checkpoint"),
    }


def derive_endpoint_status(gates):
    return endpoint_status(gates)


def derive_task_status(base_status, full_status):
    return task_status(base_status, full_status)


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
    parser.add_argument("--checkpoint-profile")
    parser.add_argument("--base-checkpoint-lambda", type=int)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-rate-ids", nargs="+")
    parser.add_argument(
        "--ours-formal", "--formal", dest="ours_formal", action="store_true",
        help="Run one combined Ours Base then Full formal task")
    parser.add_argument(
        "--formal-stop-after-base", action="store_true",
        help="Preflight only: publish Base, mark Full not attempted, and stop")
    parser.add_argument("--sample-id")
    parser.add_argument("--chunk-order", type=int)
    parser.add_argument("--chunk-path")
    parser.add_argument("--chunk-sha256")
    parser.add_argument("--chunk-count", type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _atomic_text(path, value, refuse_existing=False):
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path)) or os.curdir
    os.makedirs(parent, exist_ok=True)
    if refuse_existing and os.path.lexists(path):
        raise FileExistsError("Refusing to overwrite " + path)
    fd, temporary = tempfile.mkstemp(dir=parent, prefix=".formal-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_rows(path, rows, fields=FIELDS, atomic=False):
    if atomic:
        fd, temporary = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(path)) or os.curdir,
            prefix=".rows-")
        os.close(fd)
        try:
            with open(temporary, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def _formal_select_mode(base_candidates, enhancement_paths, scalable_paths,
                        run_official):
    if run_official:
        raise ValueError("--ours-formal cannot run official endpoints")
    if len(base_candidates) > 1:
        raise ValueError("--ours-formal accepts at most one Base checkpoint")
    if len(enhancement_paths) + len(scalable_paths) != 1:
        raise ValueError(
            "--ours-formal requires exactly one Enhancement or joint checkpoint")
    if enhancement_paths:
        return "independent"
    if base_candidates:
        raise ValueError(
            "joint formal mode uses --base-synthesis-checkpoint as bootstrap")
    return "joint"


def _formal_timed(timings, name, function, *args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        torch.cuda.synchronize()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def _formal_sparse_snapshot(value):
    """Keep exact validation state on CPU, outside codec timing/GPU residency."""
    return (value.C.detach().cpu().clone(), value.F.detach().cpu().clone(),
            tuple(value.tensor_stride))


def _formal_snapshot_difference(snapshot, value):
    coordinates, features, stride = snapshot
    if tuple(value.tensor_stride) != stride:
        return float("inf"), float("inf")
    if value.C.shape != coordinates.shape or value.F.shape != features.shape:
        return float("inf"), float("inf")
    current_coordinates = value.C.detach().cpu()
    current_features = value.F.detach().cpu()
    c_diff = float((current_coordinates - coordinates).abs().max().item()) \
        if current_coordinates.numel() else 0.0
    f_diff = float((current_features - features).abs().max().item()) \
        if current_features.numel() else 0.0
    return f_diff, c_diff


def _formal_snapshot_max_difference(snapshot, value, label):
    """Apply the former sparse exact-roundtrip gate to a CPU snapshot."""
    coordinates, features, stride = snapshot
    if tuple(value.tensor_stride) != stride:
        raise RuntimeError(label + " tensor strides differ")
    if value.C.shape != coordinates.shape or value.F.shape != features.shape:
        raise RuntimeError(label + " tensor shapes differ")
    current_coordinates = value.C.detach().cpu()
    current_features = value.F.detach().cpu()
    if not torch.equal(coordinates, current_coordinates):
        raise RuntimeError(label + " coordinates differ")
    return float((features - current_features).abs().max().item()) \
        if current_features.numel() else 0.0


def _formal_streamed_hard_reconstruct(model, attribute):
    """Run one hard Base+Full path without retaining encoded validation tensors.

    CPU validation copies happen after Enhancement encode timing and before
    Enhancement decode timing.  Codec payload, model operations, bit accounting,
    and decoded endpoint semantics are unchanged.
    """
    base = model.base_forward(attribute, hard=True)
    embedding = model._embedding(attribute.device)  # pylint: disable=protected-access
    encoded = model.enhancement.encode(
        base["Base"], attribute, base["F_B"], base["d5p"], embedding)
    encoded_full_snapshot = _formal_sparse_snapshot(encoded["x_out"])
    payload = {
        "strings": encoded["strings"],
        "min_v": encoded["min_v"],
        "max_v": encoded["max_v"],
    }
    del encoded
    torch.cuda.empty_cache()
    decoded = model.enhancement.decode(
        payload, base["Base"], base["F_B"], base["d5p"], embedding)
    result = model._result(base, decoded)  # pylint: disable=protected-access
    enhancement_bits = int(len(payload["strings"]) * 8)
    result.update({
        "enhancement_payload": payload,
        "enhancement_bits": enhancement_bits,
        "base_bits": base["prefix_rate"]["base_bits"],
        "full_bits": base["prefix_rate"]["base_bits"] + enhancement_bits,
        "prefix_rate": base["prefix_rate"],
        "encoded_Full_snapshot": encoded_full_snapshot,
    })
    return result


def _formal_hard_once(model, attribute, timings, on_base, identity,
                      stop_after_base=False):
    """Run hard reconstruction exactly once while timing its four components."""
    originals = []

    def wrap(owner, name, component, callback=None):
        original = getattr(owner, name)

        def replacement(*args, **kwargs):
            if callback:
                callback("before", args, kwargs)
            result = _formal_timed(
                timings, component, original, *args, **kwargs)
            if callback:
                callback("after", args, kwargs)
            return result

        setattr(owner, name, replacement)
        originals.append((owner, name, original))

    def encode_callback(phase, args, kwargs):
        supplied = args[0] if args else kwargs.get("base")
        if supplied is None or "snapshot" not in identity:
            return
        if phase == "before":
            identity["encode_before"] = _formal_snapshot_difference(
                identity["snapshot"], supplied)
        else:
            identity["encode_after"] = _formal_snapshot_difference(
                identity["snapshot"], supplied)

    def decode_callback(phase, args, kwargs):
        supplied = args[1] if len(args) > 1 else kwargs.get("base")
        if supplied is None or "snapshot" not in identity:
            return
        if phase == "before":
            identity["decode_before"] = _formal_snapshot_difference(
                identity["snapshot"], supplied)
        else:
            identity["decode_after"] = _formal_snapshot_difference(
                identity["snapshot"], supplied)

    # The released Prefix exposes physical encode+decode as one hard_forward.
    # Time its entropy/G-PCC encode call directly, then assign all remaining
    # hard-prefix work (entropy decode plus state completion) to decode.  This
    # keeps Base runtime exhaustive without double-counting nested timers.
    wrap(model.base.prefix.model, "forward", "prefix_encode_seconds")
    original_hard_forward = model.base.prefix.hard_forward

    def hard_forward(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return original_hard_forward(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            total = time.perf_counter() - started
            encoded = timings.get("prefix_encode_seconds", 0.0)
            timings["prefix_decode_seconds"] = (
                timings.get("prefix_decode_seconds", 0.0) +
                max(0.0, total - encoded))

    model.base.prefix.hard_forward = hard_forward
    originals.append((model.base.prefix, "hard_forward", original_hard_forward))
    wrap(model.base, "reconstruct_from_state", "base_synthesis_seconds")
    wrap(model.enhancement, "encode", "enhancement_encode_seconds",
         encode_callback)
    wrap(model.enhancement, "decode", "enhancement_decode_seconds",
         decode_callback)
    original_base_forward = model.base_forward

    def base_forward(attribute_value, hard=False):
        result = original_base_forward(attribute_value, hard=hard)
        if hard:
            identity["snapshot"] = _formal_sparse_snapshot(result["Base"])
            on_base(result)
            if stop_after_base:
                raise _FormalStopAfterBase(
                    "isolated Base-only preflight completed")
        return result

    model.base_forward = base_forward
    try:
        return _formal_streamed_hard_reconstruct(model, attribute)
    finally:
        del model.base_forward
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _formal_write_ply(path, reconstruction):
    temporary = path + ".tmp"
    write_ply_ascii(
        temporary, reconstruction.C[:, 1:].detach().cpu().numpy(),
        reconstruction_rgb(reconstruction))
    os.replace(temporary, path)


def _safe_endpoint(endpoint):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", endpoint).strip("_")


def _formal_reconstruction_path(args, endpoint):
    path = os.path.join(
        args.output_dir, "reconstruction_{}.ply".format(_safe_endpoint(endpoint)))
    if os.path.lexists(path):
        raise FileExistsError("Refusing to overwrite " + path)
    return path


def _formal_base_row(args, endpoint, source, step, rate, status, task_state,
                     points, timings, model_load_seconds, gates, reason=""):
    row = {key: "" for key in FORMAL_FIELDS}
    row.update({
        "sequence": args.sequence, "frame": args.frame,
        "operating_point": args.point or "lambda_{}".format(args.conditioning_lambda),
        "endpoint": endpoint, "source": source,
        "checkpoint_profile": args.checkpoint_profile,
        "lambda": int(args.conditioning_lambda), "checkpoint_step": int(step),
        "endpoint_status": status, "task_status": task_state,
        "gate_failures": reason, "gates": json.dumps(gates, sort_keys=True),
        "points": int(points), "base_bits": int(rate["base_bits"]),
        "x_low_bits": int(rate["x_low_bits"]),
        "r1_bits": int(rate["residual_bits"][0]),
        "r2_bits": int(rate["residual_bits"][1]),
        "r3_bits": int(rate["residual_bits"][2]),
        "r4_bits": int(rate["residual_bits"][3]),
        "num_base_residual_streams": 4, "num_native_r5_streams": 0,
        "prefix_encode_seconds": timings.get("prefix_encode_seconds", 0.0),
        "prefix_decode_seconds": timings.get("prefix_decode_seconds", 0.0),
        "base_synthesis_seconds": timings.get("base_synthesis_seconds", 0.0),
        "enhancement_encode_seconds": timings.get("enhancement_encode_seconds", 0.0),
        "enhancement_decode_seconds": timings.get("enhancement_decode_seconds", 0.0),
        "model_load_seconds": float(model_load_seconds),
    })
    return row


def _formal_publish_success(
        args, output_csv, gt_path, rows, endpoint_json, endpoint, source,
        reconstruction, physical_bits, rate, enhancement_bits, full_bits,
        step, timings, model_load_seconds, points, gates,
        identity_differences=None, roundtrip=""):
    reconstruction_path = _formal_reconstruction_path(args, endpoint)
    io_started = time.perf_counter()
    _formal_write_ply(reconstruction_path, reconstruction)
    reconstruction_bytes = os.path.getsize(reconstruction_path)
    reconstruction_hash = _sha256_file(reconstruction_path)
    io_seconds = time.perf_counter() - io_started
    metric_started = time.perf_counter()
    quality = metric(gt_path, reconstruction_path)
    metric_seconds = time.perf_counter() - metric_started
    gates.update({
        "reconstruction_written_and_hashed": bool(reconstruction_hash),
        "metric_complete": all(key in quality for key in (
            "y_mse", "u_mse", "v_mse", "y_psnr", "u_psnr", "v_psnr",
            "yuv_psnr_611")),
        "physical_bpp_uses_endpoint_input_points": int(points) > 0,
    })
    if endpoint_status(gates) != FORMAL_REUSABLE:
        raise RuntimeError(endpoint + " publication gate failed")
    codec = derive_codec_timings(
        timings["prefix_encode_seconds"], timings["prefix_decode_seconds"],
        timings["base_synthesis_seconds"],
        timings["enhancement_encode_seconds"],
        timings["enhancement_decode_seconds"])
    is_full = full_bits != ""
    row = _formal_base_row(
        args, endpoint, source, step, rate, FORMAL_REUSABLE, "RUNNING",
        points, timings, model_load_seconds, gates)
    row.update(quality)
    row.update({
        "physical_bits": int(physical_bits),
        "physical_bpp": float(physical_bits) / int(points),
        "enhancement_bits": int(enhancement_bits) if is_full else 0,
        "full_bits": int(full_bits) if is_full else "",
        "reconstruction_ply": reconstruction_path,
        "reconstruction_sha256": reconstruction_hash,
        "reconstruction_bytes": int(reconstruction_bytes),
        "base_codec_seconds": codec["base_codec_seconds"],
        "full_codec_seconds": codec["full_codec_seconds"],
        "metric_seconds": metric_seconds,
        "base_metric_seconds": metric_seconds if not is_full else "",
        "full_metric_seconds": metric_seconds if is_full else "",
        "io_seconds": io_seconds,
        "base_io_seconds": io_seconds if not is_full else "",
        "full_io_seconds": io_seconds if is_full else "",
    })
    if roundtrip != "":
        row["hard_roundtrip_max_abs_difference"] = roundtrip
    if identity_differences:
        row.update(identity_differences)
    payload = {"status": FORMAL_REUSABLE, "endpoint": endpoint, "row": row,
               "gates": gates}
    _atomic_text(
        endpoint_json, json.dumps(payload, indent=2, sort_keys=True) + "\n",
        refuse_existing=True)
    rows.append(row)
    write_rows(output_csv, rows, fields=FORMAL_FIELDS, atomic=True)
    return row


def _formal_publish_failure(
        args, output_csv, rows, endpoint_json, endpoint, status, source,
        reason, rate, step, points, timings, model_load_seconds, gates,
        identity_differences=None):
    row = _formal_base_row(
        args, endpoint, source, step, rate or {
            "base_bits": 0, "x_low_bits": 0, "residual_bits": [0, 0, 0, 0]},
        status, "RUNNING", points, timings, model_load_seconds, gates, reason)
    row["gate_failures"] = reason
    if identity_differences:
        row.update(identity_differences)
    payload = {"status": status, "endpoint": endpoint, "reason": reason,
               "row": row, "gates": gates}
    _atomic_text(
        endpoint_json, json.dumps(payload, indent=2, sort_keys=True) + "\n",
        refuse_existing=True)
    rows.append(row)
    write_rows(output_csv, rows, fields=FORMAL_FIELDS, atomic=True)
    return row


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_chunk_metadata(args, points):
    values = (args.chunk_order, args.chunk_path, args.chunk_sha256,
              args.chunk_count)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("chunk order/path/SHA-256/count must be supplied together")
    if args.chunk_order < 0 or args.chunk_count < 0:
        raise ValueError("chunk order/count must be non-negative")
    expected_hash = args.chunk_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("chunk SHA-256 must contain 64 hexadecimal characters")
    if os.path.realpath(args.chunk_path) != os.path.realpath(args.input_ply):
        raise ValueError("chunk path must identify --input-ply")
    if _sha256_file(args.input_ply) != expected_hash:
        raise ValueError("chunk SHA-256 does not match --input-ply")
    if int(points) != args.chunk_count:
        raise ValueError("chunk count does not match evaluator input points")
    return {
        "order": int(args.chunk_order),
        "path": os.path.realpath(args.chunk_path),
        "sha256": expected_hash,
        "count": int(args.chunk_count),
    }


def _formal_chunk_endpoint(row):
    status = row["endpoint_status"]
    if status != FORMAL_REUSABLE:
        return {"status": status, "reason": row.get("gate_failures", "")}
    common = {
        "status": status,
        "reconstruction_ply": row["reconstruction_ply"],
        "reconstruction_sha256": row["reconstruction_sha256"],
        "physical_bits": int(row["physical_bits"]),
    }
    runtime = {
        "prefix_encode": float(row["prefix_encode_seconds"]),
        "prefix_decode": float(row["prefix_decode_seconds"]),
        "base_synthesis": float(row["base_synthesis_seconds"]),
    }
    if row["endpoint"] == "Ours Base":
        common["components"] = {
            "x_low": int(row["x_low_bits"]), "r1": int(row["r1_bits"]),
            "r2": int(row["r2_bits"]), "r3": int(row["r3_bits"]),
            "r4": int(row["r4_bits"]),
        }
    else:
        common["components"] = {
            "base": int(row["base_bits"]),
            "enhancement": int(row["enhancement_bits"]),
        }
        runtime.update({
            "enhancement_encode": float(row["enhancement_encode_seconds"]),
            "enhancement_decode": float(row["enhancement_decode_seconds"]),
        })
    common["runtime_components"] = runtime
    return common


def _run_ours_formal(args, operating_point, canonical_profile, released_checkpoint,
                     enhancement_checkpoints, base_candidates,
                     scalable_checkpoints):
    """Run one strict Base+Full task and retain Base if Full fails."""
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Ours evaluation requires CUDA")
    enhancement_paths = list(enhancement_checkpoints)
    for value in (args.enhancement_step1763, args.enhancement_step3525):
        if value:
            enhancement_paths.append((None, value))
    mode = _formal_select_mode(
        base_candidates, enhancement_paths, scalable_checkpoints,
        args.run_official)
    if args.official_rate_ids:
        raise ValueError("--ours-formal cannot use --official-rate-ids")
    args.checkpoint_profile = canonical_profile
    for label, path in (("released checkpoint", released_checkpoint),
                        ("Base checkpoint", args.base_synthesis_checkpoint)):
        if not os.path.isfile(path):
            raise FileNotFoundError("{} not found: {}".format(label, path))
    if mode == "independent":
        selected_base = (base_candidates[0][1] if base_candidates
                         else args.base_synthesis_checkpoint)
        enhancement_label, enhancement_path = enhancement_paths[0]
    else:
        selected_base = args.base_synthesis_checkpoint
        enhancement_label, enhancement_path = scalable_checkpoints[0]
    if not os.path.isfile(selected_base):
        raise FileNotFoundError("Base checkpoint not found: " + selected_base)
    if not os.path.isfile(enhancement_path):
        raise FileNotFoundError("Enhancement checkpoint not found: " + enhancement_path)
    frozen_identity = _validate_formal_checkpoint_selection(
        args.point, mode, canonical_profile, args.conditioning_lambda,
        selected_base, enhancement_path, released_checkpoint)
    output_csv = os.path.join(args.output_dir, "physical_rd.csv")
    output_json = os.path.join(args.output_dir, "physical_rd.json")
    if os.path.lexists(output_csv) or os.path.lexists(output_json):
        raise FileExistsError("Refusing to overwrite formal output")
    os.makedirs(args.output_dir, exist_ok=True)
    _atomic_text(os.path.join(args.output_dir, "resolved_args.json"),
                 json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
                 refuse_existing=True)
    _atomic_text(os.path.join(args.output_dir, "command.txt"),
                 shlex.join([sys.executable] + sys.argv) + "\n",
                 refuse_existing=True)
    gpcc_link = os.path.join(args.output_dir, "tmc3_v21")
    if os.path.lexists(gpcc_link):
        if not os.path.exists(gpcc_link):
            raise FileExistsError("G-PCC link is broken: " + gpcc_link)
    else:
        os.symlink(args.gpcc_binary, gpcc_link)
    if operating_point is not None:
        _atomic_text(os.path.join(args.output_dir, "operating_point.json"),
                     json.dumps(operating_point, indent=2) + "\n",
                     refuse_existing=True)
    os.chdir(args.output_dir)
    task_started = time.perf_counter()
    model_started = time.perf_counter()
    base_state = torch.load(selected_base, map_location="cpu")
    base = CanonicalBaseModel(
        released_checkpoint, BaseSynthesisConfig(**base_state["config"])).cuda()
    if mode == "independent":
        enhancement_state = torch.load(enhancement_path, map_location="cpu")
        validate_independent_enhancement_metadata(
            enhancement_state, selected_base, released_checkpoint,
            args.conditioning_lambda, frozen_lineage={
                "base_synthesis_checkpoint": frozen_identity["base"],
                "released_checkpoint": frozen_identity["released"],
            })
        from scalable_attribute.canonical.scalable_model import (  # pylint: disable=import-outside-toplevel
            CanonicalScalableModel)
        load_frozen_base(base, selected_base, released_checkpoint,
                         args.base_checkpoint_lambda,
                         released_checkpoint_lineage=frozen_identity["released"])
        model = CanonicalScalableModel(base, args.conditioning_lambda).cuda().eval()
        model.enhancement.vae.load_state_dict(
            enhancement_state["enhancement_vae"], strict=True)
        checkpoint_step_value = int(enhancement_state["step"])
        source = "OURS_INDEPENDENT"
        checkpoint_metadata = {
            "base_synthesis_checkpoint": os.path.realpath(selected_base),
            "released_checkpoint": os.path.realpath(released_checkpoint),
        }
    else:
        joint_state = torch.load(enhancement_path, map_location="cpu")
        validate_joint_checkpoint_metadata(
            joint_state, args.base_synthesis_checkpoint, released_checkpoint,
            args.conditioning_lambda, canonical_profile, frozen_lineage={
                "base_synthesis_initialization": frozen_identity["base"],
                "released_checkpoint": frozen_identity["released"],
            })
        from scalable_attribute.canonical.scalable_model import (  # pylint: disable=import-outside-toplevel
            CanonicalScalableModel, load_finetuned_scalable)
        load_frozen_base(base, args.base_synthesis_checkpoint,
                         released_checkpoint, args.base_checkpoint_lambda,
                         released_checkpoint_lineage=frozen_identity["released"])
        model = CanonicalScalableModel(base, args.conditioning_lambda).cuda().eval()
        loaded_state = load_finetuned_scalable(
            model, enhancement_path, args.conditioning_lambda)
        checkpoint_step_value = int(loaded_state["step"])
        source = "OURS_JOINT"
        checkpoint_metadata = {
            "base_synthesis_initialization": os.path.realpath(
                args.base_synthesis_checkpoint),
            "released_checkpoint": os.path.realpath(released_checkpoint),
        }
    model.requires_grad_(False)
    model_load_seconds = time.perf_counter() - model_started

    input_started = time.perf_counter()
    coords, rgb = read_ply_ascii(args.input_ply)
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    batch_coords, batch_feats = ME.utils.sparse_collate([coords], [yuv])
    attribute = ME.SparseTensor(features=batch_feats, coordinates=batch_coords,
                                tensor_stride=1, device="cuda")
    chunk_metadata = _formal_chunk_metadata(args, len(attribute))
    gt_path = os.path.join(args.output_dir, "metric_gt.ply")
    _formal_write_ply(gt_path, type("GT", (), {
        "C": torch.as_tensor(batch_coords), "F": torch.as_tensor(yuv)
    })())
    input_io_seconds = time.perf_counter() - input_started
    torch.cuda.reset_peak_memory_stats()
    timings = {name: 0.0 for name in (
        "prefix_encode_seconds", "prefix_decode_seconds",
        "base_synthesis_seconds",
        "enhancement_encode_seconds", "enhancement_decode_seconds")}
    rows, endpoint_json_paths = [], {}
    base_endpoint, full_endpoint = "Ours Base", "Ours Full"
    base_json = os.path.join(args.output_dir, "endpoint_Ours_Base.json")
    full_json = os.path.join(args.output_dir, "endpoint_Ours_Full.json")
    base_box, identity = {}, {}

    def publish_base(result):
        rate = validate_base_rate_details(result["prefix_rate"])
        base_box["rate"] = rate
        gates = {"base_bits_xlow_plus_r1_r4": True,
                 "base_residual_streams_are_r1_r4": True,
                 "base_native_r5_streams_zero": True}
        if endpoint_status(gates) != FORMAL_REUSABLE:
            raise RuntimeError("Base endpoint gate failed")
        _formal_publish_success(
            args, output_csv, gt_path, rows, base_json, base_endpoint, source,
            result["Base"], rate["base_bits"], rate, 0, "",
            checkpoint_step_value, timings, model_load_seconds, len(attribute),
            gates)
        endpoint_json_paths[base_endpoint] = base_json
        base_box["published"] = True

    try:
        hard = _formal_hard_once(
            model, attribute, timings, publish_base, identity,
            stop_after_base=args.formal_stop_after_base)
        if not base_box.get("published"):
            raise RuntimeError("Base endpoint was not published before Full")
        encode_values = [identity.get(name, (float("inf"), float("inf")))
                         for name in ("encode_before", "encode_after")]
        decode_values = [identity.get(name, (float("inf"), float("inf")))
                         for name in ("decode_before", "decode_after")]
        encode_diff = tuple(max(value[index] for value in encode_values)
                            for index in range(2))
        decode_diff = tuple(max(value[index] for value in decode_values)
                            for index in range(2))
        identity_differences = {
            "base_encode_feature_max_abs_difference": encode_diff[0],
            "base_encode_coordinate_max_abs_difference": encode_diff[1],
            "base_decode_feature_max_abs_difference": decode_diff[0],
            "base_decode_coordinate_max_abs_difference": decode_diff[1],
        }
        roundtrip = _formal_snapshot_max_difference(
            hard["encoded_Full_snapshot"], hard["Full"], "Ours Full hard")
        full_gates = {
            "full_bits_base_plus_enhancement": (
                int(hard["full_bits"]) == int(hard["base_bits"]) +
                int(hard["enhancement_bits"])),
            "base_encode_tensor_unchanged": all(value == 0.0 for value in encode_diff),
            "base_decode_tensor_unchanged": all(value == 0.0 for value in decode_diff),
            "full_hard_roundtrip_exact": roundtrip == 0.0,
        }
        full_status = endpoint_status(full_gates)
        if full_status != FORMAL_REUSABLE:
            raise RuntimeError("Full endpoint gates failed")
        _formal_publish_success(
            args, output_csv, gt_path, rows, full_json, full_endpoint, source,
            hard["Full"],
            validate_full_bit_identity(
                hard["base_bits"], hard["enhancement_bits"],
                hard["full_bits"])["full_bits"],
            base_box["rate"], hard["enhancement_bits"], hard["full_bits"],
            checkpoint_step_value,
            timings, model_load_seconds, len(attribute), full_gates,
            identity_differences, roundtrip)
        endpoint_json_paths[full_endpoint] = full_json
    except Exception as error:
        intentional_base_stop = isinstance(error, _FormalStopAfterBase)
        if not base_box.get("published"):
            _formal_publish_failure(
                args, output_csv, rows, base_json, base_endpoint,
                ENDPOINT_FAILED, source, str(error), base_box.get("rate"),
                checkpoint_step_value, len(attribute), timings,
                model_load_seconds, {"base_endpoint_completed": False})
            endpoint_json_paths[base_endpoint] = base_json
        if full_endpoint not in endpoint_json_paths:
            _formal_publish_failure(
                args, output_csv, rows, full_json, full_endpoint,
                (ENDPOINT_NOT_ATTEMPTED if intentional_base_stop else
                 ENDPOINT_FAILED if base_box.get("published") else
                 ENDPOINT_NOT_ATTEMPTED),
                source, str(error) if base_box.get("published")
                else "Base failed; Full was not attempted", base_box.get("rate"),
                checkpoint_step_value, len(attribute), timings,
                model_load_seconds, {"full_endpoint_completed": False},
                {"base_encode_feature_max_abs_difference": identity.get("encode_after", ("", ""))[0],
                 "base_encode_coordinate_max_abs_difference": identity.get("encode_after", ("", ""))[1],
                 "base_decode_feature_max_abs_difference": identity.get("decode_after", ("", ""))[0],
                 "base_decode_coordinate_max_abs_difference": identity.get("decode_after", ("", ""))[1]})
            endpoint_json_paths[full_endpoint] = full_json

    statuses = {row["endpoint"]: row["endpoint_status"] for row in rows}
    combined = task_status(
        statuses.get(base_endpoint, ENDPOINT_FAILED),
        statuses.get(full_endpoint, ENDPOINT_NOT_ATTEMPTED))
    task_wall = time.perf_counter() - task_started
    codec = derive_codec_timings(**timings)
    peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    for row in rows:
        row["task_status"] = combined
        row["task_wall_seconds"] = task_wall
        row["model_load_seconds"] = model_load_seconds
        row["peak_vram_bytes"] = peak_vram
        row.update({key: codec[key] for key in (
            "prefix_encode_seconds", "prefix_decode_seconds",
            "base_synthesis_seconds", "base_codec_seconds")})
        if row["endpoint"] == base_endpoint:
            row["enhancement_encode_seconds"] = 0.0
            row["enhancement_decode_seconds"] = 0.0
            row["full_codec_seconds"] = ""
        else:
            row["enhancement_encode_seconds"] = codec[
                "enhancement_encode_seconds"]
            row["enhancement_decode_seconds"] = codec[
                "enhancement_decode_seconds"]
            row["full_codec_seconds"] = codec["full_codec_seconds"]
    write_rows(output_csv, rows, fields=FORMAL_FIELDS, atomic=True)
    for endpoint, path in endpoint_json_paths.items():
        row = next(item for item in rows if item["endpoint"] == endpoint)
        _atomic_text(path, json.dumps({"status": row["endpoint_status"],
                                        "task_status": combined, "row": row},
                                       indent=2, sort_keys=True) + "\n")
    summary = {
        "status": combined, "task_status": combined,
        "sequence": args.sequence, "frame": args.frame,
        "operating_point": args.point,
        "mode": mode, "endpoint_order": [base_endpoint, full_endpoint],
        "base_endpoint_status": statuses.get(base_endpoint, ENDPOINT_FAILED),
        "full_endpoint_status": statuses.get(full_endpoint, ENDPOINT_NOT_ATTEMPTED),
        "hard_reconstruct_invocations": 1,
        "checkpoint_metadata": checkpoint_metadata,
        "input_io_seconds": input_io_seconds,
        "timings": {**codec, "model_load_seconds": model_load_seconds,
                     "task_wall_seconds": task_wall, "peak_vram_bytes": peak_vram},
        "rows": rows,
    }
    if chunk_metadata is not None:
        chunk_payload = {
            "schema_version": 1,
            "sample": args.sample_id or args.sequence,
            "method": "Ours",
            "operating_point": args.point or
            "lambda_{}".format(args.conditioning_lambda),
            "chunk": chunk_metadata,
            "status": combined,
            "endpoints": {
                row["endpoint"]: _formal_chunk_endpoint(row) for row in rows
            },
        }
        _atomic_text(
            os.path.join(args.output_dir, "chunk_result.json"),
            json.dumps(chunk_payload, indent=2, sort_keys=True) + "\n",
            refuse_existing=True)
    _atomic_text(output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n",
                 refuse_existing=True)
    return 0 if combined == TASK_PASS else 1


def main():
    args = parse_args()
    if args.formal_stop_after_base and not args.ours_formal:
        raise ValueError("--formal-stop-after-base requires --ours-formal")
    for name in (
            "input_ply", "released_checkpoint_root",
            "base_synthesis_checkpoint",
            "gpcc_binary", "output_dir"):
        setattr(args, name, os.path.abspath(os.path.expandvars(getattr(args, name))))
    if args.chunk_path:
        args.chunk_path = os.path.abspath(os.path.expandvars(args.chunk_path))
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
    if args.checkpoint_profile is not None:
        canonical_profile = args.checkpoint_profile
    if args.base_checkpoint_lambda is None:
        args.base_checkpoint_lambda = args.conditioning_lambda
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
    released_checkpoint = os.path.join(
        args.released_checkpoint_root, canonical_profile, "epoch_last.pth")
    if args.ours_formal:
        return _run_ours_formal(
            args, operating_point, canonical_profile, released_checkpoint,
            enhancement_checkpoints, base_candidates, scalable_checkpoints)
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
    write_ply_ascii(gt_path, coords, rgb)
    rows = []

    def record(endpoint, reconstruction, bits, source, started, **extra):
        safe_endpoint = re.sub(r"[^A-Za-z0-9_.-]+", "_", endpoint)
        rec_path = os.path.join(
            args.output_dir, "reconstruction_{}.ply".format(safe_endpoint))
        if os.path.exists(rec_path):
            raise FileExistsError("Refusing to overwrite " + rec_path)
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
            "reconstruction_ply": rec_path,
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
        args.base_checkpoint_lambda)

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
            args.base_checkpoint_lambda)
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
    raise SystemExit(main())
