#!/usr/bin/env python3
"""Validate the frozen formal static RGB evaluation contract.

This is deliberately a static, standard-library-only gate.  It validates the
sample and checkpoint manifests without importing the evaluation stack, and it
does not submit work or access the network.
"""

import argparse
import csv
import hashlib
import json
import ntpath
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path


EXPECTED_SAMPLE_COLUMNS = (
    "dataset",
    "sample_id",
    "source_path",
    "points",
    "sha256",
    "source_evidence",
    "formal_preprocessing",
)
EXPECTED_DATASET_COUNTS = {
    "8iVFB": 4,
    "Owlii": 4,
    "CTC": 12,
}
EXPECTED_PREPROCESSING = {
    "8iVFB": "identity",
    "Owlii": "identity",
    "CTC": "kdtree_800k_global_no_recenter",
}
EXPECTED_OPERATING_POINTS = (
    ("32k", 32768),
    ("16k", 16384),
    ("8k", 8192),
    ("4k", 4096),
    ("2k", 2048),
    ("1k", 1024),
    ("512", 512),
)
EXPECTED_RELEASED_PROFILES = ("32k8k", "8k256", "2k128")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

SCRIPT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES_TSV = SCRIPT_ROOT / "configs/evaluation/formal_static_rgb_v1_samples.tsv"
DEFAULT_CHECKPOINTS_JSON = (
    SCRIPT_ROOT / "configs/scalable_attribute/formal_static_rgb_v1_checkpoints.json"
)


class ContractError(ValueError):
    """Raised when a manifest violates the frozen contract."""


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _nonempty_string(value, label):
    _require(isinstance(value, str), label + " must be a string")
    _require(value != "" and value.strip() == value,
             label + " must be a non-empty string")
    return value


def _validate_sha256(value, label):
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
             label + " must be 64 lowercase hexadecimal characters")
    return value


def _load_json(path):
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("duplicate JSON key: {!r}".format(key))
            result[key] = value
        return result

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=no_duplicate_keys)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("cannot read checkpoints JSON {}: {}".format(
            path, exc))


def _source_identity(source_path):
    """Return a stable lexical identity for duplicate source detection."""
    return os.path.normcase(os.path.normpath(source_path))


def read_samples(path):
    """Read and validate the sample TSV, returning normalized row dictionaries."""
    rows = []
    seen_pairs = {}
    seen_sources = {}
    try:
        handle = open(path, "r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ContractError("cannot read samples TSV {}: {}".format(path, exc))

    with handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            header = next(reader)
        except StopIteration:
            raise ContractError("samples TSV is empty")
        except (csv.Error, UnicodeError) as exc:
            raise ContractError("invalid samples TSV header: {}".format(exc))
        if tuple(header) != EXPECTED_SAMPLE_COLUMNS:
            raise ContractError(
                "samples TSV columns must be exactly {} in that order; got {}".format(
                    list(EXPECTED_SAMPLE_COLUMNS), header))

        try:
            for line_number, values in enumerate(reader, start=2):
                if not values:
                    raise ContractError(
                        "samples TSV line {} is blank".format(line_number))
                if len(values) != len(EXPECTED_SAMPLE_COLUMNS):
                    raise ContractError(
                        "samples TSV line {} has {} columns; expected {}".format(
                            line_number, len(values), len(EXPECTED_SAMPLE_COLUMNS)))
                row = dict(zip(EXPECTED_SAMPLE_COLUMNS, values))
                for field in EXPECTED_SAMPLE_COLUMNS:
                    _nonempty_string(row[field],
                                     "samples TSV line {} {}".format(
                                         line_number, field))

                dataset = row["dataset"]
                _require(dataset in EXPECTED_DATASET_COUNTS,
                         "samples TSV line {} has unknown dataset {!r}".format(
                             line_number, dataset))
                source_path = row["source_path"]
                _require("\x00" not in source_path,
                         "samples TSV line {} source_path must not contain NUL".format(
                             line_number))
                pair = (dataset, row["sample_id"])
                if pair in seen_pairs:
                    raise ContractError(
                        "duplicate dataset/sample {} at lines {} and {}".format(
                            pair, seen_pairs[pair], line_number))
                source_key = _source_identity(source_path)
                if source_key in seen_sources:
                    raise ContractError(
                        "duplicate source path {!r} at lines {} and {}".format(
                            source_path, seen_sources[source_key], line_number))
                seen_pairs[pair] = line_number
                seen_sources[source_key] = line_number

                points_text = row["points"]
                _require(re.fullmatch(r"[0-9]+", points_text) is not None,
                         "samples TSV line {} points must be a positive integer".format(
                             line_number))
                points = int(points_text)
                _require(points > 0,
                         "samples TSV line {} points must be positive".format(
                             line_number))
                _validate_sha256(
                    row["sha256"],
                    "samples TSV line {} sha256".format(line_number))
                expected_preprocessing = EXPECTED_PREPROCESSING[dataset]
                _require(row["formal_preprocessing"] == expected_preprocessing,
                         "samples TSV line {} preprocessing for {} must be {!r}".format(
                             line_number, dataset, expected_preprocessing))

                rows.append({
                    "dataset": dataset,
                    "sample_id": row["sample_id"],
                    "source_path": source_path,
                    "points": points,
                    "sha256": row["sha256"],
                    "source_evidence": row["source_evidence"],
                    "formal_preprocessing": row["formal_preprocessing"],
                    "line_number": line_number,
                })
        except (csv.Error, UnicodeError) as exc:
            raise ContractError("invalid samples TSV: {}".format(exc))

    counts = Counter(row["dataset"] for row in rows)
    if dict(counts) != EXPECTED_DATASET_COUNTS:
        raise ContractError(
            "dataset counts must be {}; got {}".format(
                EXPECTED_DATASET_COUNTS, dict(counts)))
    return rows


def _validate_relative_artifact_path(value, label):
    _nonempty_string(value, label)
    _require("\x00" not in value, label + " must not contain NUL")
    # Check both POSIX and Windows spellings even though this contract is
    # evaluated on Linux.  A drive-qualified path or UNC path is not relative.
    drive, _ = ntpath.splitdrive(value)
    slash_value = value.replace("\\", "/")
    _require(not os.path.isabs(value) and not slash_value.startswith("/"),
             label + " must be relative")
    _require(drive == "", label + " must be relative")
    _require(".." not in slash_value.split("/"),
             label + " must not contain path traversal")


def _validate_artifact(descriptor, label):
    _require(isinstance(descriptor, dict), label + " must be an object")
    missing = [
        field for field in ("path", "size_bytes", "sha256")
        if field not in descriptor
    ]
    _require(not missing,
             "{} is missing {}".format(label, ", ".join(missing)))
    path = descriptor["path"]
    _validate_relative_artifact_path(path, label + ".path")
    size_bytes = descriptor["size_bytes"]
    _require(_is_positive_int(size_bytes),
             label + ".size_bytes must be a positive integer")
    sha256 = descriptor["sha256"]
    _validate_sha256(sha256, label + ".sha256")
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _artifact_field(key):
    return key == "checkpoint" or key.endswith("_checkpoint")


def _validate_checkpoints(manifest):
    _require(isinstance(manifest, dict), "checkpoints JSON top level must be an object")
    if "schema_version" in manifest:
        _require(manifest["schema_version"] == 1,
                 "checkpoints schema_version must be 1")
    _nonempty_string(manifest.get("status"), "checkpoints status")
    _require(re.fullmatch(r"frozen(?:_|$).*", manifest["status"]) is not None,
             "checkpoints status must be frozen")
    origin_root = _nonempty_string(
        manifest.get("origin_root"), "checkpoints origin_root")
    _require("\x00" not in origin_root,
             "checkpoints origin_root must not contain NUL")

    released = manifest.get("released_checkpoints")
    _require(isinstance(released, dict),
             "released_checkpoints must be an object")
    _require(set(released) == set(EXPECTED_RELEASED_PROFILES),
             "released_checkpoints profiles must be exactly {}; got {}".format(
                 list(EXPECTED_RELEASED_PROFILES), list(released)))
    released_artifacts = {}
    artifact_records = []
    for profile in EXPECTED_RELEASED_PROFILES:
        label = "released_checkpoints.{}".format(profile)
        artifact = _validate_artifact(released[profile], label)
        released_artifacts[profile] = artifact
        artifact_records.append({"location": label, **artifact})

    operating_points = manifest.get("operating_points")
    _require(isinstance(operating_points, list),
             "operating_points must be a list")
    _require(len(operating_points) == len(EXPECTED_OPERATING_POINTS),
             "operating_points must contain exactly seven entries")

    validated_points = []
    by_point = {}
    for index, (operation, expected) in enumerate(
            zip(operating_points, EXPECTED_OPERATING_POINTS)):
        label = "operating_points[{}]".format(index)
        _require(isinstance(operation, dict), label + " must be an object")
        expected_point, expected_lambda = expected
        _require(operation.get("point") == expected_point,
                 "{} point must be {!r}".format(label, expected_point))
        _require(operation.get("lambda") == expected_lambda and
                 _is_positive_int(operation.get("lambda")),
                 "{} lambda must be {}".format(label, expected_lambda))
        released_profile = operation.get("released_profile")
        _require(isinstance(released_profile, str),
                 "{}.released_profile must be a string".format(label))
        _require(released_profile in released,
                 "{} released_profile must name a released checkpoint".format(label))
        base_loader = operation.get("base_loader")
        full_loader = operation.get("full_loader")
        _nonempty_string(base_loader, label + ".base_loader")
        _nonempty_string(full_loader, label + ".full_loader")

        point_artifacts = {}
        for field in ("base_checkpoint", "full_checkpoint"):
            _require(field in operation,
                     "{} is required in {}".format(field, label))
        # Validate the canonical fields in a stable order, then any additional
        # checkpoint fields so an unexpected bootstrap/checkpoint is not left
        # unchecked.
        artifact_fields = [
            field for field in ("bootstrap_checkpoint", "base_checkpoint",
                                "full_checkpoint")
            if field in operation
        ]
        artifact_fields.extend(sorted(
            field for field in operation
            if _artifact_field(field) and field not in artifact_fields))
        for field in artifact_fields:
            artifact_label = "{}.{}".format(label, field)
            artifact = _validate_artifact(operation[field], artifact_label)
            point_artifacts[field] = artifact
            artifact_records.append({"location": artifact_label, **artifact})

        validated = {
            "point": expected_point,
            "lambda": expected_lambda,
            "released_profile": released_profile,
            "base_loader": base_loader,
            "full_loader": full_loader,
            "artifacts": point_artifacts,
        }
        if "base_checkpoint_lambda" in operation:
            base_lambda = operation["base_checkpoint_lambda"]
            _require(_is_positive_int(base_lambda),
                     label + ".base_checkpoint_lambda must be a positive integer")
            validated["base_checkpoint_lambda"] = base_lambda
        validated_points.append(validated)
        by_point[expected_point] = validated

    _require("32k8k" in released_artifacts,
             "released 8192 policy requires the 32k8k profile")
    for point in validated_points:
        if point["lambda"] == 8192:
            _require(point["released_profile"] == "32k8k",
                     "the released 8192 policy requires released_profile 32k8k")

    four_k = by_point["4k"]
    _require(four_k["base_loader"] == "joint_scalable_base" and
             four_k["full_loader"] == "joint_scalable_full",
             "4k must use joint_scalable_base and joint_scalable_full")
    _require(four_k.get("base_checkpoint_lambda") == 8192,
             "4k must declare base_checkpoint_lambda 8192")
    base_4k = four_k["artifacts"]["base_checkpoint"]
    full_4k = four_k["artifacts"]["full_checkpoint"]
    _require(base_4k == full_4k,
             "4k base_checkpoint and full_checkpoint must be the same artifact")

    two_k = by_point["2k"]
    _require(two_k["base_loader"] == "rescued_base" and
             two_k["full_loader"] == "independent_enhancement",
             "2k must use the rescued_base and independent_enhancement contract")

    return {
        "manifest": manifest,
        "released_artifacts": released_artifacts,
        "operating_points": validated_points,
        "artifact_records": artifact_records,
    }


def _resolve_under_root(root, relative_path, label):
    try:
        candidate = (root / relative_path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError("cannot resolve {}: {}".format(label, exc))
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ContractError(
            "{} resolves outside origin_root: {}".format(label, candidate))
    return candidate


def _stream_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint_file(path, artifact, label):
    if not path.is_file():
        raise ContractError("{} does not exist as a regular file: {}".format(
            label, path))
    try:
        before = path.stat().st_size
        actual_sha = _stream_sha256(path)
        after = path.stat().st_size
    except OSError as exc:
        raise ContractError("cannot verify {}: {}".format(label, exc))
    if before != after:
        raise ContractError("{} changed while being hashed".format(label))
    if before != artifact["size_bytes"]:
        raise ContractError(
            "{} size mismatch: expected {}, got {}".format(
                label, artifact["size_bytes"], before))
    if actual_sha != artifact["sha256"]:
        raise ContractError(
            "{} sha256 mismatch: expected {}, got {}".format(
                label, artifact["sha256"], actual_sha))
    return {"resolved_path": str(path), "actual_size_bytes": before,
            "actual_sha256": actual_sha, "verified": True}


def _verify_source_file(row):
    source_path = row["source_path"]
    _require(os.path.isabs(source_path),
             "source path for {} must be absolute with --verify-files".format(
                 row["sample_id"]))
    try:
        path = Path(source_path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError("cannot resolve source {}: {}".format(
            row["sample_id"], exc))
    if not path.is_file():
        raise ContractError(
            "source file for {} does not exist as a regular file: {}".format(
                row["sample_id"], path))
    try:
        before = path.stat().st_size
        actual_sha = _stream_sha256(path)
        after = path.stat().st_size
    except OSError as exc:
        raise ContractError("cannot verify source {}: {}".format(
            row["sample_id"], exc))
    if before != after:
        raise ContractError("source {} changed while being hashed".format(
            row["sample_id"]))
    if actual_sha != row["sha256"]:
        raise ContractError(
            "source {} sha256 mismatch: expected {}, got {}".format(
                row["sample_id"], row["sha256"], actual_sha))
    return {
        "resolved_path": str(path),
        "actual_size_bytes": before,
        "actual_sha256": actual_sha,
        "verified": True,
    }


def _verify_files(rows, checkpoint_info):
    manifest = checkpoint_info["manifest"]
    origin_root_value = manifest["origin_root"]
    _require(os.path.isabs(origin_root_value),
             "origin_root must be absolute with --verify-files")
    try:
        origin_root = Path(origin_root_value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError("cannot resolve origin_root: {}".format(exc))

    source_evidence = []
    for row in rows:
        evidence = {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "path": row["source_path"],
            "sha256": row["sha256"],
            "points": row["points"],
        }
        evidence.update(_verify_source_file(row))
        source_evidence.append(evidence)

    checkpoint_evidence = []
    for record in checkpoint_info["artifact_records"]:
        path = _resolve_under_root(
            origin_root, record["path"], record["location"])
        evidence = dict(record)
        evidence.update(_verify_checkpoint_file(path, record, record["location"]))
        checkpoint_evidence.append(evidence)
    return {
        "origin_root": str(origin_root),
        "sources": source_evidence,
        "checkpoints": checkpoint_evidence,
    }


def validate_contract(samples_tsv, checkpoints_json, verify_files=False):
    """Validate both manifests and return a JSON-serializable evidence summary."""
    rows = read_samples(samples_tsv)
    checkpoint_info = _validate_checkpoints(_load_json(checkpoints_json))

    verification = {
        "enabled": bool(verify_files),
        "sources": [],
        "checkpoints": [],
    }
    if verify_files:
        verification.update(_verify_files(rows, checkpoint_info))

    summary = {
        "status": "PASS",
        "samples_tsv": str(Path(samples_tsv).resolve()),
        "checkpoints_json": str(Path(checkpoints_json).resolve()),
        "verify_files": bool(verify_files),
        "sample_count": len(rows),
        "dataset_counts": dict(EXPECTED_DATASET_COUNTS),
        "operating_points": [
            {
                "point": point["point"],
                "lambda": point["lambda"],
                "released_profile": point["released_profile"],
            }
            for point in checkpoint_info["operating_points"]
        ],
        "checkpoint_count": len(checkpoint_info["artifact_records"]),
        "checkpoint_artifacts": checkpoint_info["artifact_records"],
        "verification": verification,
    }
    return summary


def _write_atomic_json(path, payload):
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(parent),
                prefix=target.name + ".tmp-", suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the frozen formal static RGB manifests.")
    parser.add_argument(
        "--samples-tsv", "--samples", dest="samples_tsv",
        default=str(DEFAULT_SAMPLES_TSV),
        help="sample TSV (default: %(default)s)")
    parser.add_argument(
        "--checkpoints-json", "--checkpoints", dest="checkpoints_json",
        default=str(DEFAULT_CHECKPOINTS_JSON),
        help="checkpoint JSON (default: %(default)s)")
    parser.add_argument(
        "--verify-files", action="store_true",
        help="verify all absolute sources and checkpoint files")
    parser.add_argument(
        "--output-json", metavar="PATH",
        help="atomically write the PASS evidence summary to PATH")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        summary = validate_contract(
            args.samples_tsv, args.checkpoints_json, args.verify_files)
        if args.output_json:
            _write_atomic_json(args.output_json, summary)
    except (ContractError, OSError) as exc:
        print("FAIL: {}".format(exc), file=sys.stderr)
        return 1

    print("PASS: formal static RGB contract validated ({} samples, {} checkpoints)".format(
        summary["sample_count"], summary["checkpoint_count"]))
    if args.output_json:
        print("Evidence: {}".format(Path(args.output_json).resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
