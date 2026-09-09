#!/usr/bin/env python3
"""Aggregate serial formal point-cloud chunk results.

This is the deliberately small, fail-closed reducer for one
``sample/method/operating-point`` task.  It accepts only normalized per-chunk
JSON records; model-specific producer output must be adapted before it reaches
this command.

Normalized per-chunk result schema (version 1)::

    {
      "schema_version": 1,
      "sample": "Thaidancer_viewdep_vox12",
      "method": "Ours",
      "operating_point": "8K",
      "chunk": {
        "order": 0,
        "path": "chunks/0000/chunk_000000.ply",
        "sha256": "<64 lowercase hexadecimal characters>",
        "count": 800000
      },
      "endpoints": {
        "Base": {
          "reconstruction_ply": "results/chunk_000000_base.ply",
          "physical_bits": 123456,
          "components": {"x_low": 1234, "r1": 122222}
        }
      }
    }

The top-level keys, chunk keys, and endpoint keys above are exact.  Every
component value is a non-negative integer and ``physical_bits`` must equal the
sum of that endpoint's component values.  Endpoint names and component names
are arbitrary non-empty strings, but all chunk records for one task must use
the same endpoint and component sets.  Relative chunk paths are resolved from
the chunk-manifest directory; relative reconstruction paths are resolved from
the result JSON's directory.

The chunk manifest is the JSON emitted by
``prepare_ctc_formal_chunks.py``.  Its selected ``sources[]`` record must
match the supplied canonical source path, source SHA-256, and point count.
Its chunks must be numbered exactly ``0..N-1``.  Each result record must match
the manifest chunk path, SHA-256, count, and order exactly once.  Results may
be listed in any order; aggregation always follows manifest order.

For every selected endpoint, reconstruction PLYs are read through the named
``x,y,z,red,green,blue`` reader from ``prepare_ctc_formal_chunks.py``.  The
RGB values may be lossy, but the multiset of global XYZ coordinates and the
point count must match the canonical source exactly.  The merged reconstruction
and metric-only source representation are six-column ASCII with float-declared
XYZ, while retaining the canonical integer-valued global coordinate and RGB
payloads.  ``pc_error`` is called once against that full canonical content.  No
per-chunk quality values are accepted or averaged.

The default metric is the repository's ``third_party.pc_error_attr.pc_error``
wrapper (resolution 1, color enabled by that wrapper).  ``--pc-error`` may
instead name an executable following the repository's ``pc_error_d`` command
convention: ``-a SOURCE -b RECON --hausdorff=1 --resolution=1 --color=1``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from scripts.scalable_attribute.evaluation import (
        prepare_ctc_formal_chunks as _prepare_chunks,
    )
except ImportError:  # pragma: no cover - useful when called from another cwd
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from scripts.scalable_attribute.evaluation import (
        prepare_ctc_formal_chunks as _prepare_chunks,
    )

try:
    from third_party.pc_error_attr import pc_error as _repository_pc_error
except ImportError:  # pragma: no cover - executable mode can supply --pc-error
    _repository_pc_error = None


# Kept as a module-level symbol so tests and callers can inject the repository
# wrapper without importing any optional model/training dependencies.
pc_error = _repository_pc_error

SCHEMA_VERSION = 1
REQUIRED_RESULT_KEYS = frozenset(
    {"schema_version", "sample", "method", "operating_point", "chunk"}
)
REQUIRED_CHUNK_KEYS = frozenset({"order", "path", "sha256", "count"})
REQUIRED_ENDPOINT_KEYS = frozenset(
    {"reconstruction_ply", "reconstruction_sha256", "physical_bits", "components"}
)
ENDPOINT_SUCCESS_STATUSES = frozenset(
    {"PASS", "SUCCESS", "COMPLETE", "FORMAL_REUSABLE", "OK"}
)
ENDPOINT_FAILURE_STATUSES = frozenset(
    {"FAIL", "FAILED", "ERROR", "MISSING", "NOT_ATTEMPTED", "PARTIAL", "HOLD"}
)
REQUIRED_METRIC_KEYS = (
    "  c[0],    F",
    "  c[1],    F",
    "  c[2],    F",
    "  c[0],PSNRF",
    "  c[1],PSNRF",
    "  c[2],PSNRF",
)
METRIC_OUTPUT_KEYS = (
    "y_mse",
    "u_mse",
    "v_mse",
    "y_psnr",
    "u_psnr",
    "v_psnr",
    "yuv_psnr_611",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CSV_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_BASE_BIT_COMPONENTS = ("x_low", "r1", "r2", "r3", "r4")
_RUNTIME_REQUIRED_BASE = ("prefix_encode", "prefix_decode", "base_synthesis")
_RUNTIME_REQUIRED_ENHANCEMENT = ("enhancement_encode", "enhancement_decode")


class AggregateError(ValueError):
    """Raised when a formal aggregation contract is not satisfied."""


def _display(path: Path) -> str:
    return str(path)


def _require_nonempty_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise AggregateError(f"{context}: {field} must be a non-empty string")
    return value


def _require_integer(value: Any, field: str, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AggregateError(f"{context}: {field} must be an integer")
    result = int(value)
    if result < minimum:
        raise AggregateError(f"{context}: {field} must be >= {minimum}")
    return result


def _require_sha256(value: Any, field: str, context: str) -> str:
    result = _require_nonempty_string(value, field, context).lower()
    if _SHA256_RE.fullmatch(result) is None:
        raise AggregateError(f"{context}: {field} must be a 64-character SHA-256")
    return result


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _lexists(path: Path) -> bool:
    # Path.exists() returns False for a dangling symlink; such a target is still
    # an existing output and must not be replaced.
    return os.path.lexists(str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise AggregateError(f"cannot read {path}: {exc}") from exc
    with handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, context: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateError(f"{context}: invalid JSON {path}") from exc


def _coordinate_records(coords: np.ndarray) -> np.ndarray:
    array = np.asarray(coords)
    if array.ndim != 2 or array.shape[1] != 3:
        raise AggregateError(f"coordinates must have shape (N, 3), got {array.shape}")
    if array.dtype != np.int32:
        try:
            array = np.asarray(array, dtype=np.int32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AggregateError("coordinates cannot be represented as int32") from exc
    records = np.empty(
        len(array),
        dtype=np.dtype(
            [("x", "<i4"), ("y", "<i4"), ("z", "<i4")], align=False
        ),
    )
    records["x"] = array[:, 0]
    records["y"] = array[:, 1]
    records["z"] = array[:, 2]
    return records


def _coordinate_multiset_sha256(coords: np.ndarray) -> str:
    """Hash sorted XYZ records while retaining duplicate coordinates."""

    records = _coordinate_records(coords)
    records.sort(order=("x", "y", "z"), kind="quicksort")
    return hashlib.sha256(records.tobytes(order="C")).hexdigest()


def _coordinate_multiset_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if len(left) != len(right):
        return False
    left_records = _coordinate_records(left)
    right_records = _coordinate_records(right)
    left_records.sort(order=("x", "y", "z"), kind="quicksort")
    right_records.sort(order=("x", "y", "z"), kind="quicksort")
    return bool(np.array_equal(left_records, right_records))


def _validate_source_and_chunks(
    source_ply: os.PathLike[str] | str,
    chunk_manifest: os.PathLike[str] | str,
) -> tuple[Path, Path, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    source_path = Path(source_ply).resolve(strict=False)
    manifest_path = Path(chunk_manifest).resolve(strict=False)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    payload = _load_json(manifest_path, "chunk manifest")
    if not isinstance(payload, Mapping):
        raise AggregateError("chunk manifest root must be an object")
    manifest_version = payload.get("manifest_version")
    if manifest_version is not None:
        version = _require_integer(
            manifest_version, "manifest_version", "chunk manifest"
        )
        if version != 1:
            raise AggregateError("chunk manifest manifest_version must be 1")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise AggregateError("chunk manifest must contain a non-empty sources list")

    matching_sources: list[Mapping[str, Any]] = []
    for index, candidate in enumerate(sources):
        context = f"chunk manifest sources[{index}]"
        if not isinstance(candidate, Mapping):
            raise AggregateError(f"{context} must be an object")
        candidate_source = candidate.get("source_path")
        if isinstance(candidate_source, str):
            candidate_path = _resolve_path(candidate_source, manifest_path.parent)
            if candidate_path == source_path:
                matching_sources.append(candidate)
    if len(matching_sources) != 1:
        if not matching_sources:
            raise AggregateError(
                f"chunk manifest has no unique source_path for {source_path}"
            )
        raise AggregateError(f"chunk manifest has duplicate source records for {source_path}")
    source_record = matching_sources[0]
    source_context = f"chunk manifest source {source_path}"

    source_sha256 = _require_sha256(
        source_record.get("source_sha256"), "source_sha256", source_context
    )
    actual_source_sha256 = _sha256_file(source_path)
    if actual_source_sha256 != source_sha256:
        raise AggregateError(
            f"{source_context}: source SHA-256 mismatch; expected {source_sha256}, "
            f"got {actual_source_sha256}"
        )
    source_coords, source_rgb = _prepare_chunks.read_ply(source_path)
    source_count = _require_integer(
        source_record.get("source_point_count"), "source_point_count", source_context
    )
    if source_count != len(source_coords):
        raise AggregateError(
            f"{source_context}: source point count mismatch; manifest {source_count}, "
            f"PLY {len(source_coords)}"
        )
    if source_count <= 0:
        raise AggregateError("canonical source must contain at least one point")

    chunks = source_record.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise AggregateError(f"{source_context}: chunks must be a non-empty list")
    expected_orders = list(range(len(chunks)))
    by_order: dict[int, dict[str, Any]] = {}
    for index, raw_chunk in enumerate(chunks):
        context = f"{source_context} chunks[{index}]"
        if not isinstance(raw_chunk, Mapping):
            raise AggregateError(f"{context} must be an object")
        missing = REQUIRED_CHUNK_KEYS - set(raw_chunk)
        if missing:
            raise AggregateError(f"{context}: missing keys {sorted(missing)}")
        order = _require_integer(raw_chunk.get("order"), "order", context)
        if order in by_order:
            raise AggregateError(f"{context}: duplicate chunk order {order}")
        if order != index:
            raise AggregateError(
                f"{context}: expected chunk order {index}, got {order}; "
                "manifest orders must be exactly 0..N-1"
            )
        relative_or_absolute = _require_nonempty_string(
            raw_chunk.get("path"), "path", context
        )
        chunk_path = _resolve_path(relative_or_absolute, manifest_path.parent)
        if not chunk_path.is_file():
            raise FileNotFoundError(chunk_path)
        chunk_sha256 = _require_sha256(raw_chunk.get("sha256"), "sha256", context)
        actual_chunk_sha256 = _sha256_file(chunk_path)
        if actual_chunk_sha256 != chunk_sha256:
            raise AggregateError(
                f"{context}: chunk SHA-256 mismatch for {chunk_path}; "
                f"expected {chunk_sha256}, got {actual_chunk_sha256}"
            )
        chunk_count = _require_integer(raw_chunk.get("count"), "count", context)
        chunk_coords, _ = _prepare_chunks.read_ply(chunk_path)
        if chunk_count != len(chunk_coords):
            raise AggregateError(
                f"{context}: chunk point count mismatch; manifest {chunk_count}, "
                f"PLY {len(chunk_coords)}"
            )
        by_order[order] = {
            "order": order,
            "path": chunk_path,
            "path_text": relative_or_absolute,
            "sha256": chunk_sha256,
            "count": chunk_count,
            "coords": chunk_coords,
        }

    if sorted(by_order) != expected_orders:
        raise AggregateError(
            f"{source_context}: expected chunk orders {expected_orders}, "
            f"got {sorted(by_order)}"
        )
    chunk_coordinates = [by_order[order]["coords"] for order in expected_orders]
    chunk_point_count = sum(len(coords) for coords in chunk_coordinates)
    if chunk_point_count != source_count:
        raise AggregateError(
            f"{source_context}: chunk point counts sum to {chunk_point_count}, "
            f"source point count is {source_count}"
        )
    merged_chunk_coordinates = np.concatenate(chunk_coordinates, axis=0)
    if not _coordinate_multiset_equal(source_coords, merged_chunk_coordinates):
        raise AggregateError(
            f"{source_context}: chunk coordinate multiset does not equal "
            "canonical source"
        )
    return (
        source_path,
        manifest_path,
        source_coords,
        source_rgb,
        {
            "source_path": source_path,
            "source_sha256": source_sha256,
            "source_point_count": source_count,
            "coordinate_multiset_sha256": _coordinate_multiset_sha256(source_coords),
        },
        [by_order[order] for order in expected_orders],
    )


def _normalise_result_paths(
    result_paths: Sequence[os.PathLike[str] | str] | os.PathLike[str] | str | None,
    input_manifest: os.PathLike[str] | str | None,
) -> tuple[list[Path], Path | None]:
    if result_paths is not None and input_manifest is not None:
        raise AggregateError("provide exactly one of result_paths or input_manifest")
    manifest_path: Path | None = None
    if input_manifest is not None:
        manifest_path = Path(input_manifest).resolve(strict=False)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if manifest_path.suffix.lower() == ".json":
            payload = _load_json(manifest_path, "result input manifest")
            if isinstance(payload, list):
                entries = payload
            elif isinstance(payload, Mapping):
                if set(payload) != {"schema_version", "results"}:
                    raise AggregateError(
                        "result input manifest object must contain exactly "
                        "schema_version and results"
                    )
                version = _require_integer(
                    payload.get("schema_version"),
                    "schema_version",
                    "result input manifest",
                )
                if version != SCHEMA_VERSION:
                    raise AggregateError(
                        f"result input manifest schema_version must be {SCHEMA_VERSION}"
                    )
                entries = payload.get("results")
            else:
                raise AggregateError("result input manifest must be a list or object")
            if not isinstance(entries, list):
                raise AggregateError("result input manifest results must be a list")
            paths: list[Path] = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, str) or not entry:
                    raise AggregateError(
                        f"result input manifest results[{index}] must be a path string"
                    )
                paths.append(_resolve_path(entry, manifest_path.parent))
        else:
            paths = []
            try:
                lines = manifest_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise AggregateError(f"cannot read result input manifest {manifest_path}") from exc
            for line_number, line in enumerate(lines, start=1):
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                # A text input manifest is intentionally one path per line.  Do
                # not silently split shell-like lines or accept hidden columns.
                if "\t" in value:
                    raise AggregateError(
                        f"result input manifest line {line_number} contains a tab; "
                        "expected one path"
                    )
                paths.append(_resolve_path(value, manifest_path.parent))
    else:
        if result_paths is None:
            raise AggregateError("one result path or input_manifest is required")
        if isinstance(result_paths, (str, os.PathLike)):
            values: Sequence[os.PathLike[str] | str] = [result_paths]
        else:
            values = result_paths
        paths = [Path(value).resolve(strict=False) for value in values]

    if not paths:
        raise AggregateError("result input list is empty")
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            raise AggregateError(f"duplicate result JSON path: {path}")
        seen.add(path)
        if path.suffix.lower() != ".json":
            raise AggregateError(f"result input must be a .json file: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths, manifest_path


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _endpoint_role(endpoint_name: str) -> str | None:
    """Return the formal Base/Full role for a conventional endpoint label."""

    token = re.sub(r"[^a-z0-9]+", "", endpoint_name.lower())
    if token.endswith("full") or token in {"f", "oursf"}:
        return "Full"
    if token.endswith("base") or token in {"b", "oursb"}:
        return "Base"
    return None


def _canonical_bit_component(component_name: str) -> str | None:
    token = _normalise_name(component_name)
    token = re.sub(r"(?:_bits|_bit)$", "", token)
    token = token.replace("bits_", "")
    if token in {"x_low", "xlow", "x_lower", "position_low", "positions_low"}:
        return "x_low"
    if token in {"r1", "r_1", "residual_1", "residual1"}:
        return "r1"
    if token in {"r2", "r_2", "residual_2", "residual2"}:
        return "r2"
    if token in {"r3", "r_3", "residual_3", "residual3"}:
        return "r3"
    if token in {"r4", "r_4", "residual_4", "residual4"}:
        return "r4"
    if token in {"base", "base_prefix", "base_total"}:
        return "base"
    if token in {"enhancement", "enh", "r5", "residual_5", "residual5"}:
        return "enhancement"
    return None


def _canonical_runtime_name(runtime_name: str) -> str:
    token = _normalise_name(runtime_name)
    if token.startswith("runtime_"):
        token = token[len("runtime_"):]
    aliases = {
        "prefix_encode": "prefix_encode",
        "prefix_encode_seconds": "prefix_encode",
        "prefix_encode_second": "prefix_encode",
        "prefix_encode_sec": "prefix_encode",
        "physical_prefix_encode": "prefix_encode",
        "physical_prefix_encode_seconds": "prefix_encode",
        "shared_prefix_encode": "prefix_encode",
        "base_prefix_encode": "prefix_encode",
        "base_prefix_encode_seconds": "prefix_encode",
        "prefix_decode": "prefix_decode",
        "prefix_decode_seconds": "prefix_decode",
        "prefix_decode_second": "prefix_decode",
        "prefix_decode_sec": "prefix_decode",
        "physical_prefix_decode": "prefix_decode",
        "physical_prefix_decode_seconds": "prefix_decode",
        "shared_prefix_decode": "prefix_decode",
        "base_prefix_decode": "prefix_decode",
        "base_prefix_decode_seconds": "prefix_decode",
        "base_synthesis": "base_synthesis",
        "base_synthesis_seconds": "base_synthesis",
        "base_synthesis_second": "base_synthesis",
        "base_synthesis_sec": "base_synthesis",
        "synthesis": "base_synthesis",
        "enhancement_encode": "enhancement_encode",
        "enhancement_encode_seconds": "enhancement_encode",
        "enhancement_encode_second": "enhancement_encode",
        "enhancement_encode_sec": "enhancement_encode",
        "enh_encode": "enhancement_encode",
        "enhancement_decode": "enhancement_decode",
        "enhancement_decode_seconds": "enhancement_decode",
        "enhancement_decode_second": "enhancement_decode",
        "enhancement_decode_sec": "enhancement_decode",
        "enh_decode": "enhancement_decode",
        "base_metric": "base_metric",
        "base_metric_seconds": "base_metric",
        "base_metric_time": "base_metric",
        "metric_base": "base_metric",
        "full_metric": "full_metric",
        "full_metric_seconds": "full_metric",
        "full_metric_time": "full_metric",
        "metric_full": "full_metric",
        "io": "io",
        "io_seconds": "io",
        "io_time": "io",
        "task_wall": "task_wall",
        "task_wall_seconds": "task_wall",
        "task_wall_time": "task_wall",
        "wall": "task_wall",
        "peak_vram": "peak_vram_bytes",
        "peak_vram_bytes": "peak_vram_bytes",
        "peak_vram_gib": "peak_vram_gib",
    }
    return aliases.get(token, token)


def _normalise_runtime(
    raw_runtime: Any, context: str
) -> dict[str, dict[str, Any]]:
    if isinstance(raw_runtime, Mapping) and isinstance(
        raw_runtime.get("components"), Mapping
    ):
        raw_runtime = raw_runtime["components"]
    if not isinstance(raw_runtime, Mapping) or not raw_runtime:
        raise AggregateError(f"{context}: runtime_components must be a non-empty object")
    values: dict[str, dict[str, Any]] = {}
    for raw_name, raw_value in raw_runtime.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise AggregateError(f"{context}: runtime component names must be strings")
        value = raw_value
        if isinstance(value, Mapping):
            if "seconds" in value:
                value = value["seconds"]
            elif "value" in value:
                value = value["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise AggregateError(
                f"{context}: runtime component {raw_name!r} must be numeric"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise AggregateError(
                f"{context}: runtime component {raw_name!r} must be finite and >= 0"
            )
        canonical = _canonical_runtime_name(raw_name)
        previous = values.get(canonical)
        if previous is not None and previous["value"] != number:
            raise AggregateError(
                f"{context}: duplicate runtime aliases for {canonical!r} disagree"
            )
        values[canonical] = {"name": raw_name, "value": number}
    return values


def _failure_evidence(
    path: Path,
    endpoint_name: str,
    status: str,
    reason: str,
    raw_endpoint: Any = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "result": str(path),
        "endpoint": endpoint_name,
        "status": status,
        "reason": reason,
    }
    if isinstance(raw_endpoint, Mapping):
        for key in ("failure", "error", "message", "status"):
            if key in raw_endpoint:
                evidence["input_" + key] = raw_endpoint[key]
    return evidence


def _normalise_endpoint(
    path: Path, endpoint_name: str, raw_endpoint: Any
) -> dict[str, Any]:
    context = f"chunk result {path} endpoint {endpoint_name!r}"
    if not isinstance(raw_endpoint, Mapping):
        return {
            "valid": False,
            "failure": _failure_evidence(
                path, endpoint_name, "FAILED", "endpoint value must be an object", raw_endpoint
            ),
        }

    # Some producers wrap the normalized record in ``record`` or ``result``.
    data: dict[str, Any] = dict(raw_endpoint)
    for wrapper in ("record", "result"):
        if isinstance(raw_endpoint.get(wrapper), Mapping):
            data = {**dict(raw_endpoint[wrapper]), **data}
            break
    declared_status = data.get("status", "PASS")
    if not isinstance(declared_status, str):
        return {
            "valid": False,
            "failure": _failure_evidence(
                path, endpoint_name, "FAILED", "endpoint status must be a string", raw_endpoint
            ),
        }
    status = declared_status.strip().upper()
    if status in ENDPOINT_FAILURE_STATUSES:
        return {
            "valid": False,
            "failure": _failure_evidence(
                path,
                endpoint_name,
                "MISSING" if status == "MISSING" else "FAILED",
                "input endpoint reported failure",
                raw_endpoint,
            ),
        }
    if status not in ENDPOINT_SUCCESS_STATUSES:
        return {
            "valid": False,
            "failure": _failure_evidence(
                path, endpoint_name, "FAILED", f"unsupported endpoint status {status!r}", raw_endpoint
            ),
        }
    if any(key in data for key in ("failure", "error")):
        return {
            "valid": False,
            "failure": _failure_evidence(
                path, endpoint_name, "FAILED", "input endpoint contains failure evidence", raw_endpoint
            ),
        }

    try:
        reconstruction_value = data.get("reconstruction_ply")
        if reconstruction_value is None:
            reconstruction_value = data.get("reconstruction_path")
        reconstruction_text = _require_nonempty_string(
            reconstruction_value, "reconstruction_ply", context
        )
        hash_value = data.get("reconstruction_sha256")
        if hash_value is None:
            hash_value = data.get("reconstruction_ply_sha256")
        if hash_value is None:
            hash_value = data.get("reconstruction_hash")
        reconstruction_sha256 = _require_sha256(
            hash_value, "reconstruction_sha256", context
        )
        physical_bits = data.get("physical_bits")
        if physical_bits is None:
            physical_bits = data.get("total_bits")
        physical_bits = _require_integer(physical_bits, "physical_bits", context)
        raw_components = data.get("components")
        if raw_components is None:
            raw_components = data.get("bit_components")
        if not isinstance(raw_components, Mapping) or not raw_components:
            raise AggregateError(f"{context}: components must be a non-empty object")
        components: dict[str, int] = {}
        for component_name, component_value in raw_components.items():
            if not isinstance(component_name, str) or not component_name:
                raise AggregateError(
                    f"{context}: component names must be non-empty strings"
                )
            components[component_name] = _require_integer(
                component_value,
                f"components[{component_name!r}]",
                context,
            )
        component_sum = sum(components.values())
        if physical_bits != component_sum:
            raise AggregateError(
                f"{context}: physical_bits {physical_bits} != "
                f"sum(components) {component_sum}"
            )
        runtime_value = data.get("runtime_components")
        if runtime_value is None:
            runtime_value = data.get("runtime")
        if runtime_value is None:
            runtime_value = data.get("timings")
        runtime_components = _normalise_runtime(runtime_value, context)
        if data.get("bit_identity_pass") is False:
            raise AggregateError(f"{context}: input bit identity is false")
    except (AggregateError, TypeError, ValueError) as exc:
        return {
            "valid": False,
            "failure": _failure_evidence(path, endpoint_name, "FAILED", str(exc), raw_endpoint),
        }

    return {
        "valid": True,
        "input_status": status,
        "reconstruction_ply": _resolve_path(reconstruction_text, path.parent),
        "reconstruction_text": reconstruction_text,
        "reconstruction_sha256": reconstruction_sha256,
        "physical_bits": physical_bits,
        "components": components,
        "runtime_components": runtime_components,
    }


def _load_result(path: Path, chunk_base: Path) -> dict[str, Any]:
    payload = _load_json(path, "chunk result")
    context = f"chunk result {path}"
    if not isinstance(payload, Mapping):
        raise AggregateError(f"{context}: root must be an object")
    missing = sorted(REQUIRED_RESULT_KEYS - set(payload))
    if missing:
        raise AggregateError(f"{context}: missing keys {missing}")
    version = _require_integer(payload.get("schema_version"), "schema_version", context)
    if version != SCHEMA_VERSION:
        raise AggregateError(f"{context}: schema_version must be {SCHEMA_VERSION}")
    sample = _require_nonempty_string(payload.get("sample"), "sample", context)
    method = _require_nonempty_string(payload.get("method"), "method", context)
    operating_point = _require_nonempty_string(
        payload.get("operating_point"), "operating_point", context
    )

    raw_chunk = payload.get("chunk")
    if not isinstance(raw_chunk, Mapping):
        raise AggregateError(f"{context}: chunk must be an object")
    if set(raw_chunk) != REQUIRED_CHUNK_KEYS:
        missing = sorted(REQUIRED_CHUNK_KEYS - set(raw_chunk))
        extra = sorted(set(raw_chunk) - REQUIRED_CHUNK_KEYS)
        detail: list[str] = []
        if missing:
            detail.append(f"missing keys {missing}")
        if extra:
            detail.append(f"unexpected keys {extra}")
        raise AggregateError(f"{context}: chunk " + "; ".join(detail))
    order = _require_integer(raw_chunk.get("order"), "chunk.order", context)
    chunk_path_text = _require_nonempty_string(raw_chunk.get("path"), "chunk.path", context)
    chunk_sha256 = _require_sha256(raw_chunk.get("sha256"), "chunk.sha256", context)
    chunk_count = _require_integer(raw_chunk.get("count"), "chunk.count", context)

    endpoints: dict[str, dict[str, Any]] = {}
    if "endpoints" in payload:
        raw_endpoints = payload.get("endpoints")
        if not isinstance(raw_endpoints, Mapping):
            raise AggregateError(f"{context}: endpoints must be an object")
        for endpoint_name, raw_endpoint in raw_endpoints.items():
            if not isinstance(endpoint_name, str) or not endpoint_name:
                raise AggregateError(f"{context}: endpoint names must be non-empty strings")
            endpoints[endpoint_name] = _normalise_endpoint(
                path, endpoint_name, raw_endpoint
            )
    elif "endpoint" in payload:
        raw_endpoint_name = payload.get("endpoint")
        if isinstance(raw_endpoint_name, Mapping):
            endpoint_name = raw_endpoint_name.get("name")
            raw_endpoint = dict(raw_endpoint_name)
            raw_endpoint.pop("name", None)
        else:
            endpoint_name = raw_endpoint_name
            raw_endpoint = payload.get("endpoint_result")
            if not isinstance(raw_endpoint, Mapping):
                reserved = set(REQUIRED_RESULT_KEYS) | {
                    "endpoint", "endpoint_result", "status", "failure", "error",
                    "message", "failure_evidence",
                }
                raw_endpoint = {
                    key: value for key, value in payload.items() if key not in reserved
                }
                for key in ("status", "failure", "error", "message"):
                    if key in payload:
                        raw_endpoint[key] = payload[key]
        if not isinstance(endpoint_name, str) or not endpoint_name:
            raise AggregateError(f"{context}: endpoint must be a non-empty string")
        endpoints[endpoint_name] = _normalise_endpoint(
            path, endpoint_name, raw_endpoint
        )
    else:
        # A task-level failure record can legitimately contain no endpoint
        # payload.  The reducer turns that into endpoint-level missing evidence.
        task_status = payload.get("status")
        if not isinstance(task_status, str) or task_status.strip().upper() not in ENDPOINT_FAILURE_STATUSES:
            raise AggregateError(f"{context}: one of endpoints or endpoint is required")

    result_failure: dict[str, Any] | None = None
    if "failure" in payload or "error" in payload or "failure_evidence" in payload:
        result_failure = {
            "result": str(path),
            "status": "FAILED",
            "reason": "input result retained failure evidence",
        }
        for key in ("status", "failure", "error", "message", "failure_evidence"):
            if key in payload:
                result_failure["input_" + key] = payload[key]

    return {
        "path": path,
        "sample": sample,
        "method": method,
        "operating_point": operating_point,
        "chunk": {
            "order": order,
            "path": _resolve_path(chunk_path_text, chunk_base),
            "path_text": chunk_path_text,
            "sha256": chunk_sha256,
            "count": chunk_count,
        },
        "endpoints": endpoints,
        "result_failure": result_failure,
    }


def _validate_results(
    result_paths: Sequence[Path],
    manifest_chunks: Sequence[Mapping[str, Any]],
    endpoints: Sequence[str] | None,
    chunk_base: Path,
) -> tuple[list[dict[str, Any]], list[str], str, str, str]:
    results = [_load_result(path, chunk_base) for path in result_paths]
    first = results[0]
    metadata = (first["sample"], first["method"], first["operating_point"])
    for result in results[1:]:
        if (result["sample"], result["method"], result["operating_point"]) != metadata:
            raise AggregateError(
                "chunk results do not share one sample, method, and operating_point"
            )

    manifest_orders = [int(item["order"]) for item in manifest_chunks]
    expected_orders = set(manifest_orders)
    by_order: dict[int, dict[str, Any]] = {}
    for result in results:
        chunk = result["chunk"]
        order = int(chunk["order"])
        if order not in expected_orders:
            raise AggregateError(f"unexpected result chunk order {order}")
        manifest_chunk = manifest_chunks[order]
        if chunk["path"] != manifest_chunk["path"]:
            raise AggregateError(
                f"chunk order {order}: input path mismatch; result {chunk['path']}, "
                f"manifest {manifest_chunk['path']}"
            )
        if chunk["sha256"] != manifest_chunk["sha256"]:
            raise AggregateError(
                f"chunk order {order}: input SHA-256 mismatch; result "
                f"{chunk['sha256']}, manifest {manifest_chunk['sha256']}"
            )
        if chunk["count"] != manifest_chunk["count"]:
            raise AggregateError(
                f"chunk order {order}: input count mismatch; result {chunk['count']}, "
                f"manifest {manifest_chunk['count']}"
            )
        grouped = by_order.setdefault(
            order,
            {
                "order": order,
                "result_paths": [],
                "endpoints": {},
                "result_failures": [],
            },
        )
        grouped["result_paths"].append(result["path"])
        if result["result_failure"] is not None:
            grouped["result_failures"].append(result["result_failure"])
        for endpoint_name, endpoint in result["endpoints"].items():
            if endpoint_name in grouped["endpoints"]:
                raise AggregateError(
                    f"duplicate result endpoint {endpoint_name!r} for chunk order {order}"
                )
            endpoint = dict(endpoint)
            endpoint["order"] = order
            endpoint["result_path"] = result["path"]
            if not endpoint["valid"]:
                failure = dict(endpoint["failure"])
                failure["order"] = order
                endpoint["failure"] = failure
            grouped["endpoints"][endpoint_name] = endpoint
    if set(by_order) != expected_orders:
        missing = sorted(expected_orders - set(by_order))
        duplicate_or_unexpected = sorted(set(by_order) - expected_orders)
        detail = f"missing result chunk orders {missing}" if missing else ""
        if duplicate_or_unexpected:
            detail += f" unexpected result chunk orders {duplicate_or_unexpected}"
        raise AggregateError(detail.strip())

    available_endpoints: list[str] = []
    for result in results:
        for endpoint_name in result["endpoints"]:
            if endpoint_name not in available_endpoints:
                available_endpoints.append(endpoint_name)
    if endpoints is None:
        formal_base = [name for name in available_endpoints if _endpoint_role(name) == "Base"]
        formal_full = [name for name in available_endpoints if _endpoint_role(name) == "Full"]
        if formal_base:
            selected = [formal_base[0]]
            # Retain a missing Full endpoint as explicit evidence when a task
            # contains the formal Base/Full pair but Full never emitted JSON.
            selected.append(formal_full[0] if formal_full else "Full")
        else:
            selected = available_endpoints
    else:
        selected = list(endpoints)
        if not selected:
            raise AggregateError("at least one endpoint is required")
        if len(set(selected)) != len(selected):
            raise AggregateError("endpoint selection contains duplicates")

    ordered: list[dict[str, Any]] = []
    for order in manifest_orders:
        grouped = by_order[order]
        endpoint_records = dict(grouped["endpoints"])
        for endpoint_name in selected:
            if endpoint_name not in endpoint_records:
                endpoint_records[endpoint_name] = {
                    "valid": False,
                    "order": order,
                    "failure": {
                        "result": [str(path) for path in grouped["result_paths"]],
                        "endpoint": endpoint_name,
                        "order": order,
                        "status": "MISSING",
                        "reason": "endpoint record is absent for this chunk",
                    },
                }
        ordered.append(
            {
                "order": order,
                "result_paths": list(grouped["result_paths"]),
                "result_failures": list(grouped["result_failures"]),
                "endpoints": endpoint_records,
            }
        )
    if not selected:
        raise AggregateError("no endpoint records were supplied")
    return ordered, selected, metadata[0], metadata[1], metadata[2]


def _semantic_components(endpoint: Mapping[str, Any], context: str) -> dict[str, int]:
    semantic: dict[str, int] = {}
    for name, value in endpoint["components"].items():
        canonical = _canonical_bit_component(name)
        if canonical is None:
            continue
        if canonical in semantic:
            raise AggregateError(
                f"{context}: duplicate bit component identity {canonical!r}"
            )
        semantic[canonical] = int(value)
    return semantic


def _validate_bit_identity(
    endpoint_name: str,
    endpoint: Mapping[str, Any],
    base_endpoint: Mapping[str, Any] | None,
    order: int,
) -> None:
    context = f"endpoint {endpoint_name!r} chunk order {order}"
    role = _endpoint_role(endpoint_name)
    semantic = _semantic_components(endpoint, context)
    if role == "Base":
        if set(semantic) != set(_BASE_BIT_COMPONENTS):
            raise AggregateError(
                f"{context}: Base components must be exactly "
                f"{list(_BASE_BIT_COMPONENTS)}"
            )
        if endpoint["physical_bits"] != sum(semantic[name] for name in _BASE_BIT_COMPONENTS):
            raise AggregateError(f"{context}: Base physical bit identity failed")
        return
    if role != "Full":
        return
    if base_endpoint is None or not base_endpoint.get("valid", False):
        raise AggregateError(f"{context}: Full endpoint lacks a valid Base dependency")
    if "enhancement" not in semantic:
        raise AggregateError(f"{context}: Full components lack Enhancement bits")
    if "base" in semantic:
        embedded_base = semantic["base"]
    elif all(name in semantic for name in _BASE_BIT_COMPONENTS):
        embedded_base = sum(semantic[name] for name in _BASE_BIT_COMPONENTS)
    else:
        raise AggregateError(
            f"{context}: Full components must expose Base bits or x_low/r1-r4"
        )
    base_bits = int(base_endpoint["physical_bits"])
    if embedded_base != base_bits:
        raise AggregateError(
            f"{context}: Full Base bits {embedded_base} != Base bits {base_bits}"
        )
    expected_full = base_bits + semantic["enhancement"]
    if int(endpoint["physical_bits"]) != expected_full:
        raise AggregateError(
            f"{context}: Full bits {endpoint['physical_bits']} != "
            f"Base + Enhancement {expected_full}"
        )


def _runtime_entry(
    endpoint: Mapping[str, Any], canonical_name: str
) -> dict[str, Any] | None:
    return endpoint["runtime_components"].get(canonical_name)


def _aggregate_runtime(
    ordered_results: Sequence[Mapping[str, Any]],
    endpoint_name: str,
    base_name: str | None,
) -> dict[str, Any]:
    role = _endpoint_role(endpoint_name)
    runtime_totals: dict[str, float] = {}
    public_names: dict[str, str] = {}
    peak_totals: dict[str, float] = {}
    base_codec_seconds = 0.0
    enhancement_seconds = 0.0

    for grouped in ordered_results:
        order = int(grouped["order"])
        endpoint = grouped["endpoints"][endpoint_name]
        if not endpoint.get("valid", False):
            raise AggregateError(
                f"endpoint {endpoint_name!r} chunk order {order}: "
                "runtime cannot be aggregated for a failed record"
            )
        values = dict(endpoint["runtime_components"])
        base_endpoint: Mapping[str, Any] | None = None
        if role == "Full":
            if base_name is None:
                raise AggregateError("Full runtime cannot be derived without Base")
            base_endpoint = grouped["endpoints"].get(base_name)
            if base_endpoint is None or not base_endpoint.get("valid", False):
                raise AggregateError(
                    f"endpoint {endpoint_name!r} chunk order {order}: "
                    "Full runtime lacks a valid Base dependency"
                )
            for required_name in _RUNTIME_REQUIRED_BASE:
                full_value = values.get(required_name)
                base_value = _runtime_entry(base_endpoint, required_name)
                if full_value is not None and base_value is not None:
                    if full_value["value"] != base_value["value"]:
                        raise AggregateError(
                            f"endpoint {endpoint_name!r} chunk order {order}: "
                            f"shared runtime component {required_name!r} disagrees "
                            "between Base and Full"
                        )
                if full_value is None:
                    if base_value is None:
                        raise AggregateError(
                            f"endpoint {endpoint_name!r} chunk order {order}: "
                            f"missing runtime component {required_name!r}"
                        )
                    values[required_name] = base_value

        if role == "Base":
            required = _RUNTIME_REQUIRED_BASE
        elif role == "Full":
            required = _RUNTIME_REQUIRED_BASE + _RUNTIME_REQUIRED_ENHANCEMENT
        else:
            required = ()
        for required_name in required:
            entry = values.get(required_name)
            if entry is None:
                raise AggregateError(
                    f"endpoint {endpoint_name!r} chunk order {order}: "
                    f"missing runtime component {required_name!r}"
                )

        if role in {"Base", "Full"}:
            base_codec_seconds += sum(
                float(values[name]["value"]) for name in _RUNTIME_REQUIRED_BASE
            )
            if role == "Full":
                enhancement_seconds += sum(
                    float(values[name]["value"])
                    for name in _RUNTIME_REQUIRED_ENHANCEMENT
                )

        for canonical_name, entry in values.items():
            public_name = public_names.setdefault(canonical_name, entry["name"])
            number = float(entry["value"])
            if canonical_name.startswith("peak_vram"):
                peak_totals[public_name] = max(peak_totals.get(public_name, 0.0), number)
            else:
                runtime_totals[public_name] = runtime_totals.get(public_name, 0.0) + number

    if role == "Base":
        codec_seconds = base_codec_seconds
        full_codec_seconds: float | None = None
    elif role == "Full":
        codec_seconds = base_codec_seconds + enhancement_seconds
        full_codec_seconds = codec_seconds
    else:
        # Non-scalable endpoint labels retain their measured components.  The
        # formal Base/Full formula above is the only derived identity.
        codec_seconds = sum(
            value
            for name, value in runtime_totals.items()
            if not any(token in _canonical_runtime_name(name) for token in ("metric", "io", "task_wall"))
        )
        full_codec_seconds = None
    runtime_totals.update(peak_totals)
    return {
        "components": dict(sorted(runtime_totals.items())),
        "codec_runtime_seconds": codec_seconds,
        "base_codec_runtime_seconds": base_codec_seconds if role in {"Base", "Full"} else None,
        "full_codec_runtime_seconds": full_codec_seconds,
        "aggregation": "sum_over_serial_chunks",
        "serial_chunks": True,
    }


def _endpoint_failure_summary(
    endpoint_name: str,
    ordered_results: Sequence[Mapping[str, Any]],
    extra_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for grouped in ordered_results:
        endpoint = grouped["endpoints"].get(endpoint_name)
        if endpoint is not None and not endpoint.get("valid", False):
            failure = dict(endpoint.get("failure", {}))
            failure.setdefault("endpoint", endpoint_name)
            failure.setdefault("order", int(grouped["order"]))
            failures.append(failure)
        if endpoint is None:
            failures.append(
                {
                    "endpoint": endpoint_name,
                    "order": int(grouped["order"]),
                    "status": "MISSING",
                    "reason": "endpoint record is absent for this chunk",
                }
            )
    if extra_failure is not None:
        failures.append(dict(extra_failure))
    if not failures:
        failures.append(
            {
                "endpoint": endpoint_name,
                "status": "MISSING",
                "reason": "endpoint record is absent",
            }
        )
    missing_only = all(item.get("status") == "MISSING" for item in failures)
    status = "MISSING" if missing_only else "FAILED"
    return {
        "status": status,
        "physical_bits": None,
        "components": {},
        "denominator_point_count": None,
        "physical_bpp": None,
        "merged_reconstruction": None,
        "reconstruction_sha256": None,
        "merged_reconstruction_sha256": None,
        "source_coordinate_multiset_sha256": None,
        "merged_coordinate_multiset_sha256": None,
        "coordinate_multiset_verified": False,
        "metrics": None,
        "runtime_components": {},
        "runtime": None,
        "failure": failures[0].get("reason", "endpoint failed"),
        "failure_evidence": failures,
    }


def _aggregate_endpoint(
    endpoint_name: str,
    ordered_results: Sequence[Mapping[str, Any]],
    manifest_chunks: Sequence[Mapping[str, Any]],
    source_path: Path,
    source_coords: np.ndarray,
    source_info: Mapping[str, Any],
    merged_path: Path,
    final_path: Path | None,
    base_name: str | None,
    *,
    pc_error_executable: os.PathLike[str] | str | None,
    pc_error_callable: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    role = _endpoint_role(endpoint_name)
    first_components: set[str] | None = None
    component_totals: dict[str, int] = {}
    component_public_names: dict[str, str] = {}
    reconstruction_coords: list[np.ndarray] = []
    reconstruction_rgb: list[np.ndarray] = []
    chunk_evidence: list[dict[str, Any]] = []
    base_endpoint_name = base_name
    for grouped, manifest_chunk in zip(ordered_results, manifest_chunks):
        order = int(grouped["order"])
        endpoint = grouped["endpoints"].get(endpoint_name)
        if endpoint is None or not endpoint.get("valid", False):
            raise AggregateError(
                f"endpoint {endpoint_name!r} chunk order {order}: endpoint is missing or failed"
            )
        base_endpoint = (
            grouped["endpoints"].get(base_endpoint_name)
            if role == "Full" and base_endpoint_name is not None
            else None
        )
        _validate_bit_identity(endpoint_name, endpoint, base_endpoint, order)
        components = endpoint["components"]
        canonical_names = {
            _canonical_bit_component(name) or _normalise_name(name)
            for name in components
        }
        if first_components is None:
            first_components = canonical_names
        elif canonical_names != first_components:
            raise AggregateError(
                f"endpoint {endpoint_name!r} does not share one component identity set"
            )
        for name, value in components.items():
            canonical_name = _canonical_bit_component(name) or _normalise_name(name)
            public_name = component_public_names.setdefault(canonical_name, name)
            component_totals[public_name] = component_totals.get(public_name, 0) + int(value)

        reconstruction_path = endpoint["reconstruction_ply"]
        if not reconstruction_path.is_file():
            raise FileNotFoundError(reconstruction_path)
        actual_reconstruction_sha256 = _sha256_file(reconstruction_path)
        if actual_reconstruction_sha256 != endpoint["reconstruction_sha256"]:
            raise AggregateError(
                f"endpoint {endpoint_name!r} chunk order {order}: reconstruction SHA-256 mismatch"
            )
        chunk_coords, chunk_rgb = _prepare_chunks.read_ply(reconstruction_path)
        expected_count = int(manifest_chunk["count"])
        if len(chunk_coords) != expected_count:
            raise AggregateError(
                f"endpoint {endpoint_name!r} chunk order {order}: reconstruction point count "
                f"{len(chunk_coords)} != input count {expected_count}"
            )
        if not _coordinate_multiset_equal(manifest_chunk["coords"], chunk_coords):
            raise AggregateError(
                f"endpoint {endpoint_name!r} chunk order {order}: reconstruction coordinate "
                "multiset does not equal input chunk"
            )
        reconstruction_coords.append(chunk_coords)
        reconstruction_rgb.append(chunk_rgb)
        chunk_evidence.append(
            {
                "order": order,
                "input_chunk_sha256": manifest_chunk["sha256"],
                "input_chunk_count": expected_count,
                "reconstruction_ply": str(reconstruction_path),
                "reconstruction_sha256": actual_reconstruction_sha256,
            }
        )

    runtime = _aggregate_runtime(ordered_results, endpoint_name, base_name)
    merged_coords = np.concatenate(reconstruction_coords, axis=0)
    merged_rgb = np.concatenate(reconstruction_rgb, axis=0)
    if len(merged_coords) != len(source_coords):
        raise AggregateError(
            f"endpoint {endpoint_name!r}: merged point count {len(merged_coords)} != "
            f"source point count {len(source_coords)}"
        )
    if not _coordinate_multiset_equal(source_coords, merged_coords):
        raise AggregateError(
            f"endpoint {endpoint_name!r}: merged reconstruction coordinate multiset "
            "does not equal canonical source"
        )
    merged_coordinate_hash = _coordinate_multiset_sha256(merged_coords)
    _write_six_column_ascii(merged_path, merged_coords, merged_rgb)
    merged_sha256 = _sha256_file(merged_path)
    quality = metric(
        source_path,
        merged_path,
        pc_error_executable=pc_error_executable,
        pc_error_callable=pc_error_callable,
    )
    return {
        "status": "FORMAL_REUSABLE",
        "physical_bits": sum(component_totals.values()),
        "components": dict(sorted(component_totals.items())),
        "denominator_point_count": int(len(source_coords)),
        "physical_bpp": sum(component_totals.values()) / len(source_coords),
        "merged_reconstruction": str(final_path) if final_path is not None else None,
        "reconstruction_sha256": merged_sha256,
        "merged_reconstruction_sha256": merged_sha256,
        "source_coordinate_multiset_sha256": source_info["coordinate_multiset_sha256"],
        "merged_coordinate_multiset_sha256": merged_coordinate_hash,
        "coordinate_multiset_verified": True,
        "metrics": quality,
        "runtime_components": runtime["components"],
        "runtime": runtime,
        "codec_runtime_seconds": runtime["codec_runtime_seconds"],
        "base_codec_runtime_seconds": runtime["base_codec_runtime_seconds"],
        "full_codec_runtime_seconds": runtime["full_codec_runtime_seconds"],
        "chunk_evidence": chunk_evidence,
    }


def _parse_number_from_line(line: str) -> float | None:
    # pc_error_attr.number_in_line returns the last parseable number.  Keep the
    # same convention for the executable path, but reject NaN rather than
    # allowing malformed quality into the final artifact.
    tokens = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", line
    )
    if not tokens:
        return None
    try:
        value = float(tokens[-1])
    except ValueError:
        return None
    return value if not math.isnan(value) else None


def _run_pc_error_executable(
    executable: os.PathLike[str] | str,
    source_path: Path,
    reconstruction_path: Path,
) -> Mapping[str, float]:
    if isinstance(executable, (str, os.PathLike)):
        command = shlex.split(os.fspath(executable))
    else:  # Defensive; public API currently documents path/string only.
        raise AggregateError("pc_error executable must be a path or command string")
    if not command:
        raise AggregateError("pc_error executable command is empty")
    command.extend(
        [
            "-a",
            str(source_path),
            "-b",
            str(reconstruction_path),
            "--hausdorff=1",
            "--resolution=1",
            "--color=1",
        ]
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise AggregateError(f"cannot execute pc_error command {command[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        raise AggregateError(
            f"pc_error command failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    values: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        for key in REQUIRED_METRIC_KEYS:
            if key in line:
                number = _parse_number_from_line(line)
                if number is not None:
                    values[key] = number
    return values


def metric(
    source_path: os.PathLike[str] | str,
    reconstruction_path: os.PathLike[str] | str,
    *,
    pc_error_executable: os.PathLike[str] | str | None = None,
    pc_error_callable: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Run one full-source author attribute metric and derive YUV611."""

    source = Path(source_path)
    reconstruction = Path(reconstruction_path)
    if pc_error_executable is not None:
        values = _run_pc_error_executable(pc_error_executable, source, reconstruction)
    else:
        evaluator = pc_error if pc_error_callable is None else pc_error_callable
        if evaluator is None:
            raise AggregateError(
                "repository pc_error wrapper is unavailable; supply --pc-error"
            )
        try:
            values = evaluator(str(source), str(reconstruction), res=1, show=False)
        except Exception as exc:
            raise AggregateError(f"pc_error wrapper failed: {exc}") from exc
    if not isinstance(values, Mapping):
        raise AggregateError("pc_error must return a mapping of metric values")
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in values]
    if missing:
        raise AggregateError("pc_error missing metrics: " + ", ".join(missing))
    numbers: list[float] = []
    for key in REQUIRED_METRIC_KEYS:
        try:
            value = float(values[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise AggregateError(f"pc_error metric {key!r} is not numeric") from exc
        if math.isnan(value):
            raise AggregateError(f"pc_error metric {key!r} is NaN")
        numbers.append(value)
    return {
        "y_mse": numbers[0],
        "u_mse": numbers[1],
        "v_mse": numbers[2],
        "y_psnr": numbers[3],
        "u_psnr": numbers[4],
        "v_psnr": numbers[5],
        "yuv_psnr_611": (6.0 * numbers[3] + numbers[4] + numbers[5]) / 8.0,
    }


def _write_six_column_ascii(path: Path, coords: np.ndarray, rgb: np.ndarray) -> None:
    coords = np.asarray(coords)
    rgb = np.asarray(rgb)
    if coords.ndim != 2 or coords.shape[1] != 3 or rgb.shape != coords.shape:
        raise AggregateError("merged reconstruction arrays must both have shape (N, 3)")
    if coords.dtype != np.int32:
        coords = np.asarray(coords, dtype=np.int32)
    if rgb.dtype != np.uint8:
        rgb = np.asarray(rgb, dtype=np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(coords)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for point, color in zip(coords, rgb):
            handle.write(
                (
                    f"{int(point[0])} {int(point[1])} {int(point[2])} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                ).encode("ascii")
            )
        handle.flush()
        os.fsync(handle.fileno())


def _new_temp_path(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, value = tempfile.mkstemp(prefix=f".{name}.", dir=str(parent))
    os.close(descriptor)
    return Path(value)


def _remove_path(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Cleanup must not hide the original validation/metric error.
        pass


def _publish_without_overwrite(temp_path: Path, target_path: Path) -> None:
    # A hard link is an atomic no-clobber publication on the same filesystem.
    # The target is never replaced, even if another process creates it after
    # the initial preflight check.
    try:
        os.link(temp_path, target_path)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite: {target_path}") from exc
    _remove_path(temp_path)


def _resolve_output_paths(
    output_dir: os.PathLike[str] | str | None,
    output_json: os.PathLike[str] | str | None,
    output_csv: os.PathLike[str] | str | None,
    merged_dir: os.PathLike[str] | str | None,
    endpoint_names: Sequence[str],
) -> dict[str, Any]:
    if output_dir is not None:
        root = Path(output_dir).resolve(strict=False)
        if output_json is None:
            output_json = root / "aggregate.json"
        if output_csv is None:
            output_csv = root / "aggregate.csv"
        if merged_dir is None:
            merged_dir = root / "merged_reconstructions"
    elif output_json is None and output_csv is None and merged_dir is None:
        return {"json": None, "csv": None, "merged": {}}

    if output_json is None and output_csv is not None:
        output_json = Path(output_csv).with_suffix(".json")
    if output_csv is None and output_json is not None:
        output_csv = Path(output_json).with_suffix(".csv")
    json_path = Path(output_json).resolve(strict=False) if output_json is not None else None
    csv_path = Path(output_csv).resolve(strict=False) if output_csv is not None else None
    if json_path is None or csv_path is None:
        # A merged-only API call is useful for diagnostics, but the CLI always
        # supplies an output directory and therefore emits both formats.
        if merged_dir is None:
            raise AggregateError("output JSON and CSV must be supplied together")
    if json_path is not None and csv_path is not None and json_path == csv_path:
        raise AggregateError("output JSON and CSV paths must differ")
    if merged_dir is None:
        merged = {}
    else:
        merged_root = Path(merged_dir).resolve(strict=False)
        merged = {}
        slugs: dict[str, str] = {}
        for endpoint_name in endpoint_names:
            slug = _CSV_SAFE_NAME_RE.sub("_", endpoint_name).strip("._") or "endpoint"
            if slug in slugs:
                raise AggregateError(
                    f"endpoint names collide in merged filenames: "
                    f"{slugs[slug]!r} and {endpoint_name!r}"
                )
            slugs[slug] = endpoint_name
            merged[endpoint_name] = (merged_root / f"merged_{slug}.ply").resolve(strict=False)

    targets = [path for path in (json_path, csv_path) if path is not None]
    targets.extend(merged.values())
    seen: set[Path] = set()
    for target in targets:
        if target in seen:
            raise AggregateError(f"output paths collide: {target}")
        seen.add(target)
        if _lexists(target):
            raise FileExistsError(f"Refusing to overwrite: {target}")
    return {"json": json_path, "csv": csv_path, "merged": merged}


def _component_column(component_name: str) -> str:
    return "component_" + (_CSV_SAFE_NAME_RE.sub("_", component_name).strip("._") or "value")


def _make_csv_rows(
    summary: Mapping[str, Any], endpoint_names: Sequence[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    base_fields = [
        "schema_version",
        "status",
        "endpoint_status",
        "sample",
        "method",
        "operating_point",
        "endpoint",
        "source_ply",
        "source_sha256",
        "source_point_count",
        "denominator_point_count",
        "chunk_count",
        "physical_bits",
        "total_bits",
        "physical_bpp",
        "bpp",
        "merged_reconstruction",
        "reconstruction_sha256",
        "source_coordinate_multiset_sha256",
        "merged_coordinate_multiset_sha256",
        "coordinate_multiset_verified",
        "y_mse",
        "u_mse",
        "v_mse",
        "y_psnr",
        "u_psnr",
        "v_psnr",
        "yuv_psnr_611",
        "components_json",
        "runtime_components_json",
        "codec_runtime_seconds",
        "failure_reason",
    ]
    component_columns: dict[str, str] = {}
    for endpoint_name in endpoint_names:
        for component_name in summary["endpoints"][endpoint_name]["components"]:
            column = _component_column(component_name)
            previous = component_columns.get(column)
            if previous is not None and previous != component_name:
                raise AggregateError(
                    f"component names collide in CSV columns: {previous!r} and "
                    f"{component_name!r}"
                )
            component_columns[column] = component_name
    fields = base_fields + sorted(component_columns)
    rows: list[dict[str, Any]] = []
    source = summary["source"]
    for endpoint_name in endpoint_names:
        endpoint = summary["endpoints"][endpoint_name]
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": summary["status"],
            "endpoint_status": endpoint["status"],
            "sample": summary["sample"],
            "method": summary["method"],
            "operating_point": summary["operating_point"],
            "endpoint": endpoint_name,
            "source_ply": source["path"],
            "source_sha256": source["sha256"],
            "source_point_count": source["point_count"],
            "denominator_point_count": source["point_count"],
            "chunk_count": summary["chunk_manifest"]["chunk_count"],
            "physical_bits": endpoint["physical_bits"],
            "total_bits": endpoint["physical_bits"],
            "physical_bpp": endpoint["physical_bpp"],
            "bpp": endpoint["physical_bpp"],
            "merged_reconstruction": endpoint["merged_reconstruction"],
            "reconstruction_sha256": endpoint["reconstruction_sha256"],
            "source_coordinate_multiset_sha256": endpoint[
                "source_coordinate_multiset_sha256"
            ],
            "merged_coordinate_multiset_sha256": endpoint[
                "merged_coordinate_multiset_sha256"
            ],
            "coordinate_multiset_verified": endpoint["coordinate_multiset_verified"],
            "components_json": json.dumps(
                endpoint["components"], sort_keys=True, separators=(",", ":")
            ),
            "runtime_components_json": json.dumps(
                endpoint.get("runtime_components", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "codec_runtime_seconds": endpoint.get("codec_runtime_seconds"),
            "failure_reason": endpoint.get("failure"),
        }
        if endpoint.get("metrics"):
            row.update(endpoint["metrics"])
        for component_name, value in endpoint["components"].items():
            row[_component_column(component_name)] = value
        rows.append(row)
    return fields, rows


def aggregate_formal_chunk_results(
    source_ply: os.PathLike[str] | str,
    chunk_manifest: os.PathLike[str] | str,
    result_paths: Sequence[os.PathLike[str] | str] | os.PathLike[str] | str | None = None,
    *,
    input_manifest: os.PathLike[str] | str | None = None,
    endpoints: Sequence[str] | None = None,
    output_dir: os.PathLike[str] | str | None = None,
    output_json: os.PathLike[str] | str | None = None,
    output_csv: os.PathLike[str] | str | None = None,
    merged_dir: os.PathLike[str] | str | None = None,
    pc_error_executable: os.PathLike[str] | str | None = None,
    pc_error_callable: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate, merge, score, and optionally publish one formal task.

    ``result_paths`` may be in any order.  The return value is the exact JSON
    object written to ``output_json`` when output publication is requested.
    """

    (
        source_path,
        manifest_path,
        source_coords,
        source_rgb,
        source_info,
        manifest_chunks,
    ) = _validate_source_and_chunks(source_ply, chunk_manifest)
    normalised_result_paths, result_manifest_path = _normalise_result_paths(
        result_paths, input_manifest
    )
    ordered_results, selected_endpoints, sample, method, operating_point = _validate_results(
        normalised_result_paths, manifest_chunks, endpoints, manifest_path.parent
    )
    outputs = _resolve_output_paths(
        output_dir, output_json, output_csv, merged_dir, selected_endpoints
    )

    # Temp merged files live beside their final targets so publication remains
    # atomic.  MPEG pc_error 0.13.4 cannot read some native CTC PLY headers, so
    # build one metric-only named-field representation from the already
    # verified canonical arrays.  Payload values, duplicates, global
    # coordinates, and point count remain unchanged for both methods.
    private_temp_dir = tempfile.TemporaryDirectory(prefix="aggregate-formal-")
    private_root = Path(private_temp_dir.name)
    metric_source_path = private_root / "canonical_metric_source.ply"
    _write_six_column_ascii(metric_source_path, source_coords, source_rgb)
    if outputs["merged"]:
        temp_merged = {
            endpoint_name: _new_temp_path(
                final_path.parent, final_path.name
            )
            for endpoint_name, final_path in outputs["merged"].items()
        }
    else:
        temp_merged = {
            endpoint_name: private_root / f"merged_{index:03d}.ply"
            for index, endpoint_name in enumerate(selected_endpoints)
        }

    temp_paths: list[Path] = [path for path in temp_merged.values() if outputs["merged"]]
    published_paths: list[Path] = []
    try:
        endpoint_summaries: dict[str, dict[str, Any]] = {}
        base_names = [
            name for name in selected_endpoints if _endpoint_role(name) == "Base"
        ]
        base_name = base_names[0] if base_names else None
        for endpoint_name in selected_endpoints:
            try:
                endpoint_summary = _aggregate_endpoint(
                    endpoint_name,
                    ordered_results,
                    manifest_chunks,
                    metric_source_path,
                    source_coords,
                    source_info,
                    temp_merged[endpoint_name],
                    outputs["merged"].get(endpoint_name),
                    base_name,
                    pc_error_executable=pc_error_executable,
                    pc_error_callable=pc_error_callable,
                )
            except Exception as exc:
                endpoint_summary = _endpoint_failure_summary(
                    endpoint_name,
                    ordered_results,
                    {
                        "endpoint": endpoint_name,
                        "status": "FAILED",
                        "reason": str(exc),
                    },
                )
                if outputs["merged"]:
                    _remove_path(temp_merged[endpoint_name])
                    if temp_merged[endpoint_name] in temp_paths:
                        temp_paths.remove(temp_merged[endpoint_name])
            else:
                if outputs["merged"]:
                    final_path = outputs["merged"][endpoint_name]
                    _publish_without_overwrite(temp_merged[endpoint_name], final_path)
                    published_paths.append(final_path)
                    temp_paths.remove(temp_merged[endpoint_name])
            endpoint_summaries[endpoint_name] = endpoint_summary

        chunk_summary = [
            {
                "order": int(chunk["order"]),
                "path": str(chunk["path"]),
                "sha256": chunk["sha256"],
                "count": int(chunk["count"]),
            }
            for chunk in manifest_chunks
        ]
        reusable = [
            endpoint_summaries[name]["status"] == "FORMAL_REUSABLE"
            for name in selected_endpoints
        ]
        if all(reusable):
            task_status = "PASS"
            formal_status = "FORMAL_REUSABLE"
        elif any(reusable):
            task_status = "PARTIAL"
            formal_status = "PARTIAL"
        else:
            task_status = "FAIL"
            formal_status = "FAIL"
        failure_evidence = [
            evidence
            for endpoint_name in selected_endpoints
            for evidence in endpoint_summaries[endpoint_name].get(
                "failure_evidence", []
            )
        ]
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": task_status,
            "formal_status": formal_status,
            "sample": sample,
            "method": method,
            "operating_point": operating_point,
            "source": {
                "path": str(source_path),
                "sha256": source_info["source_sha256"],
                "point_count": source_info["source_point_count"],
                "coordinate_multiset_sha256": source_info[
                    "coordinate_multiset_sha256"
                ],
            },
            "chunk_manifest": {
                "path": str(manifest_path),
                "chunk_count": len(chunk_summary),
                "expected_orders": [item["order"] for item in chunk_summary],
                "chunks": chunk_summary,
            },
            "result_inputs": {
                "manifest": str(result_manifest_path) if result_manifest_path else None,
                "paths": [str(path) for path in normalised_result_paths],
                "orders_reordered_to_manifest": [int(result["order"]) for result in ordered_results],
            },
            "metric": {
                "implementation": (
                    "external_pc_error_executable"
                    if pc_error_executable is not None
                    else "third_party.pc_error_attr.pc_error"
                ),
                "resolution": 1,
                "quality_aggregation": "one_full_source_call_per_endpoint",
                "per_chunk_quality_averaging": False,
                "yuv_psnr_611": "(6Y+U+V)/8",
            },
            "endpoints": endpoint_summaries,
            "endpoint_statuses": {
                endpoint_name: endpoint_summaries[endpoint_name]["status"]
                for endpoint_name in selected_endpoints
            },
            "failure_evidence": failure_evidence,
        }

        fields, rows = _make_csv_rows(summary, selected_endpoints)
        if outputs["json"] is not None:
            json_temp = _new_temp_path(outputs["json"].parent, outputs["json"].name)
            temp_paths.append(json_temp)
            with json_temp.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _publish_without_overwrite(json_temp, outputs["json"])
            published_paths.append(outputs["json"])
            temp_paths.remove(json_temp)
        if outputs["csv"] is not None:
            csv_temp = _new_temp_path(outputs["csv"].parent, outputs["csv"].name)
            temp_paths.append(csv_temp)
            with csv_temp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            _publish_without_overwrite(csv_temp, outputs["csv"])
            published_paths.append(outputs["csv"])
            temp_paths.remove(csv_temp)
        return summary
    except Exception:
        for path in temp_paths:
            _remove_path(path)
        for path in published_paths:
            _remove_path(path)
        raise
    finally:
        private_temp_dir.cleanup()


# Short aliases make the public operation discoverable without duplicating the
# implementation name in callers.
aggregate = aggregate_formal_chunk_results
aggregate_chunk_results = aggregate_formal_chunk_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate one formal sample/method/operating-point across serial chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-ply",
        "--canonical-source",
        "--input-ply",
        dest="source_ply",
        required=True,
        help="canonical full-sample source PLY",
    )
    parser.add_argument("--chunk-manifest", required=True)
    parser.add_argument(
        "--result",
        "--chunk-result",
        dest="result_paths",
        action="append",
        help="one normalized per-chunk result JSON; repeat in any order",
    )
    parser.add_argument(
        "--results",
        dest="result_paths_many",
        nargs="+",
        help="ordered normalized per-chunk result JSON paths",
    )
    parser.add_argument(
        "--input-manifest",
        "--result-manifest",
        dest="input_manifest",
        help="text or JSON manifest listing normalized per-chunk result JSONs",
    )
    parser.add_argument(
        "--endpoint",
        dest="endpoints",
        action="append",
        required=True,
        help="endpoint to aggregate; repeat for multiple endpoints",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--merged-dir")
    parser.add_argument(
        "--pc-error",
        "--pc-error-executable",
        dest="pc_error_executable",
        help="optional pc_error_d-compatible executable/wrapper command",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.result_paths and args.result_paths_many:
        parser.error("use only one of --result/--chunk-result or --results")
    if args.input_manifest is not None and (args.result_paths or args.result_paths_many):
        parser.error("use either result paths or --input-manifest, not both")
    result_values: Sequence[str] | None
    if args.result_paths_many is not None:
        result_values = args.result_paths_many
    else:
        result_values = args.result_paths
    if result_values is None and args.input_manifest is None:
        parser.error("one of --result/--results or --input-manifest is required")
    if args.output_dir is None and args.output_json is None and args.output_csv is None:
        parser.error("one of --output-dir, --output-json, or --output-csv is required")
    try:
        summary = aggregate_formal_chunk_results(
            args.source_ply,
            args.chunk_manifest,
            result_values,
            input_manifest=args.input_manifest,
            endpoints=args.endpoints,
            output_dir=args.output_dir,
            output_json=args.output_json,
            output_csv=args.output_csv,
            merged_dir=args.merged_dir,
            pc_error_executable=args.pc_error_executable,
        )
    except (AggregateError, FileExistsError, FileNotFoundError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
