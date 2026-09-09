#!/usr/bin/env python3
"""Prepare lossless, global-coordinate CTC chunks.

The released attribute loader only consumes six-column ASCII PLY files, while
the official CTC inputs include binary little-endian PLY files and do not use a
single vertex-property order.  This module performs only that representation
adaptation and the pinned KD-tree partition.  In particular, it never shifts,
quantizes, or deduplicates points.

The reader is intentionally conservative.  It accepts a PLY containing one
``vertex`` element made up of scalar properties, and rejects schemas for which
the vertex payload cannot be unambiguously located or decoded.  Extra scalar
properties on the vertex (for example normals before or after RGB) are fine;
extra PLY elements are rejected because this adapter does not parse their
payloads.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import hashlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_MAX_NUM = 800_000
PINNED_PARTITION_SHA256 = "ec3c5c347e289b364acb1c41bbd388c172bff6b142fc87f630a7460d67389dec"
EXPECTED_PARTITION_SHA256 = PINNED_PARTITION_SHA256
REQUIRED_PROPERTIES = ("x", "y", "z", "red", "green", "blue")
OUTPUT_FORMAT = "ascii 1.0"
MULTISET_HASH_ALGORITHM = (
    "sha256(sorted six-tuples; little-endian int32 x,y,z,red,green,blue bytes)"
)


# PLY's canonical names are the first seven entries.  The fixed-width aliases
# occur in real-world PLY headers and are unambiguous, so accepting them keeps
# the parser useful without falling back to platform-dependent native types.
_PLY_DTYPES: dict[str, np.dtype] = {
    "char": np.dtype("<i1"),
    "int8": np.dtype("<i1"),
    "uchar": np.dtype("<u1"),
    "uint8": np.dtype("<u1"),
    "short": np.dtype("<i2"),
    "int16": np.dtype("<i2"),
    "ushort": np.dtype("<u2"),
    "uint16": np.dtype("<u2"),
    "int": np.dtype("<i4"),
    "int32": np.dtype("<i4"),
    "uint": np.dtype("<u4"),
    "uint32": np.dtype("<u4"),
    "float": np.dtype("<f4"),
    "float32": np.dtype("<f4"),
    "double": np.dtype("<f8"),
    "float64": np.dtype("<f8"),
    "long": np.dtype("<i8"),
    "int64": np.dtype("<i8"),
    "ulong": np.dtype("<u8"),
    "uint64": np.dtype("<u8"),
}
_INTEGER_TYPES = {
    name for name, dtype in _PLY_DTYPES.items() if dtype.kind in "iu"
}
_FLOAT_TYPES = {
    name for name, dtype in _PLY_DTYPES.items() if dtype.kind == "f"
}
_INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    name: (int(np.iinfo(dtype).min), int(np.iinfo(dtype).max))
    for name, dtype in _PLY_DTYPES.items()
    if dtype.kind in "iu"
}
_INTEGER_TOKEN = re.compile(r"^[+-]?[0-9]+$")
_DECIMAL_TOKEN = re.compile(
    r"^[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?$"
)


class PLYError(ValueError):
    """Raised when a PLY is not supported by the formal adapter."""


class PartitionProvenanceError(ValueError):
    """Raised when the pinned partition implementation cannot be verified."""


@dataclass(frozen=True)
class _Property:
    name: str
    type_name: str
    list_property: bool = False
    count_type: str | None = None
    item_type: str | None = None


@dataclass(frozen=True)
class _Element:
    name: str
    count: int
    properties: tuple[_Property, ...]


@dataclass(frozen=True)
class _Header:
    format_name: str
    format_version: str
    elements: tuple[_Element, ...]
    payload_offset: int

    @property
    def format_label(self) -> str:
        return f"{self.format_name} {self.format_version}"

    @property
    def vertex(self) -> _Element:
        vertices = [element for element in self.elements if element.name == "vertex"]
        if len(vertices) != 1:
            raise PLYError(
                "exactly one vertex element is required"
            )
        nonempty_extras = [
            element for element in self.elements
            if element.name != "vertex" and element.count != 0
        ]
        if nonempty_extras:
            raise PLYError(
                "non-empty non-vertex elements are unsupported because they "
                "change the PLY payload"
            )
        return vertices[0]


def _fail(path: Path, message: str) -> PLYError:
    return PLYError(f"{path}: {message}")


def _parse_count(token: str, path: Path, what: str) -> int:
    if not _INTEGER_TOKEN.fullmatch(token):
        raise _fail(path, f"invalid {what} count: {token!r}")
    value = int(token, 10)
    if value < 0:
        raise _fail(path, f"negative {what} count")
    return value


def _parse_header(handle: Any, path: Path) -> _Header:
    """Read and validate the PLY header, leaving the handle at its payload."""

    first = handle.readline()
    if not first:
        raise _fail(path, "empty file")
    try:
        first_text = first.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise _fail(path, "header is not ASCII") from exc
    if first_text != "ply":
        raise _fail(path, "missing PLY magic")

    format_name: str | None = None
    format_version: str | None = None
    elements: list[_Element] = []
    current_name: str | None = None
    current_count: int | None = None
    current_properties: list[_Property] = []
    saw_end_header = False

    def finish_element() -> None:
        nonlocal current_name, current_count, current_properties
        if current_name is not None:
            assert current_count is not None
            elements.append(
                _Element(current_name, current_count, tuple(current_properties))
            )
        current_name = None
        current_count = None
        current_properties = []

    while True:
        raw = handle.readline()
        if not raw:
            raise _fail(path, "missing end_header")
        try:
            line = raw.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise _fail(path, "header is not ASCII") from exc
        words = line.strip().split()
        if not words:
            # Blank header lines are harmless and do not affect the payload
            # offset, but any data before end_header is still rejected below.
            continue
        directive = words[0]
        if directive == "end_header":
            if len(words) != 1:
                raise _fail(path, "malformed end_header")
            finish_element()
            saw_end_header = True
            break
        if directive == "format":
            if len(words) != 3 or format_name is not None:
                raise _fail(path, "malformed or duplicate format declaration")
            if words[1] not in {"ascii", "binary_little_endian"}:
                if words[1] == "binary_big_endian":
                    raise _fail(path, "binary_big_endian is unsupported")
                raise _fail(path, f"unsupported PLY format: {words[1]!r}")
            if words[2] != "1.0":
                raise _fail(path, f"unsupported PLY version: {words[2]!r}")
            format_name, format_version = words[1], words[2]
            continue
        if directive in {"comment", "obj_info"}:
            continue
        if directive == "element":
            if len(words) != 3:
                raise _fail(path, "malformed element declaration")
            finish_element()
            current_name = words[1]
            current_count = _parse_count(words[2], path, "element")
            continue
        if directive == "property":
            if current_name is None:
                raise _fail(path, "property declared outside an element")
            if len(words) == 5 and words[1] == "list":
                count_type, item_type, name = words[2:]
                if count_type not in _PLY_DTYPES or _PLY_DTYPES[count_type].kind not in "iu":
                    raise _fail(path, f"unsupported list count type: {count_type!r}")
                if item_type not in _PLY_DTYPES:
                    raise _fail(path, f"unsupported list item type: {item_type!r}")
                if current_name == "vertex":
                    raise _fail(path, "vertex list properties are unsupported")
                current_properties.append(
                    _Property(
                        name=name,
                        type_name="list",
                        list_property=True,
                        count_type=count_type,
                        item_type=item_type,
                    )
                )
                continue
            if len(words) != 3:
                raise _fail(path, "malformed scalar property declaration")
            type_name, name = words[1:]
            if type_name not in _PLY_DTYPES:
                raise _fail(path, f"unsupported scalar property type: {type_name!r}")
            current_properties.append(_Property(name=name, type_name=type_name))
            continue
        raise _fail(path, f"unsupported header directive: {directive!r}")

    if not saw_end_header or format_name is None or format_version is None:
        raise _fail(path, "incomplete PLY header")
    header = _Header(format_name, format_version, tuple(elements), handle.tell())
    vertex = header.vertex  # Allows only payload-free zero-count extra elements.
    if not vertex.properties:
        raise _fail(path, "vertex element has no properties")
    names = [prop.name for prop in vertex.properties]
    if len(names) != len(set(names)):
        raise _fail(path, "duplicate vertex property name")
    missing = [name for name in REQUIRED_PROPERTIES if name not in names]
    if missing:
        raise _fail(path, f"missing required vertex properties: {', '.join(missing)}")
    if any(prop.list_property for prop in vertex.properties):
        raise _fail(path, "vertex list properties are unsupported")
    return header


def _schema(header: _Header) -> dict[str, Any]:
    vertex = header.vertex
    names = [prop.name for prop in vertex.properties]
    return {
        "vertex": {
            "count": vertex.count,
            "properties": [
                {"name": prop.name, "type": prop.type_name, "kind": "scalar"}
                for prop in vertex.properties
            ],
            "selected_indices": {
                name: names.index(name) for name in REQUIRED_PROPERTIES
            },
        },
        "required_properties": list(REQUIRED_PROPERTIES),
    }


def _parse_ascii_scalar(token: str, prop: _Property, path: Path) -> int | float | Decimal:
    if prop.list_property:
        raise _fail(path, "vertex list properties are unsupported")
    if prop.type_name in _INTEGER_TYPES:
        if not _INTEGER_TOKEN.fullmatch(token):
            raise _fail(path, f"invalid {prop.type_name} value: {token!r}")
        value = int(token, 10)
        lower, upper = _INTEGER_BOUNDS[prop.type_name]
        if value < lower or value > upper:
            raise _fail(path, f"{prop.name} is outside {prop.type_name} range")
        return value
    if prop.type_name not in _FLOAT_TYPES or not _DECIMAL_TOKEN.fullmatch(token):
        raise _fail(path, f"invalid {prop.type_name} value: {token!r}")
    try:
        exact = Decimal(token)
    except InvalidOperation as exc:
        raise _fail(path, f"invalid {prop.type_name} value: {token!r}") from exc
    if not exact.is_finite():
        raise _fail(path, f"non-finite {prop.name} value")
    try:
        value = float(exact)
    except OverflowError as exc:
        raise _fail(path, f"{prop.name} overflows {prop.type_name}") from exc
    if not np.isfinite(value):
        raise _fail(path, f"{prop.name} overflows {prop.type_name}")
    if prop.type_name in {"float", "float32"}:
        value32 = np.asarray(value, dtype="<f4")
        if not np.isfinite(value32):
            raise _fail(path, f"{prop.name} overflows float32")
    return exact


def _exact_integer(value: int | float | Decimal, path: Path, name: str, lower: int, upper: int) -> int:
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise _fail(path, f"{name} is not an exact integer")
        integer = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or value != np.rint(value):
            raise _fail(path, f"{name} is not an exact integer")
        integer = int(value)
    else:
        integer = int(value)
    if integer < lower or integer > upper:
        raise _fail(path, f"{name} is outside [{lower}, {upper}]")
    return integer


def _read_ascii(handle: Any, path: Path, header: _Header) -> tuple[np.ndarray, np.ndarray]:
    vertex = header.vertex
    names = [prop.name for prop in vertex.properties]
    selected = {name: names.index(name) for name in REQUIRED_PROPERTIES}
    coords = np.empty((vertex.count, 3), dtype=np.int32)
    rgb = np.empty((vertex.count, 3), dtype=np.uint8)

    for row_index in range(vertex.count):
        raw = handle.readline()
        if not raw:
            raise _fail(path, f"vertex payload ended at row {row_index}")
        try:
            tokens = raw.decode("ascii").split()
        except UnicodeDecodeError as exc:
            raise _fail(path, f"vertex row {row_index} is not ASCII") from exc
        if len(tokens) != len(vertex.properties):
            raise _fail(
                path,
                f"vertex row {row_index} has {len(tokens)} fields; "
                f"expected {len(vertex.properties)}",
            )
        parsed: list[int | float | Decimal] = [
            _parse_ascii_scalar(token, prop, path)
            for token, prop in zip(tokens, vertex.properties)
        ]
        coords[row_index, 0] = _exact_integer(
            parsed[selected["x"]], path, "x", np.iinfo(np.int32).min, np.iinfo(np.int32).max
        )
        coords[row_index, 1] = _exact_integer(
            parsed[selected["y"]], path, "y", np.iinfo(np.int32).min, np.iinfo(np.int32).max
        )
        coords[row_index, 2] = _exact_integer(
            parsed[selected["z"]], path, "z", np.iinfo(np.int32).min, np.iinfo(np.int32).max
        )
        rgb[row_index, 0] = _exact_integer(parsed[selected["red"]], path, "red", 0, 255)
        rgb[row_index, 1] = _exact_integer(parsed[selected["green"]], path, "green", 0, 255)
        rgb[row_index, 2] = _exact_integer(parsed[selected["blue"]], path, "blue", 0, 255)

    trailing = handle.read()
    if trailing.strip(b" \t\r\n"):
        raise _fail(path, "unexpected bytes after vertex payload")
    return coords, rgb


def _binary_dtype(vertex: _Element, path: Path) -> np.dtype:
    fields: list[tuple[str, np.dtype]] = []
    for prop in vertex.properties:
        if prop.list_property:
            raise _fail(path, "vertex list properties are unsupported")
        fields.append((prop.name, _PLY_DTYPES[prop.type_name]))
    return np.dtype(fields, align=False)


def _binary_exact_integer_array(
    values: np.ndarray,
    prop: _Property,
    path: Path,
    lower: int,
    upper: int,
    output_dtype: np.dtype,
) -> np.ndarray:
    values = values[prop.name]
    if prop.type_name in _FLOAT_TYPES:
        if not np.all(np.isfinite(values)):
            raise _fail(path, f"non-finite {prop.name} value")
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise _fail(path, f"{prop.name} is not an exact integer")
        values = rounded
    if np.any(values < lower) or np.any(values > upper):
        raise _fail(path, f"{prop.name} is outside [{lower}, {upper}]")
    return np.asarray(values, dtype=output_dtype).copy()


def _read_binary(handle: Any, path: Path, header: _Header) -> tuple[np.ndarray, np.ndarray]:
    vertex = header.vertex
    dtype = _binary_dtype(vertex, path)
    payload = handle.read()
    expected_bytes = vertex.count * dtype.itemsize
    if len(payload) != expected_bytes:
        raise _fail(
            path,
            f"binary vertex payload has {len(payload)} bytes; expected {expected_bytes}",
        )
    records = np.frombuffer(payload, dtype=dtype, count=vertex.count)
    # Validate every floating scalar, including unselected normals, so malformed
    # values cannot be silently hidden by name-based extraction.
    for prop in vertex.properties:
        if prop.type_name in _FLOAT_TYPES and not np.all(np.isfinite(records[prop.name])):
            raise _fail(path, f"non-finite {prop.name} value")
    names = [prop.name for prop in vertex.properties]
    prop_by_name = {prop.name: prop for prop in vertex.properties}
    coords = np.empty((vertex.count, 3), dtype=np.int32)
    rgb = np.empty((vertex.count, 3), dtype=np.uint8)
    for axis, name in enumerate(("x", "y", "z")):
        coords[:, axis] = _binary_exact_integer_array(
            records,
            prop_by_name[name],
            path,
            np.iinfo(np.int32).min,
            np.iinfo(np.int32).max,
            np.dtype("<i4"),
        )
    for channel, name in enumerate(("red", "green", "blue")):
        rgb[:, channel] = _binary_exact_integer_array(
            records, prop_by_name[name], path, 0, 255, np.dtype("<u1")
        )
    del names
    return coords, rgb


def read_ply_with_metadata(path: os.PathLike[str] | str) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Read named PLY vertex properties and return ``(xyz, rgb, format, schema)``."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as handle:
        header = _parse_header(handle, source)
        if header.format_name == "ascii":
            coords, rgb = _read_ascii(handle, source, header)
        elif header.format_name == "binary_little_endian":
            coords, rgb = _read_binary(handle, source, header)
        else:  # Defensive: _parse_header already rejects this.
            raise _fail(source, f"unsupported PLY format: {header.format_label}")
    return coords, rgb, header.format_label, _schema(header)


def read_ply(path: os.PathLike[str] | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a supported PLY and return exact ``int32`` coordinates and ``uint8`` RGB."""

    coords, rgb, _, _ = read_ply_with_metadata(path)
    return coords, rgb


# Explicit aliases make the small reader convenient to use from tests and
# downstream preflight scripts without exposing the private parser classes.
read_ply_vertices = read_ply
read_ply_scalar_properties = read_ply


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _function_body_dump(path: Path, function_name: str) -> str:
    """Return an AST-normalized top-level function body, without importing path."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PartitionProvenanceError(
            f"cannot parse partition source {path}: {exc}"
        ) from exc
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(functions) != 1:
        raise PartitionProvenanceError(
            f"partition source {path} must contain exactly one top-level "
            f"{function_name} function"
        )
    # Comments, whitespace, and line locations are intentionally omitted.  The
    # nested KD-tree helper remains in the body, so this still checks the full
    # executed split algorithm rather than only the outer signature.
    normalized = ast.dump(
        ast.Module(body=functions[0].body, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    return normalized


def _partition_provenance(
    repo_root: Path,
    partition_source: os.PathLike[str] | str | None,
    expected_partition_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    """Verify the pinned source and return the repository-local callable.

    The upstream file is parsed and hashed only; it is never imported.  This
    avoids pulling optional dependencies from an external checkout.  Execution
    is deliberately through the repository's pinned ``data_utils`` module
    after its function body has been compared with the supplied source.
    """

    if partition_source is None:
        raise PartitionProvenanceError(
            "partition_source is required for formal preprocessing"
        )
    source_path = Path(partition_source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    expected = str(expected_partition_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PartitionProvenanceError(
            "expected_partition_sha256 must be a 64-character hexadecimal SHA-256"
        )
    actual_hash = _sha256_file(source_path)
    if actual_hash != expected:
        raise PartitionProvenanceError(
            f"partition source SHA-256 mismatch for {source_path}: "
            f"expected {expected}, got {actual_hash}"
        )

    repo_root = repo_root.resolve()
    repo_root_text = str(repo_root)
    if repo_root_text in sys.path:
        sys.path.remove(repo_root_text)
    sys.path.insert(0, repo_root_text)
    try:
        from data_utils.attribute import partition as local_partition
    except ImportError as exc:
        raise PartitionProvenanceError(
            "cannot import repository data_utils.attribute.partition"
        ) from exc
    partition_function = getattr(local_partition, "kdtree_partition", None)
    if not callable(partition_function):
        raise PartitionProvenanceError(
            "repository data_utils.attribute.partition has no kdtree_partition"
        )
    local_source_value = inspect.getsourcefile(partition_function) or getattr(
        local_partition, "__file__", None
    )
    if local_source_value is None:
        raise PartitionProvenanceError("cannot locate executed partition source")
    local_source_path = Path(local_source_value).resolve()
    if not local_source_path.is_file():
        raise PartitionProvenanceError(
            f"executed partition source is not a file: {local_source_path}"
        )
    try:
        local_source_path.relative_to(repo_root)
    except ValueError as exc:
        raise PartitionProvenanceError(
            "executed partition implementation is outside the repository: "
            f"{local_source_path}"
        ) from exc
    upstream_body = _function_body_dump(source_path, "kdtree_partition")
    local_body = _function_body_dump(local_source_path, "kdtree_partition")
    upstream_body_hash = hashlib.sha256(upstream_body.encode("utf-8")).hexdigest()
    local_body_hash = hashlib.sha256(local_body.encode("utf-8")).hexdigest()
    body_equivalent = upstream_body == local_body
    if not body_equivalent:
        raise PartitionProvenanceError(
            "partition kdtree_partition function body is not equivalent to "
            f"the repository implementation: {source_path} vs {local_source_path}"
        )
    return partition_function, {
        "source_path": str(source_path),
        "source_sha256": actual_hash,
        "expected_sha256": expected,
        "sha256_verified": True,
        "function": "kdtree_partition",
        "executed_algorithm": "data_utils.attribute.partition.kdtree_partition",
        "executed_source_path": str(local_source_path),
        "executed_source_sha256": _sha256_file(local_source_path),
        "source_function_body_sha256": upstream_body_hash,
        "executed_function_body_sha256": local_body_hash,
        "body_equivalent": True,
    }


def _multiset_sha256(coords: np.ndarray, rgb: np.ndarray) -> str:
    """Hash sorted six-tuples, retaining duplicates and ignoring row order."""

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) != len(coords):
        raise ValueError("RGB must have shape (N, 3) matching coordinates")
    rows = np.empty(
        len(coords),
        dtype=np.dtype(
            [
                ("x", "<i4"),
                ("y", "<i4"),
                ("z", "<i4"),
                ("red", "<i4"),
                ("green", "<i4"),
                ("blue", "<i4"),
            ],
            align=False,
        ),
    )
    rows["x"] = np.asarray(coords[:, 0], dtype="<i4")
    rows["y"] = np.asarray(coords[:, 1], dtype="<i4")
    rows["z"] = np.asarray(coords[:, 2], dtype="<i4")
    rows["red"] = np.asarray(rgb[:, 0], dtype="<i4")
    rows["green"] = np.asarray(rgb[:, 1], dtype="<i4")
    rows["blue"] = np.asarray(rgb[:, 2], dtype="<i4")
    rows.sort(order=("x", "y", "z", "red", "green", "blue"), kind="quicksort")
    return hashlib.sha256(rows.tobytes(order="C")).hexdigest()


def _safe_stem(path: Path) -> str:
    stem = path.stem or "source"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return safe or "source"


def _chunk_relative_path(output_root: Path, source_index: int, source_path: Path, order: int) -> Path:
    return Path("chunks") / f"{source_index:04d}_{_safe_stem(source_path)}" / f"chunk_{order:06d}.ply"


def _write_ascii_chunk(path: Path, coords: np.ndarray, rgb: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if coords.dtype != np.int32 or rgb.dtype != np.uint8:
        raise TypeError("chunk arrays must be int32 coordinates and uint8 RGB")
    if coords.shape != (len(coords), 3) or rgb.shape != (len(coords), 3):
        raise ValueError("chunk arrays must have shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(coords)}\n"
        # MPEG pc_error 0.13.4 rejects ASCII PLY references declaring integer
        # XYZ properties.  Declare the exact integer-valued coordinates as
        # float, matching the official 8i/Owlii PLY convention; the payload
        # tokens and global coordinate values remain unchanged.
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("xb") as handle:
        handle.write(header)
        for point, color in zip(coords, rgb):
            handle.write(
                (
                    f"{int(point[0])} {int(point[1])} {int(point[2])} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                ).encode("ascii")
            )


def _read_input_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid input manifest: {path}") from exc
        entries: Any = payload
        if isinstance(payload, Mapping):
            for key in ("sources", "inputs", "files", "paths"):
                if key in payload:
                    entries = payload[key]
                    break
            else:
                raise ValueError(f"input JSON manifest has no sources list: {path}")
        if not isinstance(entries, list):
            raise ValueError(f"input JSON manifest list is invalid: {path}")
        result: list[Path] = []
        for entry in entries:
            if isinstance(entry, str):
                value = Path(entry)
                result.append(value if value.is_absolute() else path.parent / value)
            elif isinstance(entry, Mapping):
                value = next(
                    (entry[key] for key in ("source_path", "path", "file") if key in entry),
                    None,
                )
                if not isinstance(value, str):
                    raise ValueError(f"invalid input manifest entry: {entry!r}")
                value_path = Path(value)
                result.append(
                    value_path if value_path.is_absolute() else path.parent / value_path
                )
            else:
                raise ValueError(f"invalid input manifest entry: {entry!r}")
        return result
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Text manifests in this repository are one path per line.  A TSV
        # source list is also unambiguous when the first field is the path.
        value = Path(stripped.split("\t", 1)[0])
        result.append(value if value.is_absolute() else path.parent / value)
    return result


def _normalise_inputs(
    input_path: os.PathLike[str] | str | Sequence[os.PathLike[str] | str] | None,
    input_manifest: os.PathLike[str] | str | None,
    input_paths: Sequence[os.PathLike[str] | str] | None,
) -> list[Path]:
    supplied = sum(value is not None for value in (input_path, input_manifest, input_paths))
    if supplied != 1:
        raise ValueError("provide exactly one of input_path, input_manifest, or input_paths")
    if input_manifest is not None:
        paths = _read_input_manifest(Path(input_manifest))
    else:
        values: Any = input_paths if input_paths is not None else input_path
        if isinstance(values, (str, os.PathLike)):
            paths = [Path(values)]
        else:
            paths = [Path(value) for value in values]
    if not paths:
        raise ValueError("input list is empty")
    return paths


def _resolve_manifest_paths(
    output_root: Path,
    manifest_json: os.PathLike[str] | str | None,
    manifest_tsv: os.PathLike[str] | str | None,
    output_manifest: os.PathLike[str] | str | None,
) -> tuple[Path, Path]:
    if output_manifest is not None:
        if manifest_json is not None or manifest_tsv is not None:
            raise ValueError("output_manifest cannot be combined with manifest_json/manifest_tsv")
        output_manifest_path = Path(output_manifest)
        if output_manifest_path.suffix.lower() == ".tsv":
            manifest_tsv = output_manifest_path
            manifest_json = output_manifest_path.with_suffix(".json")
        else:
            manifest_json = output_manifest_path
            manifest_tsv = output_manifest_path.with_suffix(".tsv")
    else:
        manifest_json = Path(manifest_json) if manifest_json is not None else output_root / "manifest.json"
        manifest_tsv = Path(manifest_tsv) if manifest_tsv is not None else output_root / "manifest.tsv"
    return Path(manifest_json), Path(manifest_tsv)


def _check_distinct_paths(paths: Iterable[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            raise ValueError(f"manifest and chunk paths collide: {path}")
        seen.add(resolved)


def _check_new_paths(paths: Iterable[Path]) -> None:
    paths = tuple(paths)
    _check_distinct_paths(paths)
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")


def _union_record(
    source_path: Path,
    source_coords: np.ndarray,
    source_rgb: np.ndarray,
    chunk_coords: Sequence[np.ndarray],
    chunk_rgb: Sequence[np.ndarray],
) -> dict[str, Any]:
    source_hash = _multiset_sha256(source_coords, source_rgb)
    if chunk_coords:
        joined_coords = np.concatenate(chunk_coords, axis=0)
        joined_rgb = np.concatenate(chunk_rgb, axis=0)
    else:
        joined_coords = np.empty((0, 3), dtype=np.int32)
        joined_rgb = np.empty((0, 3), dtype=np.uint8)
    chunks_hash = _multiset_sha256(joined_coords, joined_rgb)
    source_count = int(len(source_coords))
    chunks_count = int(len(joined_coords))
    equal = source_count == chunks_count and source_hash == chunks_hash
    result = {
        "algorithm": MULTISET_HASH_ALGORITHM,
        "source_multiset_sha256": source_hash,
        "chunks_multiset_sha256": chunks_hash,
        "source_count": source_count,
        "chunks_count": chunks_count,
        "verified": equal,
        "source_path": str(source_path),
    }
    if not equal:
        raise RuntimeError(f"union verification failed for {source_path}: {result}")
    return result


def _write_json(path: Path, manifest: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


_TSV_FIELDS = (
    "source_order",
    "source_path",
    "source_sha256",
    "source_format",
    "source_schema_json",
    "source_point_count",
    "chunk_order",
    "chunk_path",
    "chunk_sha256",
    "chunk_count",
    "max_num",
    "partition_source_path",
    "partition_source_sha256",
    "partition_expected_sha256",
    "partition_executed_algorithm",
    "partition_sha256_verified",
    "partition_body_equivalent",
    "global_coordinates",
    "no_recenter",
    "union_verified",
    "source_multiset_sha256",
    "chunks_multiset_sha256",
)


def _write_tsv(
    path: Path,
    sources: Sequence[Mapping[str, Any]],
    max_num: int,
    partition: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_tsv(sources, max_num, partition), encoding="utf-8")


def _render_tsv(
    sources: Sequence[Mapping[str, Any]],
    max_num: int,
    partition: Mapping[str, Any],
) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=_TSV_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for source_order, source in enumerate(sources):
        union = source["union_verification"]
        chunks = source["chunks"]
        for chunk in chunks:
            writer.writerow(
                {
                        "source_order": source_order,
                        "source_path": source["source_path"],
                        "source_sha256": source["source_sha256"],
                        "source_format": source["source_format"],
                        "source_schema_json": json.dumps(source["source_schema"], sort_keys=True, separators=(",", ":")),
                        "source_point_count": source["source_point_count"],
                        "chunk_order": chunk["order"],
                        "chunk_path": chunk["path"],
                        "chunk_sha256": chunk["sha256"],
                        "chunk_count": chunk["count"],
                        "max_num": max_num,
                        "partition_source_path": partition["source_path"],
                        "partition_source_sha256": partition["source_sha256"],
                        "partition_expected_sha256": partition["expected_sha256"],
                        "partition_executed_algorithm": partition["executed_algorithm"],
                        "partition_sha256_verified": partition["sha256_verified"],
                        "partition_body_equivalent": partition["body_equivalent"],
                        "global_coordinates": True,
                        "no_recenter": True,
                        "union_verified": union["verified"],
                        "source_multiset_sha256": union["source_multiset_sha256"],
                        "chunks_multiset_sha256": union["chunks_multiset_sha256"],
                }
            )
    return handle.getvalue()


def _verify_existing_json(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"existing manifest JSON is unreadable: {path}: {exc}") from exc
    if actual != expected:
        raise RuntimeError(f"existing manifest JSON identity mismatch: {path}")


def _verify_existing_tsv(path: Path, expected: str) -> None:
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"existing manifest TSV is unreadable: {path}: {exc}") from exc
    if actual != expected:
        raise RuntimeError(f"existing manifest TSV identity mismatch: {path}")


def prepare_ctc_formal_chunks(
    input_path: os.PathLike[str] | str | Sequence[os.PathLike[str] | str] | None = None,
    output_root: os.PathLike[str] | str | None = None,
    *,
    input_manifest: os.PathLike[str] | str | None = None,
    input_paths: Sequence[os.PathLike[str] | str] | None = None,
    max_num: int = DEFAULT_MAX_NUM,
    partition_source: os.PathLike[str] | str | None = None,
    expected_partition_sha256: str = EXPECTED_PARTITION_SHA256,
    manifest_json: os.PathLike[str] | str | None = None,
    manifest_tsv: os.PathLike[str] | str | None = None,
    output_manifest: os.PathLike[str] | str | None = None,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Convert one PLY or a manifest list into deterministic formal chunks.

    The returned dictionary is the exact JSON manifest written to disk.  The
    output always has a ``sources`` list, even for one input, so consumers do
    not need a single-input special case.
    """

    if output_root is None:
        raise ValueError("output_root is required")
    if isinstance(max_num, bool) or not isinstance(max_num, (int, np.integer)) or int(max_num) <= 0:
        raise ValueError("max_num must be a positive integer")
    max_num = int(max_num)
    source_paths = _normalise_inputs(input_path, input_manifest, input_paths)
    output_root_path = Path(output_root)
    json_path, tsv_path = _resolve_manifest_paths(
        output_root_path, manifest_json, manifest_tsv, output_manifest
    )
    # Default behavior remains create-only. Explicit reuse validates complete
    # manifests and chunks rather than treating existence as sufficient.
    if reuse_existing:
        _check_distinct_paths((json_path, tsv_path))
    else:
        _check_new_paths((json_path, tsv_path))

    # Verify the pinned source without importing an external checkout, then
    # execute the repository-local function whose complete body was compared.
    repo_root = Path(__file__).resolve().parents[3]
    kdtree_partition, partition_record = _partition_provenance(
        repo_root, partition_source, expected_partition_sha256
    )

    source_records: list[dict[str, Any]] = []
    for source_order, source_input in enumerate(source_paths):
        source_path = source_input.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha256 = _sha256_file(source_path)
        coords, rgb, source_format, source_schema = read_ply_with_metadata(source_path)
        points = np.hstack((coords, rgb.astype(np.int32, copy=False)))
        if len(points):
            parts = kdtree_partition(points, max_num=max_num)
        else:
            parts = []
        target_paths = [
            output_root_path / _chunk_relative_path(output_root_path, source_order, source_path, order)
            for order in range(len(parts))
        ]
        all_output_paths = (*target_paths, json_path, tsv_path)
        if reuse_existing:
            _check_distinct_paths(all_output_paths)
        else:
            _check_new_paths(all_output_paths)

        chunk_records: list[dict[str, Any]] = []
        readback_coords: list[np.ndarray] = []
        readback_rgb: list[np.ndarray] = []
        for order, (part, target_path) in enumerate(zip(parts, target_paths)):
            part = np.asarray(part)
            if part.ndim != 2 or part.shape[1] != 6:
                raise RuntimeError(f"partition returned invalid shape for {source_path}: {part.shape}")
            if len(part) > max_num:
                raise RuntimeError(
                    f"partition exceeds max_num for {source_path}: {len(part)} > {max_num}"
                )
            part_coords = np.asarray(part[:, :3], dtype=np.int32).copy()
            part_rgb = np.asarray(part[:, 3:], dtype=np.uint8).copy()
            was_existing = target_path.exists()
            if not was_existing:
                _write_ascii_chunk(target_path, part_coords, part_rgb)
            chunk_hash = _sha256_file(target_path)
            decoded_coords, decoded_rgb, decoded_format, decoded_schema = read_ply_with_metadata(target_path)
            if decoded_format != OUTPUT_FORMAT or decoded_schema["vertex"]["properties"] != [
                {"name": "x", "type": "float", "kind": "scalar"},
                {"name": "y", "type": "float", "kind": "scalar"},
                {"name": "z", "type": "float", "kind": "scalar"},
                {"name": "red", "type": "uchar", "kind": "scalar"},
                {"name": "green", "type": "uchar", "kind": "scalar"},
                {"name": "blue", "type": "uchar", "kind": "scalar"},
            ]:
                raise RuntimeError(f"chunk schema readback mismatch: {target_path}")
            if not np.array_equal(decoded_coords, part_coords) or not np.array_equal(decoded_rgb, part_rgb):
                qualifier = "existing " if reuse_existing and was_existing else ""
                raise RuntimeError(f"{qualifier}chunk value readback mismatch: {target_path}")
            readback_coords.append(decoded_coords)
            readback_rgb.append(decoded_rgb)
            relative_path = target_path.relative_to(output_root_path).as_posix()
            chunk_records.append(
                {
                    "order": order,
                    "path": relative_path,
                    "sha256": chunk_hash,
                    "count": int(len(decoded_coords)),
                }
            )

        union = _union_record(source_path, coords, rgb, readback_coords, readback_rgb)
        source_record: dict[str, Any] = {
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "source_format": source_format,
            "source_schema": source_schema,
            "source_point_count": int(len(coords)),
            "chunks": chunk_records,
            "union_verification": union,
        }
        source_records.append(source_record)

    overall_verified = all(
        bool(source["union_verification"]["verified"]) for source in source_records
    )
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "tool": "prepare_ctc_formal_chunks",
        "max_num": max_num,
        "global_coordinates": True,
        "no_recenter": True,
        "recenter": False,
        "quantize": False,
        "deduplicate": False,
        "partition_provenance": partition_record,
        "sources": source_records,
        "source_count": len(source_records),
        "union_verification": {
            "algorithm": MULTISET_HASH_ALGORITHM,
            "verified": overall_verified,
            "per_source": [source["union_verification"] for source in source_records],
        },
        "manifest_json": str(json_path.absolute()),
        "manifest_tsv": str(tsv_path.absolute()),
    }
    tsv_content = _render_tsv(source_records, max_num, partition_record)
    if reuse_existing and json_path.exists():
        _verify_existing_json(json_path, manifest)
    else:
        _write_json(json_path, manifest)
    if reuse_existing and tsv_path.exists():
        _verify_existing_tsv(tsv_path, tsv_content)
    else:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with tsv_path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(tsv_content)
    return manifest


# Short alias for callers that prefer a verb-like API.
prepare = prepare_ctc_formal_chunks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare lossless global-coordinate CTC formal PLY chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "--input-path", nargs="+", help="one or more input PLY paths")
    group.add_argument(
        "--input-manifest",
        "--manifest-list",
        "--input-list",
        help="text or JSON list of input PLY paths",
    )
    parser.add_argument("--output-root", "--output-dir", required=True)
    parser.add_argument(
        "--output-manifest",
        help="JSON or TSV manifest path; the sibling format is emitted too",
    )
    parser.add_argument("--manifest-json", help="explicit JSON manifest path")
    parser.add_argument("--manifest-tsv", help="explicit TSV manifest path")
    parser.add_argument(
        "--partition-source",
        required=True,
        help="pinned upstream partition.py used for provenance verification",
    )
    parser.add_argument(
        "--expected-partition-sha256",
        "--partition-sha256",
        dest="expected_partition_sha256",
        default=EXPECTED_PARTITION_SHA256,
        help="expected SHA-256 of --partition-source",
    )
    parser.add_argument(
        "--max-num",
        "--max_num",
        "--max-points",
        type=int,
        default=DEFAULT_MAX_NUM,
        help="KD-tree maximum chunk size; formal default is 800000",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=("verify and reuse existing canonical chunk files; identity "
              "mismatches fail without overwriting"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = prepare_ctc_formal_chunks(
        input_path=args.input,
        input_manifest=args.input_manifest,
        output_root=args.output_root,
        max_num=args.max_num,
        partition_source=args.partition_source,
        expected_partition_sha256=args.expected_partition_sha256,
        manifest_json=args.manifest_json,
        manifest_tsv=args.manifest_tsv,
        output_manifest=args.output_manifest,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
