from __future__ import annotations

import csv
import hashlib
import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest


MODULE = importlib.import_module(
    "scripts.scalable_attribute.evaluation.prepare_ctc_formal_chunks"
)
PARTITION_SOURCE = Path(__file__).parents[1] / "data_utils/attribute/partition.py"
PARTITION_SHA256 = hashlib.sha256(PARTITION_SOURCE.read_bytes()).hexdigest()


def _partition_kwargs() -> dict[str, object]:
    return {
        "partition_source": PARTITION_SOURCE,
        "expected_partition_sha256": PARTITION_SHA256,
    }


def _write_ply(
    path: Path,
    properties: list[tuple[str, str]],
    rows: list[tuple[object, ...]],
    *,
    format_name: str = "ascii",
    extra_header: list[str] | None = None,
) -> None:
    header = [
        "ply",
        f"format {format_name} 1.0",
        f"element vertex {len(rows)}",
    ]
    header.extend(f"property {type_name} {name}" for name, type_name in properties)
    if extra_header:
        header.extend(extra_header)
    header.extend(["end_header"])
    header_bytes = ("\n".join(header) + "\n").encode("ascii")
    if format_name == "ascii":
        payload = "".join(" ".join(str(value) for value in row) + "\n" for row in rows)
        path.write_bytes(header_bytes + payload.encode("ascii"))
        return

    formats = {
        "char": "b",
        "uchar": "B",
        "short": "h",
        "ushort": "H",
        "int": "i",
        "uint": "I",
        "float": "f",
        "double": "d",
    }
    try:
        payload_format = "<" + "".join(formats[type_name] for _, type_name in properties)
    except KeyError as exc:
        raise AssertionError(f"fixture type is not supported by helper: {exc}") from exc
    payload = b"".join(struct.pack(payload_format, *row) for row in rows)
    path.write_bytes(header_bytes + payload)


def _normal_before_rgb_properties() -> list[tuple[str, str]]:
    return [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]


def _rgb_before_normal_properties() -> list[tuple[str, str]]:
    return [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
    ]


def test_named_rgb_extraction_accepts_both_thaidancer_property_orders(tmp_path: Path) -> None:
    expected_coords = np.array([[100, -20, 7], [101, -19, 8]], dtype=np.int32)
    expected_rgb = np.array([[1, 2, 255], [250, 128, 3]], dtype=np.uint8)
    rows_rgb_first = [
        (100, -20, 7, 1, 2, 255, 0.0, 1.0, 0.0),
        (101, -19, 8, 250, 128, 3, 0.0, 0.0, 1.0),
    ]
    rows_normal_first = [
        (100, -20, 7, 0.0, 1.0, 0.0, 1, 2, 255),
        (101, -19, 8, 0.0, 0.0, 1.0, 250, 128, 3),
    ]
    rgb_first = tmp_path / "Thaidancer_rgb_before_normal.ply"
    normal_first = tmp_path / "other_normal_before_rgb.ply"
    _write_ply(rgb_first, _rgb_before_normal_properties(), rows_rgb_first)
    _write_ply(normal_first, _normal_before_rgb_properties(), rows_normal_first)

    for path in (rgb_first, normal_first):
        coords, rgb, format_name, schema = MODULE.read_ply_with_metadata(path)
        np.testing.assert_array_equal(coords, expected_coords)
        np.testing.assert_array_equal(rgb, expected_rgb)
        assert coords.dtype == np.int32
        assert rgb.dtype == np.uint8
        assert format_name == "ascii 1.0"
        assert schema["vertex"]["selected_indices"] == {
            "x": 0,
            "y": 1,
            "z": 2,
            "red": 3 if path == rgb_first else 6,
            "green": 4 if path == rgb_first else 7,
            "blue": 5 if path == rgb_first else 8,
        }


def test_binary_little_endian_named_extraction(tmp_path: Path) -> None:
    cases = [
        (
            "normal_before_rgb_binary.ply",
            _normal_before_rgb_properties(),
            [(12, -3, 4096, -1.0, 0.0, 1.0, 7, 8, 9), (-8, 4, 5, 0.0, 1.0, 0.0, 250, 251, 252)],
        ),
        (
            "rgb_before_normal_binary.ply",
            _rgb_before_normal_properties(),
            [(12, -3, 4096, 7, 8, 9, -1.0, 0.0, 1.0), (-8, 4, 5, 250, 251, 252, 0.0, 1.0, 0.0)],
        ),
    ]
    for name, properties, rows in cases:
        path = tmp_path / name
        _write_ply(path, properties, rows, format_name="binary_little_endian")
        coords, rgb = MODULE.read_ply(path)
        np.testing.assert_array_equal(coords, [[12, -3, 4096], [-8, 4, 5]])
        np.testing.assert_array_equal(rgb, [[7, 8, 9], [250, 251, 252]])
        assert coords.dtype == np.int32
        assert rgb.dtype == np.uint8


def test_prepare_preserves_global_coordinates_and_verifies_multiset(tmp_path: Path) -> None:
    source = tmp_path / "global_coordinates.ply"
    properties = [
        ("x", "int"),
        ("y", "int"),
        ("z", "int"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]
    # The first and last rows are identical on purpose: count-only validation
    # would not prove that this duplicate survived chunking.
    rows = [
        (-100, 20, 3000, 4, 5, 6),
        (11, -12, 13, 14, 15, 16),
        (700, 701, -702, 200, 201, 202),
        (-100, 20, 3000, 4, 5, 6),
        (99, 98, 97, 250, 0, 1),
    ]
    _write_ply(source, properties, rows)
    output_root = tmp_path / "prepared"

    manifest = MODULE.prepare_ctc_formal_chunks(
        source, output_root, max_num=2, **_partition_kwargs()
    )
    first_chunk = output_root / manifest["sources"][0]["chunks"][0]["path"]
    header = first_chunk.read_text(encoding="ascii").split("end_header", 1)[0]
    assert "property float x" in header
    assert "property float y" in header
    assert "property float z" in header
    assert manifest["max_num"] == 2
    assert manifest["global_coordinates"] is True
    assert manifest["no_recenter"] is True
    assert manifest["recenter"] is False
    assert manifest["quantize"] is False
    assert manifest["deduplicate"] is False
    assert manifest["union_verification"]["verified"] is True
    assert manifest["partition_provenance"]["source_path"] == str(PARTITION_SOURCE.resolve())
    assert manifest["partition_provenance"]["source_sha256"] == PARTITION_SHA256
    assert manifest["partition_provenance"]["sha256_verified"] is True
    assert manifest["partition_provenance"]["body_equivalent"] is True

    source_record = manifest["sources"][0]
    assert source_record["source_point_count"] == len(rows)
    assert source_record["source_format"] == "ascii 1.0"
    assert source_record["source_schema"]["required_properties"] == list(MODULE.REQUIRED_PROPERTIES)
    chunks = source_record["chunks"]
    assert [chunk["order"] for chunk in chunks] == list(range(len(chunks)))
    assert len(chunks) >= 2
    assert all(chunk["count"] <= 2 for chunk in chunks)
    assert all((output_root / chunk["path"]).is_file() for chunk in chunks)

    decoded = [MODULE.read_ply(output_root / chunk["path"]) for chunk in chunks]
    decoded_coords = np.concatenate([item[0] for item in decoded], axis=0)
    decoded_rgb = np.concatenate([item[1] for item in decoded], axis=0)
    source_coords, source_rgb = MODULE.read_ply(source)
    assert MODULE._multiset_sha256(decoded_coords, decoded_rgb) == MODULE._multiset_sha256(
        source_coords, source_rgb
    )
    # The global values, including the negative and large coordinates, are
    # present exactly; no per-chunk minimum subtraction occurred.
    assert {tuple(row) for row in decoded_coords} == {tuple(row) for row in source_coords}
    assert sum(chunk["count"] for chunk in chunks) == len(rows)
    assert all(chunk["sha256"] == hashlib.sha256((output_root / chunk["path"]).read_bytes()).hexdigest() for chunk in chunks)

    manifest_json = output_root / "manifest.json"
    manifest_tsv = output_root / "manifest.tsv"
    assert json.loads(manifest_json.read_text(encoding="utf-8")) == manifest
    with manifest_tsv.open(newline="", encoding="utf-8") as handle:
        tsv_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(row["chunk_order"]) for row in tsv_rows] == list(range(len(chunks)))
    assert {row["union_verified"] for row in tsv_rows} == {"True"}


def test_manifest_list_has_stable_sources_shape(tmp_path: Path) -> None:
    properties = [
        ("x", "int"),
        ("y", "int"),
        ("z", "int"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]
    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"
    _write_ply(first, properties, [(1, 2, 3, 4, 5, 6)])
    _write_ply(second, properties, [(7, 8, 9, 10, 11, 12)])
    input_list = tmp_path / "inputs.txt"
    input_list.write_text(f"{first}\n{second}\n", encoding="utf-8")

    manifest = MODULE.prepare_ctc_formal_chunks(
        input_manifest=input_list,
        output_root=tmp_path / "many",
        **_partition_kwargs(),
    )
    assert len(manifest["sources"]) == 2
    assert [record["source_path"] for record in manifest["sources"]] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert manifest["max_num"] == MODULE.DEFAULT_MAX_NUM
    assert all(record["union_verification"]["verified"] for record in manifest["sources"])


@pytest.mark.parametrize(
    ("name", "format_name", "properties", "rows", "extra_header"),
    [
        (
            "missing_blue.ply",
            "ascii",
            [("x", "int"), ("y", "int"), ("z", "int"), ("red", "uchar"), ("green", "uchar")],
            [(1, 2, 3, 4, 5)],
            None,
        ),
        (
            "vertex_list.ply",
            "ascii",
            [("x", "int"), ("y", "int"), ("z", "int"), ("red", "uchar"), ("green", "uchar"), ("blue", "uchar")],
            [(1, 2, 3, 4, 5, 6)],
            ["property list uchar int ignored"],
        ),
        (
            "extra_element.ply",
            "ascii",
            [("x", "int"), ("y", "int"), ("z", "int"), ("red", "uchar"), ("green", "uchar"), ("blue", "uchar")],
            [(1, 2, 3, 4, 5, 6)],
            ["element face 1", "property list uchar int vertex_indices"],
        ),
        (
            "fractional_coordinate.ply",
            "ascii",
            [("x", "float"), ("y", "float"), ("z", "float"), ("red", "uchar"), ("green", "uchar"), ("blue", "uchar")],
            [(1.5, 2.0, 3.0, 4, 5, 6)],
            None,
        ),
    ],
)
def test_invalid_vertex_schema_fails_closed(
    tmp_path: Path,
    name: str,
    format_name: str,
    properties: list[tuple[str, str]],
    rows: list[tuple[object, ...]],
    extra_header: list[str] | None,
) -> None:
    path = tmp_path / name
    _write_ply(path, properties, rows, format_name=format_name, extra_header=extra_header)
    with pytest.raises((MODULE.PLYError, ValueError)):
        MODULE.read_ply(path)


def test_zero_count_face_element_is_payload_free_and_accepted(tmp_path: Path) -> None:
    path = tmp_path / "zero_faces.ply"
    properties = [
        ("x", "float"), ("y", "float"), ("z", "float"),
        ("red", "uchar"), ("green", "uchar"), ("blue", "uchar"),
    ]
    _write_ply(
        path, properties, [(1, 2, 3, 4, 5, 6)],
        extra_header=["element face 0", "property list uchar int vertex_indices"],
    )
    coords, rgb = MODULE.read_ply(path)
    assert coords.tolist() == [[1, 2, 3]]
    assert rgb.tolist() == [[4, 5, 6]]


def test_big_endian_is_rejected_before_payload_decode(tmp_path: Path) -> None:
    path = tmp_path / "big_endian.ply"
    properties = [
        ("x", "int"),
        ("y", "int"),
        ("z", "int"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]
    _write_ply(path, properties, [(1, 2, 3, 4, 5, 6)], format_name="binary_big_endian")
    with pytest.raises(MODULE.PLYError, match="big_endian"):
        MODULE.read_ply(path)


def test_prepare_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    properties = [
        ("x", "int"),
        ("y", "int"),
        ("z", "int"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]
    _write_ply(source, properties, [(1, 2, 3, 4, 5, 6)])
    output_root = tmp_path / "out"
    MODULE.prepare_ctc_formal_chunks(source, output_root, **_partition_kwargs())
    before = {
        path: path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        MODULE.prepare_ctc_formal_chunks(source, output_root, **_partition_kwargs())
    after = {
        path: path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_library_requires_partition_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    properties = [
        ("x", "int"),
        ("y", "int"),
        ("z", "int"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]
    _write_ply(source, properties, [(1, 2, 3, 4, 5, 6)])
    with pytest.raises(MODULE.PartitionProvenanceError, match="partition_source"):
        MODULE.prepare_ctc_formal_chunks(source, tmp_path / "without_provenance")


def test_partition_provenance_rejects_wrong_file_hash() -> None:
    with pytest.raises(MODULE.PartitionProvenanceError, match="SHA-256 mismatch"):
        MODULE._partition_provenance(
            PARTITION_SOURCE.parents[2], PARTITION_SOURCE, "0" * 64
        )


def test_partition_provenance_rejects_non_equivalent_body(tmp_path: Path) -> None:
    altered = tmp_path / "partition.py"
    altered.write_text(
        PARTITION_SOURCE.read_text(encoding="utf-8").replace(
            "return parts", "return list(reversed(parts))", 1
        ),
        encoding="utf-8",
    )
    altered_hash = hashlib.sha256(altered.read_bytes()).hexdigest()
    with pytest.raises(MODULE.PartitionProvenanceError, match="not equivalent"):
        MODULE._partition_provenance(
            PARTITION_SOURCE.parents[2], altered, altered_hash
        )


def test_cli_requires_partition_source() -> None:
    with pytest.raises(SystemExit):
        MODULE._build_parser().parse_args(
            ["--input", "source.ply", "--output-root", "out"]
        )
