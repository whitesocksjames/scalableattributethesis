from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


MODULE = importlib.import_module(
    "scripts.scalable_attribute.evaluation.aggregate_formal_chunk_results"
)


PROPERTIES = [
    ("x", "int"),
    ("y", "int"),
    ("z", "int"),
    ("red", "uchar"),
    ("green", "uchar"),
    ("blue", "uchar"),
]


def _write_ply(path: Path, rows: list[tuple[int, int, int, int, int, int]]) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(rows)}",
        *(f"property {kind} {name}" for name, kind in PROPERTIES),
        "end_header",
    ]
    payload = "\n".join(" ".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(header) + "\n" + payload + "\n", encoding="ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, list[Path], list[Path]]:
    source = tmp_path / "source.ply"
    source_rows = [
        (0, 0, 0, 10, 20, 30),
        (1, 1, 1, 40, 50, 60),
        (2, 2, 2, 70, 80, 90),
        (3, 3, 3, 100, 110, 120),
    ]
    _write_ply(source, source_rows)

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk_rows = [source_rows[:2], source_rows[2:]]
    chunks: list[Path] = []
    for order, rows in enumerate(chunk_rows):
        path = chunks_dir / f"chunk_{order:06d}.ply"
        _write_ply(path, rows)
        chunks.append(path)
    manifest = tmp_path / "chunk_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "sources": [
                    {
                        "source_path": str(source),
                        "source_sha256": _sha256(source),
                        "source_point_count": len(source_rows),
                        "chunks": [
                            {
                                "order": order,
                                "path": str(path),
                                "sha256": _sha256(path),
                                "count": len(rows),
                            }
                            for order, (path, rows) in enumerate(zip(chunks, chunk_rows))
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rec_dir = tmp_path / "reconstructions"
    rec_dir.mkdir()
    reconstructions: list[Path] = []
    for order, rows in enumerate(chunk_rows):
        # Deliberately lossy colors are allowed; global coordinates are retained.
        path = rec_dir / f"chunk_{order:06d}.ply"
        _write_ply(
            path,
            [
                (x, y, z, red + 1, green + 2, blue + 3)
                for x, y, z, red, green, blue in rows
            ],
        )
        reconstructions.append(path)
    return source, manifest, chunks, reconstructions


def _write_results(
    tmp_path: Path,
    chunks: list[Path],
    reconstructions: list[Path],
    orders: list[int] | None = None,
    *,
    coordinate_override: dict[int, Path] | None = None,
) -> list[Path]:
    if orders is None:
        orders = list(range(len(chunks)))
    results: list[Path] = []
    for index, order in enumerate(orders):
        chunk = chunks[order]
        reconstruction = (
            coordinate_override[order]
            if coordinate_override is not None and order in coordinate_override
            else reconstructions[order]
        )
        result = tmp_path / f"result_{index:06d}.json"
        result.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sample": "fixture",
                    "method": "Ours",
                    "operating_point": "8K",
                    "chunk": {
                        "order": order,
                        "path": str(chunk),
                        "sha256": _sha256(chunk),
                        "count": 2,
                    },
                    "endpoints": {
                        "Unicorn": {
                            "reconstruction_ply": str(reconstruction),
                            "reconstruction_sha256": _sha256(reconstruction),
                            "physical_bits": 10 + order,
                            "components": {"geometry": 4 + order, "attribute": 6},
                            "runtime_components": {"codec": 0.5 + order},
                        },
                        "Ours Base": {
                            "reconstruction_ply": str(reconstruction),
                            "reconstruction_sha256": _sha256(reconstruction),
                            "physical_bits": 20 + order,
                            "components": {
                                "x_low": 5,
                                "r1": 3,
                                "r2": 4,
                                "r3": 4,
                                "r4": 4 + order,
                            },
                            "runtime_components": {
                                "prefix_encode": 1.0 + order,
                                "prefix_decode": 2.0,
                                "base_synthesis": 3.0,
                            },
                        },
                        "Ours Full": {
                            "reconstruction_ply": str(reconstruction),
                            "reconstruction_sha256": _sha256(reconstruction),
                            "physical_bits": 30 + order,
                            "components": {"base": 20 + order, "enhancement": 10},
                            "runtime_components": {
                                "prefix_encode": 1.0 + order,
                                "prefix_decode": 2.0,
                                "base_synthesis": 3.0,
                                "enhancement_encode": 4.0,
                                "enhancement_decode": 5.0,
                            },
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        results.append(result)
    return results


def _fake_pc_error(calls: list[tuple[str, str]]):
    def fake(source: str, reconstruction: str, *, res: int, show: bool):
        calls.append((source, reconstruction))
        return {
            "  c[0],    F": 0.1,
            "  c[1],    F": 0.2,
            "  c[2],    F": 0.3,
            "  c[0],PSNRF": 30.0,
            "  c[1],PSNRF": 40.0,
            "  c[2],PSNRF": 50.0,
        }

    return fake


def test_aggregate_sums_bits_uses_original_denominator_and_scores_once_per_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest, chunks, reconstructions = _make_fixture(tmp_path)
    results = _write_results(tmp_path, chunks, reconstructions, orders=[1, 0])
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(MODULE, "pc_error", _fake_pc_error(calls))

    output_json = tmp_path / "aggregate.json"
    output_csv = tmp_path / "aggregate.csv"
    summary = MODULE.aggregate_formal_chunk_results(
        source,
        manifest,
        results,
        endpoints=["Unicorn", "Ours Base", "Ours Full"],
        output_json=output_json,
        output_csv=output_csv,
    )

    assert summary["source"]["point_count"] == 4
    assert summary["metric"]["per_chunk_quality_averaging"] is False
    assert len(calls) == 3
    for endpoint, expected_bits in (
        ("Unicorn", 21),
        ("Ours Base", 41),
        ("Ours Full", 61),
    ):
        row = summary["endpoints"][endpoint]
        assert row["physical_bits"] == expected_bits
        assert row["denominator_point_count"] == 4
        assert row["physical_bpp"] == expected_bits / 4
        assert row["coordinate_multiset_verified"] is True
        assert row["metrics"]["yuv_psnr_611"] == 35.0
    assert summary["endpoints"]["Unicorn"]["components"] == {
        "attribute": 12,
        "geometry": 9,
    }

    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written == summary
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["endpoint"] for row in rows] == ["Unicorn", "Ours Base", "Ours Full"]
    assert [int(row["denominator_point_count"]) for row in rows] == [4, 4, 4]
    assert [int(row["physical_bits"]) for row in rows] == [21, 41, 61]

    for endpoint in ("Unicorn", "Ours Base", "Ours Full"):
        merged = Path(summary["endpoints"][endpoint]["merged_reconstruction"])
        coords, rgb = MODULE._prepare_chunks.read_ply(merged)
        assert coords.shape == (4, 3)
        assert rgb.shape == (4, 3)
        assert merged.read_text(encoding="ascii").splitlines()[1] == "format ascii 1.0"


@pytest.mark.parametrize("orders", ([0], [0, 0]))
def test_missing_or_duplicate_chunk_orders_fail_closed(
    tmp_path: Path, orders: list[int]
) -> None:
    source, manifest, chunks, reconstructions = _make_fixture(tmp_path)
    results = _write_results(tmp_path, chunks, reconstructions, orders=orders)
    with pytest.raises(MODULE.AggregateError, match="chunk|result count"):
        MODULE.aggregate_formal_chunk_results(
            source,
            manifest,
            results,
            endpoints=["Unicorn"],
            pc_error_callable=_fake_pc_error([]),
        )


def test_coordinate_multiset_mismatch_fails_before_pc_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest, chunks, reconstructions = _make_fixture(tmp_path)
    bad = tmp_path / "bad_reconstruction.ply"
    _write_ply(
        bad,
        [
            (0, 0, 0, 1, 2, 3),
            (1, 1, 1, 1, 2, 3),
            (2, 2, 2, 1, 2, 3),
            (99, 99, 99, 1, 2, 3),
        ],
    )
    # One bad chunk still has the expected point count, so only the global
    # coordinate multiset gate can reject it.
    bad_chunk = tmp_path / "bad_chunk.ply"
    _write_ply(bad_chunk, [(2, 2, 2, 1, 2, 3), (99, 99, 99, 1, 2, 3)])
    # Keep the manifest input identity valid while using the bad reconstruction.
    results = _write_results(
        tmp_path,
        chunks,
        reconstructions,
        coordinate_override={1: bad_chunk},
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(MODULE, "pc_error", _fake_pc_error(calls))
    with pytest.raises(MODULE.AggregateError, match="coordinate multiset"):
        MODULE.aggregate_formal_chunk_results(
            source,
            manifest,
            results,
            endpoints=["Unicorn"],
            output_dir=tmp_path / "out",
        )
    assert calls == []


def test_chunk_sha_and_count_are_checked_against_manifest(
    tmp_path: Path,
) -> None:
    source, manifest, chunks, reconstructions = _make_fixture(tmp_path)
    results = _write_results(tmp_path, chunks, reconstructions)
    payload = json.loads(results[0].read_text(encoding="utf-8"))
    payload["chunk"]["count"] = 1
    results[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MODULE.AggregateError, match="input count mismatch"):
        MODULE.aggregate_formal_chunk_results(
            source,
            manifest,
            results,
            endpoints=["Unicorn"],
            pc_error_callable=_fake_pc_error([]),
        )


def test_full_missing_keeps_base_formal_and_publishes_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest, chunks, reconstructions = _make_fixture(tmp_path)
    results = _write_results(tmp_path, chunks, reconstructions)
    for result_path in results:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        del payload["endpoints"]["Ours Full"]
        result_path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(MODULE, "pc_error", _fake_pc_error(calls))

    output_dir = tmp_path / "partial"
    summary = MODULE.aggregate_formal_chunk_results(
        source,
        manifest,
        results,
        endpoints=["Ours Base", "Ours Full"],
        output_dir=output_dir,
    )

    assert summary["status"] == "PARTIAL"
    assert summary["endpoints"]["Ours Base"]["status"] == "FORMAL_REUSABLE"
    assert summary["endpoints"]["Ours Full"]["status"] == "MISSING"
    assert summary["endpoints"]["Ours Base"]["codec_runtime_seconds"] == 13.0
    assert len(calls) == 1
    assert Path(summary["endpoints"]["Ours Base"]["merged_reconstruction"]).is_file()
    assert summary["endpoints"]["Ours Full"]["merged_reconstruction"] is None
    assert summary["failure_evidence"]
    assert (output_dir / "aggregate.json").is_file()
    assert (output_dir / "aggregate.csv").is_file()
