"""Synthetic end-to-end smoke for CTC chunk preparation and aggregation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.scalable_attribute.evaluation import aggregate_formal_chunk_results
from scripts.scalable_attribute.evaluation import prepare_ctc_formal_chunks


ROOT = Path(__file__).resolve().parents[1]
PARTITION_SOURCE = ROOT / "data_utils/attribute/partition.py"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_ply(path, rows):
    header = [
        "ply", "format ascii 1.0", "element vertex {}".format(len(rows)),
        "property int x", "property int y", "property int z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    body = [" ".join(str(value) for value in row) for row in rows]
    Path(path).write_text("\n".join(header + body) + "\n", encoding="ascii")


def _fake_pc_error(source, reconstruction, res=1, show=False):
    del res, show
    for path in (source, reconstruction):
        header = Path(path).read_text(encoding="ascii").split("end_header", 1)[0]
        assert "property float x" in header
        assert "property float y" in header
        assert "property float z" in header
    return {
        "  c[0],    F": 1.0, "  c[1],    F": 2.0,
        "  c[2],    F": 3.0, "  c[0],PSNRF": 30.0,
        "  c[1],PSNRF": 31.0, "  c[2],PSNRF": 32.0,
    }


class FormalCTCPipelineSmoke(unittest.TestCase):
    def test_prepare_then_aggregate_base_and_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.ply"
            _write_ply(source, [
                (-100, 2, 3000, 1, 2, 3),
                (7, -8, 9, 10, 11, 12),
                (1000, 20, -30, 200, 201, 202),
                (-100, 2, 3000, 1, 2, 3),
            ])
            prepared = root / "prepared"
            manifest = prepare_ctc_formal_chunks.prepare_ctc_formal_chunks(
                source, prepared, max_num=2,
                partition_source=PARTITION_SOURCE,
                expected_partition_sha256=_sha256(PARTITION_SOURCE),
            )
            self.assertTrue(manifest["union_verification"]["verified"])
            first_chunk = prepared / manifest["sources"][0]["chunks"][0]["path"]
            header = first_chunk.read_text(encoding="ascii").split("end_header", 1)[0]
            self.assertIn("property float x", header)

            result_paths = []
            for chunk in manifest["sources"][0]["chunks"]:
                chunk_path = prepared / chunk["path"]
                result_path = root / "result_{:03d}.json".format(chunk["order"])
                result_path.write_text(json.dumps({
                    "schema_version": 1,
                    "sample": "synthetic_ctc",
                    "method": "Ours",
                    "operating_point": "8K",
                    "chunk": {
                        "order": chunk["order"], "path": str(chunk_path),
                        "sha256": chunk["sha256"], "count": chunk["count"],
                    },
                    "endpoints": {
                        "Ours Base": {
                            "status": "FORMAL_REUSABLE",
                            "reconstruction_ply": str(chunk_path),
                            "reconstruction_sha256": chunk["sha256"],
                            "physical_bits": 15,
                            "components": {
                                "x_low": 1, "r1": 2, "r2": 3,
                                "r3": 4, "r4": 5,
                            },
                            "runtime_components": {
                                "prefix_encode": 0.1,
                                "prefix_decode": 0.2,
                                "base_synthesis": 0.3,
                            },
                        },
                        "Ours Full": {
                            "status": "FORMAL_REUSABLE",
                            "reconstruction_ply": str(chunk_path),
                            "reconstruction_sha256": chunk["sha256"],
                            "physical_bits": 21,
                            "components": {"base": 15, "enhancement": 6},
                            "runtime_components": {
                                "prefix_encode": 0.1,
                                "prefix_decode": 0.2,
                                "base_synthesis": 0.3,
                                "enhancement_encode": 0.4,
                                "enhancement_decode": 0.5,
                            },
                        },
                    },
                }), encoding="utf-8")
                result_paths.append(result_path)

            summary = aggregate_formal_chunk_results.aggregate_formal_chunk_results(
                source, manifest["manifest_json"], result_paths,
                output_dir=root / "aggregate",
                pc_error_callable=_fake_pc_error,
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(
                summary["endpoints"]["Ours Base"]["status"],
                "FORMAL_REUSABLE")
            self.assertEqual(
                summary["endpoints"]["Ours Full"]["status"],
                "FORMAL_REUSABLE")
            self.assertEqual(
                summary["endpoints"]["Ours Base"]["physical_bits"], 30)
            self.assertEqual(
                summary["endpoints"]["Ours Full"]["physical_bits"], 42)

            failed_payload = json.loads(result_paths[0].read_text(encoding="utf-8"))
            failed_payload["endpoints"]["Ours Full"] = {
                "status": "FAILED", "reason": "synthetic Full failure",
            }
            result_paths[0].write_text(
                json.dumps(failed_payload), encoding="utf-8")
            partial = aggregate_formal_chunk_results.aggregate_formal_chunk_results(
                source, manifest["manifest_json"], result_paths,
                pc_error_callable=_fake_pc_error,
            )
            self.assertEqual(partial["status"], "PARTIAL")
            self.assertEqual(
                partial["endpoints"]["Ours Base"]["status"],
                "FORMAL_REUSABLE")
            self.assertEqual(
                partial["endpoints"]["Ours Full"]["status"], "FAILED")

    def test_aggregate_cli_propagates_formal_failure_exit_status(self):
        source = Path(aggregate_formal_chunk_results.__file__)
        text = source.read_text(encoding="utf-8")
        self.assertIn('return 0 if summary["status"] == "PASS" else 1', text)
        self.assertIn("raise SystemExit(main())", text)


if __name__ == "__main__":
    unittest.main()
