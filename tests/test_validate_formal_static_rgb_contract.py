import copy
import csv
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.scalable_attribute.evaluation.validate_formal_static_rgb_contract import (
    ContractError,
    EXPECTED_OPERATING_POINTS,
    EXPECTED_SAMPLE_COLUMNS,
    main,
    validate_contract,
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _artifact(path, data):
    return {
        "path": path,
        "size_bytes": len(data),
        "sha256": _sha256(data),
    }


def _write_fixture(directory):
    directory = Path(directory)
    origin_root = directory / "origin"
    checkpoint_path = origin_root / "checkpoints" / "shared.pth"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_data = b"small checkpoint fixture\n"
    checkpoint_path.write_bytes(checkpoint_data)

    source_directory = directory / "sources"
    source_directory.mkdir()
    rows = []
    for dataset, count in (("8iVFB", 4), ("Owlii", 4), ("CTC", 12)):
        for index in range(count):
            sample_id = "sample_{}".format(index)
            source_path = source_directory / "{}-{}.ply".format(dataset, index)
            source_data = ("{} {}\n".format(dataset, index)).encode("ascii")
            source_path.write_bytes(source_data)
            rows.append({
                "dataset": dataset,
                "sample_id": sample_id,
                "source_path": str(source_path),
                "points": str(100 + index),
                "sha256": _sha256(source_data),
                "source_evidence": "fixture",
                "formal_preprocessing": (
                    "kdtree_800k_global_no_recenter"
                    if dataset == "CTC" else "identity"
                ),
            })

    samples_path = directory / "samples.tsv"
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(EXPECTED_SAMPLE_COLUMNS), delimiter="\t",
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_artifact = _artifact(
        "checkpoints/shared.pth", checkpoint_data)
    released = {
        profile: dict(checkpoint_artifact)
        for profile in ("32k8k", "8k256", "2k128")
    }
    operating_points = []
    for point, point_lambda in EXPECTED_OPERATING_POINTS:
        if point == "4k":
            base_loader = "joint_scalable_base"
            full_loader = "joint_scalable_full"
        elif point == "2k":
            base_loader = "rescued_base"
            full_loader = "independent_enhancement"
        else:
            base_loader = "base_synthesis"
            full_loader = "independent_enhancement"
        operation = {
            "point": point,
            "lambda": point_lambda,
            "released_profile": "32k8k",
            "base_loader": base_loader,
            "full_loader": full_loader,
            "base_checkpoint": dict(checkpoint_artifact),
            "full_checkpoint": dict(checkpoint_artifact),
        }
        if point == "4k":
            operation["base_checkpoint_lambda"] = 8192
        operating_points.append(operation)

    checkpoints_path = directory / "checkpoints.json"
    checkpoints_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "frozen",
        "origin_root": str(origin_root),
        "released_checkpoints": released,
        "operating_points": operating_points,
    }, indent=2), encoding="utf-8")
    return samples_path, checkpoints_path


class ValidateFormalStaticRgbContractTests(unittest.TestCase):
    def test_valid_contract_and_atomic_cli_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            samples_path, checkpoints_path = _write_fixture(directory)
            summary = validate_contract(samples_path, checkpoints_path)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["sample_count"], 20)
            self.assertEqual(summary["dataset_counts"], {
                "8iVFB": 4, "Owlii": 4, "CTC": 12,
            })
            self.assertEqual(
                [(item["point"], item["lambda"])
                 for item in summary["operating_points"]],
                list(EXPECTED_OPERATING_POINTS))

            evidence_path = Path(directory) / "nested" / "evidence.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                return_code = main([
                    "--samples-tsv", str(samples_path),
                    "--checkpoints-json", str(checkpoints_path),
                    "--output-json", str(evidence_path),
                ])
            self.assertEqual(return_code, 0)
            self.assertIn("PASS", stdout.getvalue())
            self.assertEqual(json.loads(evidence_path.read_text()), summary)

    def test_verify_files_checks_sources_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            samples_path, checkpoints_path = _write_fixture(directory)
            summary = validate_contract(
                samples_path, checkpoints_path, verify_files=True)
            verification = summary["verification"]
            self.assertTrue(verification["enabled"])
            self.assertEqual(len(verification["sources"]), 20)
            self.assertEqual(
                len(verification["checkpoints"]),
                len(summary["checkpoint_artifacts"]))
            self.assertTrue(all(item["verified"]
                                for item in verification["sources"]))
            self.assertTrue(all(item["verified"]
                                for item in verification["checkpoints"]))

            source = Path(verification["sources"][0]["resolved_path"])
            source.write_bytes(b"changed fixture\n")
            with self.assertRaisesRegex(ContractError, "sha256 mismatch"):
                validate_contract(samples_path, checkpoints_path, verify_files=True)

    def test_fail_closed_on_tsv_columns_and_dataset_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            samples_path, checkpoints_path = _write_fixture(directory)
            original = Path(samples_path).read_text(encoding="utf-8")

            malformed = original.replace(
                "dataset\tsample_id", "sample_id\tdataset", 1)
            Path(samples_path).write_text(malformed, encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "columns must be exactly"):
                validate_contract(samples_path, checkpoints_path)

            Path(samples_path).write_text(original, encoding="utf-8")
            lines = original.splitlines()
            Path(samples_path).write_text(
                "\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "dataset counts"):
                validate_contract(samples_path, checkpoints_path)

    def test_fail_closed_on_artifact_and_lineage_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            samples_path, checkpoints_path = _write_fixture(directory)
            original = json.loads(Path(checkpoints_path).read_text())

            cases = [
                ("4k base lambda", lambda data: data["operating_points"][3].update(
                    {"base_checkpoint_lambda": 4096}),
                 "4k must declare base_checkpoint_lambda 8192"),
                ("4k artifact identity", lambda data: data["operating_points"][3][
                    "full_checkpoint"].update({"path": "checkpoints/other.pth"}),
                 "same artifact"),
                ("2k rescued loader", lambda data: data["operating_points"][4].update(
                    {"base_loader": "base_synthesis"}),
                 "2k must use"),
                ("traversal", lambda data: data["operating_points"][0][
                    "base_checkpoint"].update({"path": "../escape.pth"}),
                 "path traversal"),
                ("checkpoint size", lambda data: data["released_checkpoints"][
                    "32k8k"].update({"size_bytes": 0}),
                 "size_bytes must be a positive integer"),
            ]
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    mutated = copy.deepcopy(original)
                    mutate(mutated)
                    Path(checkpoints_path).write_text(
                        json.dumps(mutated), encoding="utf-8")
                    with self.assertRaisesRegex(ContractError, message):
                        validate_contract(samples_path, checkpoints_path)


if __name__ == "__main__":
    unittest.main()
