from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.scalable_attribute.evaluation.evaluate_preflight_gates import (
    BRANCHES,
    DuplicateCheckError,
    main,
    plan_preflight_gates,
)


def _check(check_id, scope_kind, scope, status="PASS", evidence="fixture"):
    return {
        "check_id": check_id,
        "scope_kind": scope_kind,
        "scope": scope,
        "status": status,
        "evidence": evidence,
    }


class EvaluatePreflightGatesTests(unittest.TestCase):
    def test_local_failure_isolation(self):
        summary = plan_preflight_gates(
            [
                _check(
                    "ctc-source",
                    "dataset",
                    "CTC",
                    "FAIL",
                    "CTC source identity mismatch",
                ),
                _check(
                    "joint-4k-memory",
                    "loader",
                    "joint_4k",
                    "FAIL",
                    {"peak_vram_gib": 25.1},
                ),
            ]
        )

        self.assertEqual(summary["status"], "HOLD")
        self.assertEqual(summary["counts"]["total_branches"], 48)
        self.assertEqual(summary["counts"]["hold_all"], 0)
        self.assertEqual(summary["counts"]["hold"], 18)
        self.assertEqual(summary["counts"]["eligible"], 30)

        by_id = {branch["branch_id"]: branch for branch in summary["branches"]}
        self.assertIn("8iVFB/Unicorn/256", by_id)
        self.assertIn("8iVFB/Unicorn/128", by_id)
        self.assertNotIn("8iVFB/Ours/standard/256", by_id)
        self.assertEqual(by_id["8iVFB/Ours/joint_4k/4k"]["decision"], "HOLD")
        self.assertEqual(by_id["8iVFB/Ours/standard/8k"]["decision"], "ELIGIBLE")
        self.assertEqual(by_id["8iVFB/Unicorn/4k"]["decision"], "ELIGIBLE")
        self.assertIn("ctc-source", by_id["CTC/Unicorn/32k"]["failed_check_ids"])
        self.assertIn("CTC source identity mismatch", by_id["CTC/Unicorn/32k"]["reasons"][0])

    def test_global_failure_holds_every_branch(self):
        summary = plan_preflight_gates(
            [
                _check(
                    "physical-rate-contract",
                    "global",
                    "scientific_contract",
                    "FAIL",
                    "physical bits identity is not established",
                ),
                _check("dataset-smoke", "dataset", "8iVFB", evidence="pass")
            ]
        )

        self.assertEqual(len(BRANCHES), 48)
        self.assertEqual(summary["status"], "HOLD_ALL")
        self.assertEqual(summary["counts"]["hold_all"], 48)
        self.assertEqual(summary["counts"]["hold"], 0)
        self.assertEqual(summary["counts"]["eligible"], 0)
        self.assertTrue(all(row["decision"] == "HOLD_ALL" for row in summary["branches"]))
        self.assertTrue(
            all(
                "physical bits identity is not established" in row["reasons"][0]
                for row in summary["branches"]
            )
        )

    def test_duplicate_check_ids_fail_closed(self):
        checks = [
            _check("same", "global", "global"),
            _check("same", "dataset", "Owlii"),
        ]
        with self.assertRaises(DuplicateCheckError):
            plan_preflight_gates(checks)

    def test_cli_writes_atomic_json_and_tsv_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checks_path = directory / "checks.json"
            output_json = directory / "nested" / "matrix.json"
            output_tsv = directory / "nested" / "matrix.tsv"
            checks_path.write_text(
                json.dumps([_check("one-point", "operating_point", "4K", "FAIL", "4K smoke failed")]),
                encoding="utf-8",
            )

            return_code = main(
                [
                    "--checks-json",
                    str(checks_path),
                    "--output-json",
                    str(output_json),
                    "--output-tsv",
                    str(output_tsv),
                ]
            )

            self.assertEqual(return_code, 0)
            written = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(written["counts"]["hold"], 6)
            self.assertEqual(written["counts"]["eligible"], 42)
            with output_tsv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 48)
            self.assertEqual(
                {row["decision"] for row in rows}, {"HOLD", "ELIGIBLE"}
            )


if __name__ == "__main__":
    unittest.main()
