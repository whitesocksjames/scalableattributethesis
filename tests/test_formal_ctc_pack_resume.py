import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (Path(__file__).resolve().parents[1] / "scripts" /
               "scalable_attribute" / "evaluation" / "formal_ctc_pack_resume.py")
SPEC = importlib.util.spec_from_file_location("formal_ctc_pack_resume", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(method="Ours", output="results/task"):
    return {"method": method, "output_rel": output, "task_id": "task",
            "sample_id": "sample"}


def write_aggregate(root, value, output="results/task"):
    path = root / output / "aggregate" / "aggregate.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))


class FormalCtcPackResumeTest(unittest.TestCase):
    def test_ours_resume_requires_both_reusable_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_aggregate(root, {
                "status": "PASS", "formal_status": "FORMAL_REUSABLE",
                "endpoint_statuses": {"Ours Base": "FORMAL_REUSABLE",
                                      "Ours Full": "FORMAL_REUSABLE"},
            })
            self.assertTrue(MODULE.formal_point_state(root, row())["reusable"])

    def test_partial_base_does_not_skip_combined_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_aggregate(root, {
                "status": "PARTIAL", "formal_status": "PARTIAL",
                "endpoint_statuses": {"Ours Base": "FORMAL_REUSABLE",
                                      "Ours Full": "FAILED"},
            })
            self.assertFalse(MODULE.formal_point_state(root, row())["reusable"])

    def test_original_requires_reusable_original_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_aggregate(root, {
                "status": "PASS", "formal_status": "FORMAL_REUSABLE",
                "endpoint_statuses": {"Original Unicorn": "FORMAL_REUSABLE"},
            })
            self.assertTrue(MODULE.formal_point_state(
                root, row("Original_Unicorn"))["reusable"])

    def test_retry_output_never_overwrites_prior_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = MODULE.next_retry_output(root, "task")
            self.assertTrue(first.endswith("attempt_01"))
            (root / first).mkdir(parents=True)
            second = MODULE.next_retry_output(root, "task")
            self.assertTrue(second.endswith("attempt_02"))

    def test_group_parser_rejects_duplicate_sample(self):
        with self.assertRaisesRegex(ValueError, "more than once"):
            MODULE.parse_groups(["A=one", "B=one"])

    def test_failed_point_process_does_not_stop_later_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "points.tsv"
            matrix.write_text(
                "task_id\tmethod\tsample_id\toutput_rel\n"
                "one\tOurs\tsample\tresults/one\n"
                "two\tOurs\tsample\tresults/two\n"
            )
            packs = root / "packs.tsv"
            packs.write_text(
                "pack_id\tpoint_indices\n"
                "pack\t0,1\n"
            )
            code = MODULE.main([
                "run", "--pack-manifest", str(packs), "--pack-index", "0",
                "--point-matrix", str(matrix),
                "--point-runner", str(root / "missing_runner.py"),
                "--run-root", str(root),
                "--evidence-root", str(root / "evidence"),
            ])
            self.assertEqual(code, 1)
            statuses = list((root / "evidence" / "pack" / "attempt_01" /
                             "point_status").glob("*.json"))
            self.assertEqual({path.stem for path in statuses}, {"one", "two"})


if __name__ == "__main__":
    unittest.main()
