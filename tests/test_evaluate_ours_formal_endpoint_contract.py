"""CPU/source-only contract checks for the Ours formal evaluator helpers."""

import ast
import hashlib
import math
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/scalable_attribute/evaluation/evaluate_8ivfb_sequence.py"


def _load_pure_helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    wanted = {
        "derive_codec_timings", "endpoint_status", "task_status",
        "validate_base_rate_details", "validate_full_bit_identity",
        "validate_independent_enhancement_metadata",
        "validate_joint_checkpoint_metadata", "validate_relocated_artifact",
        "frozen_artifact_lineage",
        "_formal_chunk_endpoint",
    }
    namespace = {
        "math": math,
        "os": os,
        "_sha256_file": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "FORMAL_REUSABLE": "FORMAL_REUSABLE",
        "ENDPOINT_FAILED": "FAILED",
        "TASK_PASS": "PASS",
        "TASK_PARTIAL": "PARTIAL",
        "TASK_FAILED": "FAIL",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(SCRIPT), "exec"), namespace)
    return namespace


HELPERS = _load_pure_helpers()


class OursFormalEndpointContractTests(unittest.TestCase):
    def test_runtime_derivation_is_component_based(self):
        result = HELPERS["derive_codec_timings"](1.25, 2.0, 2.5, 3.0, 4.0)
        self.assertEqual(result["base_codec_seconds"], 5.75)
        self.assertEqual(result["full_codec_seconds"], 12.75)
        with self.assertRaises(ValueError):
            HELPERS["derive_codec_timings"](-1, 0, 0)

    def test_base_full_statuses_preserve_reusable_base(self):
        status = HELPERS["endpoint_status"]({"bits": True, "roundtrip": True})
        failed = HELPERS["endpoint_status"]({"bits": True, "roundtrip": False})
        self.assertEqual(status, "FORMAL_REUSABLE")
        self.assertEqual(failed, "FAILED")
        self.assertEqual(HELPERS["task_status"](status, failed), "PARTIAL")
        self.assertEqual(HELPERS["task_status"](status, status), "PASS")

    def test_physical_bit_gates_are_explicit(self):
        rate = HELPERS["validate_base_rate_details"]({
            "num_residual_streams": 4,
            "residual_bits": [2, 3, 5, 7],
            "bits_xlow": 11,
            "base_bits": 28,
        })
        self.assertEqual(rate["residual_bits"], [2, 3, 5, 7])
        self.assertEqual(
            HELPERS["validate_full_bit_identity"](28, 13, 41)["full_bits"], 41)
        with self.assertRaises(ValueError):
            HELPERS["validate_base_rate_details"]({
                "num_residual_streams": 4,
                "residual_bits": [2, 3, 5, 7],
                "bits_xlow": 11,
                "base_bits": 29,
            })

    def test_enhancement_lineage_uses_realpath_and_lambda(self):
        with self.subTest("accepted"):
            self.assertTrue(HELPERS["validate_independent_enhancement_metadata"](
                {
                    "architecture": "canonical_independent_enhancement",
                    "base_synthesis_checkpoint": "/tmp/frozen/../frozen/base.pth",
                    "released_checkpoint": "/tmp/released.pth",
                    "conditioning_lambda": 8192,
                },
                "/tmp/frozen/base.pth", "/tmp/released.pth", 8192))
        with self.assertRaises(ValueError):
            HELPERS["validate_independent_enhancement_metadata"](
                {
                    "architecture": "canonical_independent_enhancement",
                    "base_synthesis_checkpoint": "/tmp/base.pth",
                    "released_checkpoint": "/tmp/released.pth",
                    "conditioning_lambda": 4096,
                },
                "/tmp/base.pth", "/tmp/released.pth", 8192)

    def test_relocated_checkpoint_identity_and_lineage_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relocated = root / "relocated" / "base.pth"
            relocated.parent.mkdir()
            payload = b"frozen checkpoint bytes\n"
            relocated.write_bytes(payload)
            descriptor = {
                "path": "experiments/family/base.pth",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            origin = "/data/frozen/origin"
            frozen_base = origin + "/experiments/family/base.pth"
            frozen_released = origin + "/released/epoch_last.pth"

            self.assertTrue(HELPERS["validate_relocated_artifact"](
                str(relocated), descriptor))
            state = {
                "architecture": "canonical_independent_enhancement",
                "base_synthesis_checkpoint": frozen_base,
                "released_checkpoint": frozen_released,
                "conditioning_lambda": 8192,
            }
            self.assertTrue(HELPERS["validate_independent_enhancement_metadata"](
                state, str(relocated), str(root / "released.pth"), 8192,
                frozen_lineage={
                    "base_synthesis_checkpoint": frozen_base,
                    "released_checkpoint": frozen_released,
                }))

            relocated.write_bytes(b"wrong checkpoint bytes\n")
            with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256 mismatch"):
                HELPERS["validate_relocated_artifact"](str(relocated), descriptor)

            wrong_parent = dict(state, base_synthesis_checkpoint=origin + "/other/base.pth")
            with self.assertRaisesRegex(ValueError, "base_synthesis_checkpoint mismatch"):
                HELPERS["validate_independent_enhancement_metadata"](
                    wrong_parent, str(relocated), str(root / "released.pth"), 8192,
                    frozen_lineage={
                        "base_synthesis_checkpoint": frozen_base,
                        "released_checkpoint": frozen_released,
                    })

            for field, value, message in (
                    ("conditioning_lambda", 4096, "lambda mismatch"),
                    ("architecture", "wrong", "architecture mismatch")):
                wrong = dict(state, **{field: value})
                with self.assertRaisesRegex(ValueError, message):
                    HELPERS["validate_independent_enhancement_metadata"](
                        wrong, str(relocated), str(root / "released.pth"), 8192,
                        frozen_lineage={
                            "base_synthesis_checkpoint": frozen_base,
                            "released_checkpoint": frozen_released,
                        })

            joint = {
                "architecture": "canonical_scalable_mvub_finetune_v1",
                "base_synthesis_initialization": frozen_base,
                "released_checkpoint": frozen_released,
                "conditioning_lambda": 4096,
                "checkpoint_profile": "32k8k",
            }
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                HELPERS["validate_joint_checkpoint_metadata"](
                    joint, str(relocated), str(root / "released.pth"), 4096,
                    "8k256", frozen_lineage={
                        "base_synthesis_initialization": frozen_base,
                        "released_checkpoint": frozen_released,
                    })

    def test_combined_and_isolated_base_paths_are_explicit(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--formal-stop-after-base", source)
        self.assertIn("stop_after_base=args.formal_stop_after_base", source)
        self.assertIn("isolated Base-only preflight completed", source)
        self.assertIn("Base endpoint was not published before Full", source)
        self.assertIn("raise SystemExit(main())", source)

    def test_chunk_envelope_keeps_endpoint_runtime_and_bits(self):
        row = {
            "endpoint": "Ours Full", "endpoint_status": "FORMAL_REUSABLE",
            "reconstruction_ply": "/tmp/full.ply",
            "reconstruction_sha256": "a" * 64, "physical_bits": 41,
            "base_bits": 28, "enhancement_bits": 13,
            "prefix_encode_seconds": 1.0, "prefix_decode_seconds": 2.0,
            "base_synthesis_seconds": 3.0,
            "enhancement_encode_seconds": 4.0,
            "enhancement_decode_seconds": 5.0,
        }
        endpoint = HELPERS["_formal_chunk_endpoint"](row)
        self.assertEqual(endpoint["components"], {"base": 28, "enhancement": 13})
        self.assertEqual(endpoint["runtime_components"]["prefix_decode"], 2.0)
        self.assertEqual(endpoint["runtime_components"]["enhancement_decode"], 5.0)


if __name__ == "__main__":
    unittest.main()
