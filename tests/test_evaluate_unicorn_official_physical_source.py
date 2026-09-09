import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/scalable_attribute/evaluation/evaluate_unicorn_official_physical.py"


def _load_source_module():
    spec = importlib.util.spec_from_file_location(
        "unicorn_official_physical_source", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_source_module()


class SourceContractTests(unittest.TestCase):
    def test_ctc_chunk_envelope_is_supported(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for option in ("--chunk-order", "--chunk-path", "--chunk-sha256",
                       "--chunk-count"):
            self.assertIn(option, source)
        self.assertIn('"Original Unicorn"', source)
        self.assertIn('CHUNK_RESULT_NAME = "chunk_result.json"', source)

    def test_importing_source_does_not_import_torch(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        self.assertNotIn("torch", imported)
        self.assertNotIn("MinkowskiEngine", imported)

    def test_no_thesis_model_imports_and_upstream_root_is_explicit(self):
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        self.assertFalse(any(name.startswith("scalable_attribute")
                             for name in imported_modules))
        self.assertIn("_load_upstream", source)
        self.assertIn("sys.path[:] = [str(Path(upstream_root).resolve())]", source)
        self.assertIn('import_module("lossy_attribute.model")', source)

    def test_static_contract_markers_are_present(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            '"--color_format", "yuv"',
            '"--normalize", "1"',
            '"--Vmode", "1"',
            '"--scale", "5"',
            '"--stage", "1"',
            '"--channels", "128"',
            '"--latent_channels", "32"',
            "read_ply_ascii",
            "model.forward(",
            "encode=True",
            "model.decode(",
            'len(item["strings"]) * 8',
            "gpcc_bits",
            "total_bits",
            "coder.write_file(",
            "pc_error(",
            "max_memory_allocated",
            "symlink_to(tmc3)",
            "os.replace(",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("coder.test(", source)

    def test_cli_shape(self):
        args = MODULE.parse_args([
            "--upstream-root", "/upstream",
            "--source-commit", "abc123",
            "--checkpoint", "/ckpt.pth",
            "--input-ply", "/input.ply",
            "--lambda-value", "32768",
            "--dataset", "8iVFB",
            "--sequence", "longdress",
            "--rate-id", "R01",
            "--checkpoint-profile", "32k8k",
            "--output-dir", "/out",
        ])
        self.assertEqual(args.lambda_value, 32768)
        self.assertEqual(args.rate_id, "R01")
        self.assertEqual(args.checkpoint_profile, "32k8k")

    def test_ascii_six_column_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.ply"
            path.write_bytes(
                b"ply\n"
                b"format ascii 1.0\n"
                b"comment fixture\n"
                b"element vertex 2\n"
                b"property float x\n"
                b"property float y\n"
                b"property float z\n"
                b"property uchar red\n"
                b"property uchar green\n"
                b"property uchar blue\n"
                b"end_header\n"
                b"1 2 3 4 5 6\n"
                b"7 8 9 10 11 12\n"
            )
            self.assertEqual(MODULE.ply_points(path), 2)

    def test_atomic_text_write_refuses_overwrite(self):
        with self.subTest("publish and refuse"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "result.json"
                MODULE._atomic_write_text(path, "first\n")
                self.assertEqual(path.read_text(encoding="utf-8"), "first\n")
                with self.assertRaises(FileExistsError):
                    MODULE._atomic_write_text(path, "second\n")


if __name__ == "__main__":
    unittest.main()
