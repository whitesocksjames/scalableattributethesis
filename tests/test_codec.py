"""Dependency-light checks for the standalone evaluation command."""
import tempfile
import unittest
from pathlib import Path

from scalable_attribute.evaluate import run_pc_error


class PcErrorCliTests(unittest.TestCase):
    def test_preserves_infinite_psnr_for_exact_reconstruction(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        output = "\n".join("  c[{}],{} : {}".format(c, field, value)
                           for c in range(3)
                           for field, value in (("    F", "0"), ("PSNRF", "inf")))
        with patch("scalable_attribute.evaluate.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout=output, stderr="")):
            metrics = run_pc_error("pc_error_d", "source.ply", "same.ply")
        self.assertEqual(metrics["y_psnr"], float("inf"))
        self.assertEqual(metrics["y_mse"], 0)

    def test_parses_yuv_metrics_and_weighted_psnr(self):
        with tempfile.TemporaryDirectory() as temp:
            exe = Path(temp) / "pc_error_stub"
            exe.write_text(
                "#!/usr/bin/env python3\n"
                "print('  c[0],    F : 1.0')\n"
                "print('  c[1],    F : 2.0')\n"
                "print('  c[2],    F : 3.0')\n"
                "print('  c[0],PSNRF : 30.0')\n"
                "print('  c[1],PSNRF : 31.0')\n"
                "print('  c[2],PSNRF : 32.0')\n",
                encoding="utf-8")
            exe.chmod(0o755)
            metrics = run_pc_error(exe, "reference.ply", "reconstruction.ply")
        self.assertEqual(metrics["y_mse"], 1.0)
        self.assertEqual(metrics["v_psnr"], 32.0)
        self.assertEqual(metrics["yuv_psnr_611"], 30.375)


if __name__ == "__main__":
    unittest.main()
