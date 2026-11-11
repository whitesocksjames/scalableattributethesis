"""Dependency-light checks for the standalone evaluation command."""
import tempfile
import importlib.util
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


class PayloadGeometryTests(unittest.TestCase):
    @staticmethod
    def codec_without_optional_libraries():
        import importlib.util
        from types import SimpleNamespace
        from unittest.mock import patch
        import sys
        spec = importlib.util.spec_from_file_location(
            "audit_codec", Path(__file__).parents[1] / "scalable_attribute/runtime/codec.py")
        module = importlib.util.module_from_spec(spec)
        torch_stub = SimpleNamespace(no_grad=lambda: lambda function: function)
        with patch.dict(sys.modules, {"torch": torch_stub, "MinkowskiEngine": SimpleNamespace()}):
            spec.loader.exec_module(module)
        return module

    def test_rejects_geometry_stride_before_decoding(self):
        from types import SimpleNamespace
        codec = self.codec_without_optional_libraries()
        payload = {"schema": "scalable-attribute-base-v1", "geometry_stride": [1, 1, 1],
                   "streams": [{}, {}, {}, {}]}
        with self.assertRaisesRegex(ValueError, "stride"):
            codec.decode_base(None, payload, SimpleNamespace(tensor_stride=[2, 2, 2]))

    def test_rejects_missing_residual_stream_before_decoding(self):
        from types import SimpleNamespace
        codec = self.codec_without_optional_libraries()
        model = SimpleNamespace(base=SimpleNamespace(prefix=SimpleNamespace(residual_stages=4)))
        payload = {"schema": "scalable-attribute-base-v1", "geometry_stride": [1, 1, 1],
                   "streams": [{}, {}, {}]}
        with self.assertRaisesRegex(ValueError, "four residual"):
            codec.decode_base(model, payload, SimpleNamespace(tensor_stride=[1, 1, 1]))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchac"),
                     "requires the declared PyTorch and torchac runtime")
class EntropyDecoderTests(unittest.TestCase):
    def test_fresh_decoder_needs_no_prior_compress_call(self):
        import torch
        from basic_models.conditional_entropy_model import SymmetricConditional
        values = torch.tensor([[-1., 0., 1.], [2., -2., 0.]])
        loc, scale = torch.zeros_like(values), torch.ones_like(values)
        encoder = SymmetricConditional()
        strings, minimum, maximum = encoder.compress(values, loc, scale)
        decoder = SymmetricConditional()
        self.assertFalse(hasattr(decoder, "_channels"))
        actual = decoder.decompress(strings, loc, scale, minimum, maximum, channels=3)
        self.assertTrue(torch.equal(values, actual))
        with self.assertRaisesRegex(ValueError, "channel count"):
            decoder.decompress(strings, loc, scale, minimum, maximum, channels=4)


if __name__ == "__main__":
    unittest.main()
