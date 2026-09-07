import os
import json
import time
import unittest
from pathlib import Path

import torch

FIXTURES = {
    "sample_h5": "SCALABLE_TEST_H5",
    "gpcc": "SCALABLE_TEST_GPCC",
    "released_2k": "SCALABLE_RELEASED_2K_R05_8K256_L2048",
    "base_2k": "SCALABLE_2K_RESCUE_U_PATH_STEP500",
    "enhancement_2k": "SCALABLE_2K_ENHANCEMENT_D111_STEP1500",
    "released_8k_bootstrap": "SCALABLE_RELEASED_8K_R03_32K8K_L8192",
    "base_8k_bootstrap": "SCALABLE_8K_BASE_D111_STEP3525",
    "joint_4k": "SCALABLE_4K_JOINT_FROM_8K_STEP3000",
}


def sparse_equal(left, right):
    if list(left.tensor_stride) != list(right.tensor_stride):
        return False
    if left.C.shape != right.C.shape:
        return False
    left_order = sorted(range(len(left.C)), key=lambda i: tuple(left.C[i].tolist()))
    right_order = sorted(range(len(right.C)), key=lambda i: tuple(right.C[i].tolist()))
    return (torch.equal(left.C[left_order], right.C[right_order]) and
            torch.equal(left.F[left_order], right.F[right_order]))


def load_attribute(path):
    import MinkowskiEngine as ME
    from data_utils.attribute.inout import read_h5

    coords, feats = read_h5(path)
    coordinates = torch.as_tensor(coords, dtype=torch.int32)
    features = torch.as_tensor(feats, dtype=torch.float32)
    if features.max() > 1.0:
        features = features / 255.0
    batch = torch.zeros((len(coordinates), 1), dtype=torch.int32)
    return ME.SparseTensor(
        features=features.cuda(), coordinates=torch.cat([batch, coordinates], 1),
        tensor_stride=1, device="cuda")


def make_2k_model(fixtures):
    from scalable_attribute.canonical.config import BaseSynthesisConfig
    from scalable_attribute.canonical.model import CanonicalBaseModel
    from scalable_attribute.canonical.scalable_model import (
        CanonicalScalableModel, load_frozen_base)

    state = torch.load(fixtures["base_2k"], map_location="cpu")
    assert state["architecture"] == "canonical_base_rescue_v1"
    assert int(state["conditioning_lambda"]) == 2048
    assert int(state["step"]) == 500
    config = BaseSynthesisConfig(**state["config"])
    base = CanonicalBaseModel(fixtures["released_2k"], config).cuda()
    load_frozen_base(base, fixtures["base_2k"], fixtures["released_2k"], 2048)
    model = CanonicalScalableModel(base, 2048).cuda().eval()
    enhancement = torch.load(fixtures["enhancement_2k"], map_location="cpu")
    assert enhancement["architecture"] == "canonical_independent_enhancement"
    assert int(enhancement["conditioning_lambda"]) == 2048
    model.enhancement.vae.load_state_dict(enhancement["enhancement_vae"], strict=True)
    return model


def make_4k_model(fixtures):
    from scalable_attribute.canonical.config import BaseSynthesisConfig
    from scalable_attribute.canonical.model import CanonicalBaseModel
    from scalable_attribute.canonical.scalable_model import (
        CanonicalScalableModel, load_finetuned_scalable, load_frozen_base)

    base_state = torch.load(fixtures["base_8k_bootstrap"], map_location="cpu")
    config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(fixtures["released_8k_bootstrap"], config).cuda()
    load_frozen_base(base, fixtures["base_8k_bootstrap"],
                     fixtures["released_8k_bootstrap"], 8192)
    model = CanonicalScalableModel(base, 4096).cuda().eval()
    state = load_finetuned_scalable(model, fixtures["joint_4k"], 4096)
    assert int(state["step"]) == 3000
    assert int(state["source_conditioning_lambda"]) == 8192
    assert int(state["conditioning_lambda"]) == 4096
    assert state.get("checkpoint_profile") == "32k8k"
    return model


def timed_hard_path_from_one_prefix(model, attribute):
    """Run one physical Prefix and reuse its decoded state for Base and Full."""
    import MinkowskiEngine as ME

    prefix = model.base.prefix
    native = prefix.model
    lmb = model.conditioning_lambda
    counters = {"prefix_physical_encode_calls": 0,
                "prefix_decode_calls": 0,
                "native_r5_encode_calls": 0,
                "native_r5_consume_calls": 0}
    timings = {}

    started = time.monotonic()
    counters["prefix_physical_encode_calls"] += 1
    encoded, x_low, gpcc_bits = native(
        attribute, training=False, lmb=lmb, encode=True,
        max_residual_stages=4)
    timings["prefix_encode_seconds"] = time.monotonic() - started
    counters["native_r5_encode_calls"] = int(len(encoded) > 4)
    if len(encoded) != 4:
        raise AssertionError("physical Prefix did not produce exactly r1-r4")

    x0 = ME.SparseTensor(
        features=torch.zeros_like(attribute.F),
        coordinate_map_key=attribute.coordinate_map_key,
        coordinate_manager=attribute.coordinate_manager,
        device=attribute.device)
    started = time.monotonic()
    counters["prefix_decode_calls"] += 1
    x4, f4, d4 = native.decode(
        x0=x0, x_low=x_low, enc_set_list=encoded, lmb=lmb,
        max_residual_stages=4, return_state=True)
    counters["native_r5_consume_calls"] = int(len(encoded) > 4)
    timings["prefix_decode_seconds"] = time.monotonic() - started
    state = prefix._complete_state(x4, f4, d4)
    residual_bits = [int(len(item["strings"]) * 8) for item in encoded]
    prefix_rate = {
        "bits_xlow": int(gpcc_bits),
        "residual_bits": residual_bits,
        "num_residual_streams": len(encoded),
        "base_bits": int(gpcc_bits + sum(residual_bits)),
    }

    started = time.monotonic()
    base = model.base.reconstruct_from_state(state)
    base["prefix_rate"] = prefix_rate
    timings["base_synthesis_seconds"] = time.monotonic() - started

    embedding = model._embedding(attribute.device)
    started = time.monotonic()
    encoded_enhancement = model.enhancement.encode(
        base["Base"], attribute, base["F_B"], base["d5p"], embedding)
    timings["enhancement_encode_seconds"] = time.monotonic() - started
    payload = {name: encoded_enhancement[name]
               for name in ("strings", "min_v", "max_v")}
    started = time.monotonic()
    decoded_enhancement = model.enhancement.decode(
        payload, base["Base"], base["F_B"], base["d5p"], embedding)
    timings["enhancement_decode_seconds"] = time.monotonic() - started
    full = model._result(base, decoded_enhancement)
    enhancement_bits = int(len(payload["strings"]) * 8)
    full.update({
        "encoded_Full": encoded_enhancement["x_out"],
        "base_bits": prefix_rate["base_bits"],
        "enhancement_bits": enhancement_bits,
        "full_bits": prefix_rate["base_bits"] + enhancement_bits,
        "prefix_rate": prefix_rate,
    })
    return base, full, counters, timings


@unittest.skipUnless(os.environ.get("SCALABLE_RUN_GPU_TESTS") == "1",
                     "GPU integration tests not requested")
class GPUIntegrationTests(unittest.TestCase):
 def setUp(self):
    missing = [variable for variable in FIXTURES.values()
               if not os.environ.get(variable)
               or not Path(os.environ[variable]).exists()]
    if missing:
        message = "missing scalable integration fixtures: " + ", ".join(missing)
        if os.environ.get("SCALABLE_STRICT_FIXTURES") == "1":
            self.fail(message)
        self.skipTest(message)
    if not torch.cuda.is_available():
        if os.environ.get("SCALABLE_STRICT_FIXTURES") == "1":
            self.fail("CUDA unavailable in strict GPU regression run")
        self.skipTest("CUDA unavailable")
    self.fixtures = {name: str(Path(os.environ[variable]).resolve())
                     for name, variable in FIXTURES.items()}

 def test_2k_real_hard_scalable_contract(self):
    self._check(make_2k_model)

 def test_4k_real_hard_scalable_contract(self):
    self._check(make_4k_model)

 def _check(self, factory):
    import tempfile
    attribute = load_attribute(self.fixtures["sample_h5"])
    model = factory(self.fixtures)
    codec = Path(tempfile.mkdtemp(prefix="scalable_codec_"))
    os.symlink(self.fixtures["gpcc"], codec / "tmc3_v21")
    (codec / "output" / "gpcc").mkdir(parents=True)
    previous = Path.cwd()
    os.chdir(codec)
    self.addCleanup(os.chdir, previous)

    base_only, full, counters, timings = timed_hard_path_from_one_prefix(
        model, attribute)
    print("SCALABLE_CONTRACT_TIMINGS " + json.dumps(
        {**counters, **timings}, sort_keys=True), flush=True)
    rate = full["prefix_rate"]
    self.assertEqual(counters["prefix_physical_encode_calls"], 1)
    self.assertEqual(counters["prefix_decode_calls"], 1)
    self.assertEqual(counters["native_r5_encode_calls"], 0)
    self.assertEqual(counters["native_r5_consume_calls"], 0)
    self.assertEqual(rate["num_residual_streams"], 4)
    self.assertEqual(len(rate["residual_bits"]), 4)
    self.assertEqual(full["base_bits"], rate["bits_xlow"] + sum(rate["residual_bits"]))
    self.assertEqual(full["full_bits"], full["base_bits"] + full["enhancement_bits"])
    self.assertTrue(sparse_equal(base_only["Base"], full["Base"]))
    self.assertEqual(base_only["prefix_rate"]["base_bits"], full["base_bits"])
    self.assertEqual(list(full["Base"].tensor_stride), [1])
    self.assertTrue(torch.equal(full["Base"].C, attribute.C))
    self.assertTrue(sparse_equal(full["encoded_Full"], full["Full"]))
