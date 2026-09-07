import os
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.scalable_contract.test_gpu_integration import load_attribute


COMMON_FIXTURES = {
    "sample_h5": "SCALABLE_TEST_H5",
}

POINT_FIXTURES = {
    "8k": {
        "released": "SCALABLE_RELEASED_8K_R03_32K8K_L8192",
        "base": "SCALABLE_8K_BASE_D111_STEP3525",
        "enhancement": "SCALABLE_8K_ENHANCEMENT_D111_STEP1763",
    },
    "4k": {
        "released": "SCALABLE_RELEASED_8K_R03_32K8K_L8192",
        "base": "SCALABLE_8K_BASE_D111_STEP3525",
        "joint": "SCALABLE_4K_JOINT_FROM_8K_STEP3000",
    },
}


def resolve_fixtures(point):
    variables = {**COMMON_FIXTURES, **POINT_FIXTURES[point]}
    missing = [variable for variable in variables.values()
               if not os.environ.get(variable)
               or not Path(os.environ[variable]).exists()]
    if missing:
        raise AssertionError(
            "missing lightweight GPU fixtures: " + ", ".join(missing))
    return {name: str(Path(os.environ[variable]).resolve())
            for name, variable in variables.items()}


def assert_exact_keys(test, module, state_dict, label):
    expected = set(module.state_dict())
    actual = set(state_dict)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    test.assertEqual(missing, [], label + " missing_keys")
    test.assertEqual(unexpected, [], label + " unexpected_keys")


def make_8k_standard(test, fixtures):
    from scalable_attribute.canonical.config import BaseSynthesisConfig
    from scalable_attribute.canonical.model import CanonicalBaseModel
    from scalable_attribute.canonical.scalable_model import (
        CanonicalScalableModel, load_frozen_base)

    base_state = torch.load(fixtures["base"], map_location="cpu")
    test.assertEqual(Path(fixtures["released"]).parent.name, "32k8k")
    test.assertEqual(base_state.get("architecture"),
                     "canonical_base_predict_correct")
    test.assertEqual(int(base_state.get("base_lambda", -1)), 8192)
    test.assertEqual(int(base_state.get("step", -1)), 3525)
    config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(fixtures["released"], config).cuda()
    assert_exact_keys(test, base.base_synthesis,
                      base_state["base_synthesis"], "8K BaseSynthesis")
    load_frozen_base(base, fixtures["base"], fixtures["released"], 8192)

    model = CanonicalScalableModel(base, 8192).cuda().eval()
    enhancement_state = torch.load(fixtures["enhancement"], map_location="cpu")
    test.assertEqual(enhancement_state.get("architecture"),
                     "canonical_independent_enhancement")
    test.assertEqual(int(enhancement_state.get("conditioning_lambda", -1)),
                     8192)
    test.assertEqual(int(enhancement_state.get("step", -1)), 1763)
    test.assertEqual(list(enhancement_state.get("distortion_weights", [])),
                     [1, 1, 1])
    assert_exact_keys(test, model.enhancement.vae,
                      enhancement_state["enhancement_vae"],
                      "8K EnhancementVAE")
    incompatible = model.enhancement.vae.load_state_dict(
        enhancement_state["enhancement_vae"], strict=True)
    test.assertEqual(list(incompatible.missing_keys), [])
    test.assertEqual(list(incompatible.unexpected_keys), [])
    model.requires_grad_(False)
    return model


def make_4k_joint(test, fixtures):
    from scalable_attribute.canonical.config import BaseSynthesisConfig
    from scalable_attribute.canonical.model import CanonicalBaseModel
    from scalable_attribute.canonical.scalable_model import (
        FINE_TUNE_ARCHITECTURE, CanonicalScalableModel,
        load_finetuned_scalable, load_frozen_base)

    base_state = torch.load(fixtures["base"], map_location="cpu")
    test.assertEqual(base_state.get("architecture"),
                     "canonical_base_predict_correct")
    test.assertEqual(int(base_state.get("base_lambda", -1)), 8192)
    test.assertEqual(int(base_state.get("step", -1)), 3525)
    config = BaseSynthesisConfig(**base_state["config"])
    base = CanonicalBaseModel(fixtures["released"], config).cuda()
    load_frozen_base(base, fixtures["base"], fixtures["released"], 8192)
    model = CanonicalScalableModel(base, 4096).cuda().eval()

    joint_state = torch.load(fixtures["joint"], map_location="cpu")
    resolved_args = joint_state.get("resolved_args", {})
    test.assertEqual(joint_state.get("architecture"), FINE_TUNE_ARCHITECTURE)
    test.assertEqual(
        joint_state.get("checkpoint_profile",
                        resolved_args.get("checkpoint_profile")),
        "32k8k")
    test.assertEqual(Path(fixtures["released"]).parent.name, "32k8k")
    test.assertEqual(int(joint_state.get("source_conditioning_lambda", -1)),
                     8192)
    test.assertEqual(int(joint_state.get("conditioning_lambda", -1)), 4096)
    test.assertEqual(int(joint_state.get("step", -1)), 3000)
    test.assertEqual(
        os.path.realpath(joint_state.get("base_synthesis_initialization", "")),
        os.path.realpath(fixtures["base"]))
    assert_exact_keys(test, model, joint_state["scalable_model"],
                      "4K complete joint model")
    load_finetuned_scalable(model, joint_state, 4096)
    model.requires_grad_(False)
    return model


@unittest.skipUnless(os.environ.get("SCALABLE_RUN_GPU_TESTS") == "1",
                     "GPU integration tests not requested")
class LightweightGPUContractTests(unittest.TestCase):
 def _check_reconstruction(self, point, factory):
    if not torch.cuda.is_available():
        self.fail("CUDA unavailable in lightweight GPU regression run")
    fixtures = resolve_fixtures(point)
    attribute = load_attribute(fixtures["sample_h5"])
    model = factory(self, fixtures)

    # These gates intentionally exercise deterministic model reconstruction,
    # never physical Prefix coding or arithmetic Enhancement coding.
    model.base.prefix.hard_forward = mock.Mock(
        side_effect=AssertionError("lightweight gate invoked hard Prefix"))
    model.enhancement.encode = mock.Mock(
        side_effect=AssertionError("lightweight gate invoked Enhancement encode"))
    model.enhancement.decode = mock.Mock(
        side_effect=AssertionError("lightweight gate invoked Enhancement decode"))
    with torch.no_grad():
        result = model.deterministic_forward(attribute)

    for name in ("Base", "Full"):
        sparse = result[name]
        self.assertEqual(list(sparse.tensor_stride), [1, 1, 1])
        self.assertTrue(torch.equal(sparse.C, attribute.C))
        self.assertEqual(int(sparse.F.shape[1]), 3)
        self.assertTrue(torch.isfinite(sparse.F).all().item())
    for name in ("F_B", "F_Full", "d5p", "dec_E"):
        sparse = result[name]
        self.assertEqual(list(sparse.tensor_stride), [1, 1, 1])
        self.assertTrue(torch.equal(sparse.C, attribute.C))
        self.assertGreater(int(sparse.F.shape[1]), 0)
        self.assertTrue(torch.isfinite(sparse.F).all().item())

 def test_4k_joint_step3000_reconstruction(self):
    self._check_reconstruction("4k", make_4k_joint)

 def test_8k_standard_step1763_reconstruction(self):
    self._check_reconstruction("8k", make_8k_standard)
