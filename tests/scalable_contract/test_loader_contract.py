import torch
import tempfile
import unittest
from pathlib import Path

from scalable_attribute.canonical.scalable_model import (
    FINE_TUNE_ARCHITECTURE, load_finetuned_scalable, load_frozen_base)


class Config:
    def to_dict(self):
        return {"input_mode": "x4_f4", "base_channels": 2}


class DummyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = torch.nn.Linear(2, 2)
        self.base_synthesis = torch.nn.Linear(2, 2)
        self.config = Config()

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self


class DummyScalable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = DummyBase()
        self.enhancement = torch.nn.Linear(2, 2)
        self.conditioning_lambda = 4096


def rescue_checkpoint(model, released):
    return {
        "architecture": "canonical_base_rescue_v1",
        "config": model.config.to_dict(),
        "conditioning_lambda": 2048,
        "released_checkpoint": str(released),
        "base_model": {name: torch.full_like(value, 7)
                       for name, value in model.state_dict().items()},
    }


def standard_8k_checkpoint(model, released):
    return {
        "architecture": "canonical_base_predict_correct",
        "config": model.config.to_dict(),
        "base_lambda": 8192,
        "base_checkpoint": str(released),
        "base_synthesis": {
            name: torch.full_like(value, 13)
            for name, value in model.base_synthesis.state_dict().items()
        },
    }


class LoaderContractTests(unittest.TestCase):
 def test_8k_standard_restores_only_base_synthesis_and_freezes_base(self):
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        released = tmp_path / "released.pth"
        released.touch()
        model = DummyBase()
        prefix_before = {
            name: value.clone() for name, value in model.prefix.state_dict().items()
        }
        checkpoint = tmp_path / "standard_8k.pth"
        torch.save(standard_8k_checkpoint(model, released), checkpoint)

        load_frozen_base(model, checkpoint, released, 8192)

        self.assertTrue(all(
            torch.equal(value, prefix_before[name])
            for name, value in model.prefix.state_dict().items()))
        self.assertTrue(all(
            torch.equal(value, torch.full_like(value, 13))
            for value in model.base_synthesis.state_dict().values()))
        self.assertFalse(any(
            parameter.requires_grad for parameter in model.parameters()))


 def test_8k_standard_rejects_missing_and_unexpected_base_synthesis_keys(self):
  for mutation in ("missing", "unexpected"):
   with self.subTest(mutation=mutation):
    with tempfile.TemporaryDirectory() as directory:
     tmp_path = Path(directory)
     released = tmp_path / "released.pth"
     released.touch()
     model = DummyBase()
     state = standard_8k_checkpoint(model, released)
     if mutation == "missing":
         state["base_synthesis"].pop(next(iter(state["base_synthesis"])))
     else:
         state["base_synthesis"]["unexpected_key"] = torch.zeros(1)
     checkpoint = tmp_path / (mutation + ".pth")
     torch.save(state, checkpoint)
     with self.assertRaises(RuntimeError):
         load_frozen_base(model, checkpoint, released, 8192)


 def test_2k_rescue_restores_prefix_and_base_synthesis_strictly(self):
    tmp_path = Path(tempfile.mkdtemp())
    released = tmp_path / "released.pth"
    released.touch()
    model = DummyBase()
    state = rescue_checkpoint(model, released)
    checkpoint = tmp_path / "rescue.pth"
    torch.save(state, checkpoint)
    load_frozen_base(model, checkpoint, released, 2048)
    self.assertTrue(all(torch.equal(value, torch.full_like(value, 7))
                        for value in model.state_dict().values()))
    self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


 def test_2k_rescue_rejects_missing_and_unexpected_keys(self):
  for mutation in ("missing", "unexpected"):
   with self.subTest(mutation=mutation):
    tmp_path = Path(tempfile.mkdtemp())
    released = tmp_path / "released.pth"
    released.touch()
    model = DummyBase()
    state = rescue_checkpoint(model, released)
    if mutation == "missing":
        state["base_model"].pop(next(iter(state["base_model"])))
    else:
        state["base_model"]["unexpected_key"] = torch.zeros(1)
    checkpoint = tmp_path / (mutation + ".pth")
    torch.save(state, checkpoint)
    with self.assertRaises(RuntimeError):
        load_frozen_base(model, checkpoint, released, 2048)


 def test_4k_joint_restores_complete_state_strictly(self):
    model = DummyScalable()
    state = {
        "architecture": FINE_TUNE_ARCHITECTURE,
        "conditioning_lambda": 4096,
        "step": 3000,
        "scalable_model": {name: torch.full_like(value, 11)
                           for name, value in model.state_dict().items()},
    }
    loaded = load_finetuned_scalable(model, state, 4096)
    self.assertEqual(loaded["step"], 3000)
    self.assertTrue(all(torch.equal(value, torch.full_like(value, 11))
                        for value in model.state_dict().values()))


 def test_4k_joint_rejects_missing_and_unexpected_keys(self):
  for mutation in ("missing", "unexpected"):
   with self.subTest(mutation=mutation):
    model = DummyScalable()
    state_dict = {name: value.clone() for name, value in model.state_dict().items()}
    if mutation == "missing":
        state_dict.pop(next(iter(state_dict)))
    else:
        state_dict["unexpected_key"] = torch.zeros(1)
    state = {"architecture": FINE_TUNE_ARCHITECTURE,
             "conditioning_lambda": 4096,
             "step": 3000, "scalable_model": state_dict}
    with self.assertRaises(RuntimeError):
        load_finetuned_scalable(model, state, 4096)
