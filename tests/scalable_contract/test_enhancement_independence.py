import torch
import unittest

from scalable_attribute.canonical.enhancement import EnhancementVAE


class DummyResidualVAE(torch.nn.Module):
    def __init__(self, stride=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))


class EnhancementIndependenceTests(unittest.TestCase):
 def test_enhancement_initialization_copies_without_parameter_sharing(self):
    released = DummyResidualVAE()
    enhancement = EnhancementVAE(released)
    self.assertIsNot(enhancement.vae, released)
    for source, copied in zip(released.parameters(), enhancement.parameters()):
        self.assertTrue(torch.equal(source, copied))
        self.assertIsNot(source, copied)
        self.assertNotEqual(source.data_ptr(), copied.data_ptr())
    with torch.no_grad():
        enhancement.vae.weight.add_(1)
    self.assertFalse(torch.equal(released.weight, enhancement.vae.weight))
