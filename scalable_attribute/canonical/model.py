import torch

from scalable_attribute.canonical.base_synthesis import BaseSynthesis
from scalable_attribute.canonical.prefix import FrozenUnicornPrefix


def _same_support(left, right, label):
    if list(left.tensor_stride) != list(right.tensor_stride):
        raise RuntimeError(label + " tensor strides differ")
    if not torch.equal(left.C, right.C):
        raise RuntimeError(label + " coordinates differ")


class CanonicalBaseModel(torch.nn.Module):
    def __init__(self, checkpoint, config, scale=5, stage=1, vmode=1):
        super().__init__()
        self.config = config
        self.prefix = FrozenUnicornPrefix(
            checkpoint, scale=scale, stage=stage, vmode=vmode)
        self.base_synthesis = BaseSynthesis(config)

    def train(self, mode=True):
        super().train(mode)
        self.prefix.eval()
        return self

    def forward(self, attribute, base_lambda, hard_prefix=False):
        if hard_prefix:
            state, prefix_bits = self.prefix.hard_forward(
                attribute, base_lambda)
        else:
            state = self.prefix(attribute, base_lambda)
            prefix_bits = None

        compensation = self.base_synthesis(state)
        _same_support(state.x5p, state.f5p, "x5p/f5p")
        _same_support(state.f5p, compensation, "f5p/c_B")
        feature, correction = self.prefix.synthesize(
            state.f5p, compensation)
        _same_support(state.x5p, feature, "x5p/F_B")
        _same_support(state.x5p, correction, "x5p/delta_B")
        base = state.x5p + correction
        _same_support(attribute, base, "A/Base")
        return {
            "Base": base,
            "B0": state.x5p,
            "F_B": feature,
            "c_B": compensation,
            "d5p": state.d5p,
            "prefix_state": state,
            "prefix_bits": prefix_bits,
        }
