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
        self._full_trainable = False
        self._trainable_scope = "base_synthesis_only"

    def train(self, mode=True):
        super().train(mode)
        self.prefix.train(mode if self._full_trainable else False)
        return self

    def set_trainable(self, enabled):
        """Enable the complete Base graph only for explicit full-unfreeze."""
        return self.set_trainable_scope(
            "base_path" if enabled else "base_synthesis_only")

    def freeze(self):
        """Freeze the complete Base for Enhancement-only optimization."""
        self.requires_grad_(False)
        self._full_trainable = False
        self.eval()
        return self

    @property
    def trainable_scope(self):
        return self._trainable_scope

    def set_trainable_scope(self, scope):
        """Select the two Base-only optimization scopes explicitly."""
        if scope not in ("base_synthesis_only", "base_path"):
            raise ValueError("Unknown Base trainable scope: " + str(scope))
        self.requires_grad_(False)
        self._full_trainable = scope == "base_path"
        self.prefix.set_trainable(self._full_trainable)
        self.base_synthesis.requires_grad_(True)
        self._trainable_scope = scope
        self.train(self.training)
        return self

    def forward(self, attribute, base_lambda, hard_prefix=False):
        if hard_prefix:
            state, prefix_bits = self.prefix.hard_forward(
                attribute, base_lambda)
        else:
            state = self.prefix(attribute, base_lambda)
            prefix_bits = None

        result = self.reconstruct_from_state(state)
        _same_support(attribute, result["Base"], "A/Base")
        result["prefix_bits"] = prefix_bits
        return result

    def forward_trainable(self, attribute, base_lambda):
        if not self._full_trainable:
            raise RuntimeError("Base forward_trainable requires full trainable scope")
        state, likelihoods = self.prefix.training_forward(
            attribute, base_lambda)
        result = self.reconstruct_from_state(state)
        _same_support(attribute, result["Base"], "A/Base")
        result["prefix_likelihoods"] = likelihoods
        return result

    def forward_base_training(self, attribute, base_lambda):
        """Uniform-noise r1-r4 Base path for rescue-screen arms.

        Both scopes deliberately use the same training-time quantization.
        In ``base_synthesis_only`` the Prefix remains in eval mode with all
        parameters frozen; its likelihoods are still returned for R_B_est.
        """
        state, likelihoods = self.prefix.training_forward(
            attribute, base_lambda,
            allow_frozen=self._trainable_scope == "base_synthesis_only")
        result = self.reconstruct_from_state(state)
        _same_support(attribute, result["Base"], "A/Base")
        result["prefix_likelihoods"] = likelihoods
        result["completed_residual_stages"] = self.prefix.residual_stages
        return result

    def reconstruct_from_state(self, state):
        compensation = self.base_synthesis(state)
        _same_support(state.x5p, state.f5p, "x5p/f5p")
        _same_support(state.f5p, compensation, "f5p/c_B")
        feature, correction = self.prefix.synthesize(
            state.f5p, compensation)
        _same_support(state.x5p, feature, "x5p/F_B")
        _same_support(state.x5p, correction, "x5p/delta_B")
        base = state.x5p + correction
        return {
            "Base": base,
            "B0": state.x5p,
            "F_B": feature,
            "c_B": compensation,
            "d5p": state.d5p,
            "prefix_state": state,
        }

    def native_baselines(self, state):
        """Return the two no-additional-bit full-resolution baselines."""
        zero = type(state.f5p)(
            features=torch.zeros_like(state.f5p.F),
            coordinate_map_key=state.f5p.coordinate_map_key,
            coordinate_manager=state.f5p.coordinate_manager,
        )
        feature, correction = self.prefix.synthesize(
            state.f5p, zero)
        native = state.x5p + correction
        _same_support(state.x5p, native, "x5p/B_native")
        return {"B_unpool": state.x5p, "B_native": native}
