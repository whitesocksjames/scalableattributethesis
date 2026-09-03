"""Canonical frozen Base plus one independent native EnhancementVAE."""

import os

import torch

from scalable_attribute.canonical.enhancement import EnhancementVAE


FINE_TUNE_ARCHITECTURE = "canonical_scalable_mvub_finetune_v1"
TRAINABLE_SCOPES = ("enhancement_only", "full")


def _same_support(left, right, label):
    if list(left.tensor_stride) != list(right.tensor_stride):
        raise RuntimeError(label + " tensor strides differ")
    if not torch.equal(left.C, right.C):
        raise RuntimeError(label + " coordinates differ")


def load_frozen_base(base_model, checkpoint, released_checkpoint, base_lambda):
    """Load either canonical Base checkpoint format and freeze the whole Base.

    ``canonical_base_rescue_v1`` owns a fine-tuned Prefix as well as
    BaseSynthesis, so its complete ``base_model`` state must be restored.
    """
    state = torch.load(checkpoint, map_location="cpu")
    architecture = state.get("architecture")
    if architecture not in (
            "canonical_base_predict_correct", "canonical_base_rescue_v1"):
        raise ValueError("Canonical Base checkpoint architecture mismatch")
    if state.get("config") != base_model.config.to_dict():
        raise ValueError("Canonical Base checkpoint config mismatch")
    checkpoint_lambda = state.get(
        "base_lambda", state.get("conditioning_lambda", -1))
    if int(checkpoint_lambda) != int(base_lambda):
        raise ValueError("Canonical Base checkpoint lambda mismatch")
    checkpoint_released = state.get(
        "base_checkpoint", state.get("released_checkpoint", ""))
    if os.path.realpath(checkpoint_released) != os.path.realpath(
            released_checkpoint):
        raise ValueError("Canonical Base released checkpoint mismatch")
    if architecture == "canonical_base_predict_correct":
        base_model.base_synthesis.load_state_dict(
            state["base_synthesis"], strict=True)
    else:
        if "base_model" not in state:
            raise ValueError("Base rescue checkpoint lacks complete model state")
        base_model.load_state_dict(state["base_model"], strict=True)
    base_model.requires_grad_(False)
    base_model.eval()
    return base_model


class CanonicalScalableModel(torch.nn.Module):
    """Own a frozen canonical Base and a trainable independent EnhancementVAE."""

    def __init__(self, frozen_base, conditioning_lambda,
                 enhancement_initialization_state=None):
        super().__init__()
        if any(parameter.requires_grad for parameter in frozen_base.parameters()):
            raise ValueError("Canonical Base must be frozen before scalable assembly")
        self.base = frozen_base
        self.conditioning_lambda = conditioning_lambda
        self.enhancement = EnhancementVAE(
            self.base.prefix.model.VAE,
            initialization_state=enhancement_initialization_state)
        self._trainable_scope = None
        self.set_trainable_scope("enhancement_only")

    def train(self, mode=True):
        super().train(mode)
        self.base.train(mode if self._trainable_scope == "full" else False)
        self.enhancement.train(mode)
        return self

    @property
    def trainable_scope(self):
        return self._trainable_scope

    def set_trainable_scope(self, scope):
        if scope not in TRAINABLE_SCOPES:
            raise ValueError("Unknown trainable scope: " + str(scope))
        self.requires_grad_(False)
        full = scope == "full"
        if full:
            self.base.set_trainable(True)
        else:
            self.base.freeze()
        self.enhancement.requires_grad_(True)
        self._trainable_scope = scope
        self.train(self.training)
        return self

    @torch.no_grad()
    def base_forward(self, attribute, hard=False):
        if hard:
            state, rate = self.base.prefix.hard_forward(
                attribute, self.conditioning_lambda, return_details=True)
            result = self.base.reconstruct_from_state(state)
            result["prefix_rate"] = rate
        else:
            result = self.base(attribute, self.conditioning_lambda)
        _same_support(attribute, result["Base"], "GT/Base")
        _same_support(attribute, result["F_B"], "GT/F_B")
        _same_support(attribute, result["d5p"], "GT/d5p")
        return result

    def _embedding(self, device):
        return self.base.prefix.lambda_embedding(
            self.conditioning_lambda, device)

    def _training_embedding(self, device):
        if self._trainable_scope == "full":
            return self.base.prefix.lambda_embedding_trainable(
                self.conditioning_lambda, device)
        return self._embedding(device)

    def forward(self, attribute):
        base = self.base_forward(attribute)
        output = self.enhancement(
            base["Base"], attribute, base["F_B"], base["d5p"],
            self._embedding(attribute.device))
        return self._result(base, output)

    def forward_base_train(self, attribute):
        """Base-only differentiable path; never invokes EnhancementVAE."""
        if self._trainable_scope != "full":
            raise RuntimeError("Base endpoint training requires full scope")
        base = self.base.forward_trainable(
            attribute, self.conditioning_lambda)
        _same_support(attribute, base["Base"], "GT/Base")
        return base

    def forward_full_train(self, attribute):
        """Explicit training path for frozen-Base or full-unfreeze runs."""
        if self._trainable_scope == "full":
            base = self.base.forward_trainable(
                attribute, self.conditioning_lambda)
        else:
            base = self.base_forward(attribute)
            base["prefix_likelihoods"] = None
        output = self.enhancement(
            base["Base"], attribute, base["F_B"], base["d5p"],
            self._training_embedding(attribute.device))
        result = self._result(base, output)
        result["prefix_likelihoods"] = base["prefix_likelihoods"]
        return result

    @torch.no_grad()
    def deterministic_forward(self, attribute):
        base = self.base_forward(attribute)
        output = self.enhancement.deterministic(
            base["Base"], attribute, base["F_B"], base["d5p"],
            self._embedding(attribute.device))
        return self._result(base, output)

    @torch.no_grad()
    def hard_reconstruct(self, attribute):
        base = self.base_forward(attribute, hard=True)
        embedding = self._embedding(attribute.device)
        encoded = self.enhancement.encode(
            base["Base"], attribute, base["F_B"], base["d5p"], embedding)
        payload = {
            "strings": encoded["strings"],
            "min_v": encoded["min_v"],
            "max_v": encoded["max_v"],
        }
        decoded = self.enhancement.decode(
            payload, base["Base"], base["F_B"], base["d5p"], embedding)
        result = self._result(base, decoded)
        enhancement_bits = int(len(payload["strings"]) * 8)
        result.update({
            "enhancement_payload": payload,
            "enhancement_bits": enhancement_bits,
            "base_bits": base["prefix_rate"]["base_bits"],
            "full_bits": base["prefix_rate"]["base_bits"] + enhancement_bits,
            "prefix_rate": base["prefix_rate"],
            "encoded_Full": encoded["x_out"],
            "encoded_F_Full": encoded["f_out"],
            "encoded_dec_E": encoded["dec"],
        })
        return result

    @staticmethod
    def _result(base, enhancement):
        _same_support(base["Base"], enhancement["x_out"], "Base/Full")
        _same_support(base["Base"], enhancement["f_out"], "Base/F_Full")
        _same_support(base["Base"], enhancement["dec"], "Base/dec_E")
        return {
            "Base": base["Base"],
            "Full": enhancement["x_out"],
            "F_B": base["F_B"],
            "F_Full": enhancement["f_out"],
            "d5p": base["d5p"],
            "dec_E": enhancement["dec"],
            "likelihood_E": enhancement.get("likelihood"),
            "Qlatent_E": enhancement.get("Qlatent"),
            "prefix_state": base["prefix_state"],
        }


def load_finetuned_scalable(model, checkpoint, conditioning_lambda=None):
    """Load one complete fine-tuned Prefix+Base+Enhancement checkpoint."""
    state = (torch.load(checkpoint, map_location="cpu")
             if isinstance(checkpoint, (str, os.PathLike)) else checkpoint)
    if state.get("architecture") != FINE_TUNE_ARCHITECTURE:
        raise ValueError("Fine-tuned scalable checkpoint architecture mismatch")
    expected_lambda = (model.conditioning_lambda if conditioning_lambda is None
                       else conditioning_lambda)
    if int(state.get("conditioning_lambda", -1)) != int(expected_lambda):
        raise ValueError("Fine-tuned scalable checkpoint lambda mismatch")
    if "scalable_model" not in state:
        raise ValueError("Fine-tuned scalable checkpoint lacks complete model state")
    model.load_state_dict(state["scalable_model"], strict=True)
    return state
