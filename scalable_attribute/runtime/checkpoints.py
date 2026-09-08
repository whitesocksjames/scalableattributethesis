"""Strict checkpoint restoration for canonical Base and scalable models."""

import os

import torch


FINE_TUNE_ARCHITECTURE = "canonical_scalable_mvub_finetune_v1"


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
