"""Strict loading of seven self-contained pretrained inference models."""

FORMAT = "scalable_attribute_inference_v1"
POINTS = {"512": 512, "1k": 1024, "2k": 2048, "4k": 4096,
          "8k": 8192, "16k": 16384, "32k": 32768}


def read_checkpoint(path):
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 1.13 compatibility.
        state = torch.load(path, map_location="cpu")
    if state.get("format") != FORMAT:
        raise ValueError("Expected a self-contained OURS inference checkpoint")
    point = state.get("point")
    if point not in POINTS or state.get("conditioning_lambda") != POINTS[point]:
        raise ValueError("Checkpoint point/lambda mismatch")
    if state.get("prefix_config") != {"scale": 5, "stage": 1, "vmode": 1}:
        raise ValueError("Unsupported Prefix architecture")
    weights = state.get("model_state")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Missing complete model state")
    for prefix in ("base.prefix.model.", "base.base_synthesis.", "enhancement.vae."):
        if not any(key.startswith(prefix) for key in weights):
            raise ValueError("Incomplete model state: " + prefix)
    if not all(torch.is_tensor(t) for t in weights.values()):
        raise ValueError("Model state must contain only tensors")
    return state


def load_model(path, device="cuda"):
    """Restore the full graph from one file without historical weight paths."""
    from scalable_attribute.models.config import BaseSynthesisConfig
    from scalable_attribute.models.base import CanonicalBaseModel
    from scalable_attribute.models.scalable import CanonicalScalableModel
    state = read_checkpoint(path)
    base = CanonicalBaseModel(None, BaseSynthesisConfig(**state["base_config"]))
    base.freeze()
    model = CanonicalScalableModel(base, state["conditioning_lambda"])
    model.load_state_dict(state["model_state"], strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()
