"""Small TAFA-style random single-endpoint training primitive."""

from dataclasses import dataclass

import torch

from basic_models.loss import get_bits


ENDPOINTS = ("Base", "Full")


@dataclass
class JointEndpointOutput:
    endpoint: str
    loss: torch.Tensor
    rate_base: torch.Tensor
    rate_enhancement: torch.Tensor
    distortion: torch.Tensor
    channel_mse: torch.Tensor


def sample_endpoint(generator, p_full):
    """Select exactly one endpoint using the caller-owned RNG."""
    if not 0.0 <= p_full <= 1.0:
        raise ValueError("p_full must lie in [0, 1]")
    return "Full" if generator.random() < p_full else "Base"


def _estimated_rate(likelihoods, points):
    if likelihoods is None or len(likelihoods) == 0:
        raise RuntimeError("Base rate requires non-empty r1-r4 likelihoods")
    return sum(get_bits(value) for value in likelihoods) / points


def _distortion(reference, reconstruction):
    channel_mse = torch.mean((reference - reconstruction) ** 2, dim=0)
    return channel_mse.mean(), channel_mse


def joint_endpoint_objective(model, attribute, endpoint,
                             lambda_base, lambda_full):
    """Forward and form the loss for one and only one selected endpoint."""
    if endpoint not in ENDPOINTS:
        raise ValueError("Unknown endpoint: " + str(endpoint))
    if lambda_base <= 0 or lambda_full <= 0:
        raise ValueError("Endpoint lambdas must be positive")

    if endpoint == "Base":
        output = model.forward_base_train(attribute)
        rate_base = _estimated_rate(
            output["prefix_likelihoods"], len(attribute))
        rate_enhancement = rate_base.new_zeros(())
        distortion, channel_mse = _distortion(
            attribute.F, output["Base"].F)
        loss = rate_base + lambda_base * distortion
    else:
        output = model.forward_full_train(attribute)
        rate_base = _estimated_rate(
            output["prefix_likelihoods"], len(attribute))
        rate_enhancement = get_bits(output["likelihood_E"]) / len(attribute)
        distortion, channel_mse = _distortion(
            attribute.F, output["Full"].F)
        loss = rate_base + rate_enhancement + lambda_full * distortion

    return JointEndpointOutput(
        endpoint=endpoint, loss=loss, rate_base=rate_base,
        rate_enhancement=rate_enhancement, distortion=distortion,
        channel_mse=channel_mse)
