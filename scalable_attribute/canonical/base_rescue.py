"""Compatibility shim; prefer :mod:`scalable_attribute.training.base_rescue`."""

from scalable_attribute.training.base_rescue import (
    SAMPLER_POLICY,
    DeterministicWeightedBatchSampler,
    base_rescue_objective,
    classify_difficulty,
    load_difficulty_scores,
    sample_key,
)

__all__ = [
    "SAMPLER_POLICY",
    "DeterministicWeightedBatchSampler",
    "base_rescue_objective",
    "classify_difficulty",
    "load_difficulty_scores",
    "sample_key",
]
