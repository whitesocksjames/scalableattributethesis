"""Compatibility shim; prefer :mod:`scalable_attribute.training.data_schedule`."""

from scalable_attribute.training.data_schedule import (
    FINGERPRINT_ALGORITHM,
    POLICY,
    ContinuationBatchSampler,
    ordered_manifest_content_fingerprint,
    require_compatible_schedule,
)

__all__ = [
    "FINGERPRINT_ALGORITHM",
    "POLICY",
    "ContinuationBatchSampler",
    "ordered_manifest_content_fingerprint",
    "require_compatible_schedule",
]
