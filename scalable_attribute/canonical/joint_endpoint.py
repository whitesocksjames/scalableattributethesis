"""Compatibility shim; prefer :mod:`scalable_attribute.training.joint_endpoint`."""

from scalable_attribute.training.joint_endpoint import (
    ENDPOINTS,
    JointEndpointOutput,
    joint_endpoint_objective,
    sample_endpoint,
)

__all__ = [
    "ENDPOINTS",
    "JointEndpointOutput",
    "joint_endpoint_objective",
    "sample_endpoint",
]
