"""Compatibility shim; prefer :mod:`scalable_attribute.runtime.operating_points`."""

from scalable_attribute.runtime.operating_points import (
    DEFAULT_CONFIG,
    OperatingPointConfig,
    load_operating_points,
    point_for_lambda,
    resolve_operating_point,
)

__all__ = [
    "DEFAULT_CONFIG",
    "OperatingPointConfig",
    "load_operating_points",
    "point_for_lambda",
    "resolve_operating_point",
]
