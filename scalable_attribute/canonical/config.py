"""Compatibility shim; prefer :mod:`scalable_attribute.models.config`."""

from scalable_attribute.models.config import (
    BaseSynthesisConfig,
    add_base_architecture_arguments,
)

__all__ = ["BaseSynthesisConfig", "add_base_architecture_arguments"]
