"""Compatibility shim for scalable composition and checkpoint loaders.

Prefer :mod:`scalable_attribute.models.scalable` for architecture and
:mod:`scalable_attribute.runtime.checkpoints` for strict state restoration.
"""

from scalable_attribute.models.scalable import (
    TRAINABLE_SCOPES,
    CanonicalScalableModel,
)
from scalable_attribute.runtime.checkpoints import (
    FINE_TUNE_ARCHITECTURE,
    load_finetuned_scalable,
    load_frozen_base,
)

__all__ = [
    "FINE_TUNE_ARCHITECTURE",
    "TRAINABLE_SCOPES",
    "CanonicalScalableModel",
    "load_finetuned_scalable",
    "load_frozen_base",
]
