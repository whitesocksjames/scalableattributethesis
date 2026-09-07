"""Compatibility imports for historical canonical paths.

New code should read and import architecture from :mod:`scalable_attribute.models`.
This package remains intentionally thin while training, evaluation, and runtime
support are separated in later reviewed batches.  See ``canonical/README.md``.
"""

from scalable_attribute.models.config import BaseSynthesisConfig
from scalable_attribute.models.base import CanonicalBaseModel

__all__ = ["BaseSynthesisConfig", "CanonicalBaseModel"]
