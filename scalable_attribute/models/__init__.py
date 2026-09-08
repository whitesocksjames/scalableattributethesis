"""Preferred reading and import surface for thesis model architecture.

The implementation follows the decoder dataflow:

``Prefix -> BaseSynthesis -> Base -> Enhancement``.

Training policy, evaluation, checkpoint loading, and command-line parsing are
separate concerns.  A small amount of legacy coupling remains during the
move-only refactor and is documented in the readability proposal.
"""

from scalable_attribute.models.config import BaseSynthesisConfig
from scalable_attribute.models.prefix import FrozenUnicornPrefix, PrefixState
from scalable_attribute.models.base_synthesis import BaseSynthesis
from scalable_attribute.models.base import CanonicalBaseModel
from scalable_attribute.models.enhancement import EnhancementVAE
from scalable_attribute.models.scalable import CanonicalScalableModel

__all__ = [
    "BaseSynthesisConfig",
    "PrefixState",
    "FrozenUnicornPrefix",
    "BaseSynthesis",
    "CanonicalBaseModel",
    "EnhancementVAE",
    "CanonicalScalableModel",
]
