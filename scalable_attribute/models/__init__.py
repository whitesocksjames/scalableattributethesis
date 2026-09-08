"""Preferred reading and import surface for thesis model architecture.

The implementation follows the decoder dataflow:

``Prefix -> BaseSynthesis -> Base -> Enhancement``.

Training entry points choose optimization scopes, while model-owned lifecycle
invariants and differentiable endpoint forwards remain beside the graph they
control. Evaluation, checkpoint loading, and command-line parsing are separate
concerns. The final boundary decisions are documented in the repository guide.
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
