"""Compatibility shim; prefer :mod:`scalable_attribute.evaluation.base_validation`."""

from scalable_attribute.evaluation.base_validation import (
    FIELDS,
    evaluate_base,
    summarize,
)

__all__ = ["FIELDS", "evaluate_base", "summarize"]
