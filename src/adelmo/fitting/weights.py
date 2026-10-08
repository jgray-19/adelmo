"""Inverse-variance loss weights and their global normalisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


def global_weight_scale(weights: Iterable[np.ndarray]) -> float:
    """The largest inverse-variance weight anywhere in a fit, or 1 if every weight is zero.

    Dividing every worker's weights by this one number keeps losses near unity
    without changing any *relative* weight, so all workers report on one scale.
    """
    largest = max((float(np.max(w)) for w in weights if w.size), default=0.0)
    if largest > 0.0:
        return largest
    LOGGER.warning("All computed weights are zero; skipping global normalisation")
    return 1.0


def variance_to_weight(variances: np.ndarray) -> np.ndarray:
    """Inverse-variance weights; a non-finite or non-positive variance gets weight zero."""
    weights = np.zeros_like(variances, dtype=np.float64)
    valid = np.isfinite(variances) & (variances > 0.0)
    np.divide(1.0, variances, out=weights, where=valid)
    return weights
