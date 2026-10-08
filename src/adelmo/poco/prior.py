"""Gaussian knob priors for the Levenberg-Marquardt fits, one strength per knob family."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def validate_prior_strengths(
    strengths: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate exact terminal knob-family prior strengths."""
    result: dict[str, float] = {}
    for family, value in (strengths or {}).items():
        family = str(family)
        if not family or "." in family:
            raise ValueError(
                f"Prior family {family!r} must be an exact terminal attribute such as 'dk1l'"
            )
        value = float(value)
        if value < 0.0:
            raise ValueError("prior strengths must be >= 0")
        result[family] = value
    return result


def prior_alphas(
    strengths: Mapping[str, float],
    data_hessian: np.ndarray,
    knob_names: list[str],
    *,
    log: bool = False,
) -> np.ndarray:
    """Return one independently scaled Tikhonov precision per knob family."""
    strengths = validate_prior_strengths(strengths)
    diagonal = np.abs(np.diag(np.asarray(data_hessian, dtype=float)))
    alphas = np.zeros(len(knob_names))
    knob_families = [name.rpartition(".")[2] for name in knob_names]
    missing = sorted(set(knob_families) - set(strengths))
    unused = sorted(set(strengths) - set(knob_families))
    if missing or unused:
        raise ValueError(
            f"Prior families must exactly cover optimised knobs; missing={missing}, unused={unused}"
        )
    families = np.asarray(knob_families)
    for family, strength in strengths.items():
        indices = np.flatnonzero(families == family)
        positive = diagonal[indices][diagonal[indices] > 0.0]
        scale = float(np.median(positive)) if positive.size else 0.0
        alphas[indices] = float(strength) * scale
        if log:
            logger.info(
                "Knob prior for %s: %d knobs, alpha=%.3e "
                "(strength=%.3e x median diag H=%.3e)",
                family,
                len(indices),
                alphas[indices[0]],
                strength,
                scale,
            )

    return alphas


def apply_prior(
    loss: float,
    grad: np.ndarray,
    hessian: np.ndarray,
    params: np.ndarray,
    prior_mean: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Add a diagonal Gaussian knob prior to the loss, gradient and Hessian.

    Implements the MAP term ``0.5·alpha·||theta - theta0||²`` consistently with
    the worker convention (``grad = dL/dtheta``, ``hessian = d²L/dtheta²``): the
    gradient and Hessian gain the corresponding fixed diagonal precision. This
    allows families with different units to use independent curvature scales.
    """
    delta = params - prior_mean
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != delta.shape:
        raise ValueError("Prior coefficients must have one entry per optimisation knob")
    grad = grad + coefficients * delta
    hessian = hessian + np.diag(coefficients)
    loss = loss + 0.5 * float(delta @ (coefficients * delta))
    return loss, grad, hessian
