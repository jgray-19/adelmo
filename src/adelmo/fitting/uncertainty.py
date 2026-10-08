"""Knob uncertainties from the normal matrix ``A`` (and noise matrix ``B``) of a fit."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Eigenvalues of the normal matrix below this floor are treated as unconstrained
# directions: flooring them keeps the inverted covariance finite and non-negative
# instead of exploding (or going negative through numerical noise).
HESSIAN_MIN_EIGENVALUE = 1e-35


# Relative eigenvalue (of the unit-diagonal normal matrix) below which a direction counts as unconstrained by the data.
SINGULAR_REL_TOLERANCE = 1e-8


def _floored_inverse(matrix: np.ndarray, min_eigenvalue: float) -> np.ndarray:
    """Inverse of the symmetrised ``matrix`` with eigenvalues floored to ``min_eigenvalue``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    sym = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    clipped = np.maximum(eigenvalues, min_eigenvalue)

    n_clipped = int(np.count_nonzero(eigenvalues < min_eigenvalue))
    if n_clipped:
        logger.warning(
            "Normal matrix had %d eigenvalue(s) below %.3e; using the floor to keep "
            "uncertainties finite and non-negative.",
            n_clipped,
            min_eigenvalue,
        )
    return (eigenvectors / clipped) @ eigenvectors.T


def warn_if_singular(
    normal_matrix: np.ndarray,
    knob_names: list[str] | None = None,
    *,
    rel_tolerance: float = SINGULAR_REL_TOLERANCE,
) -> int:
    """Warn when the data normal matrix ``JᵀWJ`` is (near-)singular; returns the number of weak directions.

    The matrix is first scaled to unit diagonal (a correlation matrix), so the test does not depend on the knobs' units.
    Directions with an eigenvalue below ``rel_tolerance`` x the largest one are not constrained by the data: the fit
    returns whatever the prior or the starting point puts there, and the reported uncertainties are only as good as the
    prior. Pass the matrix *without* any prior added.
    """
    matrix = np.asarray(normal_matrix, dtype=np.float64)
    sym = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(sym)
    unconstrained = diagonal <= 0.0
    scale = np.where(unconstrained, 1.0, np.sqrt(np.abs(diagonal)))
    eigenvalues, eigenvectors = np.linalg.eigh(sym / np.outer(scale, scale))
    weak = eigenvalues < rel_tolerance * max(float(eigenvalues[-1]), np.finfo(float).tiny)
    n_weak = int(np.count_nonzero(weak))
    if n_weak or unconstrained.any():
        message = (
            f"Normal matrix is near-singular: {n_weak} of {len(eigenvalues)} directions have eigenvalue < "
            f"{rel_tolerance:.0e} x largest (unit-diagonal scaling), {int(unconstrained.sum())} knob(s) have no sensitivity at all. "
            "The data do not constrain them; the fit is set there by the prior."
        )
        if knob_names is not None and n_weak:
            weight = (eigenvectors[:, weak] ** 2).sum(axis=1)
            worst = np.argsort(weight)[::-1][:5]
            message += " Knobs most involved: " + ", ".join(
                f"{knob_names[i]} ({weight[i]:.2f})" for i in worst
            )
        logger.warning(message)
    return n_weak


def sandwich_uncertainties(
    normal_matrix: np.ndarray,
    noise_matrix: np.ndarray,
    *,
    min_eigenvalue: float = HESSIAN_MIN_EIGENVALUE,
) -> np.ndarray:
    """1-sigma uncertainties ``sqrt(diag(A⁻¹ B A⁻¹))``.

    Invariant to the loss-weight scale (``A ∝ w``, ``B ∝ w²``), so ``A`` and ``B`` may
    be built from the globally normalised loss weights.
    """
    inverse = _floored_inverse(normal_matrix, min_eigenvalue)
    covariance = inverse @ np.asarray(noise_matrix, dtype=np.float64) @ inverse
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def hessian_uncertainties(
    normal_matrix: np.ndarray,
    *,
    min_eigenvalue: float = HESSIAN_MIN_EIGENVALUE,
) -> np.ndarray:
    """1-sigma parameter uncertainties from a Gauss-Newton normal matrix.

    ``normal_matrix`` must be the weighted normal matrix ``A = JᵀWJ`` built with
    *physical* inverse-variance weights ``W = 1/σ²`` (units ``1/measurement²``).
    Its inverse is then the parameter covariance and ``sqrt(diag(A⁻¹))`` gives the
    1-sigma uncertainties in real parameter units - so callers must pass the
    un-normalised, true-variance Hessian, never one rescaled by an arbitrary
    loss-normalisation (which would put the result in a meaningless space).

    Note the factor-of-2 convention: for a chi-square ``Σ w r²`` the second
    derivative is ``2 JᵀWJ``; pass ``A = JᵀWJ`` here (half that), i.e. the normal
    matrix / Fisher information, not the raw chi-square Hessian.

    The matrix is symmetrised and its eigenvalues floored to ``min_eigenvalue`` so
    weakly-constrained or rank-deficient directions yield large-but-finite,
    non-negative uncertainties rather than blowing up or turning negative through
    accumulated numerical noise.
    """
    covariance = _floored_inverse(normal_matrix, min_eigenvalue)
    variances = np.clip(np.diag(covariance), 0.0, None)
    return np.sqrt(variances)
