"""Detect degenerate (unconstrained) knob directions before optimising.

Background
----------
A weighted least-squares fit ``min_x  Σ w r(x)²`` is well-posed only if the
measurement constrains every knob combination. Otherwise the Gauss-Newton normal
matrix ``A = JᵀWJ`` (``J`` the residual Jacobian, ``W`` the inverse-variance
weights) is rank-deficient or ill-conditioned, with near-zero eigenvalues whose
eigenvectors are knob combinations the data cannot constrain. An optimiser started
on such a problem has no unique minimum: the knobs drift along the flat direction
with little change in loss.

``A`` is the matrix already accumulated for parameter uncertainties
(:func:`aba_optimiser.fitting.uncertainty.hessian_uncertainties`). Its eigenspectrum at
the initial knobs identifies a degenerate fit, and the knob combinations
responsible, before optimisation, so the user can regularise, remove knobs or add
independent data.

Unit scaling
------------
The raw ``A`` mixes knobs of different physical scale (dipole ``k0``, quad ``k1``,
``pt``), so its condition number reflects unit disparity as well as degeneracy. By
default the analysis uses the symmetrically scaled matrix
``Ã = D^{-1/2} A D^{-1/2}`` with ``D = diag(A)``. It is dimensionless with unit
diagonal for every constrained knob, so a small eigenvalue indicates an
unconstrained *combination* rather than a small-valued knob. Knobs with zero
sensitivity (``diag(A) == 0``) are reported separately as fully unconstrained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aba_optimiser.fitting.uncertainty import hessian_uncertainties

# Default relative tolerance for calling a (scaled) eigenvalue "null". Scaled
# eigenvalues are O(1) for well-constrained directions, so this compares against
# the largest eigenvalue rather than an absolute floor.
DEFAULT_REL_TOL = 1e-8

# Below this fraction of the largest self-sensitivity, a knob's diagonal entry is
# treated as a structural zero (no sensitivity at all). This is deliberately tiny:
# it exists only to catch genuinely dead columns (and numerical fuzz on them), not
# to demote knobs that are merely far less sensitive than the strongest one - unit
# disparity between real knobs is handled by the symmetric scaling instead.
DEFAULT_ZERO_SENSITIVITY_REL = 1e-15


@dataclass(frozen=True)
class DegenerateDirection:
    """One near-null direction of the normal matrix - an unconstrained combination.

    Attributes:
        eigenvalue: Scaled eigenvalue (``≈ 0`` for a perfectly flat direction).
        relative_eigenvalue: ``eigenvalue / largest eigenvalue``; the size of this
            direction's curvature relative to the best-constrained direction.
        components: Combination coefficients in knob space, one per knob, unit-norm.
            The knobs trade off along ``Σ components[i] * knob[i]``.
        contributions: ``(knob_name, coefficient)`` pairs for the knobs that
            dominate this direction, largest ``|coefficient|`` first.
    """

    eigenvalue: float
    relative_eigenvalue: float
    components: np.ndarray
    contributions: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class DegeneracyReport:
    """Result of :func:`analyse_degeneracy`.

    Attributes:
        knob_names: Knob ordering matching every per-knob array below.
        scaled: Whether the eigen-analysis used the unit-scaled matrix.
        eigenvalues: Eigenvalues of the (scaled) normal matrix, descending.
        eigenvectors: Corresponding eigenvectors as columns, in knob space and
            aligned with ``eigenvalues``.
        condition_number: ``λ_max / λ_min`` over the positive eigenvalues; ``inf``
            when any direction is exactly flat.
        numerical_rank: Number of eigenvalues above ``rel_tol * λ_max``.
        n_degenerate: ``len(knob_names) - numerical_rank`` - the number of
            unconstrained knob combinations.
        degenerate_directions: The near-null directions, weakest first.
        zero_sensitivity_knobs: Knobs whose data sensitivity is (numerically) zero.
        uncertainties: Physical 1σ uncertainty per knob from the *raw* matrix.
        rel_tol: Relative tolerance used for the rank decision.
    """

    knob_names: list[str]
    scaled: bool
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    numerical_rank: int
    n_degenerate: int
    degenerate_directions: list[DegenerateDirection]
    zero_sensitivity_knobs: list[str]
    uncertainties: dict[str, float]
    rel_tol: float

    @property
    def is_degenerate(self) -> bool:
        """True when at least one knob combination is unconstrained by the data."""
        return self.n_degenerate > 0

    def worst_constrained_knobs(self, count: int = 10) -> list[tuple[str, float]]:
        """Knobs with the largest physical uncertainty, worst first."""
        ordered = sorted(self.uncertainties.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:count]

    def summary(self, *, max_directions: int = 10, max_knobs: int = 8) -> str:
        """Human-readable overview suitable for logging before an optimisation."""
        lines: list[str] = []
        verdict = "DEGENERATE" if self.is_degenerate else "well-posed"
        lines.append(
            f"Degeneracy check ({'scaled' if self.scaled else 'raw'}): {verdict} - "
            f"{self.numerical_rank}/{len(self.knob_names)} knob directions constrained, "
            f"{self.n_degenerate} unconstrained."
        )
        cond = self.condition_number
        cond_str = "inf" if not np.isfinite(cond) else f"{cond:.3e}"
        lines.append(f"  condition number = {cond_str} (rel_tol = {self.rel_tol:.1e})")

        if self.zero_sensitivity_knobs:
            preview = ", ".join(self.zero_sensitivity_knobs[:max_knobs])
            extra = len(self.zero_sensitivity_knobs) - max_knobs
            if extra > 0:
                preview += f", ... (+{extra} more)"
            lines.append(f"  zero-sensitivity knobs (data cannot see them): {preview}")

        if self.degenerate_directions:
            lines.append("  unconstrained knob combinations (weakest first):")
            for i, direction in enumerate(self.degenerate_directions[:max_directions]):
                terms = ", ".join(
                    f"{coeff:+.2f}*{name}"
                    for name, coeff in direction.contributions[:max_knobs]
                )
                lines.append(
                    f"    [{i}] rel_eig={direction.relative_eigenvalue:.2e}: {terms}"
                )
            hidden = len(self.degenerate_directions) - max_directions
            if hidden > 0:
                lines.append(f"    ... (+{hidden} more directions)")

        worst = self.worst_constrained_knobs(max_knobs)
        if worst:
            worst_str = ", ".join(f"{name}={sigma:.2e}" for name, sigma in worst)
            lines.append(f"  largest 1σ uncertainties: {worst_str}")
        return "\n".join(lines)


def _extract_contributions(
    components: np.ndarray,
    knob_names: list[str],
    *,
    contribution_threshold: float,
) -> list[tuple[str, float]]:
    """Return dominant ``(knob, coefficient)`` pairs of a unit-norm direction."""
    magnitudes = np.abs(components)
    peak = float(magnitudes.max()) if magnitudes.size else 0.0
    if peak == 0.0:
        return []
    cutoff = contribution_threshold * peak
    order = np.argsort(magnitudes)[::-1]
    return [
        (knob_names[i], float(components[i]))
        for i in order
        if magnitudes[i] >= cutoff
    ]


def analyse_degeneracy(
    normal_matrix: np.ndarray,
    knob_names: list[str],
    *,
    scale: bool = True,
    rel_tol: float = DEFAULT_REL_TOL,
    contribution_threshold: float = 0.1,
    zero_sensitivity_rel: float = DEFAULT_ZERO_SENSITIVITY_REL,
) -> DegeneracyReport:
    """Diagnose unconstrained knob directions from a Gauss-Newton normal matrix.

    Args:
        normal_matrix: The weighted normal matrix ``A = JᵀWJ`` (symmetric, positive
            semi-definite), as accumulated by the tracking workers. Evaluate it at
            the *initial* knobs to diagnose the problem before optimising.
        knob_names: Names of the knobs, in the row/column order of ``normal_matrix``.
        scale: If True (default), analyse the dimensionless symmetrically scaled
            matrix ``D^{-1/2} A D^{-1/2}`` so the verdict reflects
            collinearity rather than unit disparity between knobs.
        rel_tol: A direction is "unconstrained" when its eigenvalue is below
            ``rel_tol * λ_max``.
        contribution_threshold: A knob is listed as contributing to a degenerate
            direction when ``|coefficient| >= contribution_threshold * max|coeff|``.
        zero_sensitivity_rel: A knob is treated as having zero sensitivity when its
            diagonal entry is below ``zero_sensitivity_rel * max(diag)``.

    Returns:
        A :class:`DegeneracyReport`.

    Raises:
        ValueError: If ``normal_matrix`` is not square or ``knob_names`` does not
            match its dimension.
    """
    matrix = np.asarray(normal_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"normal_matrix must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    if len(knob_names) != n:
        raise ValueError(
            f"knob_names has {len(knob_names)} entries but normal_matrix is {n}x{n}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("normal_matrix contains non-finite entries (NaN or inf)")
    knob_names = list(knob_names)

    sym = 0.5 * (matrix + matrix.T)

    # Physical 1σ uncertainties come from the *raw* matrix (real parameter units).
    uncertainty_values = hessian_uncertainties(sym)
    uncertainties = dict(zip(knob_names, (float(u) for u in uncertainty_values)))

    diag = np.diag(sym)
    diag_max = float(diag.max()) if n else 0.0
    zero_sens_mask = diag <= (zero_sensitivity_rel * diag_max if diag_max > 0.0 else 0.0)
    zero_sensitivity_knobs = [knob_names[i] for i in np.nonzero(zero_sens_mask)[0]]

    if scale:
        # Symmetric scaling: unit diagonal for every knob with sensitivity, and an
        # all-zero row/column (hence an exact zero eigenvalue with a unit-vector
        # eigenvector) for every zero-sensitivity knob.
        inv_sqrt = np.zeros(n, dtype=np.float64)
        good = ~zero_sens_mask
        inv_sqrt[good] = 1.0 / np.sqrt(diag[good])
        work = sym * np.outer(inv_sqrt, inv_sqrt)
    else:
        work = sym

    eigenvalues, eigenvectors = np.linalg.eigh(work)
    # eigh returns ascending; reorder to descending for readability.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    lambda_max = float(eigenvalues[0]) if n else 0.0
    # Numerical noise can push tiny eigenvalues slightly negative; clamp for reporting.
    clamped = np.clip(eigenvalues, 0.0, None)
    threshold = rel_tol * lambda_max if lambda_max > 0.0 else 0.0
    constrained_mask = clamped > threshold
    numerical_rank = int(np.count_nonzero(constrained_mask))
    n_degenerate = n - numerical_rank

    positive = clamped[clamped > threshold]
    if positive.size and positive.min() > 0.0 and numerical_rank == n:
        condition_number = float(lambda_max / positive.min())
    else:
        # At least one flat/near-flat direction: treat as singular.
        condition_number = float("inf")

    degenerate_directions: list[DegenerateDirection] = []
    for idx in range(n):
        if constrained_mask[idx]:
            continue
        components = eigenvectors[:, idx]
        # Normalise sign so the dominant component is positive (stable presentation).
        peak_idx = int(np.argmax(np.abs(components)))
        if components[peak_idx] < 0:
            components = -components
        contributions = _extract_contributions(
            components, knob_names, contribution_threshold=contribution_threshold
        )
        degenerate_directions.append(
            DegenerateDirection(
                eigenvalue=float(eigenvalues[idx]),
                relative_eigenvalue=(
                    float(clamped[idx] / lambda_max) if lambda_max > 0.0 else 0.0
                ),
                components=components.copy(),
                contributions=contributions,
            )
        )
    # Weakest (most degenerate) direction first.
    degenerate_directions.sort(key=lambda d: d.relative_eigenvalue)

    return DegeneracyReport(
        knob_names=knob_names,
        scaled=scale,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition_number,
        numerical_rank=numerical_rank,
        n_degenerate=n_degenerate,
        degenerate_directions=degenerate_directions,
        zero_sensitivity_knobs=zero_sensitivity_knobs,
        uncertainties=uncertainties,
        rel_tol=rel_tol,
    )


__all__ = [
    "DEFAULT_REL_TOL",
    "DegeneracyReport",
    "DegenerateDirection",
    "analyse_degeneracy",
]
