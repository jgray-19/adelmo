"""Common data structures and utilities for all worker types.

This module defines shared data structures, configurations, and utility functions
used across different worker implementations (tracking and optics modes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator

logger = logging.getLogger(__name__)

# Eigenvalues of the normal matrix below this floor are treated as unconstrained
# directions: flooring them keeps the inverted covariance finite and non-negative
# instead of exploding (or going negative through numerical noise).
HESSIAN_MIN_EIGENVALUE = 1e-35
# Relative eigenvalue (of the unit-diagonal normal matrix) below which a direction counts as unconstrained by the data.
SINGULAR_REL_TOLERANCE = 1e-8

class KickPlane(str, Enum):
    """Kick-plane options for worker routing and payload selection."""

    X = "x"
    Y = "y"
    XY = "xy"


@dataclass
class WorkerConfig:
    """Configuration shared by all worker processes.

    The accelerator object bundles machine-specific setup, while the remaining
    fields describe the local BPM range, tracking direction, and optional input
    files needed by the worker.

    """

    accelerator: Accelerator
    tracking_start_bpm: str
    tracking_end_bpm: str
    magnet_range: str
    # Per-measurement keyword arguments forwarded to the MAD-NG interface, e.g.
    # machine_state, b2_errors.
    interface_options: dict[str, Any] = field(default_factory=dict)
    observation_range_start_bpm: str | None = None
    initial_condition_marker: str | None = None
    # Whether to cycle the sequence so tracking starts at this worker's init
    # marker. Closed-twiss workers fit the whole ring from ``$start`` and set it
    # False; every tracking plan cycles.
    cycle_sequence: bool = True
    sdir: int = 1
    kick_plane: KickPlane = KickPlane.XY
    bad_bpms: list[str] | None = None
    debug: bool = False
    mad_logfile: Path | None = None
    python_logfile: Path | None = None
    tracking_anchor_mode: str | None = None
    tracking_anchor_sources: list[str] | None = None
    observed_tracking_anchor_markers: list[str] | None = None
    cycle_marker: str | None = None


@dataclass
class PrecomputedTrackingWeights:
    """Per-observable, globally normalised weights for the loss and gradient.

    ``scale`` undoes the normalisation (``weight · scale = 1/σ²``), so the
    uncertainty propagation can report a physical normal matrix.
    """

    x: np.ndarray
    y: np.ndarray
    px: np.ndarray
    py: np.ndarray
    scale: float


@dataclass
class TrackingData:
    """Reference data for a tracking-loss evaluation.

    Position and momentum comparison arrays use shape
    ``(n_particles, n_data_points, 2)``, with the last axis storing the two
    transverse components for each observable family.

    Reading ids identify one measured grid cell (file, turn, marker) across all
    workers, so the uncertainty propagation can add up every use of the same noisy
    reading -- as an observation or as a start coordinate.
    """

    position_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    position_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    init_coords: np.ndarray  # Shape: (n_particles, 6)
    init_pts: np.ndarray  # Shape: (n_particles,)
    reading_ids: np.ndarray  # Shape: (n_particles, n_data_points)
    init_reading_ids: np.ndarray  # Shape: (n_particles,)
    init_variances: np.ndarray  # Shape: (n_particles, 2), var of the start x, y
    precomputed_weights: PrecomputedTrackingWeights | None


@dataclass
class UncertaintyPart:
    """One worker's contribution to the propagated knob covariance.

    ``normal`` is ``Σ w J Jᵀ``. Each row ``r`` of ``sensitivities`` is the change in
    that worker's gradient per unit noise on reading ``reading_ids[r]`` (an
    observation, a start coordinate, or both), with ``variances[r]`` its declared
    variance.
    """

    normal: np.ndarray  # Shape: (n_knobs, n_knobs)
    reading_ids: np.ndarray  # Shape: (n_rows,)
    sensitivities: np.ndarray  # Shape: (n_rows, n_knobs)
    variances: np.ndarray  # Shape: (n_rows,)

    @classmethod
    def empty(cls, n_knobs: int) -> UncertaintyPart:
        """A worker that contributes nothing (disabled, or uncertainty not requested)."""
        return cls(
            normal=np.zeros((n_knobs, n_knobs)),
            reading_ids=np.zeros(0, dtype=np.int64),
            sensitivities=np.zeros((0, n_knobs)),
            variances=np.zeros(0),
        )


class ObservableKind(str, Enum):
    """How a per-BPM observable is turned into a residual.

    ``POINTWISE``
        The model value at each BPM is compared directly against the measurement:
        closed orbit, beta, alpha, dispersion.
    ``ADVANCE``
        Consecutive BPM values are differenced first, so a cumulative model
        quantity is compared against a measured *advance*. Used for phase, where
        only the BPM-to-BPM advance is measurable and the absolute value carries
        an arbitrary origin.
    """

    POINTWISE = "pointwise"
    ADVANCE = "advance"


#: Observables the MAD-NG side knows how to evaluate on the closed twiss, mapped
#: to how their residual is formed. Names are ``gphys.optfun`` function names,
#: except the closed-orbit coordinates which are read off the map's constant part.
#:
#: ``betx``/``bety``/``alfx``/``alfy`` are the coupled Edwards-Teng/physical-plane
#: projections, requested via ``coupling=true`` on the ``twiss{}`` call in
#: ``run_closed_twiss_init.mad``. They are computed live as part of that twiss
#: (the beam is attached throughout), not read back off a bare saved map
#: afterward -- only the orbit coordinates (``x``/``y``/``px``/``py``, via
#: ``ORBIT_COORDS``) take that route, for their knob Jacobian. Never use the
#: uncoupled Ripken ``beta11``/``beta22``/``alfa11``/``alfa22`` columns here:
#: under real coupling their mode/plane identity is not guaranteed to track x/y,
#: so they are not comparable to an omc3 ``BETX``/``ALFX`` measurement by name
#: coincidence alone.
#:
#: ``mu1``/``mu2`` stay mode-indexed: MAD-NG has no ``mux``/``muy`` optical
#: function at all (confirmed against its own name list -- only ``mu``/``dmu``
#: with a mode-index suffix exist), so there is no physical-plane phase-advance
#: column to request in the first place.
#:
#: ``dx``/``dy`` are ``d(x)/d(pt)`` and ``d(y)/d(pt)`` - the MAD-X ``DX`` convention.
#: ``dpx``/``dpy`` are available but should normally be left out when fitting
#: omc3 output: omc3 derives its ``DPY`` from ``DY`` through the *model* transfer
#: matrix, so including both double-counts one measurement.
OBSERVABLE_KINDS: dict[str, ObservableKind] = {
    "x": ObservableKind.POINTWISE,
    "y": ObservableKind.POINTWISE,
    "px": ObservableKind.POINTWISE,
    "py": ObservableKind.POINTWISE,
    "betx": ObservableKind.POINTWISE,
    "bety": ObservableKind.POINTWISE,
    "alfx": ObservableKind.POINTWISE,
    "alfy": ObservableKind.POINTWISE,
    "dx": ObservableKind.POINTWISE,
    "dy": ObservableKind.POINTWISE,
    "dpx": ObservableKind.POINTWISE,
    "dpy": ObservableKind.POINTWISE,
    "mu1": ObservableKind.ADVANCE,
    "mu2": ObservableKind.ADVANCE,
}


@dataclass
class Observable:
    """One measured observable family, aligned to ``ClosedTwissData.bpm_names``.

    ``targets`` and ``variances`` have one entry per BPM for a ``POINTWISE``
    observable and one per *interval* (``n_bpms - 1``) for an ``ADVANCE`` one.
    A non-finite or non-positive variance drops that point from the fit, which is
    how partially-measured planes are handled without special-casing.
    """

    name: str
    targets: np.ndarray
    variances: np.ndarray

    @property
    def kind(self) -> ObservableKind:
        """Residual form for this observable."""
        try:
            return OBSERVABLE_KINDS[self.name]
        except KeyError:
            raise ValueError(
                f"Unknown observable '{self.name}'. Known: {sorted(OBSERVABLE_KINDS)}"
            ) from None


@dataclass
class ClosedTwissData:
    """Reference closed-twiss measurements for one momentum, for one worker.

    Arrays are ordered to match the model BPM ordering (the order the sequence's
    monitors are observed by twiss). ``bpm_names`` records that order so the
    worker can align the twiss output to these comparisons by name.

    ``pt`` is the known MAD-NG momentum coordinate of this measurement.
    It is a fixed input to twiss (``x0map.pt``), never an optimisation knob, so
    both the off-momentum bend response and the dispersive orbit come from the
    physics. Fitting several ``pt`` values jointly makes the per-magnet
    Jacobians independent and lifts the single-measurement rank deficiency.

    ``weight_scale`` and ``total_points`` are the *global* loss normalisation and
    must be identical across every worker in a fit. Each worker divides its
    inverse-variance weights by ``weight_scale`` and its loss/gradient/Hessian by
    ``total_points``. If derived per worker (from its own largest weight and point count), the
    optimiser would minimise ``sum_w (1/(max_w * N_w)) * chi2_w`` instead of the
    pooled ``sum_w chi2_w``. The most precisely measured momentum would then be
    down-weighted, and the reported 1-sigma (built from the un-normalised
    ``JᵀWJ``) would not describe the estimator that was minimised. They are
    computed once, over every worker's observables, by
    :func:`create_worker_payloads`.
    """

    bpm_names: list[str]
    observables: list[Observable]
    pt: float = 0.0
    weight_scale: float = 1.0
    total_points: int = 1


class WeightProcessor:
    """Utility class for processing and normalizing measurement weights.

    Provides static methods for converting variances to weights, normalizing,
    and aggregating weights for use in loss functions and Hessian approximations.
    """

    @staticmethod
    def variance_to_weight(variances: np.ndarray) -> np.ndarray:
        """Convert variances to inverse-variance weights.

        Invalid or non-positive variances are set to zero weight.

        Args:
            variances: Array of variance values

        Returns:
            Array of weights (1/variance for valid entries, 0 for invalid)
        """
        weights = np.zeros_like(variances, dtype=np.float64)
        valid = np.isfinite(variances) & (variances > 0.0)
        np.divide(1.0, variances, out=weights, where=valid)
        return weights

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
            message += " Knobs most involved: " + ", ".join(f"{knob_names[i]} ({weight[i]:.2f})" for i in worst)
        logger.warning(message)
    return n_weak


def merge_uncertainty_parts(parts: list[UncertaintyPart], n_knobs: int) -> UncertaintyPart:
    """Sum the normal matrices and merge rows that share a reading id.

    ``G_e`` adds every part's sensitivity to reading ``e`` (shared between workers, or
    used as both an observation and a start), so :func:`noise_matrix` counts its noise
    once. Merging a merged part with further parts gives the same result.
    """
    normal = np.zeros((n_knobs, n_knobs), dtype=np.float64)
    for part in parts:
        normal += part.normal
    ids = np.concatenate([part.reading_ids for part in parts]) if parts else np.zeros(0, np.int64)
    if ids.size == 0:
        return UncertaintyPart(normal, ids, np.zeros((0, n_knobs)), np.zeros(0))

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    sensitivities = np.concatenate([part.sensitivities for part in parts])[order]
    return UncertaintyPart(
        normal=normal,
        reading_ids=sorted_ids[starts],
        sensitivities=np.add.reduceat(sensitivities, starts, axis=0),
        variances=np.concatenate([part.variances for part in parts])[order][starts],
    )


def noise_matrix(part: UncertaintyPart) -> np.ndarray:
    """``B = Σ_e σ_e² G_e G_eᵀ`` for a merged part; readings without a finite variance add nothing."""
    variances = np.where(np.isfinite(part.variances), part.variances, 0.0)
    return (part.sensitivities * variances[:, None]).T @ part.sensitivities


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
