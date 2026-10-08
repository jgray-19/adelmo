"""Closed-twiss worker: fit knobs to a measured periodic optics solution.

Runs one MAD-NG ``twiss`` per iteration with the optimisation knobs installed as
TPSA parameters on the sequence. twiss finds the parametric closed orbit with
``cofind`` and normalises the parametric one-turn map, so the closed orbit, beta,
alpha, phase and dispersion at every BPM are the *periodic* solution of the ring
as a function of the knobs.

When only orbit coordinates are observed the normal form is dead weight, so the
worker calls ``cofind`` and a single parametric ``track`` instead
(``compute_closed_orbit`` in ``run_closed_twiss_init.mad``), on a first-order map.
It returns the same orbit and Jacobian as twiss.

The optical functions and their knob derivatives are requested through twiss's
own ``trkopt`` list, which fills one mtable column per requested name; the knob
monomial is encoded in the name, so ``beta11_`` is the value and
``beta11_0..010..0`` is its derivative with respect to knob ``i``. The closed
orbit is the exception - ``x``/``y`` are not optical functions, so their scalar
comes from the ordinary twiss column and their Jacobian from the saved map.

Nothing is seeded from the measurement. There is no starting point to propagate
from, so no measurement noise enters as an initial condition and no error can be
absorbed by an assumed anchor; every residual is attributable to the magnets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from adelmo.fitting.protocol import GradReply
from adelmo.fitting.weights import variance_to_weight
from adelmo.fitting.worker import AbstractWorker
from adelmo.machine.mad.scripts import CLOSED_TWISS_INIT

if TYPE_CHECKING:
    from pymadng import MAD

    from adelmo.fitting.protocol import Evaluate

LOGGER = logging.getLogger(__name__)

#: Observables that are plain phase-space coordinates rather than optical
#: functions. ``gphys.optfun`` has no entry for these, so their value comes from
#: the ordinary twiss column and their knob derivative from the saved map.
ORBIT_COORDS = ("x", "px", "y", "py")


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


def read_orbit_only(mad: MAD, n_coords: int, n_knobs: int):
    """Receive BPM names, orbit values and Jacobian from ``compute_closed_orbit``."""
    mad.send("send_orbit_only()")
    names = list(mad.recv())
    values = np.empty((n_coords, len(names)))
    jacobian = np.empty((n_coords, len(names), n_knobs))
    for i in range(n_coords):
        values[i] = np.asarray(mad.recv(), dtype=float).ravel()
        jacobian[i] = np.asarray(mad.recv(), dtype=float).reshape(len(names), n_knobs)
    return names, values, jacobian


class ClosedTwissWorker(AbstractWorker[ClosedTwissData]):
    """Worker that fits knobs to a measured closed-twiss solution."""

    def prepare_data(self, data: ClosedTwissData) -> None:
        """Store the measured observables and load the MAD-NG init script."""
        if not data.observables:
            raise ValueError("ClosedTwissData carries no observables to fit")

        LOGGER.debug(
            "Worker %s: closed-twiss data for %d BPMs, observables %s",
            self.worker_id,
            len(data.bpm_names),
            [obs.name for obs in data.observables],
        )

        self.observables = list(data.observables)
        # Closed-orbit coordinates are not optical functions, so they take the
        # saved-map route; everything else goes through twiss's trkopt columns.
        self.orbit_coords = [obs.name for obs in self.observables if obs.name in ORBIT_COORDS]
        self.optics_names = [obs.name for obs in self.observables if obs.name not in ORBIT_COORDS]
        # No optical function to normalise for: skip twiss. The MAD-side map order
        # is chosen from the same condition (an empty ``optics_columns``).
        self.closed_solver = (
            "compute_closed_twiss()" if self.optics_names else "compute_closed_orbit()"
        )
        self._measured_index = {name: i for i, name in enumerate(data.bpm_names)}
        # Known momentum offset of this measurement; pinned on x0map.pt (not a knob).
        self.pt = float(data.pt)
        # Filled on the first compute once the twiss BPM ordering is known.
        self._twiss_bpm_order: list[str] | None = None
        # Global loss normalisation, identical for every worker in the fit. See
        # ``ClosedTwissData`` for why it must not be derived per worker.
        self.weight_scale = float(data.weight_scale)
        if not np.isfinite(self.weight_scale) or self.weight_scale <= 0.0:
            raise ValueError(f"Worker {self.worker_id}: weight_scale must be finite and positive")
        self.normalisation_points = max(1, int(data.total_points))

        self.init_text = self._strip_comment_lines(CLOSED_TWISS_INIT.read_text())

    @staticmethod
    def _strip_comment_lines(text: str) -> str:
        """Remove full-line comments and blank lines before sending to MAD-NG."""
        kept = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("--") or stripped.startswith("!"):
                continue
            kept.append(line)
        return "\n".join(kept)

    def _setup_da_maps(self, mad: MAD) -> None:
        """Build the parametric knob map and observable readout in MAD-NG.

        ``pt`` is never an optimisation knob here: the momentum is a fixed input
        pinned on the parametric map's ``pt`` coordinate so cofind returns this
        worker's off-momentum closed solution (dispersion + off-momentum bend kick).
        """
        knob_names = list(mad["knob_names"])
        if "pt" in knob_names:
            knob_names.remove("pt")
            mad["knob_names"] = knob_names

        # One trkopt column per (optical function, knob monomial), values first so
        # the returned frame splits cleanly into a value block and a Jacobian block.
        self.n_knobs = len(knob_names)
        self.value_columns = [f"{name}_" for name in self.optics_names]
        self.derivative_columns = [
            f"{name}_{_knob_monomial(i, self.n_knobs)}"
            for name in self.optics_names
            for i in range(self.n_knobs)
        ]
        mad["optics_columns"] = self.value_columns + self.derivative_columns
        mad["orbit_coords"] = self.orbit_coords
        mad.send(self.init_text)
        mad.send(f"x0map.pt:set0({self.pt:.15e})")

    def _solve(self, mad: MAD, observables: list[Observable], *, context: str = ""):
        """Solve the closed orbit (and twiss) at the loaded knobs.

        Returns ``(bpm_names, values, jacobian)`` with ``values`` of shape
        ``(n_observables, n_bpms)`` and ``jacobian`` of shape ``(n_observables,
        n_bpms, n_knobs)``, rows in the order of ``observables``; ``None`` when the
        closed orbit is lost or MAD raised (an unstable trial point), so the
        optimiser backtracks.
        """
        mad.send(self.closed_solver)
        try:
            found = mad.recv()
        except RuntimeError as exc:
            LOGGER.warning(
                "Worker %s: MAD error in the closed-orbit solve%s (%s); flagging step for backtrack",
                self.worker_id,
                context,
                exc,
            )
            return None
        if not found:
            LOGGER.warning(
                "Worker %s: closed orbit not found%s; flagging step for backtrack",
                self.worker_id,
                context,
            )
            return None

        n_knobs = self.n_knobs
        if not self.optics_names:  # orbit coordinates only: already in observable order
            return read_orbit_only(mad, len(self.orbit_coords), n_knobs)

        columns = ["name", *self.orbit_coords, *self.value_columns, *self.derivative_columns]
        frame = mad.closed_tbl.to_df(columns=columns)
        twiss_names = list(frame["name"])
        n_bpms = len(twiss_names)
        n_optics = len(self.optics_names)
        optics_values = frame[self.value_columns].to_numpy(dtype=float).T
        optics_jacobian = (
            frame[self.derivative_columns]
            .to_numpy(dtype=float)
            .reshape(n_bpms, n_optics, n_knobs)
            .transpose(1, 0, 2)
        )
        orbit_values = frame[list(self.orbit_coords)].to_numpy(dtype=float).T
        orbit_jacobian = np.empty((len(self.orbit_coords), n_bpms, n_knobs))
        if self.orbit_coords:
            mad.send("send_orbit_jacobian()")
            for i in range(len(self.orbit_coords)):
                orbit_jacobian[i] = np.asarray(mad.recv(), dtype=float).reshape(n_bpms, n_knobs)

        # Re-interleave into the caller's observable order.
        values = np.empty((len(observables), n_bpms))
        jacobian = np.empty((len(observables), n_bpms, n_knobs))
        orbit_iter, optics_iter = iter(range(len(self.orbit_coords))), iter(range(n_optics))
        for i, obs in enumerate(observables):
            if obs.name in ORBIT_COORDS:
                source = next(orbit_iter)
                values[i], jacobian[i] = orbit_values[source], orbit_jacobian[source]
            else:
                source = next(optics_iter)
                values[i], jacobian[i] = optics_values[source], optics_jacobian[source]
        return twiss_names, values, jacobian

    def _reply(
        self, grad: np.ndarray, loss: float, hessian: np.ndarray, normal_matrix: np.ndarray, extra=None
    ) -> GradReply:
        """Normalise by the global point count; the physical ``normal`` stays un-normalised.

        Summed across workers, the inverse of ``normal`` is the covariance in real units.
        """
        points = self.normalisation_points
        return GradReply(self.worker_id, loss / points, grad / points, hessian / points, normal_matrix, extra)

    def _failure(self) -> GradReply:
        """Reply for a lost closed orbit: NaN loss, so the optimiser backtracks."""
        n = self.n_knobs
        return self._reply(np.zeros(n), float("nan"), np.zeros((n, n)), np.zeros((n, n)))

    def evaluate(self, mad: MAD, message: Evaluate) -> GradReply:
        """Closed twiss at ``message.knobs`` and its weighted least-squares derivatives."""
        self.send_knobs(mad, message.knobs)
        solved = self._solve(mad, self.observables)
        if solved is None:
            return self._failure()
        twiss_names, model, jacobian = solved
        if self._twiss_bpm_order != twiss_names:
            self.targets, self.raw_weights, self.weights = align_observables(
                self.observables,
                self._measured_index,
                twiss_names,
                self.weight_scale,
                worker_id=self.worker_id,
            )
            self._twiss_bpm_order = twiss_names
        # Every observable family contributes an independent block to the same
        # ``2 JᵀWJ`` normal equations; the inverse-variance weights make families in
        # different units commensurable, so no per-family scaling is applied.
        return self._reply(
            *weighted_loss_gradient_hessian(
                model,
                jacobian,
                self.observables,
                self.targets,
                self.weights,
                self.raw_weights,
                self.weight_scale,
            )
        )


def align_observables(
    observables,
    measured_index: dict[str, int],
    twiss_names: list[str],
    weight_scale: float,
    *,
    worker_id: int,
):
    """Align observable targets and weights to the model BPM ordering."""
    missing = [name for name in twiss_names if name not in measured_index]
    if missing:
        raise RuntimeError(
            f"Worker {worker_id}: {len(missing)} observed BPMs have no measurement, "
            f"e.g. {missing[:5]}"
        )
    order = np.array([measured_index[name] for name in twiss_names])
    if not np.all(np.diff(order) == 1):
        raise RuntimeError(
            f"Worker {worker_id}: the observed BPMs are not a contiguous run of "
            "the measured ordering; phase advances cannot be aligned by interval."
        )

    targets = []
    raw_weights = []
    for observable in observables:
        indices = order if observable.kind is ObservableKind.POINTWISE else order[:-1]
        targets.append(np.asarray(observable.targets, dtype=float)[indices])
        raw_weights.append(
            variance_to_weight(
                np.asarray(observable.variances, dtype=float)[indices]
            )
        )
    return targets, raw_weights, [weight / weight_scale for weight in raw_weights]


def weighted_loss_gradient_hessian(
    model: np.ndarray,
    jacobian: np.ndarray,
    observables,
    targets,
    weights,
    raw_weights,
    weight_scale: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Evaluate weighted least squares for explicitly supplied observable blocks.

    ``weights`` are ``raw_weights / weight_scale`` (see :func:`align_observables`), so
    the Gauss-Newton Hessian ``2 JᵀWJ`` is the physical normal matrix ``Jᵀ W_raw J``
    times ``2 / weight_scale``. Only the latter is formed, once per observable, as a
    symmetric ``Aᵀ A`` product of the sqrt-weighted Jacobian.
    """
    n_knobs = jacobian.shape[-1]
    grad = np.zeros(n_knobs)
    normal_matrix = np.zeros((n_knobs, n_knobs))
    loss = 0.0

    for index, observable in enumerate(observables):
        values, jac = model[index], jacobian[index]
        if observable.kind is ObservableKind.ADVANCE:
            values, jac = _to_advance(values, jac)

        weight, raw_weight = weights[index], raw_weights[index]
        if not np.allclose(weight * weight_scale, raw_weight, rtol=1e-9, atol=0.0):
            raise ValueError("weights must equal raw_weights / weight_scale")
        residual = np.where(weight > 0.0, values - targets[index], 0.0)
        grad += 2.0 * (weight * residual) @ jac
        weighted_jac = jac * np.sqrt(raw_weight)[:, None]
        normal_matrix += weighted_jac.T @ weighted_jac  # numpy uses a symmetric rank-k update
        loss += float(np.sum(weight * residual**2))

    return grad, loss, (2.0 / weight_scale) * normal_matrix, normal_matrix


def _knob_monomial(index: int, n_knobs: int) -> str:
    """Parameter-monomial suffix selecting d/d(knob ``index``) in a trkopt name.

    ``gphys.nf_pk`` reads the ``_`` separator as the six phase-space slots, so
    only the parameter part belongs in the suffix.
    """
    return "0" * index + "1" + "0" * (n_knobs - index - 1)


def _to_advance(values: np.ndarray, jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a cumulative per-BPM quantity into BPM-to-BPM advances.

    ``gphys`` returns the phase from ``atan2``, so it is wrapped into a unit
    interval rather than accumulated around the ring. Reducing the consecutive
    difference modulo 1 undoes that, which is unambiguous as long as no BPM pair
    is separated by a full unit of phase - true of every real BPM layout, and the
    same assumption omc3 makes when it reports an advance in ``[0, 1)``.

    The modulo is locally the identity, so the Jacobian of the advance is the
    difference of consecutive Jacobian rows.
    """
    return np.mod(np.diff(values), 1.0), np.diff(jacobian, axis=0)
