"""Closed-orbit worker for absolute and reference-subtracted orbit series."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.machine_state import assign_state, read_state
from aba_optimiser.mad.scripts import PYTHON_IN_MAD
from aba_optimiser.workers.closed_twiss import (
    ORBIT_COORDS,
    ClosedTwissWorker,
    align_observables,
    weighted_loss_gradient_hessian,
)
from aba_optimiser.workers.common import ClosedTwissData, Observable
from aba_optimiser.workers.shared_reference import ReferencePublisher, ReferenceReader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pymadng import MAD

    from aba_optimiser.workers.protocol import Evaluate, GradReply

LOGGER = logging.getLogger(__name__)
ORBIT_OBSERVABLES = ("x", "y")
_NO_MESSAGE = object()  # no reference message received yet in this iteration
#: BPM-to-BPM phase advance, from a plain twiss with no reference subtraction --
#: see ClosedOrbitWorker's phase-only series.
PHASE_OBSERVABLES = ("mu1", "mu2")
SUPPORTED_OBSERVABLES = ORBIT_OBSERVABLES + PHASE_OBSERVABLES


@dataclass
class ClosedOrbitMeasurementData:
    """Internal prepared measurement and the model states it compares."""

    observables: list[Observable]
    pt: float = 0.0
    reference_pt: float = 0.0


@dataclass
class ClosedOrbitSeriesData:
    """Internal worker payload for measurements sharing a machine state."""

    bpm_names: list[str]
    measurements: list[ClosedOrbitMeasurementData]
    #: MAD-X globals (quadrupole strengths, corrector kicks, ...) this series is measured at.
    machine_state: dict[str, float] = field(default_factory=dict)
    #: The state the reference orbit is solved at; a relative plane is the change from it to ``machine_state``.
    reference_state: dict[str, float] = field(default_factory=dict)
    absolute_planes: tuple[str, ...] = ()
    weight_scale: float = 1.0
    total_points: int = 1
    #: Gain layout for :class:`~aba_optimiser.workers.calibrated_closed_orbit.CalibratedClosedOrbitWorker`.
    calibration: object | None = None
    #: Take the reference orbit from the reference worker (see :mod:`aba_optimiser.workers.shared_reference`)
    #: instead of solving it here. The worker then expects one reference message per iteration.
    shared_reference: bool = False
    #: This worker only solves and publishes the reference orbits, once per iteration, for every series.
    reference_only: bool = False

    @property
    def needs_reference(self) -> bool:
        """Whether any orbit plane of this series is fitted relative to a reference state."""
        observables = self.measurements[0].observables
        return any(o.name in ORBIT_OBSERVABLES and o.name not in self.absolute_planes for o in observables)

    @property
    def observables(self) -> list[Observable]:
        """Observable blocks of every measurement, as :attr:`ClosedTwissData.observables`."""
        return [
            observable
            for measurement in self.measurements
            for observable in measurement.observables
        ]


@dataclass
class SeriesState:
    """What a worker keeps for one of its series: the prepared data and its alignment to the model BPMs."""

    #: Position of the series in its worker; part of the closed-orbit warm-start key.
    index: int
    data: ClosedOrbitSeriesData
    #: Observables of every measurement of the series, in the order the rows are evaluated.
    observables: list[Observable]
    measured_index: dict[str, int]
    #: 1 for an orbit plane fitted relative to the reference state, 0 for an absolute one.
    subtract: np.ndarray
    #: Model BPM order the alignments below were made for.
    twiss_bpm_order: list[str] | None = None
    #: Per measurement: ``(observables, targets, weights, raw_weights)`` in model BPM order.
    alignment: list[tuple] | None = None

    def align(self, twiss_names: list[str], weight_scale: float, worker_id: int) -> None:
        """Align every measurement's targets and weights to the model BPM order."""
        alignment = []
        for measurement in self.data.measurements:
            targets, raw_weights, weights = align_observables(
                measurement.observables,
                self.measured_index,
                twiss_names,
                weight_scale,
                worker_id=worker_id,
            )
            alignment.append((measurement.observables, targets, weights, raw_weights))
        self.alignment = alignment
        self.twiss_bpm_order = twiss_names


def _validate_series(data: ClosedOrbitSeriesData) -> tuple[str, ...]:
    """Check a series' observables and planes; return its absolute planes."""
    if not data.measurements:
        raise ValueError("ClosedOrbitSeriesData needs at least one measurement")
    first = data.measurements[0]
    if not first.observables:
        raise ValueError("A closed-orbit measurement needs at least one observable")

    names = tuple(observable.name for observable in first.observables)
    unsupported = [name for name in names if name not in SUPPORTED_OBSERVABLES]
    if unsupported:
        raise ValueError(f"ClosedOrbitWorker supports {SUPPORTED_OBSERVABLES}, got {unsupported}")
    if any(name in PHASE_OBSERVABLES for name in names) and any(
        name in ORBIT_OBSERVABLES for name in names
    ):
        # ``subtract`` is sized to the orbit coordinates alone; a mixed series would
        # silently apply the orbit reference-subtraction mask to the phase rows too.
        raise ValueError(f"A closed-orbit series cannot mix {ORBIT_OBSERVABLES} and {PHASE_OBSERVABLES}: got {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate closed-orbit observables: {names}")
    for measurement in data.measurements[1:]:
        other = tuple(observable.name for observable in measurement.observables)
        if other != names:
            raise ValueError(
                "Every measurement in a closed-orbit series must use the same "
                f"observable order, got {other} after {names}"
            )

    # Phase advance is BPM-to-BPM within a single twiss, never a delta against a
    # reference state, so a phase-only series takes exactly one measurement.
    if all(name in PHASE_OBSERVABLES for name in names):
        if len(data.measurements) != 1:
            raise ValueError("A phase-only closed-orbit series takes exactly one measurement")
        return ()

    absolute = tuple(dict.fromkeys(data.absolute_planes))
    unknown = set(absolute) - set(names)
    if unknown:
        raise ValueError(f"Unknown absolute plane(s) {sorted(unknown)}")
    has_relative = any(name not in absolute for name in names)
    has_momentum_signal = any(
        measurement.pt != measurement.reference_pt for measurement in data.measurements
    )
    if (
        data.machine_state == data.reference_state
        and not has_momentum_signal
        and has_relative
        and not absolute
    ):
        raise ValueError(
            "A closed-orbit series needs a machine state that differs from the reference, "
            "a momentum change, or an absolute plane"
        )
    return absolute


class ClosedOrbitWorker(ClosedTwissWorker):
    """Fit one or more closed-orbit series in one MAD-NG process.

    Each series is a :class:`ClosedOrbitSeriesData`: its machine state and all its
    measurements. Every measurement keeps its own target, signal momentum and
    reference momentum. The losses, gradients and Hessians of the series are
    summed, so the fit is the one a process per series would give. Repeated model
    states are cached within one series evaluation only.
    """

    def prepare_data(self, data: list[ClosedOrbitSeriesData]) -> None:
        if not data:
            raise ValueError("A closed-orbit worker needs at least one series")
        self.series: list[SeriesState] = []
        for index, series in enumerate(data):
            absolute = _validate_series(series)
            first = series.measurements[0]
            # The closed-twiss setup takes the first series: every series in a worker
            # fits the same observables (checked below) on one global normalisation.
            if index == 0:
                super().prepare_data(
                    ClosedTwissData(
                        bpm_names=series.bpm_names,
                        observables=first.observables,
                        pt=first.pt,
                        weight_scale=series.weight_scale,
                        total_points=series.total_points,
                    )
                )
            names = [o.name for o in first.observables]
            if [n for n in names if n in ORBIT_COORDS] + [n for n in names if n not in ORBIT_COORDS] != (
                self.orbit_coords + self.optics_names
            ):
                raise ValueError(f"Every series in a worker needs the same observables, got {names}")
            self.series.append(
                SeriesState(
                    index=index,
                    data=series,
                    observables=list(first.observables),
                    measured_index={name: i for i, name in enumerate(series.bpm_names)},
                    subtract=np.array(
                        [float(name not in absolute) for name in self.orbit_coords], dtype=float
                    ),
                )
            )
        self.shared_reference = data[0].shared_reference
        self.reference_only = data[0].reference_only
        #: Model values of every global a series state sets, read on first use.
        self._baseline: dict[str, float] = {}
        self._states: dict[str, dict[str, float]] = {}
        self._reference_message = _NO_MESSAGE
        self._publisher: ReferencePublisher | None = None
        self._reference_reader: ReferenceReader | None = None

    # ------------------------------------------------------------------
    # Machine states
    # ------------------------------------------------------------------

    @contextmanager
    def _in_series(self, mad: MAD, series: SeriesState):
        """Make the series' signal and reference states available; on exit restore every global touched.

        The signal state is the model's own values under ``machine_state``, the
        reference state the same under ``reference_state``. The restore is what
        stops one series of a worker leaking its settings into the next.
        """
        machine_state, reference_state = series.data.machine_state, series.data.reference_state
        unread = [name for name in (*machine_state, *reference_state) if name not in self._baseline]
        if unread:
            self._baseline.update(read_state(mad, PYTHON_IN_MAD, unread))
        self._states = {
            "signal": {**self._baseline, **machine_state},
            "reference": {**self._baseline, **reference_state},
        }
        try:
            yield
        finally:
            assign_state(mad, self._baseline)

    def _enter_state(self, mad: MAD, role: str, knob_updates: dict[str, float]) -> None:
        """Put the machine in the series' ``"signal"`` state or in the ``"reference"`` state."""
        del knob_updates
        assign_state(mad, self._states[role])

    @staticmethod
    def _set_pt(mad: MAD, value: float) -> None:
        mad.send(f"x0map.pt:set0({value:.15e})")

    @staticmethod
    def _set_co_key(mad: MAD, series: SeriesState, role: str, pt: float) -> None:
        """Warm-start the next closed-orbit solve from this machine state's own last orbit.

        The state is (series index in this worker, signal/reference role, momentum).
        """
        mad.send(f"co_key = '{series.index}:{role}:{pt:.12e}'")

    # ------------------------------------------------------------------
    # Solves
    # ------------------------------------------------------------------

    def _model_and_jacobian(self, mad: MAD, series: SeriesState, *, context: str = ""):
        """``(values, jacobian)`` of the series' observables at the loaded state, or ``None`` if the orbit is lost."""
        solved = self._solve(mad, series.observables, context=context)
        if solved is None:
            return None
        twiss_names, values, jacobian = solved
        if series.twiss_bpm_order != twiss_names:
            series.align(twiss_names, self.weight_scale, self.worker_id)
        return values, jacobian

    def _orbit_plain(self, mad: MAD, series: SeriesState, *, context: str = ""):
        """Closed-orbit values ``(n_coords, n_bpms)`` without knob derivatives (``compute_orbit_plain``); ``None`` if the orbit is lost.

        The knobs must be plain constants (``knobs_to_plain``) and the series orbit-only.
        """
        if self.optics_names:
            raise NotImplementedError("loss-only evaluation supports orbit observables only")
        mad.send("compute_orbit_plain()")
        try:
            found = mad.recv()
        except RuntimeError as exc:  # MAD raised: a failed trial step, like a lost orbit
            LOGGER.warning("Worker %s: MAD error in the loss-only orbit%s (%s); flagging step for backtrack", self.worker_id, context, exc)
            return None
        if not found:
            LOGGER.warning("Worker %s: closed orbit not found%s; flagging step for backtrack", self.worker_id, context)
            return None
        if series.twiss_bpm_order is None:
            raise RuntimeError("loss-only orbit needs the BPM order from a prior twiss evaluation")
        values = np.empty((len(self.orbit_coords), len(series.twiss_bpm_order)))
        for index in range(len(self.orbit_coords)):
            values[index] = np.asarray(mad.recv(), dtype=float).ravel()
        return values

    def _models(
        self,
        mad: MAD,
        series: SeriesState,
        knob_updates: dict[str, float],
        solve: Callable,
    ) -> Iterator[tuple[int, object]]:
        """Yield ``(measurement index, model)`` for every measurement of the series; the model is ``None`` once an orbit is lost.

        ``solve(mad, series, context=...)`` evaluates the loaded state. A relative
        plane compares the signal state with the reference state; each ``(role,
        pt)`` state is solved once. Call inside :meth:`_in_series`.
        """
        cache = {}

        def evaluate(role: str, pt: float):
            key = (role, pt)
            if key not in cache:
                self._enter_state(mad, role, knob_updates)
                self._set_pt(mad, pt)
                self._set_co_key(mad, series, role, pt)
                cache[key] = solve(mad, series, context=f" ({role}, pt={pt:+.9g})")
            return cache[key]

        for index, measurement in enumerate(series.data.measurements):
            signal = evaluate("signal", float(measurement.pt))
            if signal is None or not np.any(series.subtract):
                yield index, signal
                if signal is None:
                    return
                continue
            reference = self._reference(evaluate, measurement)
            if reference is None:
                yield index, None
                return
            yield index, self._compare_to_reference(series, signal, reference)

    @staticmethod
    def _compare_to_reference(series: SeriesState, signal, reference):
        """Subtract the reference only in the relative observable planes."""
        if isinstance(signal, np.ndarray):  # values alone (loss-only)
            return signal - reference * series.subtract[:, None]
        model = signal[0] - reference[0] * series.subtract[:, None]
        jacobian = signal[1] - reference[1] * series.subtract[:, None, None]
        return model, jacobian

    # ------------------------------------------------------------------
    # Shared reference
    # ------------------------------------------------------------------

    @contextmanager
    def _reference_round(self):
        """One iteration of a signal worker: exactly one reference message is consumed, however it ends."""
        self._reference_message = _NO_MESSAGE
        try:
            yield
        finally:
            if self.shared_reference and self._reference_message is _NO_MESSAGE:
                self._reference_message = self.conn.recv()

    def _reference(self, evaluate, measurement):
        """Reference ``(orbit, jacobian)`` of one measurement: solved here, or taken from the reference worker."""
        reference_pt = float(measurement.reference_pt)
        if not self.shared_reference:
            return evaluate("reference", reference_pt)
        if self._reference_message is _NO_MESSAGE:
            self._reference_message = self.conn.recv()
        message = self._reference_message
        if message is None:  # the reference worker lost the closed orbit
            return None
        if self._reference_reader is None:
            self._reference_reader = ReferenceReader()
        block = self._reference_reader.view(message)[message.reference_pts.index(reference_pt)]
        rows = message.rows(self.orbit_coords)
        return block[rows, :, 0], block[rows, :, 1:]

    def _publish_reference(self, mad: MAD, knob_updates: dict[str, float]) -> GradReply:
        """Reference worker: solve the reference state of every distinct reference momentum and publish them."""
        if self._publisher is None:
            self._publisher = ReferencePublisher()
        series = self.series[0]
        reference_pts = tuple(dict.fromkeys(float(m.reference_pt) for m in series.data.measurements))
        states = []
        n = self.n_knobs
        with self._in_series(mad, series):
            for reference_pt in reference_pts:
                self._enter_state(mad, "reference", knob_updates)
                self._set_pt(mad, reference_pt)
                self._set_co_key(mad, series, "reference", reference_pt)
                state = self._model_and_jacobian(mad, series, context=f" (shared reference, pt={reference_pt:+.9g})")
                if state is None:
                    return self._reply(np.zeros(n), float("nan"), np.zeros((n, n)), np.zeros((n, n)))
                states.append(state)
        published = self._publisher.publish(states, reference_pts, tuple(self.orbit_coords))
        return self._reply(np.zeros(n), 0.0, np.zeros((n, n)), np.zeros((n, n)), extra=published)

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
        if self._reference_reader is not None:
            self._reference_reader.close()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, mad: MAD, message: Evaluate) -> GradReply:
        """Sum the loss, gradient and Hessians of every series at ``message.knobs``."""
        self.send_knobs(mad, message.knobs)
        if self.reference_only:
            return self._publish_reference(mad, message.knobs)
        n = self.n_knobs
        gradient = np.zeros(n)
        loss = 0.0
        hessian = np.zeros((n, n))
        normal_matrix = np.zeros((n, n))
        with self._reference_round():
            for series in self.series:
                part = self._evaluate_series(mad, series, message.knobs)
                if part is None:
                    return self._failure()
                gradient += part[0]
                loss += part[1]
                hessian += part[2]
                normal_matrix += part[3]
        return self._reply(gradient, loss, hessian, normal_matrix)

    def _evaluate_series(self, mad: MAD, series: SeriesState, knob_updates: dict[str, float]):
        """Loss, gradient and Hessians of one series; ``None`` if a closed orbit is lost."""
        n = self.n_knobs
        gradient = np.zeros(n)
        loss = 0.0
        hessian = np.zeros((n, n))
        normal_matrix = np.zeros((n, n))
        with self._in_series(mad, series):
            for index, model in self._models(mad, series, knob_updates, self._model_and_jacobian):
                if model is None:
                    return None
                observables, targets, weights, raw_weights = series.alignment[index]
                part = weighted_loss_gradient_hessian(
                    *model, observables, targets, weights, raw_weights, self.weight_scale
                )
                gradient += part[0]
                loss += part[1]
                hessian += part[2]
                normal_matrix += part[3]
        return gradient, loss, hessian, normal_matrix
