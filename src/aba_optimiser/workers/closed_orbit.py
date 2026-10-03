"""Closed-orbit worker for absolute and reference-subtracted orbit series."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.workers.closed_twiss import (
    ORBIT_COORDS,
    ClosedTwissWorker,
    _align_observables,
    _weighted_loss_gradient_hessian,
    read_orbit_only,
)
from aba_optimiser.mad.scripts import PYTHON_IN_MAD
from aba_optimiser.workers.common import ClosedTwissData, Observable
from aba_optimiser.workers.shared_reference import ReferencePublisher, ReferenceReader

if TYPE_CHECKING:
    from pymadng import MAD

LOGGER = logging.getLogger(__name__)
ORBIT_OBSERVABLES = ("x", "y")
_NO_MESSAGE = object()  # no reference message received yet in this iteration
#: BPM-to-BPM phase advance, from a plain twiss with no reference subtraction --
#: see ClosedOrbitWorker.prepare_data's phase-only branch.
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
    """Internal worker payload for measurements sharing a control setting."""

    bpm_names: list[str]
    measurements: list[ClosedOrbitMeasurementData]
    control_knob: str | None = None
    control_nominal: float = 0.0
    control_delta: float = 0.0
    absolute_planes: tuple[str, ...] = ()
    weight_scale: float = 1.0
    total_points: int = 1
    #: Gain layout for :class:`~aba_optimiser.workers.calibrated_closed_orbit.CalibratedClosedOrbitBatchWorker`.
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
    def all_observables(self) -> list[Observable]:
        """Observable blocks contributing to this worker's loss."""
        return [
            observable
            for measurement in self.measurements
            for observable in measurement.observables
        ]


class ClosedOrbitWorker(ClosedTwissWorker):
    """Fit several closed-orbit measurements in one MAD-NG process.

    Every measurement retains its own target, signal momentum, and reference
    momentum. Repeated model states are cached only within an iteration; this
    makes a shared global reference cheap without conflating the signal closed
    orbits at different momenta.
    """

    def prepare_data(self, data: ClosedOrbitSeriesData) -> None:
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
            # _subtract is sized to orbit_coords alone; a mixed series would
            # silently apply the orbit reference-subtraction mask to the phase
            # rows too. Not needed here (orbit and phase are separate series),
            # so it is refused rather than left to misbehave quietly.
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

        # Phase advance is BPM-to-BPM within a single twiss, never a delta
        # against a reference state, so a phase-only series carries no control
        # knob and exactly one measurement -- the degenerate-signal check below
        # (built for x/y's absolute/relative distinction) does not apply to it.
        is_phase_only = all(name in PHASE_OBSERVABLES for name in names)
        if is_phase_only:
            if data.control_knob is not None or data.control_delta != 0.0:
                raise ValueError("A phase-only closed-orbit series cannot carry a control knob")
            if len(data.measurements) != 1:
                raise ValueError(
                    "A phase-only closed-orbit series takes exactly one measurement"
                )
            absolute: tuple[str, ...] = ()
        else:
            absolute = tuple(dict.fromkeys(data.absolute_planes))
            unknown = set(absolute) - set(names)
            if unknown:
                raise ValueError(f"Unknown absolute plane(s) {sorted(unknown)}")
            has_relative = any(name not in absolute for name in names)
            has_momentum_signal = any(
                measurement.pt != measurement.reference_pt for measurement in data.measurements
            )
            if (
                data.control_delta == 0.0
                and not has_momentum_signal
                and has_relative
                and not absolute
            ):
                raise ValueError(
                    "A closed-orbit series needs a control delta, a momentum change, "
                    "or an absolute plane"
                )

        proxy = ClosedTwissData(
            bpm_names=data.bpm_names,
            observables=first.observables,
            pt=first.pt,
            weight_scale=data.weight_scale,
            total_points=data.total_points,
        )
        super().prepare_data(proxy)
        self.series_measurements = list(data.measurements)
        self.control_knob = data.control_knob
        self.control_nominal = float(data.control_nominal)
        self.control_delta = float(data.control_delta)
        self.absolute_planes = absolute
        self.calibration = data.calibration
        self.shared_reference = data.shared_reference
        self.reference_only = data.reference_only
        self._reference_message = _NO_MESSAGE
        self._checked_controls: set[str] = set()
        self._subtract = np.array(
            [float(name not in absolute) for name in self.orbit_coords], dtype=float
        )
        self._measurement_alignment = None

    def _set_control(self, mad: MAD, value: float) -> None:
        if self.control_knob is not None:
            mad.send(f"MADX['{self.control_knob}'] = {value:.15e}")

    @staticmethod
    def _set_pt(mad: MAD, value: float) -> None:
        mad.send(f"x0map.pt:set0({value:.15e})")

    def _align_measurements(self, twiss_names: list[str]) -> None:
        alignments = []
        for measurement in self.series_measurements:
            targets, raw_weights, weights = _align_observables(
                measurement.observables,
                self._measured_index,
                twiss_names,
                self.weight_scale,
                worker_id=self.worker_id,
            )
            alignments.append((measurement.observables, targets, weights, raw_weights))
        self._measurement_alignment = alignments
        self._twiss_bpm_order = twiss_names

    def _model_and_jacobian(self, mad: MAD, *, context: str = ""):
        mad.send(self.closed_solver)
        if not mad.recv():
            LOGGER.warning(
                "Worker %s: closed orbit not found%s; flagging step for backtrack",
                self.worker_id,
                context,
            )
            return None

        n_knobs = self.n_knobs
        n_optics = len(self.optics_names)
        if self.optics_names:
            columns = ["name", *self.orbit_coords, *self.value_columns, *self.derivative_columns]
            frame = mad.closed_tbl.to_df(columns=columns)
            twiss_names = list(frame["name"])
        else:
            twiss_names, orbit_values, orbit_jacobian = read_orbit_only(
                mad, len(self.orbit_coords), n_knobs
            )
        if self._twiss_bpm_order != twiss_names:
            self._align_measurements(twiss_names)

        n_bpms = len(twiss_names)
        if self.optics_names:
            orbit_values = frame[list(self.orbit_coords)].to_numpy(dtype=float).T
            orbit_jacobian = np.empty((len(self.orbit_coords), n_bpms, n_knobs))
            if self.orbit_coords:
                mad.send("send_orbit_jacobian()")
                for index in range(len(self.orbit_coords)):
                    orbit_jacobian[index] = np.asarray(mad.recv(), dtype=float).reshape(
                        n_bpms, n_knobs
                    )
            optics_values = frame[self.value_columns].to_numpy(dtype=float).T
            optics_jacobian = (
                frame[self.derivative_columns]
                .to_numpy(dtype=float)
                .reshape(n_bpms, n_optics, n_knobs)
                .transpose(1, 0, 2)
            )
        else:
            optics_values = np.empty((0, n_bpms))
            optics_jacobian = np.empty((0, n_bpms, n_knobs))

        # Re-interleave into the caller's observable order, exactly as
        # ClosedTwissWorker.compute_gradients_and_loss does for its own single
        # measurement -- here duplicated because this worker caches signal and
        # reference states independently before comparing them.
        values = np.empty((len(self.observables), n_bpms))
        jacobian = np.empty((len(self.observables), n_bpms, n_knobs))
        orbit_iter, optics_iter = iter(range(len(self.orbit_coords))), iter(range(n_optics))
        for index, observable in enumerate(self.observables):
            if observable.name in ORBIT_COORDS:
                source = next(orbit_iter)
                values[index], jacobian[index] = orbit_values[source], orbit_jacobian[source]
            else:
                source = next(optics_iter)
                values[index], jacobian[index] = optics_values[source], optics_jacobian[source]
        return values, jacobian

    def _orbit_plain(self, mad: MAD, *, context: str = ""):
        """Closed-orbit values ``(n_coords, n_bpms)`` without knob derivatives (``compute_orbit_plain``); ``None`` if the orbit is lost.

        The knobs must be plain constants (``knobs_to_plain``) and the series orbit-only. Rows follow ``self.observables``.
        """
        if self.optics_names:
            raise NotImplementedError("loss-only evaluation supports orbit observables only")
        mad.send("compute_orbit_plain()")
        if not mad.recv():
            LOGGER.warning("Worker %s: closed orbit not found%s; flagging step for backtrack", self.worker_id, context)
            return None
        names = list(mad.recv())
        values = np.empty((len(self.orbit_coords), len(names)))
        for index in range(len(self.orbit_coords)):
            values[index] = np.asarray(mad.recv(), dtype=float).ravel()
        if self._twiss_bpm_order != names:
            self._align_measurements(names)
        return values

    def _compare_to_reference(self, signal, reference):
        """Subtract the reference only in the relative observable planes."""
        model = signal[0] - reference[0] * self._subtract[:, None]
        jacobian = signal[1] - reference[1] * self._subtract[:, None, None]
        return model, jacobian

    def _apply_knobs(self, mad: MAD, knob_updates: dict[str, float]) -> None:
        commands = [
            f"loaded_sequence['{name}']:set0({value:.15e})"
            for name, value in knob_updates.items()
            if name in self.knob_name_set
        ]
        if commands:
            mad.send("\n".join(commands))

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        del batch
        self._apply_knobs(mad, knob_updates)
        if self.reference_only:
            return self._publish_reference(mad)
        with self._reference_round():
            return self._evaluate_series(mad)

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
            return evaluate(self.control_nominal, reference_pt, "reference")
        if self._reference_message is _NO_MESSAGE:
            self._reference_message = self.conn.recv()
        message = self._reference_message
        if message is None:  # the reference worker lost the closed orbit
            return None
        block = self._reader().view(message)[message.reference_pts.index(reference_pt)]
        rows = message.rows(self.orbit_coords)
        return block[rows, :, 0], block[rows, :, 1:]

    def _reader(self) -> ReferenceReader:
        if getattr(self, "_reference_reader", None) is None:
            self._reference_reader = ReferenceReader()
        return self._reference_reader

    def _check_control_at_nominal(self, mad: MAD) -> None:
        """A shared reference holds every corrector at its starting value, so each must start at its nominal."""
        if self.control_knob is None or self.control_knob in self._checked_controls:
            return
        mad.send(f"{PYTHON_IN_MAD}:send(MADX['{self.control_knob}'])")
        start = float(mad.recv())
        if start != self.control_nominal:
            raise ValueError(
                f"shared_reference needs {self.control_knob} to start at its nominal "
                f"{self.control_nominal}, but it is {start}"
            )
        self._checked_controls.add(self.control_knob)

    def _publish_reference(self, mad: MAD):
        """Reference worker: solve the reference state of every distinct reference momentum and publish them."""
        if getattr(self, "_publisher", None) is None:
            self._publisher = ReferencePublisher()
        reference_pts = tuple(dict.fromkeys(float(m.reference_pt) for m in self.series_measurements))
        states = []
        for reference_pt in reference_pts:
            self._set_control(mad, self.control_nominal)
            self._set_pt(mad, reference_pt)
            state = self._model_and_jacobian(mad, context=f" (shared reference, pt={reference_pt:+.9g})")
            if state is None:
                return np.zeros(1), float("nan"), np.zeros((1, 1)), None
            states.append(state)
        published = self._publisher.publish(states, reference_pts, tuple(self.orbit_coords))
        # same dummy-slot convention as the calibrated worker: the payload rides in the last slot
        return np.zeros(1), 0.0, np.zeros((1, 1)), published

    def run(self) -> None:
        try:
            super().run()
        finally:
            if getattr(self, "_publisher", None) is not None:
                self._publisher.close()
            if getattr(self, "_reference_reader", None) is not None:
                self._reference_reader.close()

    def _evaluate_series(
        self, mad: MAD
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Loss, gradient and Hessians of the active series: trim, every momentum, restore."""
        failure = (
            np.zeros(self.n_knobs),
            float("nan"),
            np.zeros((self.n_knobs, self.n_knobs)),
            np.zeros((self.n_knobs, self.n_knobs)),
        )
        cache = {}

        def evaluate(control: float, pt: float, role: str):
            key = (control, pt)
            if key not in cache:
                self._set_control(mad, control)
                self._set_pt(mad, pt)
                cache[key] = self._model_and_jacobian(
                    mad, context=f" ({role}, control={control:+.9g}, pt={pt:+.9g})"
                )
            return cache[key]

        gradient = np.zeros(self.n_knobs)
        loss = 0.0
        hessian = np.zeros((self.n_knobs, self.n_knobs))
        normal_matrix = np.zeros((self.n_knobs, self.n_knobs))
        signal_control = self.control_nominal + self.control_delta
        if self.shared_reference:
            self._check_control_at_nominal(mad)

        try:
            for index, measurement in enumerate(self.series_measurements):
                signal = evaluate(signal_control, float(measurement.pt), "signal")
                if signal is None:
                    return failure

                if np.any(self._subtract):
                    reference = self._reference(evaluate, measurement)
                    if reference is None:
                        return failure
                    model, jacobian = self._compare_to_reference(signal, reference)
                else:
                    model, jacobian = signal

                observables, targets, weights, raw_weights = self._measurement_alignment[index]
                part_grad, part_loss, part_hessian, part_normal = (
                    _weighted_loss_gradient_hessian(
                        model,
                        jacobian,
                        observables,
                        targets,
                        weights,
                        raw_weights,
                        self.weight_scale,
                    )
                )
                gradient += part_grad
                loss += part_loss
                hessian += part_hessian
                normal_matrix += part_normal
        finally:
            self._set_control(mad, self.control_nominal)

        return gradient, loss, hessian, normal_matrix


@dataclass
class ClosedOrbitBatchData:
    """Corrector settings evaluated one after another in a single process."""

    series: list[ClosedOrbitSeriesData]


class ClosedOrbitBatchWorker(ClosedOrbitWorker):
    """One MAD-NG process looping over a list of corrector settings.

    Each setting is a :class:`ClosedOrbitSeriesData` (its trim and all its
    momenta). The losses, gradients and Hessians are summed, so the fit is the
    one a process per setting would give.
    """

    #: Attributes ``ClosedOrbitWorker.prepare_data`` sets per series.
    _SERIES_STATE = (
        "observables",
        "orbit_coords",
        "optics_names",
        "closed_solver",
        "_measured_index",
        "pt",
        "_twiss_bpm_order",
        "weight_scale",
        "normalisation_points",
        "series_measurements",
        "control_knob",
        "control_nominal",
        "control_delta",
        "absolute_planes",
        "calibration",
        "shared_reference",
        "reference_only",
        "_subtract",
        "_measurement_alignment",
    )

    def prepare_data(self, data: ClosedOrbitBatchData) -> None:
        if not data.series:
            raise ValueError("ClosedOrbitBatchData needs at least one series")
        self._series_states: list[dict] = []
        for series in data.series:
            super().prepare_data(series)
            self._series_states.append(self._save_state())
        names = {tuple(state["orbit_coords"] + state["optics_names"]) for state in self._series_states}
        if len(names) != 1:
            raise ValueError(f"Every series in a batch needs the same observables, got {sorted(names)}")

    def _save_state(self) -> dict:
        return {name: getattr(self, name) for name in self._SERIES_STATE}

    def _load_state(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self, name, value)

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        del batch
        self._apply_knobs(mad, knob_updates)
        self._load_state(self._series_states[0])
        if self.reference_only:
            return self._publish_reference(mad)
        gradient = np.zeros(self.n_knobs)
        loss = 0.0
        hessian = np.zeros((self.n_knobs, self.n_knobs))
        normal_matrix = np.zeros((self.n_knobs, self.n_knobs))
        with self._reference_round():
            for state in self._series_states:
                self._load_state(state)
                part = self._evaluate_series(mad)
                state.update(self._save_state())
                if np.isnan(part[1]):
                    return part
                gradient += part[0]
                loss += part[1]
                hessian += part[2]
                normal_matrix += part[3]
        return gradient, loss, hessian, normal_matrix
