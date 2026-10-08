"""Closed-orbit worker with BPM gain and corrector calibration parameters (see :mod:`adelmo.poco.calibration`)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from multiprocessing import shared_memory

from adelmo.fitting.protocol import GradReply
from adelmo.machine.mad.machine_state import assign_state
from adelmo.poco.calibration import (
    PLANES,
    CalibrationBlocks,
    add_series,
    bpm_gain_name,
    corrector_gain_knob_name,
    corrector_gain_name,
)
from adelmo.poco.workers.closed_orbit import ClosedOrbitWorker

if TYPE_CHECKING:
    from pymadng import MAD

    from adelmo.fitting.protocol import Evaluate
    from adelmo.poco.workers.closed_orbit import SeriesState

LOGGER = logging.getLogger(__name__)


class CalibratedClosedOrbitWorker(ClosedOrbitWorker):
    """Worker that fits ``(1 + b_bpm) * model(q, kicks * (1 + g_corrector))`` to the measured orbit changes.

    Every calibrated corrector that is on (a non-zero ``k_<corrector>`` global) in either the series' state or its
    reference state has its whole kick scaled by its gain, in both states.

    Assumes the baseline lattice has no enabled correctors: a kick that is non-zero only in the baseline (not set by
    either state) gets no gain knob and is never scaled.

    The knob dict it receives carries the gains next to the magnet knobs (names from :mod:`adelmo.poco.calibration`);
    MAD-NG never sees them. Its reply carries the blocks' header (``b`` coupling, diagonal, gradient, loss-normalisation
    constant) in :attr:`~adelmo.fitting.protocol.GradReply.extra`, followed on the pipe by the blocks' arrays.
    """

    def _setup_da_maps(self, mad: MAD) -> None:
        """One more knob per corrector that is on in this worker's series: its gain offset, after the magnet knobs."""
        correctors = list(
            dict.fromkeys(
                name[2:].upper()
                for series in self.series
                for name, value in (*series.data.machine_state.items(), *series.data.reference_state.items())
                if name.startswith("k_") and name[2:].upper() in series.data.calibration.correctors and value != 0.0
            )
        )
        gain_knobs = [corrector_gain_knob_name(corrector) for corrector in correctors]
        mad.send("\n".join(f"loaded_sequence['{knob}'] = 0" for knob in gain_knobs))
        mad["knob_names"] = [name for name in mad["knob_names"] if name != "pt"] + gain_knobs
        super()._setup_da_maps(mad)
        #: Number of magnet knobs; ``n_knobs`` also counts the gain knobs.
        self.n_q = self.n_knobs - len(gain_knobs)
        #: The correctors with a gain knob, in Jacobian column order (columns ``n_q`` onwards).
        self._correctors = correctors

    def _enter_state(self, mad: MAD, role: str, knob_updates: dict[str, float]) -> None:
        """Enter *role*'s state with every corrector kick ``k`` set to ``k * (1 + g + gain knob)``.

        The gain knob is 0, so the value is ``k * (1 + g)``, and its Jacobian column is d(orbit)/d(g). The kick is a deferred
        expression, like the magnet strengths, so it follows the knobs when they are made plain (``knobs_to_plain``).
        """
        state = self._states[role]
        assign_state(mad, state)
        commands = []
        for name, kick in state.items():
            corrector = name[2:].upper()
            if name.startswith("k_") and corrector in self._correctors and kick != 0.0:
                gain = 1.0 + float(knob_updates.get(corrector_gain_name(corrector), 0.0))
                knob = corrector_gain_knob_name(corrector)
                commands.append(f"MADX['{name}'] = \\-> {kick:.15e} * ({gain:.15e} + loaded_sequence['{knob}'])")
        if commands:
            mad.send("\n".join(commands))

    def _loss_unit(self) -> float:
        return 1.0 / (self.weight_scale * self.normalisation_points)

    def evaluate(self, mad: MAD, message: Evaluate) -> GradReply:
        if message.loss_only:
            return self._loss_only(mad, message.knobs)
        knob_updates = message.knobs
        self.send_knobs(mad, knob_updates)
        spec = self.series[0].data.calibration
        n_g, n_b = len(spec.correctors), len(PLANES) * len(spec.bpms)
        blocks = CalibrationBlocks(self.n_q, n_g, n_b, self._shared_buffer(CalibrationBlocks.nbytes(self.n_q, n_g, n_b)))
        for series in self.series:
            if not self._evaluate_calibrated(mad, series, knob_updates, spec, blocks):
                return GradReply(self.worker_id, float("nan"), np.zeros(1), extra={})
        blocks.symmetrise()
        blocks.scale(1.0 / self.normalisation_points)
        payload = blocks.to_payload()
        payload["loss_unit"] = self._loss_unit()
        payload["shared_memory"] = self._shared.name
        # The fitter reads the arrays from the shared block and everything else from the payload (already normalised).
        return GradReply(self.worker_id, blocks.loss, np.zeros(1), extra=payload)

    _shared: shared_memory.SharedMemory | None = None

    def _shared_buffer(self, size: int):
        """The worker's block of shared memory the blocks are accumulated in, created on first use and reused every iteration."""
        if self._shared is None:
            self._shared = shared_memory.SharedMemory(create=True, size=size)
        return self._shared.buf

    def close(self) -> None:
        super().close()
        if self._shared is not None:
            self._shared.close()
            self._shared.unlink()
            self._shared = None

    def _loss_only(self, mad: MAD, knob_updates: dict[str, float]) -> GradReply:
        """Data loss alone, from plain closed-orbit solves (no knob parameters, no Jacobian, no Hessian).

        The same loss as the blocks' ``loss`` of a full evaluation; the reply's ``extra`` is
        ``{"loss": normalised loss, "loss_unit": ...}``.
        """
        self.send_knobs(mad, knob_updates)
        total = 0.0
        mad.send("knobs_to_plain()")
        try:
            for series in self.series:
                part = self._evaluate_calibrated_loss(mad, series, knob_updates)
                if part is None:
                    return GradReply(self.worker_id, float("nan"), np.zeros(1), extra={})
                total += part
        finally:
            mad.send("knobs_to_param()")
        loss = total / self.normalisation_points
        return GradReply(self.worker_id, loss, np.zeros(1), extra={"loss": loss, "loss_unit": self._loss_unit()})

    def _evaluate_calibrated_loss(
        self, mad: MAD, series: SeriesState, knob_updates: dict[str, float]
    ) -> float | None:
        """One series' ``sum w ((1 + b) * model - target)^2`` as in :func:`add_series`, from orbit values only."""
        loss = 0.0
        names = series.twiss_bpm_order
        with self._in_series(mad, series):
            for index, model in self._models(mad, series, knob_updates, self._orbit_plain):
                if model is None:
                    return None
                observables, targets, weights, _raw = series.alignment[index]
                for o, observable in enumerate(observables):
                    gains = np.array([knob_updates.get(bpm_gain_name(observable.name, name), 0.0) for name in names])
                    residual = (1.0 + gains) * model[o] - np.asarray(targets[o], dtype=float)
                    w = np.asarray(weights[o], dtype=float)
                    loss += float(np.sum(w * np.where(w > 0.0, residual, 0.0) ** 2))
        return loss

    def _evaluate_calibrated(
        self, mad: MAD, series: SeriesState, knob_updates: dict[str, float], spec, blocks: CalibrationBlocks
    ) -> bool:
        """Add one series' blocks to ``blocks``; ``False`` if its closed orbit is lost."""
        bpm_index = {name: i for i, name in enumerate(spec.bpms)}
        corrector_indices = [spec.correctors.index(corrector) for corrector in self._correctors]
        with self._in_series(mad, series):
            for index, model in self._models(mad, series, knob_updates, self._model_and_jacobian):
                if model is None:
                    return False
                values, jacobian = model
                observables, targets, weights, _raw = series.alignment[index]
                names = series.twiss_bpm_order
                cols = np.empty((len(observables), len(names)), dtype=int)
                gains = np.empty((len(observables), len(names)))
                for o, observable in enumerate(observables):
                    plane = PLANES.index(observable.name)
                    for k, name in enumerate(names):
                        cols[o, k] = plane * len(spec.bpms) + bpm_index[name]
                        gains[o, k] = knob_updates.get(bpm_gain_name(observable.name, name), 0.0)
                q_jac, d_gain = jacobian[..., : self.n_q], jacobian[..., self.n_q :]
                add_series(blocks, values, q_jac, d_gain, targets, weights, gains, cols, corrector_indices)
        return True
