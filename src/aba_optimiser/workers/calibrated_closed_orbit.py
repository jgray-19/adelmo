"""Closed-orbit worker with BPM gain and corrector calibration parameters (see :mod:`aba_optimiser.calibration`)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.calibration import (
    PLANES,
    CalibrationBlocks,
    add_series,
    bpm_gain_name,
    corrector_gain_knob_name,
    corrector_gain_name,
)
from aba_optimiser.workers.closed_orbit import ClosedOrbitBatchWorker
from aba_optimiser.workers.protocol import LOSS_ONLY

if TYPE_CHECKING:
    from pymadng import MAD

LOGGER = logging.getLogger(__name__)


class CalibratedClosedOrbitBatchWorker(ClosedOrbitBatchWorker):
    """Batch worker that fits ``(1 + b_bpm) * model(q, kicks * (1 + g_corrector))`` to the measured orbit changes.

    Every calibrated corrector that is on (a non-zero ``k_<corrector>`` global) in either the series' state or its
    reference state has its whole kick scaled by its gain, in both states.

    Assumes the baseline lattice has no enabled correctors: a kick that is non-zero only in the baseline (not set by
    either state) gets no gain knob and is never scaled.

    The knob dict it receives carries the gains next to the magnet knobs (names from :mod:`aba_optimiser.calibration`);
    MAD-NG never sees them. It returns the ``u = (q, g)`` gradient and Hessian in the usual slots and the remaining
    blocks (``b`` coupling, diagonal, gradient, loss-normalisation constant) as a payload in the last slot.
    """

    def _setup_da_maps(self, mad: MAD) -> None:
        """One more knob per corrector that is on in this worker's series: its gain offset, after the magnet knobs."""
        correctors = list(
            dict.fromkeys(
                name[2:].upper()
                for state in self._series_states
                for name, value in (*state["machine_state"].items(), *state["reference_state"].items())
                if name.startswith("k_") and name[2:].upper() in state["calibration"].correctors and value != 0.0
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

    def _enter_scaled(self, mad: MAD, role: str, knob_updates: dict[str, float]) -> None:
        """Enter *role*'s state with every corrector kick ``k`` set to ``k * (1 + g + gain knob)``.

        The gain knob is 0, so the value is ``k * (1 + g)``, and its Jacobian column is d(orbit)/d(g).
        """
        state = self._states[role]
        self._assign_state(mad, state)
        commands = []
        for name, kick in state.items():
            corrector = name[2:].upper()
            if name.startswith("k_") and corrector in self._correctors and kick != 0.0:
                gain = 1.0 + float(knob_updates.get(corrector_gain_name(corrector), 0.0))
                knob = corrector_gain_knob_name(corrector)
                commands.append(f"MADX['{name}'] = {kick:.15e} * ({gain:.15e} + loaded_sequence['{knob}'])")
        if commands:
            mad.send("\n".join(commands))

    def setup_mad_interface(self, knob_values):
        """The startup message also carries the gains, which are not model values."""
        magnet_knobs = {k: v for k, v in knob_values.items() if not k.startswith(("bpmgain.", "corrgain."))}
        return super().setup_mad_interface(magnet_knobs)

    def compute_gradients_and_loss(self, mad: MAD, knob_updates: dict[str, float], batch: int):
        if batch == LOSS_ONLY:
            return self._loss_only(mad, knob_updates)
        self._apply_knobs(mad, knob_updates)
        spec = self._series_states[0]["calibration"]
        n_b = len(PLANES) * len(spec.bpms)
        blocks = CalibrationBlocks(self.n_q, len(spec.correctors), n_b)
        for state in self._series_states:
            self._load_state(state)
            part = self._evaluate_calibrated(mad, knob_updates, spec)
            state.update(self._save_state())
            if part is None:
                nan = float("nan")
                return np.zeros(1), nan, np.zeros((1, 1)), {}
            blocks.add(part)
        blocks.scale(1.0 / self.normalisation_points)
        payload = blocks.to_payload()
        payload["loss_unit"] = 1.0 / (self.weight_scale * self.normalisation_points)
        # The master reads everything from the payload (already normalised); the standard slots are dummies.
        return np.zeros(1), blocks.loss * self.normalisation_points, np.zeros((1, 1)), payload

    def _loss_only(self, mad: MAD, knob_updates: dict[str, float]):
        """Data loss alone, from plain closed-orbit solves (no knob parameters, no Jacobian, no Hessian).

        Same loss as the blocks' ``loss`` of :meth:`compute_gradients_and_loss`; returned in the standard dummy slots with
        ``{"loss": normalised loss, "loss_unit": ...}`` as the payload.
        """
        self._apply_knobs(mad, knob_updates)
        spec = self._series_states[0]["calibration"]
        total = 0.0
        mad.send("knobs_to_plain()")
        try:
            for state in self._series_states:
                self._load_state(state)
                part = self._evaluate_calibrated_loss(mad, knob_updates, spec)
                state.update(self._save_state())
                if part is None:
                    nan = float("nan")
                    return np.zeros(1), nan, np.zeros((1, 1)), {}
                total += part
        finally:
            mad.send("knobs_to_param()")
        payload = {"loss": total / self.normalisation_points, "loss_unit": 1.0 / (self.weight_scale * self.normalisation_points)}
        return np.zeros(1), total, np.zeros((1, 1)), payload

    def _evaluate_calibrated_loss(self, mad: MAD, knob_updates: dict[str, float], spec) -> float | None:
        """One series' ``sum w ((1 + b) * model - target)^2`` as in :func:`add_series`, from orbit values only."""
        cache = {}

        def evaluate(role: str, pt: float):
            key = (role, pt)
            if key not in cache:
                self._enter_scaled(mad, role, knob_updates)
                self._set_pt(mad, pt)
                cache[key] = self._orbit_plain(mad, context=f" ({role}, pt={pt:+.9g})")
            return cache[key]

        loss = 0.0
        with self._in_series(mad):
            for index, measurement in enumerate(self.series_measurements):
                signal = evaluate("signal", float(measurement.pt))
                if signal is None:
                    return None
                if np.any(self._subtract):
                    reference = evaluate("reference", float(measurement.reference_pt))
                    if reference is None:
                        return None
                    model = signal - reference * self._subtract[:, None]
                else:
                    model = signal
                observables, targets, weights, _raw = self._measurement_alignment[index]
                names = self._twiss_bpm_order
                for o, observable in enumerate(observables):
                    gains = np.array([knob_updates.get(bpm_gain_name(observable.name, name), 0.0) for name in names])
                    residual = (1.0 + gains) * model[o] - np.asarray(targets[o], dtype=float)
                    w = np.asarray(weights[o], dtype=float)
                    loss += float(np.sum(w * np.where(w > 0.0, residual, 0.0) ** 2))
        return loss

    def _evaluate_calibrated(self, mad: MAD, knob_updates: dict[str, float], spec) -> CalibrationBlocks | None:
        n_b = len(PLANES) * len(spec.bpms)
        blocks = CalibrationBlocks(self.n_q, len(spec.correctors), n_b)
        bpm_index = {name: i for i, name in enumerate(spec.bpms)}
        corrector_indices = [spec.correctors.index(corrector) for corrector in self._correctors]
        cache = {}

        def evaluate(role: str, pt: float):
            key = (role, pt)
            if key not in cache:
                self._enter_scaled(mad, role, knob_updates)
                self._set_pt(mad, pt)
                cache[key] = self._model_and_jacobian(mad, context=f" ({role}, pt={pt:+.9g})")
            return cache[key]

        with self._in_series(mad):
            for index, measurement in enumerate(self.series_measurements):
                signal = evaluate("signal", float(measurement.pt))
                if signal is None:
                    return None
                if np.any(self._subtract):
                    reference = evaluate("reference", float(measurement.reference_pt))
                    if reference is None:
                        return None
                    model, jacobian = self._compare_to_reference(signal, reference)
                else:
                    model, jacobian = signal
                observables, targets, weights, _raw = self._measurement_alignment[index]
                names = self._twiss_bpm_order
                cols = np.empty((len(observables), len(names)), dtype=int)
                gains = np.empty((len(observables), len(names)))
                for o, observable in enumerate(observables):
                    plane = PLANES.index(observable.name)
                    for k, name in enumerate(names):
                        cols[o, k] = plane * len(spec.bpms) + bpm_index[name]
                        gains[o, k] = knob_updates.get(bpm_gain_name(observable.name, name), 0.0)
                q_jac, d_gain = jacobian[..., : self.n_q], jacobian[..., self.n_q :]
                add_series(blocks, model, q_jac, d_gain, targets, weights, gains, cols, corrector_indices)
        return blocks
