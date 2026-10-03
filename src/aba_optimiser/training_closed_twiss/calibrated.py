"""Closed-orbit fitter that also fits BPM gains and corrector kick calibrations (see :mod:`aba_optimiser.calibration`)."""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.calibration import (
    PLANES,
    CalibrationBlocks,
    CalibrationSpec,
    apply_prior,
    back_substitute,
    bpm_gain_name,
    corrector_gain_name,
    reduce_blocks,
)
from aba_optimiser.optimisers.levenberg_marquardt import LevenbergMarquardtOptimiser
from aba_optimiser.training_closed_twiss.closed_orbit import ClosedOrbitFitter
from aba_optimiser.workers.calibrated_closed_orbit import CalibratedClosedOrbitBatchWorker
from aba_optimiser.workers.closed_orbit import ClosedOrbitBatchData
from aba_optimiser.workers.protocol import LOSS_ONLY, distribute

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)


class CalibratedClosedOrbitFitter(ClosedOrbitFitter):
    """:class:`ClosedOrbitFitter` with ``(1 + b_bpm)(1 + g_corrector)`` calibration parameters.

    Every series' ``control_knob`` must be ``k_<corrector>``; the corrector's gain is named after it. The BPM gains are
    eliminated by a Schur complement each iteration and recovered by back-substitution, so the Levenberg-Marquardt solve
    is over the magnet knobs and the corrector gains only. ``sigma_bpm`` / ``sigma_corrector`` are Gaussian priors that
    also fix the ``b``/``g`` scale degeneracy. ``knob_sigmas`` (knob name -> absolute width, in the knob's units) adds a
    Gaussian prior on those magnet knobs; knobs it does not name stay free. After the fit, :attr:`calibration_result` holds every gain.
    """

    def __init__(
        self,
        *args,
        sigma_bpm: float = 1e-2,
        sigma_corrector: float = 1e-2,
        knob_sigmas: Mapping[str, float] | None = None,
        loss_only_trials: bool = True,
        **kwargs,
    ) -> None:
        self.knob_sigmas = {name.lower(): float(sigma) for name, sigma in (knob_sigmas or {}).items()}
        if any(sigma <= 0 for sigma in self.knob_sigmas.values()):
            raise ValueError("knob_sigmas must be > 0")
        #: Evaluate each trial point with the loss alone (plain closed-orbit solves, no knob derivatives) and run the full
        #: gradient/Hessian evaluation only at points that improve on the best loss. Same accepted path, rejected trials ~6x cheaper.
        self.loss_only_trials = bool(loss_only_trials)
        self.sigma_bpm = float(sigma_bpm)
        self.sigma_corrector = float(sigma_corrector)
        self.calibration_result: dict[str, float] = {}
        self.calibration_spec: CalibrationSpec | None = None
        #: ``(magnet knobs, loss)`` of every accepted iteration
        self.history: list[tuple[dict[str, float], float]] = []
        super().__init__(*args, **kwargs)

    def _group_payloads(self, payloads):
        """Always batch workers, each series tagged with the global gain layout."""
        bpms = tuple(payloads[0][1].bpm_names)
        if any(tuple(data.bpm_names) != bpms for _, data in payloads):
            raise ValueError("Calibration fits need every series to observe the same BPMs")
        if any(not (data.control_knob or "").startswith("k_") for _, data in payloads):
            raise ValueError("Calibration fits need every series' control_knob to be 'k_<corrector>'")
        correctors = tuple(dict.fromkeys(data.control_knob[2:].upper() for _, data in payloads))
        self.calibration_spec = CalibrationSpec(bpms, correctors)
        for _, data in payloads:
            data.calibration = self.calibration_spec
        groups = distribute([len(item.measurements) for item in self.series], self.n_workers)
        self.worker_class = CalibratedClosedOrbitBatchWorker
        LOGGER.info("Calibration fit: %d BPMs x 2 planes, %d correctors, %d worker(s)", len(bpms), len(correctors), len(groups))
        return [
            (payloads[group[0]][0], ClosedOrbitBatchData([payloads[i][1] for i in group])) for group in groups
        ]

    def _collect_calibrated(self, channels, knobs: dict[str, float]) -> tuple[CalibrationBlocks, float] | None:
        """Sum every worker's blocks; ``None`` if any worker lost its closed orbit."""
        channels.send_all((knobs, 0))
        results = channels.recv_all()
        if not results:
            raise RuntimeError("No closed-orbit workers returned results")
        total, unit = None, 1.0
        lost = False
        for result in results:
            if not isinstance(result, tuple) or len(result) != 5:
                raise RuntimeError(f"Unexpected closed-orbit worker payload: {result!r}")
            _, _, loss, _, payload = result
            if loss == float("inf"):
                raise RuntimeError("Worker error detected during calibrated closed-orbit optimisation")
            if np.isnan(loss):
                lost = True
                continue
            part = CalibrationBlocks.from_payload(payload)
            unit = payload["loss_unit"]
            if total is None:
                total = part
            else:
                total.add(part)
        return None if lost or total is None else (total, unit)

    def _collect_loss_only(self, channels, knobs: dict[str, float]) -> tuple[float, float] | None:
        """Sum every worker's data loss (no derivatives); ``None`` if any worker lost its closed orbit. Returns ``(loss, loss_unit)``."""
        channels.send_all((knobs, LOSS_ONLY))
        results = channels.recv_all()
        if not results:
            raise RuntimeError("No closed-orbit workers returned results")
        total, unit, lost = 0.0, 1.0, False
        for result in results:
            if not isinstance(result, tuple) or len(result) != 5:
                raise RuntimeError(f"Unexpected closed-orbit worker payload: {result!r}")
            _, _, loss, _, payload = result
            if loss == float("inf"):
                raise RuntimeError("Worker error detected during calibrated closed-orbit optimisation")
            if np.isnan(loss):
                lost = True
                continue
            total += payload["loss"]
            unit = payload["loss_unit"]
        return None if lost else (total, unit)

    def _gauss_newton(self, channels, writer):
        spec = self.calibration_spec
        knob_names = list(self.config_manager.knob_names)
        n_q, n_g, n_b = len(knob_names), len(spec.correctors), len(PLANES) * len(spec.bpms)
        g_names = [corrector_gain_name(c) for c in spec.correctors]
        b_names = [bpm_gain_name(plane, bpm) for plane in PLANES for bpm in spec.bpms]

        unknown = sorted(set(self.knob_sigmas) - {name.lower() for name in knob_names})
        if unknown:
            raise ValueError(f"knob_sigmas names knobs that are not optimised: {unknown[:5]}")
        sigma_q = np.array([self.knob_sigmas.get(name.lower(), 0.0) for name in knob_names])

        u = np.zeros(n_q + n_g)
        u[:n_q] = [float(self.initial_knobs[name]) for name in knob_names]
        b = np.zeros(n_b)
        best_b, best_blocks = b.copy(), None
        optimiser = LevenbergMarquardtOptimiser(self.lm_config, initial_params=u)
        zero_grad, zero_hess = (np.zeros(n_q + n_g), np.zeros((n_q + n_g,) * 2)) if self.loss_only_trials else (None, None)
        run_start = time.time()
        last_update, completed = None, 0

        for iteration in range(self.lm_config.max_iterations):
            completed = iteration + 1
            knobs = dict(zip(knob_names + g_names + b_names, np.concatenate([u, b]).tolist(), strict=True))
            update = None
            if self.loss_only_trials and optimiser.best_hessian is not None:
                # Trial point: the loss alone decides. Only a point that improves on the best loss gets the (expensive) full
                # evaluation below, so a rejected trial costs one plain closed-orbit solve instead of a parametric one + Hessian.
                trial = self._collect_loss_only(channels, knobs)
                if trial is None:
                    update = optimiser.update(u, float("nan"), zero_grad, zero_hess, True)
                else:
                    shell = SimpleNamespace(loss=trial[0], n_q=n_q)
                    apply_prior(shell, u, b, self.sigma_corrector, self.sigma_bpm, trial[1], sigma_q, loss_only=True)
                    if os.environ.get("ABA_VERIFY_LOSS_ONLY"):  # debugging: compare with the full evaluation at the same point
                        full = self._collect_calibrated(channels, knobs)
                        if full is not None:
                            check = SimpleNamespace(loss=full[0].loss, n_q=n_q)
                            apply_prior(check, u, b, self.sigma_corrector, self.sigma_bpm, full[1], sigma_q, loss_only=True)
                            LOGGER.warning("verify: loss-only %.10e  full %.10e  rel diff %.2e (data %.10e vs %.10e)",
                                           shell.loss, check.loss, (shell.loss - check.loss) / check.loss, trial[0], full[0].loss)
                    if not shell.loss < optimiser.best_loss:
                        update = optimiser.update(u, shell.loss, zero_grad, zero_hess, False)
            if update is None:
                collected = self._collect_calibrated(channels, knobs)
                if collected is None:
                    blocks, loss, grad, hess, lost = None, float("nan"), np.zeros(n_q + n_g), np.zeros((n_q + n_g,) * 2), True
                else:
                    blocks, unit = collected
                    apply_prior(blocks, u, b, self.sigma_corrector, self.sigma_bpm, unit, sigma_q)
                    loss = blocks.loss
                    grad, hess = reduce_blocks(blocks)
                    lost = False
                update = optimiser.update(u, loss, grad, hess, lost)
            last_update = update
            if update.accepted:
                best_b, best_blocks = b.copy(), blocks
                self.history.append((dict(zip(knob_names, u[:n_q].tolist(), strict=True)), float(loss)))
            u = update.next_params

            if update.converged and not update.accepted:
                LOGGER.warning("Levenberg-Marquardt stopped at iter %d (%s)", iteration, update.reason)
                break
            if best_blocks is not None:
                b = best_b + back_substitute(best_blocks, u - optimiser.best_params)
            if update.reason == "failed" or not update.accepted:
                LOGGER.info("Iter %d: step rejected or closed orbit lost; retrying from best (lam=%.1e)", iteration, update.damping)
                continue
            self._log_gn_iteration(writer, iteration, update.loss, update.grad_norm, update.damping, run_start)
            if update.converged:
                LOGGER.info("Levenberg-Marquardt converged (%s) at iter %d", update.reason, iteration)
                break

        best_u = optimiser.best_params
        best_knobs = dict(zip(knob_names, best_u[:n_q].tolist(), strict=True))
        self.calibration_result = dict(zip(g_names, best_u[n_q:].tolist(), strict=True))
        self.calibration_result.update(zip(b_names, best_b.tolist(), strict=True))
        self.diagnostics = {
            "converged": bool(last_update is not None and last_update.converged),
            "reason": None if last_update is None else last_update.reason,
            "iterations": completed,
            "best_loss": float(optimiser.best_loss),
        }
        return best_knobs, None, knob_names
