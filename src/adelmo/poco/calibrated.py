"""Closed-orbit fitter that also fits BPM gains and corrector kick calibrations (see :mod:`adelmo.poco.calibration`)."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import resource_tracker, shared_memory
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from adelmo.fitting.protocol import Evaluate, GradReply, distribute
from adelmo.machine.mad.machine_state import resolve_machine_state
from adelmo.poco.calibration import (
    PLANES,
    CalibrationBlocks,
    CalibrationSpec,
    apply_prior,
    back_substitute,
    bpm_gain_name,
    corrector_gain_name,
    reduce_blocks,
)
from adelmo.poco.closed_orbit import CLOSED_ORBIT_OBSERVABLES, ClosedOrbitFitter
from adelmo.poco.lm_loop import LMPoint, run_levenberg_marquardt
from adelmo.poco.workers.calibrated import CalibratedClosedOrbitWorker

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from adelmo.fitting.config import OutputConfig, SequenceConfig
    from adelmo.machine.accelerators import Accelerator
    from adelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
    from adelmo.poco.closed_orbit import ClosedOrbitSeries

LOGGER = logging.getLogger(__name__)


class CalibratedClosedOrbitFitter(ClosedOrbitFitter):
    """:class:`ClosedOrbitFitter` with ``(1 + b_bpm)(1 + g_corrector)`` calibration parameters.

    Every series' own ``machine_state`` must set exactly one ``k_<corrector>`` kick (its change from the fitter's
    ``machine_state`` is worked out internally); the corrector's gain is named after it. The BPM gains are
    eliminated by a Schur complement each iteration and recovered by back-substitution, so the Levenberg-Marquardt solve
    is over the magnet knobs and the corrector gains only. ``sigma_bpm`` / ``sigma_corrector`` are Gaussian priors that
    also fix the ``b``/``g`` scale degeneracy. ``knob_sigmas`` (knob name -> absolute width, in the knob's units) adds a
    Gaussian prior on those magnet knobs; knobs it does not name stay free. After the fit, :attr:`calibration_result` holds every gain.
    """

    worker_class = CalibratedClosedOrbitWorker

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        series: list[ClosedOrbitSeries] | tuple[ClosedOrbitSeries, ...],
        observables: tuple[str, ...] = CLOSED_ORBIT_OBSERVABLES,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        machine_state: Path | Mapping[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        prior_strengths: Mapping[str, float] | None = None,
        output_config: OutputConfig | None = None,
        max_workers: int | None = None,
        *,
        sigma_bpm: float = 1e-2,
        sigma_corrector: float = 1e-2,
        knob_sigmas: Mapping[str, float] | None = None,
        loss_only_trials: bool = True,
    ) -> None:
        self.knob_sigmas = {name.lower(): float(sigma) for name, sigma in (knob_sigmas or {}).items()}
        if any(sigma <= 0 for sigma in self.knob_sigmas.values()):
            raise ValueError("knob_sigmas must be > 0")
        #: Evaluate each trial point with the loss alone (plain closed-orbit solves, no knob derivatives) and run the full
        #: gradient/Hessian evaluation only at points that improve on the best loss. Same accepted path, rejected trials ~6x cheaper.
        self.loss_only_trials = bool(loss_only_trials)
        self.sigma_bpm = float(sigma_bpm)
        self.sigma_corrector = float(sigma_corrector)
        self._attached: dict[str, shared_memory.SharedMemory] = {}  # workers' shared blocks, mapped once
        self.calibration_result: dict[str, float] = {}
        self.calibration_spec: CalibrationSpec | None = None
        # The gains are grouped into every worker's batch, so the shared-reference worker never applies.
        super().__init__(
            accelerator,
            sequence_config,
            series,
            observables=observables,
            lm_config=lm_config,
            initial_knob_strengths=initial_knob_strengths,
            machine_state=machine_state,
            true_strengths=true_strengths,
            use_errors=use_errors,
            prior_strengths=prior_strengths,
            output_config=output_config,
            max_workers=max_workers,
        )

    def _group_payloads(self, payloads):
        """Always batch workers, each series tagged with the global gain layout."""
        bpms = tuple(payloads[0][1].bpm_names)
        if any(tuple(data.bpm_names) != bpms for _, data in payloads):
            raise ValueError("Calibration fits need every series to observe the same BPMs")
        kicks = [
            [name for name in resolve_machine_state(item.machine_state) if name.startswith("k_")]
            for item in self.series
        ]
        if any(len(names) != 1 for names in kicks):
            raise ValueError(
                f"Calibration fits need every series' machine_state to set exactly one 'k_<corrector>' kick, got {kicks}"
            )
        correctors = tuple(dict.fromkeys(names[0][2:].upper() for names in kicks))
        self.calibration_spec = CalibrationSpec(bpms, correctors)
        for _, data in payloads:
            data.calibration = self.calibration_spec
        groups = distribute([len(item.measurements) for item in self.series], self.n_workers)
        LOGGER.info("Calibration fit: %d BPMs x 2 planes, %d correctors, %d worker(s)", len(bpms), len(correctors), len(groups))
        return [(payloads[group[0]][0], [payloads[i][1] for i in group]) for group in groups]

    def _attach(self, name: str) -> shared_memory.SharedMemory:
        """Map a worker's shared block once, without taking over its deletion (see :mod:`adelmo.fitting.shared_reference`)."""
        if name not in self._attached:
            if sys.version_info >= (3, 13):
                self._attached[name] = shared_memory.SharedMemory(name=name, track=False)
            else:
                block = shared_memory.SharedMemory(name=name)
                resource_tracker.unregister(block._name, "shared_memory")  # noqa: SLF001
                self._attached[name] = block
        return self._attached[name]

    def _collect_calibrated(self, channels, knobs: dict[str, float]) -> tuple[CalibrationBlocks, float] | None:
        """Sum every worker's blocks; ``None`` if any worker lost its closed orbit."""
        channels.send_all(Evaluate(knobs))
        replies = channels.recv_all()
        payloads = []
        for result in replies:
            if not isinstance(result, GradReply):
                raise RuntimeError(f"Unexpected closed-orbit worker payload: {result!r}")
            if result.loss == float("inf"):
                raise RuntimeError("Worker error detected during calibrated closed-orbit optimisation")
            payloads.append(result.extra)
        if any(np.isnan(result.loss) for result in replies) or not payloads:
            return None
        first = payloads[0]
        shape = (first["n_q"], first["n_g"], first["n_b"])
        parts = [CalibrationBlocks(*shape, self._attach(p["shared_memory"]).buf, zero=False) for p in payloads]
        total = CalibrationBlocks(*shape)
        total.loss = sum(p["loss"] for p in payloads)
        # Memory-bound sum of ~170 MB per worker: split by rows over threads (numpy releases the GIL).
        def add_rows(name: str, rows: slice) -> None:
            out = getattr(total, name)[rows]
            np.copyto(out, getattr(parts[0], name)[rows])
            for part in parts[1:]:
                np.add(out, getattr(part, name)[rows], out=out)

        step = 64
        jobs = [(name, slice(i, i + step)) for name in CalibrationBlocks.ARRAYS for i in range(0, getattr(total, name).shape[0], step)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda job: add_rows(*job), jobs))
        return total, first["loss_unit"]

    def _collect_loss_only(self, channels, knobs: dict[str, float]) -> tuple[float, float] | None:
        """Sum every worker's data loss (no derivatives); ``None`` if any worker lost its closed orbit. Returns ``(loss, loss_unit)``."""
        channels.send_all(Evaluate(knobs, loss_only=True))
        results = channels.recv_all()
        if not results:
            raise RuntimeError("No closed-orbit workers returned results")
        total, unit, lost = 0.0, 1.0, False
        for result in results:
            if not isinstance(result, GradReply):
                raise RuntimeError(f"Unexpected closed-orbit worker payload: {result!r}")
            loss, payload = result.loss, result.extra
            if loss == float("inf"):
                raise RuntimeError("Worker error detected during calibrated closed-orbit optimisation")
            if np.isnan(loss):
                lost = True
                continue
            total += payload["loss"]
            unit = payload["loss_unit"]
        return None if lost else (total, unit)

    def _extra_result(self) -> dict[str, float]:
        return dict(self.calibration_result)

    def _solve(self, channels, writer):
        spec = self.calibration_spec
        knob_names = list(self.machine.knob_names)
        n_q, n_g, n_b = len(knob_names), len(spec.correctors), len(PLANES) * len(spec.bpms)
        g_names = [corrector_gain_name(c) for c in spec.correctors]
        b_names = [bpm_gain_name(plane, bpm) for plane in PLANES for bpm in spec.bpms]

        unknown = sorted(set(self.knob_sigmas) - {name.lower() for name in knob_names})
        if unknown:
            raise ValueError(f"knob_sigmas names knobs that are not optimised: {unknown[:5]}")
        sigma_q = np.array([self.knob_sigmas.get(name.lower(), 0.0) for name in knob_names])

        u = np.zeros(n_q + n_g)
        u[:n_q] = [float(self.machine.initial_knobs[name]) for name in knob_names]
        #: The BPM gains: eliminated from each solve, then back-substituted from the best point's blocks.
        gains = {"b": np.zeros(n_b), "best_b": np.zeros(n_b), "best_blocks": None}

        def knobs_at(params: np.ndarray) -> dict[str, float]:
            values = np.concatenate([params, gains["b"]]).tolist()
            return dict(zip(knob_names + g_names + b_names, values, strict=True))

        def trial_loss(params: np.ndarray) -> float | None:
            trial = self._collect_loss_only(channels, knobs_at(params))
            if trial is None:
                return None
            shell = SimpleNamespace(loss=trial[0], n_q=n_q)
            apply_prior(shell, params, gains["b"], self.sigma_corrector, self.sigma_bpm, trial[1], sigma_q, loss_only=True)
            return shell.loss

        def evaluate(params: np.ndarray, iteration: int) -> LMPoint:
            del iteration
            collected = self._collect_calibrated(channels, knobs_at(params))
            if collected is None:
                return LMPoint(float("nan"), np.zeros(n_q + n_g), np.zeros((n_q + n_g,) * 2), failed=True)
            blocks, unit = collected
            apply_prior(blocks, params, gains["b"], self.sigma_corrector, self.sigma_bpm, unit, sigma_q)
            grad, hess = reduce_blocks(blocks)
            return LMPoint(blocks.loss, grad, hess, extra=blocks)

        def on_accept(params: np.ndarray, point: LMPoint) -> None:
            gains["best_b"], gains["best_blocks"] = gains["b"].copy(), point.extra
            self.history.append((dict(zip(knob_names, params[:n_q].tolist(), strict=True)), float(point.loss)))

        def after_step(params: np.ndarray, optimiser) -> None:
            if gains["best_blocks"] is not None:
                gains["b"] = gains["best_b"] + back_substitute(gains["best_blocks"], params - optimiser.best.value)

        optimiser, self.diagnostics = run_levenberg_marquardt(
            u,
            evaluate,
            self.lm_config,
            trial_loss=trial_loss if self.loss_only_trials else None,
            on_accept=on_accept,
            after_step=after_step,
            writer=writer,
        )
        best_u = optimiser.best.value
        self.calibration_result = dict(zip(g_names, best_u[n_q:].tolist(), strict=True))
        self.calibration_result.update(zip(b_names, gains["best_b"].tolist(), strict=True))
        # TODO: report uncertainties. The gains' covariance needs the Schur-reduced normal matrix of the physical weights.
        return dict(zip(knob_names, best_u[:n_q].tolist(), strict=True)), None
