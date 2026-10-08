"""Fitter for closed-twiss optimisation (match a measured periodic optics solution)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from adelmo.config import SimulationConfig
from adelmo.fitting.lifecycle import run_with_workers
from adelmo.fitting.pool import WorkerPool
from adelmo.fitting.protocol import Evaluate, GradReply, Start
from adelmo.fitting.reduction import reduce_replies
from adelmo.fitting.results import FitDiagnostics, FitResult
from adelmo.fitting.setup import MachineSetup
from adelmo.fitting.uncertainty import hessian_uncertainties, warn_if_singular
from adelmo.fitting.weights import global_weight_scale, variance_to_weight
from adelmo.fitting.worker import WorkerConfig
from adelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
from adelmo.poco.lm_loop import LMPoint, run_levenberg_marquardt
from adelmo.poco.prior import (
    apply_prior,
    prior_alphas,
    validate_prior_strengths,
)
from adelmo.poco.workers.closed_twiss import (
    ClosedTwissData,
    ClosedTwissWorker,
    Observable,
    ObservableKind,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from adelmo.fitting.config import OutputConfig, SequenceConfig
    from adelmo.machine.accelerators import Accelerator

logger = logging.getLogger(__name__)


#: Measurement columns backing each fittable observable, as
#: ``observable -> (value column, error column)`` in the frame produced by
#: ``tmom_recon.build_twiss_from_measurements``.
#:
#: ``mu1``/``mu2`` are special: the measured frame carries the *cumulative* phase
#: and its cumulative variance, which are differenced into per-interval advances
#: (with the variance difference) by :func:`_advance_targets`.
#:
#: ``betx``/``bety``/``alfx``/``alfy`` are the physical-plane projections, never
#: ``beta11``/``beta22``/``alfa11``/``alfa22`` (MAD-NG's generalised-mode
#: columns, only meaningful with ``coupling=true`` -- see
#: ``run_closed_twiss_init.mad``). Phase advance has no such projected name in
#: MAD-NG (confirmed against its own optical-function name list: only
#: ``mu``/``dmu`` with a mode-index suffix exist, no ``mux``/``muy``), so
#: ``mu1``/``mu2`` remain mode-indexed on both sides here.
MEASUREMENT_COLUMNS: dict[str, tuple[str, str]] = {
    "x": ("X", "ERRX"),
    "y": ("Y", "ERRY"),
    "betx": ("BETX", "ERRBETX"),
    "bety": ("BETY", "ERRBETY"),
    "alfx": ("ALFX", "ERRALFX"),
    "alfy": ("ALFY", "ERRALFY"),
    "dx": ("DX", "ERRDX"),
    "dy": ("DY", "ERRDY"),
    "dpx": ("DPX", "ERRDPX"),
    "dpy": ("DPY", "ERRDPY"),
    # ``MUX``/``MUY`` are what ``build_twiss_from_measurements`` names the
    # cumulative phase it accumulates from the measured adjacent advances, paired
    # with the cumulative ``mu1_var``/``mu2_var``. Both are differenced back into
    # per-interval advances by :func:`_advance_targets`.
    "mu1": ("MUX", "mu1_var"),
    "mu2": ("MUY", "mu2_var"),
}

#: Observables fitted when the caller does not choose: everything the closed
#: twiss and an omc3 optics measurement independently have in common. All of
#: them are fitted by default: each family constrains a different combination of
#: the knobs, and inverse-variance weighting gives a noisy family little weight.
#: Narrow this only to isolate an effect, as the tests do.
#:
#: ``dpx``/``dpy`` are the one exclusion, and it is required for correctness:
#: omc3 does not measure them independently but *derives* them from
#: ``DX``/``DY`` through the model transfer matrix, so including them counts one
#: measurement twice. In the vertical it is worse than redundant - the derivation
#: assumes no vertical dispersion source exists anywhere, which is exactly the
#: hypothesis a ``Dy`` fit is testing. They remain available for callers whose
#: dispersion errors come from somewhere else.
DEFAULT_OBSERVABLES: tuple[str, ...] = (
    "x",
    "y",
    "betx",
    "bety",
    "alfx",
    "alfy",
    "mu1",
    "mu2",
    "dx",
    "dy",
)


class LMFitter:
    """Shared lifecycle and Levenberg-Marquardt solve of the full-ring closed-orbit/closed-twiss fitters.

    Subclasses set ``worker_class`` and build :attr:`worker_payloads`, one
    ``(WorkerConfig, data)`` per worker, in their own ``__init__``.
    """

    worker_class: type
    log_suffix: str
    fit_label: str

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        *,
        num_workers: int,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        prior_strengths: Mapping[str, float] | None = None,
        output_config: OutputConfig | None = None,
    ) -> None:
        if accelerator.optimise_energy:
            raise ValueError(
                f"{type(self).__name__} does not support accelerator.optimise_energy; "
                "measurement momenta are fixed pt inputs"
            )

        self.lm_config = lm_config or LevenbergMarquardtConfig()
        self.diagnostics: FitDiagnostics | None = None
        #: ``(knobs, loss)`` of every accepted iteration, in order
        self.history: list[tuple[dict[str, float], float]] = []
        self.machine = MachineSetup(
            accelerator,
            SimulationConfig(num_workers=num_workers, num_batches=1, use_fixed_bpm=True),
            sequence_config,
            output_config=output_config,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
        )
        self.worker_payloads: list = []

        self.use_errors = use_errors
        self.prior_strengths = validate_prior_strengths(prior_strengths)

    def close(self) -> None:
        """Shut down the MAD process of the model; call once the fit (and any use of its model) is done."""
        self.machine.close()

    def run(self) -> FitResult:
        """Run the Levenberg-Marquardt fit on one worker per payload."""
        writer = self.machine.make_writer(self.log_suffix)
        pool = WorkerPool()
        knob_names = list(self.machine.knob_names)
        initial_knobs = self.machine.initial_knobs

        def body() -> tuple[dict[str, float], np.ndarray | None]:
            for worker_id, (config, data) in enumerate(self.worker_payloads):
                pool.spawn(
                    self.worker_class, worker_id, data, config, self.machine.simulation_config
                ).send(Start(dict(initial_knobs)))
            return self._solve(pool.channels, writer)

        knobs, normal_matrix = run_with_workers(
            body,
            fallback=lambda: (dict(initial_knobs), None),
            stop_workers=pool.stop,
            writer=writer,
        )
        logger.info("%s optimisation complete.", self.fit_label)
        return FitResult(
            knobs=self.machine.accelerator.format_result_knobs(knobs),
            uncertainties=self.machine.accelerator.format_result_knobs(
                _hessian_uncertainties(normal_matrix, knob_names)
            ),
            diagnostics=self.diagnostics,
            extra=self._extra_result(),
        )

    def _extra_result(self) -> dict[str, float]:
        """Fitted parameters that are not model knobs."""
        return {}

    def _solve(self, channels, writer) -> tuple[dict[str, float], np.ndarray | None]:
        """Levenberg-Marquardt solve over the shared knobs; returns the best knobs and the physical normal matrix there.

        The closed twiss is close to linear in the knob strengths over the range
        a fit explores, so a curvature-preconditioned step converges in a few
        iterations and, unlike plain gradient descent, resolves the weakly
        conditioned directions made identifiable by the multi-delta measurements.
        """
        knob_names = list(self.machine.knob_names)
        n_knobs = len(knob_names)
        initial = np.array(
            [float(self.machine.initial_knobs[name]) for name in knob_names], dtype=float
        )
        # Error knobs are regularised toward the ideal zero-error lattice. The
        # optimisation start is independent and must not redefine that prior.
        prior_mean = np.zeros_like(initial)
        alphas: list[np.ndarray] = []

        def evaluate(params: np.ndarray, iteration: int) -> LMPoint:
            knobs = dict(zip(knob_names, (float(v) for v in params), strict=False))
            reduced = reduce_replies(self._evaluate_workers(channels, knobs), n_knobs)
            loss, grad, hessian = reduced.loss, reduced.grad, reduced.hessian
            if iteration == 0 and not reduced.particle_lost:
                warn_if_singular(reduced.normal, knob_names)  # before the prior is added
            if self.prior_strengths and not reduced.particle_lost:
                if not alphas:
                    alphas.append(prior_alphas(self.prior_strengths, hessian, knob_names, log=True))
                loss, grad, hessian = apply_prior(loss, grad, hessian, params, prior_mean, alphas[0])
            return LMPoint(loss, grad, hessian, reduced.particle_lost)

        def on_accept(params: np.ndarray, point: LMPoint) -> None:
            self.history.append(
                (dict(zip(knob_names, (float(v) for v in params), strict=False)), float(point.loss))
            )

        optimiser, self.diagnostics = run_levenberg_marquardt(
            initial, evaluate, self.lm_config, on_accept=on_accept, writer=writer
        )
        best_knobs = dict(zip(knob_names, (float(v) for v in optimiser.best.value), strict=False))
        # The optimiser's Hessian is in the worker's normalised weight space, so
        # its inverse is not a covariance in physical knob units. Re-evaluate the
        # physical normal matrix ``JᵀWJ`` (true inverse-variance weights) at the
        # solution and regularise it with the same isotropic prior the fit used,
        # so the reported 1-sigma is the MAP posterior width in real units.
        reduced = reduce_replies(self._evaluate_workers(channels, best_knobs), n_knobs)
        normal_matrix = reduced.normal
        if not reduced.particle_lost and self.prior_strengths:
            normal_matrix = normal_matrix + np.diag(
                prior_alphas(self.prior_strengths, normal_matrix, knob_names)
            )
        return best_knobs, normal_matrix

    def _evaluate_workers(self, channels, knobs: dict[str, float]) -> list[GradReply]:
        """Send ``knobs`` to the workers and return the replies to be summed, in order."""
        channels.send_all(Evaluate(knobs))
        return channels.recv_all()


class ClosedTwissFitter(LMFitter):
    """Optimise knobs so periodic model optics match measured closed twiss."""

    worker_class = ClosedTwissWorker
    log_suffix = "closed_twiss_opt"
    fit_label = "Closed-twiss"

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        measurements: dict[float, str | Path | pd.DataFrame],
        observables: tuple[str, ...] = DEFAULT_OBSERVABLES,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        machine_state: Path | Mapping[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        prior_strengths: Mapping[str, float] | None = None,
        output_config: OutputConfig | None = None,
    ) -> None:
        if not measurements:
            raise ValueError("measurements must contain at least one measured optics set")
        observables = tuple(observables)
        if not observables:
            raise ValueError("At least one observable must be fitted")
        unknown = [name for name in observables if name not in MEASUREMENT_COLUMNS]
        if unknown:
            raise ValueError(
                f"Unknown observables {unknown}; known: {sorted(MEASUREMENT_COLUMNS)}"
            )

        self.observable_names = observables
        super().__init__(
            accelerator,
            sequence_config,
            num_workers=len(measurements),
            lm_config=lm_config,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            use_errors=use_errors,
            prior_strengths=prior_strengths,
            output_config=output_config,
        )
        self.measurements = {
            float(pt): load_measurement(source, observables)
            for pt, source in measurements.items()
        }
        interface_options = {} if machine_state is None else {"machine_state": machine_state}
        self.worker_payloads = create_worker_payloads(
            self.measurements,
            observables,
            self.machine.all_bpms,
            sequence_config.magnet_range,
            sequence_config.bad_bpms,
            accelerator,
            interface_options,
            self.use_errors,
            self.machine.output_config.mad_logfile,
            self.machine.output_config.python_logfile,
        )


def _hessian_uncertainties(
    normal_matrix: np.ndarray | None, knob_names: list[str]
) -> dict[str, float]:
    """1-sigma knob uncertainties from the physical normal matrix ``JᵀWJ``.

    Delegates the covariance numerics to the shared :func:`hessian_uncertainties`
    (symmetrise + eigenvalue floor) so weakly-constrained directions yield finite,
    non-negative uncertainties in real knob units. Returns NaN for an absent
    matrix (e.g. an interrupted fit).
    """
    if normal_matrix is None:
        return {name: float("nan") for name in knob_names}
    sigmas = hessian_uncertainties(normal_matrix)
    return dict(zip(knob_names, (float(s) for s in sigmas), strict=False))


def load_measurement(
    source: str | Path | pd.DataFrame, observables: tuple[str, ...]
) -> pd.DataFrame:
    """Load a measured optics set as a DataFrame indexed by BPM name.

    Args:
        source: Either an omc3 measurement folder or a DataFrame already carrying
            the columns listed in :data:`MEASUREMENT_COLUMNS`.
        observables: Observable names that must be present.

    Returns:
        DataFrame indexed by BPM name. Missing error columns are filled with NaN.
        Individual NaN errors drop their own points; an observable whose errors
        are *all* unusable is rejected by :func:`_build_observable` unless the
        caller passes ``use_errors=False``, because a family with no variances
        cannot be weighted commensurably against families that have them.
    """
    if isinstance(source, pd.DataFrame):
        measurement = source.copy()
    else:
        from tmom_recon import build_twiss_from_measurements

        logger.info("Loading optics from measurement folder %s", source)
        measurement, has_dispersion = build_twiss_from_measurements(
            Path(source), include_errors=True
        )
        if not has_dispersion and any(name in ("dx", "dy") for name in observables):
            raise ValueError(
                f"Dispersion observables requested but {source} carries no dispersion data"
            )

    missing = [
        MEASUREMENT_COLUMNS[name][0]
        for name in observables
        if MEASUREMENT_COLUMNS[name][0] not in measurement.columns
    ]
    if missing:
        raise ValueError(f"Measurement is missing required columns: {sorted(missing)}")

    for name in observables:
        err_column = MEASUREMENT_COLUMNS[name][1]
        if err_column not in measurement.columns:
            measurement[err_column] = np.nan
    return measurement


def _advance_targets(
    measurement: pd.DataFrame, bpms: list[str], value_column: str, var_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval phase advance and its variance from a cumulative measurement.

    ``build_twiss_from_measurements`` accumulates the measured adjacent advances
    into a cumulative phase (and a cumulative variance), so differencing
    consecutive BPMs recovers exactly the independent advances that were measured
    - and their independent variances. The modulo keeps the result in ``[0, 1)``
    to match the wrapped model advance.
    """
    phase = measurement.loc[bpms, value_column].to_numpy(dtype=float)
    cumulative_var = measurement.loc[bpms, var_column].to_numpy(dtype=float)
    return np.mod(np.diff(phase), 1.0), np.diff(cumulative_var)


def create_worker_payloads(
    measurements: dict[float, pd.DataFrame],
    observables: tuple[str, ...],
    all_bpms: list[str],
    magnet_range: str,
    bad_bpms: list[str] | None,
    accelerator: Accelerator,
    interface_options: dict,
    use_errors: bool,
    mad_logfile: Path | None,
    python_logfile: Path | None,
) -> list[tuple[WorkerConfig, ClosedTwissData]]:
    """Build one full-ring closed-twiss worker payload per measured momentum.

    Every worker shares the same knobs and full-ring config; they differ only in
    their measurement and its fixed MAD-NG momentum coordinate ``pt``.

    The loss normalisation is stamped on afterwards, from every worker's
    observables at once: it has to be one common constant, or the workers are not
    minimising the same objective. See :func:`stamp_global_normalisation`.
    """
    payloads = [
        build_worker_payload(
            pt,
            measurement,
            observables,
            all_bpms,
            magnet_range,
            bad_bpms,
            accelerator,
            interface_options,
            use_errors,
            mad_logfile,
            python_logfile,
        )
        for pt, measurement in measurements.items()
    ]
    stamp_global_normalisation(payloads)
    return payloads


def stamp_global_normalisation(payloads: list[tuple[WorkerConfig, ClosedTwissData]]) -> None:
    """Give every worker the same weight scale and point count.

    The workers' losses are summed by the fitter, so any per-worker scaling of a
    worker's own loss is a re-weighting of that momentum in the joint fit. Both
    normalisations therefore have to be global constants:

    ``weight_scale``
        The largest inverse-variance weight anywhere in the fit. Dividing by it
        keeps the numbers near unity without changing any *relative* weight, so
        the argmin is exactly that of the pooled chi-square. Taking each worker's
        own maximum instead would divide each momentum by a different number.
    ``total_points``
        The total number of weighted points in the fit. This converts the summed
        chi-square into a mean, which is what the Levenberg-Marquardt damping
        scale is tuned against; dividing by each worker's own count would make a
        momentum measured at fewer BPMs count for more per point.

    Both cancel out of the Gauss-Newton step ``H^-1 g`` when applied uniformly, so
    they must be uniform: the fit is then invariant to them and only the physical
    inverse-variance weights determine the result.
    """
    weights = [
        variance_to_weight(np.asarray(observable.variances, dtype=float))
        for _config, data in payloads
        for observable in data.observables
    ]
    scale = global_weight_scale(weights)
    points = max(1, sum(int(np.count_nonzero(w)) for w in weights))
    logger.info(
        "Global loss normalisation over %d worker(s): weight scale %.6e, %d weighted points",
        len(payloads),
        scale,
        points,
    )
    for _config, data in payloads:
        data.weight_scale = scale
        data.total_points = points


def build_worker_payload(
    pt: float,
    measurement: pd.DataFrame,
    observables: tuple[str, ...],
    all_bpms: list[str],
    magnet_range: str,
    bad_bpms: list[str] | None,
    accelerator: Accelerator,
    interface_options: dict,
    use_errors: bool,
    mad_logfile: Path | None,
    python_logfile: Path | None,
) -> tuple[WorkerConfig, ClosedTwissData]:
    """Build a single full-ring closed-twiss worker payload for one momentum.

    BPMs without a measurement are added to ``bad_bpms`` so twiss does not observe
    them, keeping the model observables aligned with the measured targets.
    """
    measured_bpms = [bpm for bpm in all_bpms if bpm in measurement.index]
    if len(measured_bpms) < 2:
        raise ValueError(f"Fewer than two model BPMs have a measurement (pt={pt}).")

    unmeasured = [bpm for bpm in all_bpms if bpm not in measurement.index]
    logger.info(
        "Closed-twiss fit (pt=%g) over %d measured BPMs (%d model BPMs unobserved)",
        pt,
        len(measured_bpms),
        len(unmeasured),
    )

    observable_data = [
        _build_observable(name, measurement, measured_bpms, use_errors, pt)
        for name in observables
    ]

    worker_bad_bpms = list(unmeasured) + (list(bad_bpms) if bad_bpms else [])
    config = WorkerConfig(
        accelerator=accelerator,
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range=magnet_range,
        interface_options=interface_options,
        cycle_sequence=False,
        sdir=1,
        bad_bpms=worker_bad_bpms,
        mad_logfile=mad_logfile,
        python_logfile=python_logfile,
    )
    data = ClosedTwissData(
        bpm_names=measured_bpms,
        observables=observable_data,
        pt=pt,
    )
    return (config, data)


def _build_observable(
    name: str,
    measurement: pd.DataFrame,
    bpms: list[str],
    use_errors: bool,
    pt: float,
) -> Observable:
    """Extract one observable's targets and variances from a measurement frame."""
    value_column, err_column = MEASUREMENT_COLUMNS[name]
    observable = Observable(name=name, targets=np.empty(0), variances=np.empty(0))

    if observable.kind is ObservableKind.ADVANCE:
        # The phase "error" column is already a cumulative *variance*, so
        # differencing gives the per-interval variance directly.
        targets, variances = _advance_targets(measurement, bpms, value_column, err_column)
    else:
        targets = measurement.loc[bpms, value_column].to_numpy(dtype=float)
        errors = measurement.loc[bpms, err_column].to_numpy(dtype=float)
        variances = errors**2

    usable = np.isfinite(variances) & (variances > 0)
    if use_errors and not np.any(usable):
        finite = np.isfinite(variances)
        raise ValueError(
            f"Observable {name!r} (pt={pt:g}) has no usable measurement errors: "
            f"n={variances.size}, finite={int(np.count_nonzero(finite))}, "
            f"positive=0, zero={int(np.count_nonzero(finite & (variances == 0)))}, "
            f"negative={int(np.count_nonzero(finite & (variances < 0)))}. "
            f"Column {err_column!r} is empty or identically zero. A 1/<target^2> "
            "fallback weight is not used: it is in this family's own physical units, "
            "so against families with real variances it would effectively remove the "
            "family from the fit. Supply real errors, or set use_errors=False so every "
            "family is normalised the same way."
        )

    if not use_errors:
        # Weight each family by the inverse of its own mean square target. Without
        # this an unweighted fit would be dominated by whichever family happens to
        # carry the largest numbers (beta in metres against orbit in millimetres),
        # which is a units artefact rather than a statement about information.
        # Safe only because *no* family carries real variances in this branch.
        scale = float(np.mean(targets[np.isfinite(targets)] ** 2)) if targets.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        variances = np.full_like(targets, scale, dtype=float)
    elif not np.all(usable):
        logger.warning(
            "Observable '%s' (pt=%g): %d/%d points have no usable error and carry "
            "zero weight.",
            name,
            pt,
            int(np.count_nonzero(~usable)),
            usable.size,
        )

    return Observable(name=name, targets=targets, variances=variances)
