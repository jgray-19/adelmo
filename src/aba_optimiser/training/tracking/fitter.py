"""Tracking fitters that recover magnet knob strengths from turn-by-turn data.

:class:`TrackingFitter` holds the mode-agnostic machinery (data + worker
management, the optimisation loop, and Hessian-based uncertainties). The public
entry points are its thin subclasses, one per tracking geometry:

* :class:`ArcByArcFitter` -- arc-by-arc ranges, with the AC dipole optionally
  accounted for (``acd_excited``).
* :class:`KickerFitter` -- forward-only tracking from a kicker marker.
* :class:`ACDMarkerFitter` -- bidirectional tracking from the AC-dipole markers.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.training.config.models import OutputConfig
from aba_optimiser.training.config.tracking import (
    TrackingModeSetup,
    acd_marker_setup,
    arc_by_arc_setup,
    kicker_setup,
)
from aba_optimiser.training.lifecycle import run_with_workers
from aba_optimiser.training.machine_setup import MachineSetup
from aba_optimiser.training.results import FitResult
from aba_optimiser.training.sgd.loop import SGDLoop
from aba_optimiser.training.tracking.data_manager import DataManager
from aba_optimiser.training.tracking.session import TrackingSession
from aba_optimiser.training.tracking.workers.setup import WorkerSetupHelper
from aba_optimiser.workers.common import sandwich_uncertainties

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.analysis import DegeneracyReport
    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.config.models import (
        CheckpointConfig,
        KickerConfig,
        MeasurementConfig,
        SequenceConfig,
    )

    #: ``(current_knobs, best_knobs) -> new x, px, y, py start coordinates or None``
    InitialConditionsCallback = Callable[[dict[str, float], dict[str, float]], np.ndarray | None]
    #: ``(epoch, train_loss, validation_loss, grad_norm, true_diff)``
    LossCallback = Callable[[int, float, float | None, float, float], None]

logger = logging.getLogger(__name__)
random.seed(42)  # For reproducibility


@dataclass(frozen=True)
class FitterOptions:
    """Optional inputs shared by every tracking fitter.

    Attributes:
        initial_knob_strengths: Starting knob values (optimisation space; a
            ``deltap`` becomes ``pt``). Knobs left out keep the model value; fixed
            strengths outside the fitted knobs are set in every model.
        true_strengths: Known answer, as a dict or knob file, for diagnostics.
        debug: Run MAD-NG in debug mode.
        output_config: Logging, TensorBoard and uncertainty settings.
        checkpoint_config: Periodic checkpointing and restore.
        initial_conditions_callback: Epoch-end hook that returns refreshed start
            coordinates for every worker, or ``None`` to keep them.
        loss_callback: Called after every epoch with its losses.
    """

    initial_knob_strengths: dict[str, float] | None = None
    true_strengths: Path | dict[str, float] | None = None
    debug: bool = False
    output_config: OutputConfig = field(default_factory=OutputConfig)
    checkpoint_config: CheckpointConfig | None = None
    initial_conditions_callback: InitialConditionsCallback | None = None
    loss_callback: LossCallback | None = None


class TrackingFitter:
    """Recovers magnet knob strengths from turn-by-turn tracking, via MAD-NG.

    Composes a :class:`~aba_optimiser.training.machine_setup.MachineSetup` (model,
    BPM ranges, starting knobs), the measurement :class:`DataManager`, a
    :class:`~aba_optimiser.training.tracking.session.TrackingSession` per run and the
    :class:`~aba_optimiser.training.sgd.loop.SGDLoop`. It is not constructed
    directly: the ``setup`` comes from one of the public subclasses
    (:class:`ArcByArcFitter`, :class:`KickerFitter`, :class:`ACDMarkerFitter`).

    All inputs and results are in optimisation space.
    """

    def __init__(
        self,
        setup: TrackingModeSetup,
        *,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        options: FitterOptions | None = None,
    ):
        options = options if options is not None else FitterOptions()
        accelerator.log_optimisation_targets()
        if setup.simulation_config.optimise_momenta:
            logger.info("Including momenta (px, py) in loss function")
        else:
            logger.info("Using position-only optimisation (x, y only)")
        optimiser_config.log_state()

        self.tracking_plan = setup.plan
        self.options = options
        self.measurement_config = measurement_config
        details = measurement_config.details
        self.interface_options = [d.interface_options for d in details]
        self.machine_deltaps = [d.machine_deltap for d in details]
        # Per-file BPM the measurement turns are recorded from; in kicker mode,
        # fall back to the marker when a file does not set its own.
        self.first_bpms = [d.first_bpm or setup.first_bpm_fallback for d in details]

        self.machine = MachineSetup(
            accelerator,
            setup.simulation_config,
            sequence_config,
            bpm_start_points=setup.bpm_start_points,
            bpm_end_points=setup.bpm_end_points,
            output_config=options.output_config,
            tracking_plan=setup.plan,
            initial_knob_strengths=options.initial_knob_strengths,
            true_strengths=options.true_strengths,
            debug=options.debug,
            **_first_set(self.interface_options, ("machine_state", "b2_errors")),
        )
        self.data_manager: DataManager | None = self._load_data()
        simulation_config = self.machine.simulation_config
        logging.getLogger("aba_optimiser.workers").setLevel(simulation_config.worker_logging_level)
        output_config = self.machine.output_config
        self.worker_setup = WorkerSetupHelper(
            accelerator=accelerator,
            all_bpms=self.machine.all_bpms,
            fixed_start=self.machine.fixed_start,
            fixed_end=self.machine.fixed_end,
            use_fixed_bpm=simulation_config.use_fixed_bpm,
            bad_bpms=sequence_config.bad_bpms,
            file_kick_planes=self.data_manager.file_kick_planes,
            magnet_range=sequence_config.magnet_range,
            interface_options_per_file=self.interface_options,
            debug=options.debug,
            mad_logfile=output_config.mad_logfile,
            python_logfile=output_config.python_logfile,
            tracking_plan=self.tracking_plan,
        )
        self.loop = SGDLoop(
            self.machine.knob_names,
            self.machine.true_strengths,
            optimiser_config,
            simulation_config.num_batches,
        )
        #: The workers of the current or last run.
        self.session: TrackingSession | None = None

    def run(self) -> FitResult:
        """Run the optimisation; knobs and uncertainties are in optimisation space."""
        writer = self.machine.make_writer("tracking_opt")
        simulation_config = self.machine.simulation_config
        include_uncertainty = self.machine.output_config.include_uncertainty
        start_knobs = self.machine.worker_start_knobs

        def body() -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
            session = self.start_session(start_knobs)
            total_turns = self.data_manager.get_total_turns()
            if simulation_config.enable_preloop_outlier_screening:
                session.screen_outliers(
                    start_knobs,
                    bpm_sigma_threshold=simulation_config.bpm_loss_outlier_sigma,
                    worker_sigma_threshold=simulation_config.worker_loss_outlier_sigma,
                )
            # The workers hold their own copies of the data now.
            self.data_manager = None
            gc.collect()

            final_knobs = self.loop.run(
                self.machine.initial_knobs,
                session.training.channels,
                total_turns=total_turns,
                writer=writer,
                checkpoint_config=self.options.checkpoint_config,
                validation_loss_fn=session.validation_loss,
                loss_callback=self.options.loss_callback,
                epoch_end_hook=self._epoch_end_hook(session),
            )

            # Workers still hold the last batch's knobs; the uncertainty must be
            # propagated at the knobs we return.
            if include_uncertainty:
                session.set_training_knobs(final_knobs)
            normal_and_noise = session.stop_and_collect_uncertainty(
                len(final_knobs), propagate_uncertainty=include_uncertainty
            )
            return final_knobs, normal_and_noise

        final_knobs, normal_and_noise = run_with_workers(
            body,
            fallback=lambda: (self.loop.best.value, None),
            stop_workers=self._terminate_session,
            writer=writer,
        )
        uncertainties = self._uncertainties(final_knobs, normal_and_noise)
        logger.info("Optimisation complete.")
        return FitResult(
            knobs=final_knobs,
            uncertainties=dict(zip(final_knobs.keys(), uncertainties)),
            diagnostics=self.loop.diagnostics,
        )

    def start_session(
        self,
        start_knobs: dict[str, float] | None = None,
        *,
        enable_validation: bool | None = None,
    ) -> TrackingSession:
        """Start the workers, as :meth:`run` does, and return their session.

        ``start_knobs`` defaults to :attr:`MachineSetup.worker_start_knobs`;
        ``enable_validation`` to the tracking plan's choice. The data are reloaded
        if a previous run released them. Stop the session (or
        :meth:`TrackingSession.terminate` it) when done.
        """
        if self.data_manager is None:
            self.data_manager = self._load_data()
        if enable_validation is None:
            enable_validation = self.tracking_plan.enable_validation
        data = self.data_manager
        self.session = TrackingSession(
            self.worker_setup,
            self.machine.simulation_config,
            turn_batches=data.turn_batches,
            validation_turn_batches=data.validation_turn_batches if enable_validation else [],
            file_turn_map=data.file_map,
            start_bpms=self.machine.start_bpms,
            end_bpms=self.machine.end_bpms,
            machine_deltaps=self.machine_deltaps,
        )
        self.session.start(
            data.tracks, self.machine.worker_start_knobs if start_knobs is None else start_knobs
        )
        return self.session

    def build_initial_normal_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Accumulate the Gauss-Newton normal matrix ``A = JᵀWJ`` at the initial knobs.

        This starts the tracking workers exactly as :meth:`run` does, then requests
        the worker-side Hessians *without taking a single optimisation step*, so the
        returned matrix describes the problem the optimiser is about to face. Workers
        are shut down before returning; the fitter can still :meth:`run` afterwards.

        Returns:
            Tuple of (normal matrix, knob names) with the knob names in the row/column
            order of the matrix.
        """
        try:
            session = self.start_session()
            normal, _ = session.stop_and_collect_uncertainty(
                len(self.machine.knob_names), propagate_uncertainty=True
            )
        except BaseException:
            self._terminate_session()
            raise
        return normal, list(self.machine.knob_names)

    def check_degeneracy(self, **analyse_kwargs) -> DegeneracyReport:
        """Diagnose unconstrained knob directions before optimising.

        Builds the normal matrix at the initial knobs and analyses its eigenspectrum.
        Keyword arguments are forwarded to
        :func:`aba_optimiser.analysis.analyse_degeneracy` (e.g. ``rel_tol``, ``scale``).
        """
        from aba_optimiser.analysis import analyse_degeneracy

        normal_matrix, knob_names = self.build_initial_normal_matrix()
        return analyse_degeneracy(normal_matrix, knob_names, **analyse_kwargs)

    def _terminate_session(self) -> None:
        if self.session is not None:
            self.session.terminate()

    def _epoch_end_hook(
        self, session: TrackingSession
    ) -> Callable[[dict[str, float], dict[str, float]], str | None] | None:
        """The epoch-end hook refreshing the workers' initial conditions, or ``None`` without a callback."""
        callback = self.options.initial_conditions_callback
        if callback is None:
            return None
        return initial_conditions_hook(
            callback, self.machine.initial_model_values, session.send_init_condition_updates
        )

    def _uncertainties(
        self,
        final_knobs: dict[str, float],
        normal_and_noise: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """1-sigma knob uncertainties in the order of ``output_knob_names``; zero when not computed.

        ``(A, B)`` are the normal matrix and the measurement noise propagated through
        every observation and start coordinate, ``Cov = A⁻¹ B A⁻¹``.
        """
        if self.machine.output_config.include_uncertainty and normal_and_noise is not None:
            uncertainties = sandwich_uncertainties(*normal_and_noise)
        else:
            uncertainties = np.zeros(len(final_knobs), dtype=np.float64)
        uncertainty_by_knob = dict(zip(self.machine.knob_names, uncertainties, strict=True))
        return np.array(
            [uncertainty_by_knob[name] for name in self.machine.output_knob_names],
            dtype=np.float64,
        )

    def _load_data(self) -> DataManager:
        """Load the measurement files, batch their turns and fix ``num_batches`` to fit them."""
        machine = self.machine
        data = DataManager(
            self.tracking_plan.observed_bpms(machine.bpms_in_range, machine.all_bpms),
            machine.all_bpms,
            machine.simulation_config,
            self.measurement_config.files,
            tracking_plan=self.tracking_plan,
            first_bpms=self.first_bpms,
            extra_markers=list(self.tracking_plan.extra_markers),
        )
        data.load_track_data()
        data.prepare_turn_batches(len(machine.start_bpms), len(machine.end_bpms))

        # No batch may need more turns than the smallest worker holds.
        min_turns_per_batch = min(len(batch) for batch in data.turn_batches)
        machine.simulation_config = dataclasses.replace(
            machine.simulation_config,
            num_batches=min(machine.simulation_config.num_batches, min_turns_per_batch),
        )
        data.simulation_config = machine.simulation_config
        return data


def initial_conditions_hook(
    callback: InitialConditionsCallback,
    fixed_model_values: dict[str, float],
    push: Callable[[np.ndarray], None],
) -> Callable[[dict[str, float], dict[str, float]], str]:
    """An epoch-end hook that pushes ``callback``'s new start coordinates with ``push``.

    The optimiser only carries this stage's knobs, but the workers track through the
    full model: ``fixed_model_values`` (fixed strengths and ``pt`` from
    ``initial_knob_strengths``) are merged into the knobs the callback sees, or it
    would rebuild initial conditions from the bare model defaults. An empty
    ``best_knobs`` stays empty so the callback can skip early epochs.

    The hook returns a fragment for the epoch log line: ``dic`` is the mean absolute
    change per component since the previous update and ``dic0`` the mean absolute
    drift since the first one. They separate an update that is converging (``dic``
    falling, ``dic0`` settling), one that never moved the conditions (both zero), and
    a callback that keeps returning ``None``, which emits no fragment.
    """
    fixed = dict(fixed_model_values)
    # First and previous pushed coordinates, set on the first successful update so
    # a failing first call cannot become the drift baseline.
    pushed: dict[str, np.ndarray] = {}

    def hook(current_knobs: dict[str, float], best_knobs: dict[str, float]) -> str | None:
        new_coords = callback(
            {**fixed, **current_knobs}, {**fixed, **best_knobs} if best_knobs else best_knobs
        )
        if new_coords is None:
            return None
        push(new_coords)

        first = pushed.setdefault("first", new_coords)
        previous = pushed.get("previous", new_coords)
        pushed["previous"] = new_coords
        # Mean over particles and components, so the number stays comparable
        # whatever the tracking mode's particle count is.
        step = float(np.abs(new_coords - previous).mean())
        drift = float(np.abs(new_coords - first).mean())
        return f"dic={step:.2e}, dic0={drift:.2e}"

    return hook


def _first_set(options_per_file: list[dict], keys: tuple[str, ...]) -> dict:
    """For each of ``keys``, the first file's non-``None`` value, so the fitter's model matches the workers'."""
    merged: dict = {}
    for key in keys:
        for options in options_per_file:
            if options.get(key) is not None:
                merged[key] = options[key]
                break
    return merged


class ArcByArcFitter(TrackingFitter):
    """Fit magnet strengths from arc-by-arc BPM ranges.

    Tracks the caller's ``bpm_start_points`` x ``bpm_end_points`` ranges. Set
    ``acd_excited`` when the data was AC-dipole excited: the exciter markers are then
    installed and any range that would straddle the AC dipole is rerouted the long
    way round the ring. Leave it ``False`` for free-oscillation data.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        *,
        acd_excited: bool = False,
        options: FitterOptions | None = None,
    ):
        setup = arc_by_arc_setup(
            accelerator=accelerator,
            simulation_config=simulation_config,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            acd_excited=acd_excited,
        )
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )


class KickerFitter(TrackingFitter):
    """Fit magnet strengths from kicker-excited turn-by-turn data.

    Runs a single worker forward-only from the kicker marker, which supplies the
    tracking initial conditions.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        kicker_config: KickerConfig,
        options: FitterOptions | None = None,
    ):
        setup = kicker_setup(kicker_config, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )


class ACDMarkerFitter(TrackingFitter):
    """Fit magnet strengths from AC-dipole data, tracked from the exciter markers.

    Tracks bidirectionally from the AC-dipole ``before``/``after`` markers (which
    supply the initial conditions) and observes the whole ring. Use
    :class:`ArcByArcFitter` with ``acd_excited=True`` instead to track AC-dipole data
    over ordinary arc ranges.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        options: FitterOptions | None = None,
    ):
        setup = acd_marker_setup(accelerator, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )
