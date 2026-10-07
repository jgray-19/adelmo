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
import time
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from aba_optimiser.training.base_fitter import BaseFitter
from aba_optimiser.training.config.models import OutputConfig
from aba_optimiser.training.config.tracking import (
    TrackingModeSetup,
    acd_marker_setup,
    arc_by_arc_setup,
    kicker_setup,
)
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.lifecycle import run_with_workers
from aba_optimiser.training.results import FitResult
from aba_optimiser.training.workers.manager import WorkerManager
from aba_optimiser.training.workers.setup import WorkerSetupHelper
from aba_optimiser.workers.common import sandwich_uncertainties

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Unpack

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.analysis import DegeneracyReport
    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.config.models import (
        CheckpointConfig,
        KickerConfig,
        MeasurementConfig,
        SequenceConfig,
    )

logger = logging.getLogger(__name__)
random.seed(42)  # For reproducibility


class FitterOptions(TypedDict, total=False):
    """Common optional inputs shared by every fitter entry point.

    Declared once here and forwarded by each subclass as
    ``**fitter_options: Unpack[FitterOptions]`` to :class:`TrackingFitter`, which
    documents and consumes them. Keeping the shared surface in one place avoids
    repeating the same optional parameters across every subclass constructor.
    """

    initial_knob_strengths: dict[str, float] | None
    true_strengths: Path | dict[str, float] | None
    debug: bool
    optimise_knobs: list[str] | None
    output_config: OutputConfig | None
    checkpoint_config: CheckpointConfig | None
    initial_conditions_callback: Callable[[dict[str, float], dict[str, float]], np.ndarray | None] | None
    loss_callback: Callable[[int, float, float | None, float, float], None] | None


class TrackingFitter(BaseFitter):
    """
    Recovers magnet knob strengths from turn-by-turn tracking, via MAD-NG.

    Holds the mode-agnostic machinery (data management, worker coordination, the
    optimisation loop, and Hessian-based uncertainties) shared by every tracking
    geometry. It is not constructed directly: the ``tracking_plan`` and rewritten
    ``simulation_config`` come from a :class:`TrackingModeSetup` built by one of the
    public subclasses (:class:`ArcByArcFitter`, :class:`KickerFitter`,
    :class:`ACDMarkerFitter`). Those subclasses forward the
    shared optional inputs here as ``**fitter_options`` (see :class:`FitterOptions`).

    Design: optimisation-space only internally and externally. All user inputs,
    internal algorithms, and reported results use the same knob coordinates.
    """

    _defer_managers = True

    def __init__(
        self,
        setup: TrackingModeSetup,
        *,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        debug: bool = False,
        optimise_knobs: list[str] | None = None,
        output_config: OutputConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        initial_conditions_callback: Callable[
            [dict[str, float], dict[str, float]], np.ndarray | None
        ]
        | None = None,
        loss_callback: Callable[[int, float, float | None, float, float], None] | None = None,
    ):
        """
        Initialise the fitter with all required managers.

        User inputs are in optimisation space and remain there throughout the run.

        Args:
            setup (TrackingModeSetup): Resolved tracking mode (plan, rewritten
                simulation config, and BPM points) built by the calling subclass.
            accelerator (Accelerator): Accelerator instance defining machine configuration.
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration.
            sequence_config (SequenceConfig): Sequence and beam configuration.
            measurement_config (MeasurementConfig): Measurement data file configuration.
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths in optimisation space.
            true_strengths (Path | dict[str, float], optional): True strengths file or dict in optimisation space.
            debug (bool, optional): Enable debug mode. Defaults to False.
            optimise_knobs (list[str] | None, optional): List of global knob names to optimise.
            output_config (OutputConfig): Output/logging behaviour.
            checkpoint_config (CheckpointConfig | None): Checkpointing configuration.
            initial_conditions_callback (Callable | None, optional): Epoch-end hook to
                refresh worker initial conditions.
        """
        simulation_config = setup.simulation_config

        # Log optimisation targets
        accelerator.log_optimisation_targets()
        if simulation_config.optimise_momenta:
            logger.info("Including momenta (px, py) in loss function")
        else:
            logger.info("Using position-only optimisation (x, y only)")

        self.tracking_plan = setup.plan

        # Resolve the per-file lists the training stack works with from the
        # measurement-file-keyed config.
        self.measurement_config = measurement_config
        self.measurement_files = measurement_config.files
        self.interface_options = [d.interface_options for d in measurement_config.details]
        self.machine_deltaps = [d.machine_deltap for d in measurement_config.details]
        # Per-file BPM the measurement turns are recorded from; in kicker mode,
        # fall back to the marker when a file does not set its own.
        self.first_bpms = [
            d.first_bpm or setup.first_bpm_fallback for d in measurement_config.details
        ]
        self.num_configs = len(self.measurement_files)
        self.output_config = output_config if output_config is not None else OutputConfig()
        self.checkpoint_config = checkpoint_config
        self.initial_conditions_callback = initial_conditions_callback
        self.loss_callback = loss_callback

        # BaseFitter will normalise the optimisation-space inputs and handle
        # tracking-specific energy parameter conversions in this subclass.
        super().__init__(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config,
            setup.bpm_start_points,
            setup.bpm_end_points,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            debug=debug,
            optimise_knobs=optimise_knobs,
            output_config=self.output_config,
        )

        # Initialise tracking-specific managers
        self._init_data_manager()

        self._init_worker_manager(
            sequence_config.magnet_range,
            sequence_config.bad_bpms,
        )
        # Initialise OptimisationLoop now that _init_data_manager
        # has finalised simulation_config.num_batches.
        BaseFitter._init_managers(self)

    def run(self) -> FitResult:
        """Run the optimisation; knobs and uncertainties are in optimisation space."""
        run_start = time.time()
        writer = self.setup_logging("tracking_opt")
        total_turns = self.data_manager.get_total_turns()
        initial_worker_values = {
            **self.config_manager.initial_model_values,
            **self.initial_knobs,
        }

        def body() -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
            self.worker_manager.start_workers(
                self.data_manager.tracks,
                self.data_manager.turn_batches,
                self.data_manager.validation_turn_batches,
                self.data_manager.file_map,
                self.config_manager.start_bpms,
                self.config_manager.end_bpms,
                self.simulation_config,
                self.machine_deltaps,
                initial_worker_values,
                enable_validation=self.tracking_plan.enable_validation,
            )

            # Pre-loop diagnostics: mask BPM and worker outliers before optimisation
            if self.simulation_config.enable_preloop_outlier_screening:
                self.worker_manager.screen_initial_outliers(
                    initial_worker_values,
                    bpm_sigma_threshold=self.simulation_config.bpm_loss_outlier_sigma,
                    worker_sigma_threshold=self.simulation_config.worker_loss_outlier_sigma,
                )

            # Clean up memory after workers are started
            self._cleanup_memory()

            final_knobs = self.optimisation_loop.run_optimisation(
                self.initial_knobs,
                self.worker_manager.training.channels,
                writer,
                run_start,
                total_turns,
                checkpoint_config=self.checkpoint_config,
                validation_loss_fn=self.worker_manager.compute_validation_loss,
                loss_callback=self.loss_callback,
                epoch_end_hook=self._make_epoch_end_hook(),
            )

            # Workers still hold the last batch's knobs; the uncertainty must be
            # propagated at the knobs we return.
            if self.output_config.include_uncertainty:
                self.worker_manager.set_training_knobs(final_knobs)
            normal_and_noise = self.worker_manager.stop_and_collect_uncertainty(
                len(final_knobs),
                propagate_uncertainty=self.output_config.include_uncertainty,
            )
            return final_knobs, normal_and_noise

        final_knobs, normal_and_noise = run_with_workers(
            body,
            fallback=lambda: (self.optimisation_loop.best.value, None),
            stop_workers=self.worker_manager.terminate_workers,
            writer=writer,
        )
        uncertainties = self._uncertainties(final_knobs, normal_and_noise)
        logger.info("Optimisation complete.")
        return FitResult(
            knobs=final_knobs,
            uncertainties=dict(zip(final_knobs.keys(), uncertainties)),
            diagnostics=self.optimisation_loop.diagnostics,
        )

    def build_initial_normal_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Accumulate the Gauss-Newton normal matrix ``A = JᵀWJ`` at the initial knobs.

        This starts the tracking workers exactly as :meth:`run` does, then requests
        the worker-side Hessians *without taking a single optimisation step*, so the
        returned matrix describes the problem the optimiser is about to face. Workers
        are shut down before returning.

        Returns:
            Tuple of (normal matrix, knob names) with the knob names in the row/column
            order of the matrix.
        """
        initial_worker_values = {
            **self.config_manager.initial_model_values,
            **self.initial_knobs,
        }
        # Note: unlike run(), this deliberately does NOT call _cleanup_memory()
        # (which deletes self.data_manager), so the instance remains usable - e.g.
        # check_degeneracy() followed by run() on the same fitter.
        workers_finalised = False
        try:
            self.worker_manager.start_workers(
                self.data_manager.tracks,
                self.data_manager.turn_batches,
                self.data_manager.validation_turn_batches,
                self.data_manager.file_map,
                self.config_manager.start_bpms,
                self.config_manager.end_bpms,
                self.simulation_config,
                self.machine_deltaps,
                initial_worker_values,
                enable_validation=self.tracking_plan.enable_validation,
            )
            total_hessian, _ = self.worker_manager.stop_and_collect_uncertainty(
                len(self.config_manager.knob_names),
                propagate_uncertainty=True,
            )
            # stop_and_collect_uncertainty shuts the workers down cleanly itself.
            workers_finalised = True
        finally:
            if not workers_finalised:
                self.worker_manager.terminate_workers()

        return total_hessian, list(self.config_manager.knob_names)

    def check_degeneracy(self, **analyse_kwargs) -> DegeneracyReport:
        """Diagnose unconstrained knob directions before optimising.

        Builds the normal matrix at the initial knobs and analyses its eigenspectrum.
        Keyword arguments are forwarded to
        :func:`aba_optimiser.analysis.analyse_degeneracy` (e.g. ``rel_tol``, ``scale``).

        Returns:
            A :class:`~aba_optimiser.analysis.DegeneracyReport`.
        """
        from aba_optimiser.analysis import analyse_degeneracy

        normal_matrix, knob_names = self.build_initial_normal_matrix()
        return analyse_degeneracy(normal_matrix, knob_names, **analyse_kwargs)

    def _make_epoch_end_hook(self):
        """Return an epoch-end callable that updates worker initial conditions, or None.

        The callable returns a fragment for the epoch log line reporting how far
        the initial conditions moved: ``dic`` is the mean absolute change per
        component since the previous update and ``dic0`` the mean absolute drift
        since the first one. Those two numbers separate the three states that are
        otherwise indistinguishable in the loss -- an update that is converging
        (``dic`` falling, ``dic0`` settling), one that never moved the conditions
        at all (both zero), and one whose callback keeps failing and returning
        ``None``, which emits no fragment.
        """
        if self.initial_conditions_callback is None:
            return None

        worker_manager = self.worker_manager
        # The optimiser state only carries this stage's knobs, but the workers track
        # through the full model - fixed strengths (and a fixed pt) supplied via
        # initial_knob_strengths live in initial_model_values. The callback rebuilds a
        # model from the knobs it is handed, so it has to see those too or it would
        # silently reconstruct initial conditions from the bare model defaults.
        fixed_model_values = dict(self.config_manager.initial_model_values)

        # [first, previous] pushed initial conditions, set on the first successful
        # update so a failing first call cannot become the drift baseline.
        pushed: dict[str, np.ndarray] = {}

        def hook(current_knobs: dict[str, float], best_knobs: dict[str, float]) -> str | None:
            new_coords = self.initial_conditions_callback(
                {**fixed_model_values, **current_knobs},
                {**fixed_model_values, **best_knobs} if best_knobs else best_knobs,
            )
            if new_coords is None:
                return None
            worker_manager.send_init_condition_updates(new_coords)

            first = pushed.setdefault("first", new_coords)
            previous = pushed.get("previous", new_coords)
            pushed["previous"] = new_coords
            # Mean over particles and components, so the number stays comparable
            # whatever the tracking mode's particle count is.
            step = float(np.abs(new_coords - previous).mean())
            drift = float(np.abs(new_coords - first).mean())
            return f"dic={step:.2e}, dic0={drift:.2e}"

        return hook

    def _cleanup_memory(self) -> None:
        """Clean up memory after worker initialisation."""
        del self.data_manager
        gc.collect()

    def _uncertainties(
        self,
        final_knobs: dict[str, float],
        normal_and_noise: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """1-sigma knob uncertainties in the order of :attr:`output_knob_names`; zero when not computed.

        ``(A, B)`` are the normal matrix and the measurement noise propagated through
        every observation and start coordinate, ``Cov = A⁻¹ B A⁻¹``.
        """
        if self.output_config.include_uncertainty and normal_and_noise is not None:
            uncertainties = sandwich_uncertainties(*normal_and_noise)
        else:
            uncertainties = np.zeros(len(final_knobs), dtype=np.float64)
        uncertainty_by_knob = dict(zip(self.config_manager.knob_names, uncertainties, strict=True))
        return np.array(
            [uncertainty_by_knob[name] for name in self.output_knob_names],
            dtype=np.float64,
        )

    def _init_data_manager(self) -> None:
        """Initialise data manager and load track data."""
        observed_bpms = self.tracking_plan.observed_bpms(
            self.config_manager.bpms_in_range,
            self.config_manager.all_bpms,
        )
        self.data_manager = DataManager(
            observed_bpms,
            self.config_manager.all_bpms,
            self.simulation_config,
            self.measurement_files,
            tracking_plan=self.tracking_plan,
            first_bpms=self.first_bpms,
            extra_markers=list(self.tracking_plan.extra_markers),
        )

        # Load track data and prepare batches
        self.data_manager.load_track_data()
        self.data_manager.prepare_turn_batches(self.config_manager)

        # Adjust num_batches to not exceed the smallest worker allocation.
        min_turns_per_batch = min(len(batch) for batch in self.data_manager.turn_batches)
        self.simulation_config = dataclasses.replace(
            self.simulation_config,
            num_batches=min(self.simulation_config.num_batches, min_turns_per_batch),
        )
        self.data_manager.simulation_config = self.simulation_config

    def _init_worker_manager(
        self,
        magnet_range: str,
        bad_bpms: list[str] | None,
    ) -> None:
        """Initialize worker manager for tracking workers."""
        # Set worker logging level
        import logging

        logging.getLogger("aba_optimiser.workers").setLevel(
            self.simulation_config.worker_logging_level
        )

        self.worker_manager = WorkerManager(
            WorkerSetupHelper(
                accelerator=self.accelerator,
                all_bpms=self.config_manager.all_bpms,
                fixed_start=self.config_manager.fixed_start,
                fixed_end=self.config_manager.fixed_end,
                use_fixed_bpm=self.simulation_config.use_fixed_bpm,
                bad_bpms=bad_bpms,
                file_kick_planes=self.data_manager.file_kick_planes,
                magnet_range=magnet_range,
                interface_options_per_file=self.interface_options,
                debug=self.debug,
                mad_logfile=self.mad_logfile,
                python_logfile=self.python_logfile,
                tracking_plan=self.tracking_plan,
            )
        )

    def _get_mad_setup_kwargs(self) -> dict:
        """Mirror the worker MAD setup when building the expected knob list."""
        merged: dict = {}
        for key in ("machine_state", "b2_errors"):
            for options in self.interface_options:
                if options.get(key) is not None:
                    merged[key] = options[key]
                    break
        return merged

    def _get_configuration_manager_kwargs(self) -> dict:
        """Pass tracking-plan planning information into configuration setup."""
        return {"tracking_plan": self.tracking_plan}


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
        **fitter_options: Unpack[FitterOptions],
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
            **fitter_options,
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
        **fitter_options: Unpack[FitterOptions],
    ):
        setup = kicker_setup(kicker_config, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            **fitter_options,
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
        **fitter_options: Unpack[FitterOptions],
    ):
        setup = acd_marker_setup(accelerator, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            **fitter_options,
        )
