"""Abstract base class for all worker process types.

This module defines the core worker interface that all specific worker
implementations must follow. It handles the process lifecycle, MAD-NG
interface setup, and communication protocol with the main process.
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import TYPE_CHECKING, Generic, TypeVar

from threadpoolctl import threadpool_limits

from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.mad.scripts import PYTHON_IN_MAD

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    import numpy as np
    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers.common import WorkerConfig
    from aba_optimiser.workers.protocol import WorkerErrorPayload

LOGGER = logging.getLogger(__name__)

# Type variable for worker data type
WorkerDataType = TypeVar("WorkerDataType")


class AbstractWorker(Process, ABC, Generic[WorkerDataType]):
    """Abstract base class for all worker process implementations.

    This class provides the core infrastructure for running optimisation workers
    in separate processes. It handles:
    - Process lifecycle management
    - MAD-NG interface initialization
    - Communication with the main process via pipes
    - Common configuration and logging

    Subclasses must implement:
    - setup_mad_sequence(): Configure MAD-NG sequence parameters
    - send_initial_conditions(): Initialize particle states in MAD-NG
    - compute_gradients_and_loss(): Core computation logic
    - prepare_data(): Process and prepare input data

    Type Parameters:
        WorkerDataType: The type of data structure this worker uses
    """

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: WorkerDataType,
        config: WorkerConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        """Initialize the worker process.

        Args:
            conn: Pipe connection for communicating with main process
            worker_id: Unique identifier for this worker
            data: Worker-specific data (tracking or optics)
            config: Configuration parameters
            simulation_config: Simulation configuration settings
        """
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn
        self.config = config
        self.simulation_config = simulation_config
        # Populated in setup_mad_interface: the knobs this worker actually created
        # (its optimisation range). Runtime knob-updates are filtered to this set so
        # values for magnets outside the worker's range are ignored rather than
        # applied to a nonexistent MAD variable.
        self.knob_name_set: set[str] = set()
        bpm_range_start = config.observation_range_start_bpm or config.tracking_start_bpm
        self.bpm_range = f"{bpm_range_start}/{config.tracking_end_bpm}"

        self.tracking_range = self.bpm_range
        if config.sdir < 0:
            self.tracking_range = f"{config.tracking_end_bpm}/{config.tracking_start_bpm}"
        if config.initial_condition_marker is not None:
            # Kicker mode: the sequence is already cycled to start at the kicker.
            # Pass nil so MAD-NG tracks through the full sequence for all N turns
            # rather than a named range that would treat elements outside it as drifts.
            self.tracking_range = None

        LOGGER.debug(
            "Initializing worker %d for BPM range %s -> %s",
            worker_id,
            config.tracking_start_bpm,
            config.tracking_end_bpm,
        )

        # Let subclasses process their specific data
        self.prepare_data(data)

    @abstractmethod
    def prepare_data(self, data: WorkerDataType) -> None:
        """Process and prepare worker-specific data.

        This method should extract relevant data from the input structure,
        compute weights, split into batches, etc.

        Args:
            data: Worker-specific data structure
        """
        pass

    @abstractmethod
    def setup_mad_sequence(self, mad: MAD) -> None:
        """Configure MAD-NG sequence for this worker type.

        This method should set worker-specific MAD-NG variables like
        number of turns, tracking range, etc.

        Args:
            mad: MAD-NG interface object
        """
        pass

    @abstractmethod
    def send_initial_conditions(self, mad: MAD) -> None:
        """Send initial particle/optics conditions to MAD-NG.

        Args:
            mad: MAD-NG interface object
        """
        pass

    @abstractmethod
    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute gradients and loss for given knob values.

        This is the core computation method that runs tracking/optics
        calculations and computes the gradient of the loss function.

        Args:
            mad: MAD-NG interface object
            knob_updates: Dictionary of knob names to values
            batch: Batch index for multi-batch processing

        Returns:
            Tuple of (gradient_array, loss_value)
        """
        pass

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """Create a base differential algebra (DA) map in MAD-NG.

        The DA map is used for automatic differentiation of tracking
        with respect to optimisation knobs.

        Args:
            mad: MAD-NG interface object
            knob_order: Order of the DA expansion (1 for linear, 2 for quadratic)
        """
        mad.send("coord_names = {'x', 'px', 'y', 'py', 't', 'pt'}")
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, "
            f"mo={knob_order}, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def build_error_payload(self, exc: BaseException, *, phase: str) -> WorkerErrorPayload:
        """Build a structured error payload for parent-side handling."""
        return {
            "worker_id": self.worker_id,
            "status": "error",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }

    def send_error_payload(self, exc: BaseException, *, phase: str) -> None:
        """Best-effort send of a structured worker failure message."""
        payload = self.build_error_payload(exc, phase=phase)
        LOGGER.error(
            "Worker %s failed during %s: %s",
            self.worker_id,
            phase,
            payload["error"],
        )
        try:
            self.conn.send(payload)
        except (BrokenPipeError, EOFError, OSError):
            LOGGER.exception(
                "Worker %s could not send error payload to parent during %s",
                self.worker_id,
                phase,
            )

    def _resolve_per_worker_logfile(self, logfile_path):
        """Return a per-worker logfile path derived from a base logfile path."""
        if logfile_path is None:
            return None

        if logfile_path.suffix:
            return logfile_path.with_name(
                f"{logfile_path.stem}_worker_{self.worker_id}{logfile_path.suffix}"
            )
        return logfile_path.with_name(f"{logfile_path.name}_worker_{self.worker_id}")

    def configure_worker_threads(self) -> None:
        """Limit this worker process's BLAS threads (``SimulationConfig.worker_blas_threads``).

        Must run inside the worker process: the pool is already initialised in the forked
        parent, so the ``OPENBLAS_NUM_THREADS`` environment variable would be too late.
        """
        if self.simulation_config.worker_blas_threads is not None:
            threadpool_limits(limits=self.simulation_config.worker_blas_threads)

    def configure_python_worker_logging(self) -> None:
        """Attach a file handler so worker Python logs land in the worker logfile."""
        worker_logfile = self._resolve_per_worker_logfile(
            self.config.python_logfile or self.config.mad_logfile
        )
        if worker_logfile is None:
            return

        worker_logfile.parent.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        level = self.simulation_config.worker_logging_level
        root_logger.setLevel(level)

        if any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == str(worker_logfile)
            for handler in root_logger.handlers
        ):
            return

        file_handler = logging.FileHandler(worker_logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "PYTHON: %(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

        LOGGER.debug(
            "Worker %s Python logging attached to %s",
            self.worker_id,
            worker_logfile,
        )

    def _cycle_target(self) -> str | None:
        """Element the sequence is cycled to before tracking, or None.

        The initial-condition marker (kicker/ACD) or the point where this
        worker's measured turn increment starts. For backward ranges that is the
        tracking end, because the payload initial coordinates are taken there.
        Validation workers must cycle identically or they track a different path.
        """
        if not self.config.cycle_sequence:
            return None
        tracking_init_bpm = (
            self.config.tracking_start_bpm
            if self.config.sdir > 0
            else self.config.tracking_end_bpm
        )
        return self.config.cycle_marker or self.config.initial_condition_marker or tracking_init_bpm

    def setup_mad_interface(self, init_knobs: dict[str, float]) -> tuple[MAD, int]:
        """Initialize and configure the MAD-NG interface.

        This method uses the accelerator's factory method to create a properly
        configured MAD interface, eliminating the need to manually pass many
        individual parameters.

        Args:
            init_knobs: Initial values for all optimisation knobs

        Returns:
            Tuple of (MAD interface object, number of BPMs)

        Raises:
            ValueError: If knob names from MAD don't match initial knobs
        """
        LOGGER.debug(f"Worker {self.worker_id}: Setting up MAD interface")
        LOGGER.debug(f"Worker {self.worker_id}: Using BPM range {self.bpm_range}")

        worker_logfile = self._resolve_per_worker_logfile(self.config.mad_logfile)

        cycle_target = self._cycle_target()

        # Use accelerator factory to create MAD interface
        mad_iface = GradientDescentMadInterface(
            accelerator=self.config.accelerator,
            magnet_range=self.config.magnet_range,
            bpm_range=self.bpm_range,
            **self.config.interface_options,
            initial_model_values=init_knobs,
            bad_bpms=self.config.bad_bpms,
            debug=self.config.debug,
            mad_logfile=worker_logfile,
            py_name=PYTHON_IN_MAD,
            tracking_anchor_mode=self.config.tracking_anchor_mode,
            tracking_anchor_markers=self.config.tracking_anchor_sources,
            observed_tracking_anchor_markers=self.config.observed_tracking_anchor_markers,
        )

        # Cycle the sequence to this worker's init marker so its tracking range is
        # one contiguous segment. Closed-twiss workers fit from $start and do not.
        if cycle_target is not None:
            mad_iface.cycle_to_start(cycle_target)

        knob_names = mad_iface.knob_names
        self.knob_name_set = set(knob_names)
        # Every knob this worker created (its optimisation range) must have an initial
        # value. The caller provides initial values for the whole optimisation, which may
        # also include magnets in this worker's tracking range that it does not optimise.
        missing = self.knob_name_set - set(init_knobs)
        if missing:
            raise ValueError(
                f"Worker {self.worker_id}: {len(missing)} MAD knobs have no initial value, "
                f"e.g. {sorted(missing)[:5]}"
            )

        # Non-optimised pt is not an element strength, so keep it as a fixed tracking
        # scalar while the optimiser updates only its own knob vector.
        self.fixed_pt = (
            float(init_knobs.get("pt", 0.0)) if "pt" not in self.knob_name_set else 0.0
        )

        mad = mad_iface.mad
        mad["knob_names"] = knob_names
        # With no tracking range (kicker mode) MAD tracks the full cycled ring and
        # observes every monitor, so the observable vectors must be sized for all
        # BPMs. The named range count would miss the BPM that wraps past the start
        # marker, undersizing the vectors and overflowing during tracking.
        nbpms = len(mad_iface.all_bpms) if self.tracking_range is None else mad_iface.nbpms
        mad["nbpms"] = nbpms
        mad["sdir"] = self.config.sdir

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix", "vector")
        mad.load("MAD.utility", "tblcat")

        # Call worker-specific sequence setup
        self.setup_mad_sequence(mad)

        # Setup differential algebra maps
        self._setup_da_maps(mad)

        return mad, nbpms

    @abstractmethod
    def _setup_da_maps(self, mad: MAD) -> None:
        """Setup differential algebra maps specific to worker type.

        Args:
            mad: MAD-NG interface object
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """Main worker run loop.

        This method handles the communication protocol:
        1. Wait for initial handshake
        2. Receive initial knob values
        3. Setup MAD interface
        4. Loop: receive knobs -> compute -> send results
        5. Cleanup on termination signal (None received)
        """
        pass

    @abstractmethod
    def _initialise_mad_computation(self, mad: MAD) -> None:
        """Initialise MAD-NG environment for computation.

        This method should load any initialisation scripts needed
        before the main computation loop.

        Args:
            mad: MAD-NG interface object
        """
        pass
