"""Abstract base class for all worker process types.

:meth:`AbstractWorker.run` owns the process lifecycle and the message loop of
:mod:`aba_optimiser.fitting.protocol`; subclasses supply the MAD-NG setup and
answer :class:`~aba_optimiser.fitting.protocol.Evaluate` and
:class:`~aba_optimiser.fitting.protocol.Command` messages.
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from threadpoolctl import threadpool_limits

from aba_optimiser.fitting.protocol import Command, ErrorReply, Evaluate, Stop
from aba_optimiser.machine.mad import GradientDescentMadInterface
from aba_optimiser.machine.mad.scripts import PYTHON_IN_MAD

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.fitting.protocol import Ack, GradReply, LossReply
    from aba_optimiser.machine.accelerators import Accelerator

LOGGER = logging.getLogger(__name__)

# Type variable for worker data type
WorkerDataType = TypeVar("WorkerDataType")


class KickPlane(str, Enum):
    """Kick-plane options for worker routing and payload selection."""

    X = "x"
    Y = "y"
    XY = "xy"


@dataclass
class WorkerConfig:
    """Configuration shared by all worker processes.

    The accelerator object bundles machine-specific setup, while the remaining
    fields describe the local BPM range, tracking direction, and optional input
    files needed by the worker.

    """

    accelerator: Accelerator
    tracking_start_bpm: str
    tracking_end_bpm: str
    magnet_range: str
    # Per-measurement keyword arguments forwarded to the MAD-NG interface, e.g.
    # machine_state, b2_errors.
    interface_options: dict[str, Any] = field(default_factory=dict)
    observation_range_start_bpm: str | None = None
    initial_condition_marker: str | None = None
    # Whether to cycle the sequence so tracking starts at this worker's init
    # marker. Closed-twiss workers fit the whole ring from ``$start`` and set it
    # False; every tracking plan cycles.
    cycle_sequence: bool = True
    sdir: int = 1
    kick_plane: KickPlane = KickPlane.XY
    bad_bpms: list[str] | None = None
    debug: bool = False
    mad_logfile: Path | None = None
    python_logfile: Path | None = None
    tracking_anchor_mode: str | None = None
    tracking_anchor_sources: list[str] | None = None
    observed_tracking_anchor_markers: list[str] | None = None
    cycle_marker: str | None = None


class AbstractWorker(Process, ABC, Generic[WorkerDataType]):
    """Abstract base class for all worker process implementations.

    Subclasses implement :meth:`prepare_data`, :meth:`_setup_da_maps` and
    :meth:`evaluate`, and override :meth:`on_start`, :meth:`handle_command`,
    :meth:`on_stop` and :meth:`close` where they need to.

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

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Set worker-specific MAD-NG variables before the DA maps are built."""

    @abstractmethod
    def evaluate(self, mad: MAD, message: Evaluate) -> GradReply:
        """Answer one :class:`~aba_optimiser.fitting.protocol.Evaluate` message."""

    def handle_command(self, mad: MAD, command: Command) -> Ack | LossReply:
        """Answer one :class:`~aba_optimiser.fitting.protocol.Command` message."""
        raise ValueError(f"Worker {self.worker_id}: unsupported command {command.kind.name}")

    def on_start(self, knobs: dict[str, float]) -> MAD:
        """Build the MAD-NG session from the :class:`~aba_optimiser.fitting.protocol.Start` knobs."""
        mad, _nbpms = self.setup_mad_interface(knobs)
        return mad

    def on_stop(self, mad: MAD, failed: bool) -> None:
        """Called once the message loop ends, before the session is closed."""

    def close(self) -> None:
        """Release resources other than the MAD-NG session."""

    def send_reply(self, reply: object) -> None:
        """Send one reply to the parent."""
        self.conn.send(reply)

    def run(self) -> None:
        """Run the worker: :class:`Start`, then evaluations and commands until :class:`Stop`."""
        mad: MAD | None = None
        try:
            self.configure_python_worker_logging()
            self.configure_worker_threads()
            message = self.conn.recv()
            if isinstance(message, Stop):
                return
            mad = self.on_start(message.knobs)
            failed = False
            while True:
                message = self.conn.recv()
                if isinstance(message, Stop):
                    LOGGER.debug("Worker %s: received termination signal", self.worker_id)
                    break
                if isinstance(message, Command):
                    self.send_reply(self.handle_command(mad, message))
                    continue
                if not isinstance(message, Evaluate):
                    raise ValueError(f"Worker {self.worker_id}: unexpected message {message!r}")
                try:
                    self.send_reply(self.evaluate(mad, message))
                except Exception as exc:  # noqa: BLE001
                    self.send_error_payload(exc, phase="computation")
                    failed = True
                    break
            self.on_stop(mad, failed)
        except Exception as exc:  # noqa: BLE001
            self.send_error_payload(exc, phase="startup")
        finally:
            LOGGER.debug("Worker %s: terminating", self.worker_id)
            self.close()
            if mad is not None:
                mad.send("shush()")

    def send_knobs(self, mad: MAD, knobs: dict[str, float], *, plain: bool = False) -> None:
        """Set this worker's own knobs; ``pt`` is never a sequence knob.

        Knobs outside the worker's optimisation range are ignored. ``plain`` assigns
        numbers (a session without knob parameters) instead of setting the constant
        part of each knob's TPSA.
        """
        template = "loaded_sequence['{}'] = {:.15e}" if plain else "loaded_sequence['{}']:set0({:.15e})"
        commands = [
            template.format(name, value)
            for name, value in knobs.items()
            if name != "pt" and name in self.knob_name_set
        ]
        if commands:
            mad.send("\n".join(commands))

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """Create a base differential algebra (DA) map in MAD-NG.

        The DA map is used for automatic differentiation of tracking
        with respect to optimisation knobs.

        Args:
            mad: MAD-NG interface object
            knob_order: Order of the DA expansion (1 for linear, 2 for quadratic)
        """
        mad.send(
            f"da_x0_base = damap{{nv=6, np=#knob_names, "
            f"mo={knob_order}, po={knob_order}, pn=knob_names}}"
        )

    def send_error_payload(self, exc: BaseException, *, phase: str) -> None:
        """Best-effort send of the failure to the parent."""
        reply = ErrorReply(
            worker_id=self.worker_id,
            phase=phase,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        LOGGER.error("Worker %s failed during %s: %s", self.worker_id, phase, reply.error)
        try:
            self.conn.send(reply)
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
