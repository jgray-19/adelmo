"""Base fitter class with shared functionality for all fitters."""

from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tensorboardX import SummaryWriter

from aba_optimiser.training.config.manager import ConfigurationManager
from aba_optimiser.training.config.models import OutputConfig
from aba_optimiser.training.optimisation.loop import OptimisationLoop
from aba_optimiser.training.utils import filter_bad_bpms, normalise_true_strengths

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypeAlias

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.config.manager import (
        ConfigurationManager as ConfigurationManagerType,
    )
    from aba_optimiser.training.config.models import SequenceConfig

    ConfigurationManagerCls: TypeAlias = type[ConfigurationManagerType]

LOGGER = logging.getLogger(__name__)


class BaseFitter(ABC):
    """Base class for all optimisation fitters.

    Provides shared functionality for:
    - Configuration management
    - Optimisation loop setup
    - Result management
    - Logging setup
    - Common initialization patterns

    Design: optimisation-space only end-to-end. User inputs, internal state, and
    reported results all use the same knob coordinates.

    Subclasses that need to complete their own setup before ``_init_managers`` is
    called should set ``_defer_managers = True`` as a class attribute.  They are
    then responsible for calling ``BaseFitter._init_managers(self)`` explicitly
    once their setup is complete.
    """

    _defer_managers: bool = False
    _configuration_manager_cls: ConfigurationManagerCls = ConfigurationManager

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        debug: bool = False,
        optimise_knobs: list[str] | None = None,
        output_config: OutputConfig | None = None,
    ):
        """Initialize base fitter.

        User inputs (initial_knob_strengths, true_strengths) are expected in
        optimisation space and are passed through unchanged.

        Args:
            accelerator: Accelerator instance defining machine configuration
            optimiser_config: Gradient descent optimiser configuration
            simulation_config: Simulation and worker configuration
            sequence_config: Sequence and BPM filtering configuration
            bpm_start_points: Start BPMs for each range
            bpm_end_points: End BPMs for each range
            initial_knob_strengths: Initial knob strengths in optimisation space.
            true_strengths: True strengths (Path, dict, or None) in optimisation space.
            debug: Enable debug mode
            optimise_knobs: List of global knob names to optimise, or None
            output_config: Output and logging configuration. Defaults to OutputConfig().
        """
        self.optimiser_config = optimiser_config
        self.simulation_config = simulation_config
        self.accelerator = accelerator
        self.debug = debug
        self.output_config = output_config if output_config is not None else OutputConfig()
        self.mad_logfile: Path | None = self.output_config.mad_logfile
        self.python_logfile: Path | None = self.output_config.python_logfile

        self.output_config.log_state()
        self.optimiser_config.log_state()
        self.simulation_config.log_state()
        sequence_config.log_state()

        if not accelerator.has_any_optimisation():
            raise ValueError("No optimisation types enabled in the accelerator configuration.")

        config_manager_kwargs = self._get_configuration_manager_kwargs()

        # Filter bad BPMs
        bpm_start_points, bpm_end_points = filter_bad_bpms(
            bpm_start_points, bpm_end_points, sequence_config.bad_bpms
        )
        tracking_plan = config_manager_kwargs.get("tracking_plan")
        start_point_label = (
            tracking_plan.start_point_label
            if tracking_plan is not None
            else "BPM tracking start points"
        )
        LOGGER.info(
            "After filtering bad BPMs, %s: %s; end points: %s",
            start_point_label,
            bpm_start_points,
            bpm_end_points,
        )

        # Initialize configuration manager
        self.config_manager = self._configuration_manager_cls(
            accelerator,
            simulation_config,
            sequence_config,
            bpm_start_points,
            bpm_end_points,
            optimise_knobs,
            **config_manager_kwargs,
        )
        mad_setup_kwargs = self._get_mad_setup_kwargs()
        self.config_manager.setup_mad_interface(
            debug,
            self.mad_logfile,
            **mad_setup_kwargs,
        )

        # Keep user-space inputs in optimisation space throughout the fitter.
        true_strengths_dict = normalise_true_strengths(true_strengths)
        true_strengths_delta = self.convert_deltap_to_pt(true_strengths_dict)
        initial_knobs_delta = self.convert_deltap_to_pt(initial_knob_strengths)
        if initial_knobs_delta is not None:
            initial_knobs_delta = self.accelerator.normalise_initial_knobs(initial_knobs_delta)

        # Initialize knob strengths in optimisation space
        self.initial_knobs, self.filtered_true_strengths = (
            self.config_manager.initialise_knob_strengths(true_strengths_delta, initial_knobs_delta)
        )
        self._validate_knob_initialisation()

        # Use initial knobs as true strengths if none provided.
        if not true_strengths_delta:
            self.filtered_true_strengths = self.initial_knobs.copy()

        # Initialize managers (may be deferred by subclasses via _defer_managers)
        if not self._defer_managers:
            self._init_managers()

    def convert_deltap_to_pt(
        self, initial_knob_strengths: dict[str, float] | None
    ) -> dict[str, float] | None:
        """Normalise user-provided initial knob strengths into optimisation space."""
        if initial_knob_strengths is None:
            return None
        initial_knob_strengths = initial_knob_strengths.copy()
        if "deltap" in initial_knob_strengths:
            initial_knob_strengths["pt"] = self.config_manager.mad_iface.dp2pt(
                initial_knob_strengths.pop("deltap")
            )
        return initial_knob_strengths

    def _init_managers(self) -> None:
        """Initialize optimisation loop and result manager."""
        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            self.optimiser_config,
            self.simulation_config,
        )

        output_knob_names = self.accelerator.format_result_knob_names(
            self.config_manager.knob_names
        )

        self.output_knob_names = output_knob_names

    def _validate_knob_initialisation(self) -> None:
        """Validate that fitter setup produced a usable knob set."""
        knob_names = self.config_manager.knob_names
        if not knob_names:
            raise ValueError(
                "No optimisation knobs were created for this fitter configuration. "
                f"Optimisation is enabled, but the MAD model returned zero knobs for "
                f"magnet range '{self.config_manager.magnet_range}'. Check that the "
                "selected optimisation flags match elements present in the loaded "
                "sequence and range."
            )

        if len(self.initial_knobs) != len(knob_names):
            raise ValueError(
                "Knob initialisation produced an inconsistent result: "
                f"{len(knob_names)} knob names but {len(self.initial_knobs)} initial values."
            )

    def _get_mad_setup_kwargs(self) -> dict:
        """Return extra kwargs for fitter-side MAD interface setup."""
        return {}

    def _get_configuration_manager_kwargs(self) -> dict:
        """Return extra kwargs for configuration manager construction."""
        return {}

    def setup_logging(self, log_suffix: str = "opt") -> SummaryWriter | None:
        """Set up TensorBoard logging.

        Args:
            log_suffix: Suffix for the log directory name

        Returns:
            TensorBoard SummaryWriter instance or None when disabled
        """
        if not self.output_config.write_tensorboard_logs:
            LOGGER.info("TensorBoard logging disabled")
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.output_config.tensorboard_root / f"{timestamp}_{log_suffix}"
        log_dir.parent.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    @abstractmethod
    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process.

        Returns:
            Tuple of (final_knobs, uncertainties)
        """
        pass
