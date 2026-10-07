"""Configuration shared by every fitter: the lattice range and the output settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aba_optimiser.config import TRAINING_RUNS_ROOT

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SequenceConfig:
    """Configuration for the sequence segment used during optimisation.

    The fields define the magnet range to expose to MAD-NG and any BPMs that
    should be ignored. Where measurement data starts (the BPM each recorded turn
    begins at) is a property of the measurement, not the sequence, so it lives on
    :class:`MeasurementDetails`.
    """

    magnet_range: str
    bad_bpms: list[str] | None = None

    def log_state(self) -> None:
        """Log the current sequence config settings."""
        logger.info("SequenceConfig: %s", self)


@dataclass
class OutputConfig:
    """Output and logging behaviour for optimisation runs.

    Attributes:
        write_tensorboard_logs: Whether to write TensorBoard event files.
        include_uncertainty: Whether to compute uncertainties. Disabling this
            skips worker-side Hessian estimation for faster execution.
        tensorboard_root: Root directory for TensorBoard event-file runs.
        mad_logfile: Optional MAD log file path.
        python_logfile: Optional Python worker log file path.
    """

    write_tensorboard_logs: bool = True
    include_uncertainty: bool = True
    tensorboard_root: Path = field(default_factory=lambda: TRAINING_RUNS_ROOT)
    mad_logfile: Path | None = None
    python_logfile: Path | None = None

    def log_state(self) -> None:
        """Log the current output config settings."""
        logger.info("OutputConfig: %s", self)
