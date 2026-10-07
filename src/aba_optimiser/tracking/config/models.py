"""Configuration dataclasses for fitter initialization.

These dataclasses group related parameters to reduce the number of
individual arguments passed to fitter constructors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class MeasurementDetails:
    """Per-measurement model setup and momentum metadata.

    ``interface_options`` are passed straight through as keyword arguments to the
    MAD-NG interface for this measurement, so any MAD interface option is
    supported (commonly ``machine_state``,
    ``b2_errors``). The bunch structure is read from the ``bunch_number`` column
    of the measurement parquet, so it is not configured here.

    ``first_bpm`` names the BPM each recorded turn begins at, used to cycle the
    measurement data to the boundary it was generated from. Leave it ``None`` to
    use the file's own first recorded BPM; set it when that row order is
    unreliable (for example ACD marker rows written after the BPMs).
    """

    interface_options: dict[str, Any] = field(default_factory=dict)
    machine_deltap: float = 0.0
    first_bpm: str | None = None


@dataclass
class MeasurementConfig:
    """Maps each measurement file to its :class:`MeasurementDetails`."""

    measurements: dict[Path, MeasurementDetails]

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("MeasurementConfig requires at least one measurement file")

    @property
    def files(self) -> list[Path]:
        """Return the measurement files in insertion order."""
        return list(self.measurements.keys())

    @property
    def details(self) -> list[MeasurementDetails]:
        """Return the per-file details in measurement-file order."""
        return list(self.measurements.values())

    def log_state(self) -> None:
        """Log the current measurement config settings."""
        logger.info("MeasurementConfig: %s", self)


@dataclass
class KickerConfig:
    """Configuration for kicker-only tracking runs.

    Attributes:
        kicker_name: Element name used as the initial-condition marker.
        turns_after_kicker: Number of turns to track after the kicker.
    """

    kicker_name: str
    turns_after_kicker: int

    def __post_init__(self) -> None:
        if self.turns_after_kicker < 1:
            raise ValueError("turns_after_kicker must be >= 1")

    def log_state(self) -> None:
        """Log the current kicker config settings."""
        logger.info("KickerConfig: %s", self)


@dataclass
class CheckpointConfig:
    """Checkpoint save/restore behaviour for optimisation runs."""

    checkpoint_path: Path
    checkpoint_every_n_epochs: int = 0
    restore_from_checkpoint: bool = False
