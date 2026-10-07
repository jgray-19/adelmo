"""Helper functions for creating fitter configurations.

This module provides convenience functions to reduce duplication when
creating fitter configuration objects for common use cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aba_optimiser.tracking.config.models import MeasurementConfig, MeasurementDetails

if TYPE_CHECKING:
    from collections.abc import Mapping


def create_arc_measurement_config(
    measurement_file: Path,
    machine_deltap: float = 0.0,
    machine_state: Mapping[str, float] | Path | None = None,
    first_bpm: str | None = None,
    b2_errors: Path | None = None,
) -> MeasurementConfig:
    """Build a single-file MeasurementConfig.

    Convenience for the common one-measurement case. Multi-file runs build the
    ``{path: MeasurementDetails(...)}`` mapping directly. The bunch structure is
    read from the parquet's ``bunch_number`` column, so it is not configured here.
    ``first_bpm`` names the BPM the recorded turns begin at; leave it ``None`` to
    use the file's own first recorded BPM. ``b2_errors`` optionally applies an LHC
    dipole b2 error table during optimisation (requires ``machine_state``, holding the tune knobs).
    """
    interface_options: dict = {}
    if machine_state is not None:
        interface_options["machine_state"] = machine_state
    if b2_errors is not None:
        interface_options["b2_errors"] = b2_errors
    return MeasurementConfig(
        {
            Path(measurement_file): MeasurementDetails(
                interface_options=interface_options,
                machine_deltap=machine_deltap,
                first_bpm=first_bpm,
            )
        }
    )
