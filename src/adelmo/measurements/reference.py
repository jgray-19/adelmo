"""Construction helpers for tmom-recon's closed-orbit reconstruction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


@dataclass(frozen=True)
class ReconstructionFrame:
    """The explicit closed-orbit inputs `tmom_recon`'s reconstruction calls take.

    tmom-recon takes ``closed_orbit_at_zero`` (measured BPM x/y at dp=0) and
    ``orbit_mode`` as keyword arguments, and generates any restored angle itself
    from the model.
    """

    closed_orbit_at_zero: pd.DataFrame
    orbit_mode: str


def reconstruction_frame(
    closed_orbit: pd.DataFrame,
    *,
    dynamic_planes: Iterable[str] = (),
) -> ReconstructionFrame:
    """Build the closed-orbit reconstruction inputs for one measurement.

    tmom-recon subtracts ``closed_orbit_at_zero`` and then, per ``orbit_mode``,
    either restores nothing (``"dynamic"``: the result stays relative to the
    measured orbit) or restores the measured positions plus model-generated
    angles (``"absolute"``): every plane dynamic gives ``orbit_mode="dynamic"``, no plane
    dynamic gives ``orbit_mode="absolute"``. A mix of the two is not supported.
    """
    orbit = closed_orbit.copy()
    if "name" in orbit.columns:
        orbit = orbit.set_index("name")
    orbit.columns = [str(column).lower() for column in orbit.columns]
    dynamic = tuple(sorted(str(plane).lower() for plane in dynamic_planes))
    unknown = set(dynamic) - {"x", "y"}
    if unknown:
        raise ValueError(f"Unknown dynamic plane(s): {sorted(unknown)}")

    if dynamic == ("x", "y"):
        return ReconstructionFrame(orbit[["x", "y"]], "dynamic")
    if not dynamic:
        return ReconstructionFrame(orbit[["x", "y"]], "absolute")
    raise ValueError(
        f"A frame that is dynamic in {dynamic} but absolute in the other plane is "
        "not supported: tmom-recon reconstructs all four coordinates as one mode."
    )


__all__ = ["ReconstructionFrame", "reconstruction_frame"]
