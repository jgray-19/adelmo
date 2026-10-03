"""Shared utilities for training fitters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from pymadng_utils.io.utils import read_knobs

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "bpm_supports_both_planes",
    "bpm_supports_plane",
    "create_bpm_range_specs",
    "extract_bpm_range_names",
    "filter_bad_bpms",
    "find_common_bpms",
    "load_tfs_files",
    "normalise_true_strengths",
]


def find_common_bpms(*dataframes: pd.DataFrame) -> list[str]:
    """Return BPM names common to all tables, preserving the first table order."""
    if not dataframes:
        return []
    common = set(dataframes[0].index)
    for dataframe in dataframes[1:]:
        common &= set(dataframe.index)
    return [str(bpm) for bpm in dataframes[0].index if bpm in common]


def filter_bad_bpms(
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    bad_bpms: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Remove bad BPMs from start and end point lists.

    Args:
        bpm_start_points: List of starting BPM names
        bpm_end_points: List of ending BPM names
        bad_bpms: Optional list of BPM names to remove

    Returns:
        Tuple of (filtered_start_points, filtered_end_points)
    """
    if bad_bpms is None:
        return bpm_start_points, bpm_end_points

    filtered_start = bpm_start_points.copy()
    filtered_end = bpm_end_points.copy()

    for bpm in bad_bpms:
        if bpm in filtered_start:
            filtered_start.remove(bpm)
            LOGGER.warning(f"Removed bad BPM {bpm} from start points")
        if bpm in filtered_end:
            filtered_end.remove(bpm)
            LOGGER.warning(f"Removed bad BPM {bpm} from end points")

    return filtered_start, filtered_end


def normalise_true_strengths(
    true_strengths: Path | dict[str, float] | None,
) -> dict[str, float]:
    """Normalise true strengths to a dictionary format.

    Args:
        true_strengths: Can be None, a Path to a file, or a dict

    Returns:
        Dictionary of true strengths (empty if None was provided)
    """
    if true_strengths is None:
        return {}
    if isinstance(true_strengths, Path):
        return read_knobs(true_strengths)
    if isinstance(true_strengths, dict):
        return true_strengths.copy()
    raise TypeError(f"Unexpected type for true_strengths: {type(true_strengths)}")


def extract_bpm_range_names(
    all_bpms: list[str],
    start_bpm: str,
    end_bpm: str,
    sdir: int,
    allow_missing_start: bool = False,
) -> list[str]:
    """Extract BPM names between start and end BPMs, handling circular wrapping.

    Args:
        all_bpms: List of all BPM names
        start_bpm: Starting BPM name
        end_bpm: Ending BPM name
        sdir: Direction (1 for forward, -1 for reverse)

    Returns:
        List of BPM names in the range
    """
    if start_bpm in all_bpms:
        start_pos = all_bpms.index(start_bpm)
    elif allow_missing_start:
        if sdir == -1:
            raise ValueError(
                f"Start marker '{start_bpm}' is not in BPM list for reverse tracking"
            )
        start_pos = 0
    else:
        raise ValueError(f"Start BPM '{start_bpm}' not found in BPM list")

    if end_bpm not in all_bpms:
        raise ValueError(f"End BPM '{end_bpm}' not found in BPM list")
    end_pos = all_bpms.index(end_bpm) + 1

    if end_pos <= start_pos:
        # Circular wrapping
        extracted = all_bpms[start_pos:] + all_bpms[:end_pos]
    else:
        extracted = all_bpms[start_pos:end_pos]
    # Reverse for negative direction
    if sdir == -1:
        extracted = extracted[::-1]

    return extracted


def load_tfs_files(
    directory: Path,
    file_specs: dict[str, tuple[str, str]],
) -> dict[str, pd.DataFrame]:
    """Load TFS files, preserving phase-table NAME/NAME2 rows."""
    no_index_keys = {key for key in file_specs if "phase" in key and "beta" not in key}
    loaded = {}
    for key, (prefix, suffix) in file_specs.items():
        path = directory / f"{prefix}{suffix}.tfs"
        if not path.exists():
            raise FileNotFoundError(path)
        loaded[key] = tfs.read(path, index=None if key in no_index_keys else "NAME")
    return loaded


def bpm_supports_plane(accelerator, bpm: str, kick_plane: str) -> bool:
    """Return whether ``bpm`` can measure the requested kick plane."""
    plane = accelerator.infer_monitor_plane(bpm)
    if kick_plane in ("x", "X"):
        return "H" in plane
    if kick_plane in ("y", "Y"):
        return "V" in plane
    if kick_plane in ("xy", "XY"):
        return ("H" in plane) or ("V" in plane)
    raise ValueError(f"Unsupported kick plane {kick_plane!r}")


def bpm_supports_both_planes(accelerator, bpm: str) -> bool:
    """Return whether ``bpm`` can measure both transverse planes."""
    return bpm_supports_plane(accelerator, bpm, "x") and bpm_supports_plane(accelerator, bpm, "y")


def create_bpm_range_specs(
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    use_fixed_bpm: bool,
    fixed_start: str | None = None,
    fixed_end: str | None = None,
) -> list[tuple[str, str, int]]:
    """Create BPM range specifications for optimisation workers.

    Args:
        bpm_start_points: List of starting BPM names
        bpm_end_points: List of ending BPM names
        use_fixed_bpm: If True, use fixed BPM pairs; if False, create cartesian product
        fixed_start: Fixed start BPM for backward tracking (only used if use_fixed_bpm=True)
        fixed_end: Fixed end BPM for forward tracking (only used if use_fixed_bpm=True)

    Returns:
        List of (start_bpm, end_bpm, sdir) tuples where sdir is 1 for forward, -1 for reverse
    """
    if use_fixed_bpm:
        LOGGER.warning("Using fixed BPM pairs for optimisation. This will create fewer, more constrained measurement combinations.")
        if fixed_start is None or fixed_end is None:
            raise ValueError("fixed_start and fixed_end must be provided when use_fixed_bpm=True")
        # Forward: start -> fixed_end; Backward: fixed_start -> end
        range_specs = [(s, fixed_end, 1) for s in bpm_start_points] + [
            (fixed_start, e, -1) for e in bpm_end_points
        ]
    else:
        # Cartesian product: every start with every end in both directions
        range_specs = [
            (s, e, sdir) for s in bpm_start_points for e in bpm_end_points for sdir in (1, -1)
        ]

    # MAD tracks "A/A" as the single element A: the worker would observe only its own
    # start point and constrain nothing.
    degenerate = sorted({s for s, e, _ in range_specs if s == e})
    if degenerate:
        raise ValueError(
            f"BPM ranges must not start and end at the same BPM: {degenerate}. "
            "Choose start and end points that differ (with use_fixed_bpm, the fixed start "
            "and end BPMs must also not appear among the end and start points)."
        )
    return range_specs
