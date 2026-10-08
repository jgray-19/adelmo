"""Reusable orchestration helpers for end-to-end AC-dipole pipelines.

The accelerator-specific parts of an integration pipeline are model creation,
the accelerator options passed to omc3, and the fitted accelerator class.  The
data plumbing is identical for PSB and LHC: convert long-form turn-by-turn data
to Harpy input, run driven and compensated optics, combine measured positions
with fitted model angles, and merge reconstructed momenta. This module keeps
that common path out of accelerator tests.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omc3.hole_in_one import hole_in_one_entrypoint
from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pandas as pd


@dataclass(frozen=True)
class ACDOpticsAnalysisConfig:
    """Configuration shared by the Harpy and two optics-analysis stages."""

    model_dir: Path
    harpy_options: Mapping[str, Any]
    optics_options: Mapping[str, Any]


def long_frame_to_tbt_data(frame: pd.DataFrame, *, source_file: Path) -> TbtData:
    """Convert long-form ``name/turn/x/y`` data into Harpy's in-memory type."""
    names = list(dict.fromkeys(frame["name"].astype(str)))
    turns = sorted(int(turn) for turn in frame["turn"].unique())

    def matrix(plane: str) -> pd.DataFrame:
        result = (
            frame.pivot(index="name", columns="turn", values=plane)
            .reindex(index=names, columns=turns)
            .astype(float)
        )
        if result.isna().any().any():
            raise ValueError(f"Incomplete {plane}-plane turn-by-turn matrix")
        return result

    return TbtData(
        matrices=[TransverseData(X=matrix("x"), Y=matrix("y"))],
        nturns=len(turns),
        bunch_ids=[0],
        meta={"file": str(source_file)},
    )


def run_driven_and_compensated_optics(
    acquisitions: Sequence[tuple[Path, pd.DataFrame]],
    *,
    output_dir: Path,
    config: ACDOpticsAnalysisConfig,
) -> tuple[Path, Path]:
    """Run Harpy over every acquisition, then driven and compensated optics.

    Each ``(source_file, frame)`` pair is one acquisition of the same machine
    state. omc3 derives its measurement errors from the *spread* across
    acquisitions: ``optics_measurements.phase._get_phases`` short-circuits on
    ``phases_meas.ndim < 2`` and writes a matrix of exact zeros when it is handed
    a single file, so a one-acquisition analysis reports ``ERRPHASEX``/
    ``ERRPHASEY`` of 0 -- which downstream reads as an infinitely precise phase
    that no inverse-variance fit can weight. Pass at least two acquisitions
    whenever the phase errors are going to be used.

    Args:
        acquisitions: ``(source_file, frame)`` pairs of long-form
            ``name/turn/x/y`` turn-by-turn data. The source file names the lin
            output; it need not exist on disk.
        output_dir: Root for the ``lin_files``, ``driven`` and ``compensated``
            subdirectories.
        config: Model directory and the Harpy/optics option sets.

    Returns:
        The ``driven`` and ``compensated`` output directories.

    Raises:
        ValueError: If no acquisitions are given, if two share a source file
            name, or if ``config`` asks for Harpy cleaning.
    """
    if not acquisitions:
        raise ValueError("At least one acquisition is required")
    source_files = [source for source, _ in acquisitions]
    counts = Counter(source.name for source in source_files)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Acquisitions must have distinct file names; repeated: {duplicates}")

    harpy_options = dict(config.harpy_options)
    if harpy_options.get("clean", False):
        raise ValueError(
            "Harpy/OMC3 cleaning is disabled for ACD pipelines; clean the "
            "turn-by-turn data before passing it to Harpy instead."
        )
    harpy_options["clean"] = False
    lin_dir = output_dir / "lin_files"
    driven_dir = output_dir / "driven"
    compensated_dir = output_dir / "compensated"
    lin_dir.mkdir(parents=True, exist_ok=True)
    input_data = [
        long_frame_to_tbt_data(frame, source_file=source) for source, frame in acquisitions
    ]

    hole_in_one_entrypoint(
        harpy=True,
        optics=False,
        files=input_data,
        outputdir=lin_dir,
        tbt_datatype="tbt_data",
        model_dir=config.model_dir,
        **harpy_options,
    )
    lin_bases = [lin_dir / source.name for source in source_files]
    common = dict(config.optics_options)
    for destination, compensation in (
        (driven_dir, "none"),
        (compensated_dir, "equation"),
    ):
        hole_in_one_entrypoint(
            harpy=False,
            optics=True,
            files=lin_bases,
            outputdir=destination,
            model_dir=config.model_dir,
            compensation=compensation,
            **common,
        )
    return driven_dir, compensated_dir

