"""Measurement data loading and turn batching for the tracking optimisation.

A measurement file becomes a :class:`FileTracks`: a dense ``(turn, marker)`` grid
of coordinate arrays plus the index maps needed to address it. Everything
downstream -- worker payloads, initial conditions, momentum refreshes -- reads
that grid directly, so there is exactly one place where measurement rows are put
into their tracking order.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aba_optimiser.config import FILE_COLUMNS
from aba_optimiser.training.tracking.workers.turn_planner import (
    WorkerTurnPlanner,
    group_turns_by_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.config.tracking import TrackingPlan

    ShuffleTurns = Callable[[list[int]], None]

LOGGER = logging.getLogger(__name__)
#: Seed of the default turn shuffle.
TURN_SHUFFLE_SEED = 42

#: Attempts (and initial backoff) for a measurement-parquet read. Measurement
#: parquets live on AFS/NFS, where a transient ``OSError`` clears on its own;
#: without a retry one blip aborts an hour-long fit.
_READ_ATTEMPTS = 3
_READ_BACKOFF_SECONDS = 1.0

#: The observables stored per marker, paired with their variance columns.
COORDINATES: tuple[str, ...] = ("x", "px", "y", "py")
VARIANCES: tuple[str, ...] = ("var_x", "var_px", "var_y", "var_py")
GRID_COLUMNS: tuple[str, ...] = COORDINATES + VARIANCES


@dataclass
class FileTracks:
    """One measurement file as a dense ``(turn, marker)`` grid.

    ``values[name][i, j]`` is observable ``name`` at ``turns[i]`` and
    ``markers[j]``. Markers are in ring order cycled to the BPM the file's turns
    were recorded from, so a contiguous tracking range maps to a monotonic column
    sequence that crosses the ring boundary exactly once -- which is how the
    payload builder infers per-turn wraps.
    """

    turns: np.ndarray
    markers: list[str]
    values: dict[str, np.ndarray]
    kick_plane: str

    turn_row: dict[int, int] = field(init=False)
    marker_col: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.turn_row = {int(turn): row for row, turn in enumerate(self.turns)}
        self.marker_col = {name: col for col, name in enumerate(self.markers)}

    @property
    def first_turn(self) -> int:
        """Return the lowest global turn number in this file."""
        return int(self.turns[0])

    def rows(self, turns: list[int]) -> np.ndarray:
        """Return the grid rows for the given global turn numbers."""
        return np.array([self.turn_row[t] for t in turns], dtype=np.int64)

    def cols(self, markers: list[str]) -> np.ndarray:
        """Return the grid columns for the given marker names."""
        return np.array([self.marker_col[m] for m in markers], dtype=np.int64)

    def with_updated_momenta(self, reconstructed: pd.DataFrame) -> FileTracks:
        """Return a copy with ``px``/``py`` (and their variances) patched.

        ``reconstructed`` is a long-form frame of ``turn, name, px, py, var_px,
        var_py`` in file-local turn numbering, as produced by the momentum
        reconstruction. Names are matched case-insensitively; rows naming a
        marker or turn this file does not hold are ignored, and markers the
        reconstruction does not cover keep their existing momenta.
        """
        upper_to_col = {name.upper(): col for name, col in self.marker_col.items()}
        values = {name: array.copy() for name, array in self.values.items()}

        frame = pd.DataFrame(reconstructed)
        rows = [self.turn_row.get(int(t) + self.first_turn) for t in frame["turn"]]
        cols = [upper_to_col.get(str(name).upper()) for name in frame["name"]]
        keep = [
            i
            for i, (r, c) in enumerate(zip(rows, cols))
            if r is not None and c is not None
        ]
        if not keep:
            return self

        row_idx = np.array([rows[i] for i in keep], dtype=np.int64)
        col_idx = np.array([cols[i] for i in keep], dtype=np.int64)
        for column in ("px", "py", "var_px", "var_py"):
            updated = frame[column].to_numpy(dtype="float64")[keep]
            present = ~np.isnan(updated)
            values[column][row_idx[present], col_idx[present]] = updated[present]

        return FileTracks(
            turns=self.turns,
            markers=self.markers,
            values=values,
            kick_plane=self.kick_plane,
        )


def infer_kick_plane(frame: pd.DataFrame) -> str:
    """Infer whether a measurement file is excited in x, y, or both planes.

    A plane counts as excited when its coordinate or momentum spans a measurable
    range; when both do, the larger has to dominate by an order of magnitude for
    the file to be treated as single-plane.
    """

    def span(coord: str, momentum: str) -> float:
        spans = []
        for column in (coord, momentum):
            values = frame[column].dropna().to_numpy(dtype="float64", copy=False)
            spans.append(float(values.max() - values.min()) if values.size else 0.0)
        return max(spans)

    x_span = span("x", "px")
    y_span = span("y", "py")

    minimum_span = 1e-12
    if x_span <= minimum_span and y_span <= minimum_span:
        return "xy"
    if x_span <= minimum_span:
        return "y"
    if y_span <= minimum_span:
        return "x"
    if max(x_span, y_span) / min(x_span, y_span) >= 10.0:
        return "x" if x_span > y_span else "y"
    return "xy"


def _read_parquet(source: Path | str, markers: list[str]) -> pd.DataFrame:
    """Read a measurement parquet's in-range marker rows, retrying transient I/O."""
    for attempt in range(_READ_ATTEMPTS):
        try:
            df = pd.read_parquet(
                source, columns=FILE_COLUMNS, filters=[("name", "in", markers)]
            )
            break
        except OSError as error:
            if attempt == _READ_ATTEMPTS - 1:
                raise
            backoff = _READ_BACKOFF_SECONDS * 2**attempt
            LOGGER.warning(
                "Transient I/O error reading %s (attempt %d/%d): %s. Retrying in %.1fs.",
                source,
                attempt + 1,
                _READ_ATTEMPTS,
                error,
                backoff,
            )
            time.sleep(backoff)

    missing = [c for c in FILE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in track data {source}: {missing}. "
            "Regenerate the measurement parquet."
        )
    return df


def _marker_order(
    frame: pd.DataFrame, ring_bpms: list[str], first_bpm: str | None
) -> list[str]:
    """Return the file's markers in ring order, cycled to where its turns start.

    The ring boundary that matters is the one the *measurement* was generated
    from: the BPM each recorded turn begins at. A tracking arc straddling that
    boundary has its early BPMs at the end of one turn and its later BPMs at the
    start of the next, and only a wrap placed at the generation boundary reads
    each side from the right turn.

    ``first_bpm`` names that BPM when the file's own row order is unreliable (ACD
    marker rows are written after all the BPMs); it is projected forward to the
    next ring BPM this file actually holds. Otherwise the file's first recorded
    ring BPM is used. Markers outside the ring order (a kicker's initial-condition
    marker) are placed immediately before the first ring BPM -- their real
    physical position, since such a marker is recorded right before tracking
    starts.
    """
    appearance = list(dict.fromkeys(frame["name"].astype(str)))
    present = set(appearance)
    ring = [bpm for bpm in ring_bpms if bpm in present]
    if not ring:
        return appearance

    natural = next(b for b in appearance if b in set(ring))
    start = natural
    if first_bpm in ring_bpms:
        n = len(ring_bpms)
        begin = ring_bpms.index(first_bpm)
        start = next(
            (
                ring_bpms[(begin + offset) % n]
                for offset in range(n)
                if ring_bpms[(begin + offset) % n] in set(ring)
            ),
            natural,
        )

    pivot = ring.index(start)
    cycled = ring[pivot:] + ring[:pivot]
    extras = [name for name in appearance if name not in set(ring)]
    return extras + cycled


def _build_grid(frame: pd.DataFrame, markers: list[str]) -> dict[str, np.ndarray]:
    """Reshape a long-form frame into dense ``(turn, marker)`` arrays.

    Rows absent from the measurement become zero-weight: the coordinate is zeroed
    and its variance set to infinity, so a missing plane at a single-plane BPM
    drops out of the loss without dropping the whole row.
    """
    turns = np.sort(frame["turn"].unique())
    grid = frame.set_index(["turn", "name"]).reindex(
        pd.MultiIndex.from_product([turns, markers], names=["turn", "name"])
    )
    shape = (len(turns), len(markers))
    # Copy to prevent a view being returned.
    values = {
        column: grid[column].to_numpy(dtype="float64", copy=True).reshape(shape)
        for column in GRID_COLUMNS
    }
    for coordinate, variance in zip(COORDINATES, VARIANCES, strict=True):
        missing = np.isnan(values[coordinate])
        values[variance][missing] = np.inf
        values[coordinate][missing] = 0.0
    return values


class DataManager:
    """Loads measurement files and plans the turn batches workers receive."""

    def __init__(
        self,
        bpms_in_range: list[str],
        all_bpms: list[str],
        simulation_config: SimulationConfig,
        measurement_files: list[Path],
        tracking_plan: TrackingPlan,
        first_bpms: list[str | None] | None = None,
        extra_markers: list[str] | None = None,
        shuffle_turns: ShuffleTurns | None = None,
    ):
        """Create a data manager for one optimisation run.

        Args:
            shuffle_turns: In-place turn ordering, also used by ``WorkerTurnPlanner``.
                Defaults to a shuffle seeded with :data:`TURN_SHUFFLE_SEED`, so
                every run with the same data uses the same turns.
        """
        self.all_bpms = all_bpms
        self.simulation_config = simulation_config
        self.measurement_files = measurement_files
        self.tracking_plan = tracking_plan
        self.first_bpms = first_bpms or [None] * len(measurement_files)
        self.observed_markers = list(
            dict.fromkeys(bpms_in_range + (extra_markers or []))
        )
        self.shuffle_turns = shuffle_turns or random.Random(TURN_SHUFFLE_SEED).shuffle

        self.tracks: dict[int, FileTracks]
        self.available_turns: list[int]
        self.turn_batches: list[list[int]]
        # Held-out validation turns grouped into per-file batches. These turns are
        # removed from ``turn_batches`` entirely so validation loss is genuinely
        # out-of-sample. Empty when validation is disabled or there is too little data.
        self.validation_turn_batches: list[list[int]] = []
        # {file_index -> {bunch_number -> sorted global turns}}
        self.bunch_turns_by_file: dict[int, dict[int, list[int]]]
        self.boundary_turns_by_file: dict[int, set[int]]
        self.file_map: dict[int, int]  # {turn -> file_index}

    # ---------- Loading ----------

    def load_track_data(self) -> None:
        """Read every measurement file into a :class:`FileTracks` grid.

        Each file keeps its own turn numbering on disk; here it is shifted into a
        disjoint global block so a turn number identifies a file as well as a turn.
        Bunch structure comes from the ``bunch_number`` column.
        """
        sources = list(self.measurement_files)
        LOGGER.info("Loading %d measurement file(s)...", len(sources))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
            frames = list(
                pool.map(lambda s: _read_parquet(s, self.observed_markers), sources)
            )

        self.tracks = {}
        self.bunch_turns_by_file = {}
        self.file_map = {}
        offset = 0
        for file_idx, frame in enumerate(frames):
            frame = frame.copy()
            frame["turn"] = (frame["turn"] - int(frame["turn"].min()) + offset).astype(
                "int32"
            )
            offset = int(frame["turn"].max()) + 1

            per_turn = frame[["turn", "bunch_number"]].drop_duplicates("turn")
            bunches: dict[int, list[int]] = {}
            for turn, bunch in zip(per_turn["turn"], per_turn["bunch_number"]):
                bunches.setdefault(int(bunch), []).append(int(turn))
            self.bunch_turns_by_file[file_idx] = {
                b: sorted(t) for b, t in bunches.items()
            }

            frame = frame.drop(columns=["bunch_number"])
            markers = _marker_order(frame, self.all_bpms, self.first_bpms[file_idx])
            turns = np.sort(frame["turn"].unique())
            values = _build_grid(frame, markers)
            self.tracks[file_idx] = FileTracks(
                turns=turns,
                markers=markers,
                values=values,
                kick_plane=infer_kick_plane(frame),
            )
            self.file_map.update(dict.fromkeys((int(t) for t in turns), file_idx))

        self.available_turns = sorted(self.file_map)
        LOGGER.info(
            "Loaded track data: %s",
            ", ".join(
                f"file_{idx}={len(t.turns)} turns, {len(t.markers)} markers ({t.kick_plane})"
                for idx, t in sorted(self.tracks.items())
            ),
        )

    @property
    def file_kick_planes(self) -> dict[int, str]:
        """Return the inferred excitation plane of each measurement file."""
        return {idx: tracks.kick_plane for idx, tracks in self.tracks.items()}

    def with_updated_momenta(
        self, reconstructions: dict[int, pd.DataFrame]
    ) -> dict[int, FileTracks]:
        """Return the track grids with reconstructed momenta patched in.

        Files absent from ``reconstructions`` are passed through unchanged, so a
        reconstruction that failed for one file this epoch keeps its last momenta.
        """
        return {
            idx: tracks.with_updated_momenta(reconstructions[idx])
            if idx in reconstructions
            else tracks
            for idx, tracks in self.tracks.items()
        }

    # ---------- Turn batching ----------

    def prepare_turn_batches(self, num_starts: int, num_ends: int) -> None:
        """Split the turns into training and validation batches for ``num_starts`` x ``num_ends`` ranges."""
        LOGGER.info("Preparing turn batches for worker distribution")

        self.boundary_turns_by_file, self.available_turns = (
            self.tracking_plan.select_available_turns(
                bunch_turns_by_file=self.bunch_turns_by_file,
                simulation_config=self.simulation_config,
                available_turns=self.available_turns,
            )
        )
        if not self.available_turns:
            raise ValueError(
                "No turns available after removing boundary turns. Check that each "
                "bunch in the measurement data has more than one turn."
            )
        LOGGER.info(
            "Removed %d boundary turns (n_run_turns=%d), %d available",
            sum(len(turns) for turns in self.boundary_turns_by_file.values()),
            self.simulation_config.n_run_turns,
            len(self.available_turns),
        )

        train_turns, validation_turns = self._split_validation_turns()
        train_turns = self._sample_training_turns(train_turns)
        self.validation_turn_batches = self._batches_per_file(validation_turns)

        batch_plan = WorkerTurnPlanner(
            self.tracking_plan,
            self.simulation_config,
            shuffle_turns=self.shuffle_turns,
        ).build_turn_batches(
            available_turns=train_turns,
            file_map=self.file_map,
            num_files=len(self.tracks),
            num_starts=num_starts,
            num_ends=num_ends,
        )
        self.turn_batches = batch_plan.turn_batches

        if not self.turn_batches:
            raise ValueError(
                f"Failed to create any training batches. Available turns: "
                f"{len(self.available_turns)}, training turns after validation split + "
                f"data_fraction={self.simulation_config.data_fraction}: {len(train_turns)}. "
                "Consider raising data_fraction, lowering validation_fraction, or using "
                "longer bunches."
            )

        self.num_workers = len(self.turn_batches)
        training_turns = sum(len(batch) for batch in self.turn_batches)
        validation_used = sum(len(batch) for batch in self.validation_turn_batches)
        LOGGER.info(
            "Created %d batches from %d files; %d range specs each = %d workers",
            self.num_workers,
            len(self.tracks),
            batch_plan.range_specs_per_batch,
            self.num_workers * batch_plan.range_specs_per_batch,
        )
        LOGGER.info(
            "Turn usage: %d training, %d held-out validation, %d unused of %d available",
            training_turns,
            validation_used,
            len(self.available_turns) - training_turns - validation_used,
            len(self.available_turns),
        )
        if validation_used == 0 and self.tracking_plan.enable_validation:
            LOGGER.warning(
                "No held-out validation turns were reserved (validation_fraction=%.3f, too "
                "little data). Validation loss falls back to training loss and CANNOT "
                "detect overfitting.",
                self.simulation_config.validation_fraction,
            )

    def _split_validation_turns(self) -> tuple[list[int], list[int]]:
        """Partition available turns into disjoint (training, validation) sets.

        Validation turns are reserved per file so every file keeps its share of
        held-out data, and at least one training turn is always retained per file.
        """
        if (
            not self.tracking_plan.enable_validation
            or self.simulation_config.validation_fraction <= 0.0
        ):
            return sorted(self.available_turns), []

        val_frac = self.simulation_config.validation_fraction
        train_turns: list[int] = []
        validation_turns: list[int] = []
        for turns in self._shuffled_turns_by_file(self.available_turns):
            n_val = max(0, min(round(val_frac * len(turns)), len(turns) - 1))
            validation_turns.extend(turns[:n_val])
            train_turns.extend(turns[n_val:])
        return sorted(train_turns), sorted(validation_turns)

    def _sample_training_turns(self, train_turns: list[int]) -> list[int]:
        """Keep ``data_fraction`` of the training turns, stratified per file."""
        frac = self.simulation_config.data_fraction
        if frac >= 1.0:
            return sorted(train_turns)

        kept: list[int] = []
        for turns in self._shuffled_turns_by_file(train_turns):
            kept.extend(turns[: max(1, round(frac * len(turns)))])
        return sorted(kept)

    def _shuffled_turns_by_file(self, turns: list[int]) -> list[list[int]]:
        """Group turns by file and shuffle each group in place."""
        by_file = group_turns_by_file(turns, self.file_map)
        groups = [list(by_file[idx]) for idx in sorted(by_file)]
        for group in groups:
            self.shuffle_turns(group)
        return groups

    def _batches_per_file(self, turns: list[int]) -> list[list[int]]:
        """Group turns into one batch per measurement file."""
        if not turns:
            return []
        by_file = group_turns_by_file(turns, self.file_map)
        return [sorted(by_file[idx]) for idx in sorted(by_file) if by_file[idx]]

    def get_total_turns(self) -> int:
        """Calculate the number of tracked turns that will actually be processed."""
        start_turns = sum(len(batch) for batch in self.turn_batches)
        return start_turns * self.simulation_config.n_run_turns
