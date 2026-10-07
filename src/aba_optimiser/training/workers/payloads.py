"""Worker payload construction.

Turns :class:`~aba_optimiser.training.data_manager.FileTracks` grids and
observation plans into the immutable arrays a tracking worker receives.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from aba_optimiser.workers import (
    PrecomputedTrackingWeights,
    TrackingData,
    WeightProcessor,
)

if TYPE_CHECKING:
    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.training.data_manager import FileTracks
    from aba_optimiser.training.workers.setup import WorkerObservationPlan
    from aba_optimiser.workers import WorkerConfig

LOGGER = logging.getLogger(__name__)

WorkerPayload: TypeAlias = tuple["TrackingData", "WorkerConfig", int]


def observation_turn_offsets(cols: np.ndarray, sdir: int) -> np.ndarray:
    """Return the per-point turn offset implied by a range's column order.

    Markers are stored in ring order cycled to the file's turn boundary, so a
    contiguous tracking range is a monotonic column sequence. Each place the
    sequence jumps backwards (forwards, when tracking in reverse) is the ring
    wrap, i.e. the point where the track moves into the next (previous) turn.
    """
    steps = np.diff(cols)
    wrapped = steps < 0 if sdir == 1 else steps > 0
    return np.cumsum(np.concatenate([[0], wrapped.astype(np.int64)])) * sdir


class WorkerPayloadBuilder:
    """Build tracking payload arrays and shared weights for workers."""

    def __init__(self, accelerator: Accelerator) -> None:
        self.accelerator = accelerator

    def compute_pt(self, file_idx: int, machine_deltaps: list[float]) -> float:
        """Return the reference ``pt`` for a measurement file's energy offset."""
        return self.accelerator.dp2pt(machine_deltaps[file_idx])

    def make_tracking_data(
        self,
        turn_batch: list[int],
        file_turn_map: dict[int, int],
        plan: WorkerObservationPlan,
        machine_deltaps: list[float],
        tracks: dict[int, FileTracks],
        n_run_turns: int,
    ) -> TrackingData:
        """Build the serialisable tracking payload for one worker plan.

        Both transverse planes are always allocated; the ones the file does not
        excite keep their infinite variance and so contribute nothing to the loss.
        """
        bpm_names = plan.bpm_names
        if not bpm_names:
            raise ValueError(
                f"No observed BPMs for worker range "
                f"{plan.range_spec.start_bpm}/{plan.range_spec.end_bpm}"
            )

        sdir = plan.range_spec.sdir
        init_marker = plan.init_bpm
        n_turns = len(turn_batch)
        n_points = len(bpm_names) * n_run_turns

        pos = np.zeros((n_turns, n_points, 2))
        mom = np.zeros((n_turns, n_points, 2))
        pos_var = np.full((n_turns, n_points, 2), np.inf)
        mom_var = np.full((n_turns, n_points, 2), np.inf)
        init_coords = np.zeros((n_turns, 6))
        pts = np.empty(n_turns)
        reading_ids = np.zeros((n_turns, n_points), dtype=np.int64)
        init_reading_ids = np.zeros(n_turns, dtype=np.int64)
        init_variances = np.full((n_turns, 2), np.inf)

        has_x = plan.kick_plane in ("x", "xy")
        has_y = plan.kick_plane in ("y", "xy")

        batch_rows_by_file: dict[int, list[int]] = defaultdict(list)
        turns_by_file: dict[int, list[int]] = defaultdict(list)
        for i, turn in enumerate(turn_batch):
            file_idx = file_turn_map[turn]
            batch_rows_by_file[file_idx].append(i)
            turns_by_file[file_idx].append(turn)
            pts[i] = self.compute_pt(file_idx, machine_deltaps)

        for file_idx, batch_rows in batch_rows_by_file.items():
            file_tracks = tracks[file_idx]
            if init_marker not in file_tracks.marker_col:
                raise ValueError(
                    f"Init marker '{init_marker}' not found in tracking data for file {file_idx}"
                )
            values = file_tracks.values

            cols = file_tracks.cols(bpm_names * n_run_turns)
            start_rows = file_tracks.rows(turns_by_file[file_idx])
            rows = start_rows[:, None] + observation_turn_offsets(cols, sdir)[None, :]

            init_rows = start_rows
            init_col = file_tracks.marker_col[init_marker]
            # One id per (file, turn, marker) grid cell, shared by every worker.
            file_base = np.int64(file_idx) << 32
            n_markers = len(file_tracks.markers)
            reading_ids[batch_rows] = file_base + rows * n_markers + cols
            init_reading_ids[batch_rows] = file_base + init_rows * n_markers + init_col
            if has_x:
                init_coords[batch_rows, 0] = values["x"][init_rows, init_col]
                init_coords[batch_rows, 1] = values["px"][init_rows, init_col]
                init_variances[batch_rows, 0] = values["var_x"][init_rows, init_col]
            if has_y:
                init_coords[batch_rows, 2] = values["y"][init_rows, init_col]
                init_coords[batch_rows, 3] = values["py"][init_rows, init_col]
                init_variances[batch_rows, 1] = values["var_y"][init_rows, init_col]
            init_coords[batch_rows, 5] = pts[batch_rows]

            # An all-zero initial state means the marker row was missing from the
            # measurement: tracking from it would silently fit noise.
            dead = np.all(init_coords[batch_rows, :4] == 0.0, axis=1)
            if dead.any():
                turn = turns_by_file[file_idx][int(np.argmax(dead))]
                raise ValueError(
                    f"Initial coordinates for turn {turn} at marker {init_marker} are all zero"
                )

            if has_x:
                pos[batch_rows, :, 0] = values["x"][rows, cols]
                mom[batch_rows, :, 0] = values["px"][rows, cols]
                pos_var[batch_rows, :, 0] = values["var_x"][rows, cols]
                mom_var[batch_rows, :, 0] = values["var_px"][rows, cols]
            if has_y:
                pos[batch_rows, :, 1] = values["y"][rows, cols]
                mom[batch_rows, :, 1] = values["py"][rows, cols]
                pos_var[batch_rows, :, 1] = values["var_y"][rows, cols]
                mom_var[batch_rows, :, 1] = values["var_py"][rows, cols]

        for array in (pos, mom, pos_var, mom_var):
            array.setflags(write=False)

        return TrackingData(
            position_comparisons=pos,
            momentum_comparisons=mom,
            position_variances=pos_var,
            momentum_variances=mom_var,
            init_coords=init_coords,
            init_pts=pts,
            reading_ids=reading_ids,
            init_reading_ids=init_reading_ids,
            init_variances=init_variances,
            precomputed_weights=None,
        )

    @staticmethod
    def attach_global_weights(
        payloads: list[WorkerPayload],
        *,
        optimise_momenta: bool = True,
    ) -> list[WorkerPayload]:
        """Precompute globally normalised weights for all tracking workers.

        Weights are inverse variances divided by the single largest weight, so all
        workers -- training and validation alike -- report losses on one comparable
        scale.
        """
        if not payloads:
            return payloads

        def active_observables(config: WorkerConfig) -> tuple[str, ...]:
            kick_plane = config.kick_plane
            if kick_plane == "x":
                return ("x", "px") if optimise_momenta else ("x",)
            if kick_plane == "y":
                return ("y", "py") if optimise_momenta else ("y",)
            return ("x", "y", "px", "py") if optimise_momenta else ("x", "y")

        observables = ("x", "y", "px", "py")
        raw_by_payload = []
        global_max = 0.0
        for data, config, _file_idx in payloads:
            active = active_observables(config)
            raw = [
                WeightProcessor.variance_to_weight(variance)
                for variance in (
                    data.position_variances[:, :, 0],
                    data.position_variances[:, :, 1],
                    data.momentum_variances[:, :, 0],
                    data.momentum_variances[:, :, 1],
                )
            ]
            global_max = max(
                global_max,
                max(
                    (np.max(raw[i]) for i, name in enumerate(observables) if name in active),
                    default=0.0,
                ),
            )
            raw_by_payload.append((data, raw))

        if global_max == 0.0:
            LOGGER.warning("All computed weights are zero; skipping global normalisation")
            global_max = 1.0

        for data, raw in raw_by_payload:
            data.precomputed_weights = PrecomputedTrackingWeights(
                x=raw[0] / global_max,
                y=raw[1] / global_max,
                px=raw[2] / global_max,
                py=raw[3] / global_max,
                scale=global_max,
            )

        LOGGER.info(
            "Global weight normalisation complete: max weight=%.3e across %d payloads",
            global_max,
            len(payloads),
        )
        return payloads

    @staticmethod
    def expand_bpm_mask(mask: np.ndarray, n_run_turns: int) -> np.ndarray:
        """Expand a per-BPM mask across repeated turns."""
        if n_run_turns <= 1:
            return mask
        return np.tile(mask, n_run_turns)

    @staticmethod
    def diagnostic_loss_per_bpm(
        loss_per_point: np.ndarray,
        bpm_names: list[str],
        n_run_turns: int,
        worker_id: int,
    ) -> np.ndarray:
        """Reduce point-wise diagnostic losses to one value per BPM."""
        expected_points = len(bpm_names) * n_run_turns
        if loss_per_point.size != expected_points:
            raise RuntimeError(
                f"Worker {worker_id}: diagnostics size mismatch "
                f"(got {loss_per_point.size}, expected {expected_points} = "
                f"{len(bpm_names)} BPMs x {n_run_turns} turns)"
            )
        if n_run_turns == 1:
            return loss_per_point
        return loss_per_point.reshape(n_run_turns, len(bpm_names)).sum(axis=0)
