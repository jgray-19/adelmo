"""Worker payload construction from the measurement grid.

These cover the one piece of index arithmetic the tracking payload depends on:
turning a BPM range into ``(turn, marker)`` grid coordinates, including the ring
wrap that carries a range into the next (or previous) recorded turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.training.tracking.data_manager import FileTracks
from aba_optimiser.training.tracking.workers.payloads import (
    WorkerPayloadBuilder,
    observation_turn_offsets,
)
from aba_optimiser.training.tracking.workers.setup import WorkerObservationPlan, WorkerRangeSpec
from aba_optimiser.workers.common import KickPlane, TrackingData, WorkerConfig

MARKERS = ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3"]


def _accelerator(tmp_path) -> PSB:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    return PSB(ring=3, sequence_file=seq_file)


def _tracks(n_turns: int = 3, kick_plane: str = "x") -> FileTracks:
    """A grid whose x value encodes its ``(turn, marker)`` position as turn.marker."""
    turns = np.arange(1, n_turns + 1)
    grid = np.array([[t + 0.1 * (m + 1) for m in range(len(MARKERS))] for t in turns])
    values = {
        "x": grid,
        "px": grid * 0.01,
        "y": np.zeros_like(grid),
        "py": np.zeros_like(grid),
        "var_x": np.full_like(grid, 1.0),
        "var_px": np.full_like(grid, 3.0),
        "var_y": np.full_like(grid, np.inf),
        "var_py": np.full_like(grid, np.inf),
    }
    return FileTracks(turns=turns, markers=list(MARKERS), values=values, kick_plane=kick_plane)


def _plan(bpm_names: list[str], sdir: int = 1, kick_plane: KickPlane = KickPlane.X):
    return WorkerObservationPlan(
        range_spec=WorkerRangeSpec(start_bpm=bpm_names[0], end_bpm=bpm_names[-1], sdir=sdir),
        file_idx=0,
        kick_plane=kick_plane,
        bpm_names=bpm_names,
        bad_bpms=None,
    )


def test_observation_turn_offsets_advances_once_per_ring_wrap() -> None:
    # Columns 1,2 then back to 0: the range wraps the ring once and enters turn+1.
    forward = observation_turn_offsets(np.array([1, 2, 0, 1]), sdir=1)
    assert forward.tolist() == [0, 0, 1, 1]

    # Tracking backwards, a forward column jump is the wrap into the previous turn.
    backward = observation_turn_offsets(np.array([1, 0, 2, 1]), sdir=-1)
    assert backward.tolist() == [0, 0, -1, -1]


def test_payload_reads_its_range_from_the_measurement_grid(tmp_path) -> None:
    builder = WorkerPayloadBuilder(_accelerator(tmp_path))

    data = builder.make_tracking_data(
        turn_batch=[1, 2],
        file_turn_map={1: 0, 2: 0},
        plan=_plan(MARKERS),
        machine_deltaps=[0.0],
        tracks={0: _tracks()},
        n_run_turns=1,
    )

    assert np.allclose(data.position_comparisons[0, :, 0], [1.1, 1.2, 1.3])
    assert np.allclose(data.position_comparisons[1, :, 0], [2.1, 2.2, 2.3])
    # Initial conditions come from the range start in the same turn.
    assert np.allclose(data.init_coords[:, 0], [1.1, 2.1])
    assert np.allclose(data.init_coords[:, 1], [0.011, 0.021])
    assert np.allclose(data.position_variances[..., 0], 1.0)
    assert np.allclose(data.momentum_variances[..., 0], 3.0)


def test_payload_leaves_the_unexcited_plane_infinitely_weighted(tmp_path) -> None:
    builder = WorkerPayloadBuilder(_accelerator(tmp_path))

    data = builder.make_tracking_data(
        turn_batch=[1],
        file_turn_map={1: 0},
        plan=_plan(MARKERS),
        machine_deltaps=[0.0],
        tracks={0: _tracks()},
        n_run_turns=1,
    )

    assert np.allclose(data.position_comparisons[..., 1], 0.0)
    assert not np.isfinite(data.position_variances[..., 1]).any()
    assert not np.isfinite(data.momentum_variances[..., 1]).any()
    assert np.allclose(data.init_coords[:, 2:4], 0.0)


def test_multi_turn_payload_continues_into_the_next_recorded_turn(tmp_path) -> None:
    builder = WorkerPayloadBuilder(_accelerator(tmp_path))

    data = builder.make_tracking_data(
        turn_batch=[1],
        file_turn_map={1: 0},
        plan=_plan(MARKERS),
        machine_deltaps=[0.0],
        tracks={0: _tracks()},
        n_run_turns=2,
    )

    assert np.allclose(
        data.position_comparisons[0, :, 0], [1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
    )


def test_backward_range_crossing_the_wrap_reads_the_previous_turn(tmp_path) -> None:
    builder = WorkerPayloadBuilder(_accelerator(tmp_path))
    # Backward from marker 2 to marker 3: the step 1 -> 2 is a forward column jump,
    # so the last point belongs to the previous recorded turn.
    reversed_range = [MARKERS[1], MARKERS[0], MARKERS[2]]

    data = builder.make_tracking_data(
        turn_batch=[2],
        file_turn_map={2: 0},
        plan=_plan(reversed_range, sdir=-1),
        machine_deltaps=[0.0],
        tracks={0: _tracks()},
        n_run_turns=1,
    )

    assert np.allclose(data.position_comparisons[0, :, 0], [2.2, 2.1, 1.3])


def test_payload_rejects_a_marker_with_no_measured_initial_state(tmp_path) -> None:
    builder = WorkerPayloadBuilder(_accelerator(tmp_path))
    tracks = _tracks()
    # A marker row missing from the measurement is zero-filled with infinite variance.
    tracks.values["x"][0, 0] = 0.0
    tracks.values["px"][0, 0] = 0.0

    with pytest.raises(ValueError, match="all zero"):
        builder.make_tracking_data(
            turn_batch=[1],
            file_turn_map={1: 0},
            plan=_plan(MARKERS),
            machine_deltaps=[0.0],
            tracks={0: tracks},
            n_run_turns=1,
        )


def test_payload_arrays_are_frozen_before_they_reach_a_worker(tmp_path) -> None:
    builder = WorkerPayloadBuilder(_accelerator(tmp_path))

    data = builder.make_tracking_data(
        turn_batch=[1],
        file_turn_map={1: 0},
        plan=_plan(MARKERS),
        machine_deltaps=[0.0],
        tracks={0: _tracks()},
        n_run_turns=1,
    )

    assert not data.position_comparisons.flags.writeable
    assert not data.momentum_comparisons.flags.writeable
    assert not data.position_variances.flags.writeable
    assert not data.momentum_variances.flags.writeable


def _weight_payload(accelerator, position_var, momentum_var, kick_plane=KickPlane.XY):
    data = TrackingData(
        position_comparisons=np.zeros((1, 1, 2)),
        momentum_comparisons=np.zeros((1, 1, 2)),
        position_variances=np.array([[position_var]], dtype=np.float64),
        momentum_variances=np.array([[momentum_var]], dtype=np.float64),
        init_coords=np.array([[1.0, 0.1, 0.0, 0.0, 0.0, 0.01]]),
        init_pts=np.array([0.01]),
        reading_ids=np.zeros((1, 1), dtype=np.int64),
        init_reading_ids=np.zeros(1, dtype=np.int64),
        init_variances=np.ones((1, 2)),
        precomputed_weights=None,
    )
    config = WorkerConfig(
        accelerator=accelerator,
        tracking_start_bpm=MARKERS[0],
        tracking_end_bpm=MARKERS[-1],
        magnet_range="$start/$end",
        kick_plane=kick_plane,
    )
    return (data, config, 0)


def test_attach_global_weights_normalises_every_observable_to_the_largest(tmp_path) -> None:
    accelerator = _accelerator(tmp_path)
    payload = _weight_payload(accelerator, [2.0, 3.0], [4.0, 5.0])

    weights = WorkerPayloadBuilder.attach_global_weights([payload])[0][0].precomputed_weights

    assert weights is not None
    assert np.allclose(weights.x, [[1.0]])
    assert np.allclose(weights.y, [[2.0 / 3.0]])
    assert np.allclose(weights.px, [[0.5]])
    assert np.allclose(weights.py, [[0.4]])


def test_position_only_runs_do_not_let_momenta_set_the_normalisation(tmp_path) -> None:
    accelerator = _accelerator(tmp_path)
    payload = _weight_payload(accelerator, [1e-8, 1e-8], [1e-12, 1e-12])

    weights = WorkerPayloadBuilder.attach_global_weights(
        [payload], optimise_momenta=False
    )[0][0].precomputed_weights

    assert weights is not None
    # Positions set the scale; the far larger momentum weights are left unscaled.
    assert np.allclose(weights.x, [[1.0]])
    assert np.allclose(weights.y, [[1.0]])
    assert np.allclose(weights.px, [[1e4]])
    assert np.allclose(weights.py, [[1e4]])


def test_diagnostic_loss_per_bpm_sums_over_tracked_turns() -> None:
    per_bpm = WorkerPayloadBuilder.diagnostic_loss_per_bpm(
        loss_per_point=np.array([1.0, 2.0, 3.0, 4.0]),
        bpm_names=MARKERS[:2],
        n_run_turns=2,
        worker_id=0,
    )

    assert per_bpm.tolist() == [4.0, 6.0]
