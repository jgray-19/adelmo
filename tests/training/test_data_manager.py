from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.config.tracking import (
    TrackingPlan,
    _boundary_turns_for_track,
)
from aba_optimiser.training.data_manager import DataManager, _marker_order
from aba_optimiser.training.workers.turn_planner import (
    _allocate_batches_per_file,
)
from aba_optimiser.training.workers.turn_planner import (
    group_turns_by_file as _group_turns_by_file,
)

_DEFAULT_TRACKING_PLAN = TrackingPlan()


def _make_track_df(turns: list[int], bpm_name: str = "BPM.1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn": turns,
            "name": [bpm_name] * len(turns),
            "x": [0.0] * len(turns),
            "px": [0.0] * len(turns),
            "y": [0.0] * len(turns),
            "py": [0.0] * len(turns),
            "var_x": [1.0] * len(turns),
            "var_y": [1.0] * len(turns),
            "var_px": [1.0] * len(turns),
            "var_py": [1.0] * len(turns),
        }
    ).set_index(["turn", "name"])


def _range_spec_plan(*, use_fixed_bpm: bool, num_starts: int, num_ends: int):
    return _DEFAULT_TRACKING_PLAN.range_specs_per_batch(
        use_fixed_bpm=use_fixed_bpm, num_starts=num_starts, num_ends=num_ends
    )


def _single_bunch_by_file(*turn_lists: list[int]) -> dict[int, dict[int, list[int]]]:
    return {file_idx: {0: turns} for file_idx, turns in enumerate(turn_lists)}


def _config_manager(start_bpms: list[str], end_bpms: list[str]) -> SimpleNamespace:
    return SimpleNamespace(start_bpms=start_bpms, end_bpms=end_bpms)


def _batch_turn_set(batches: list[list[int]]) -> set[int]:
    return {turn for batch in batches for turn in batch}


def test_prepare_turn_batches_distributes_all_training_turns_across_workers() -> None:
    """With data_fraction=1 and no validation holdout, every training turn is used.

    Arc-by-arc has range_specs_per_batch=1, so num_workers=6 asks for 6 turn
    batches; num_batches=1 means no even-trim, so all available turns are used
    with none duplicated across batches.
    """
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=6,
            num_batches=1,
            n_run_turns=1,
            validation_fraction=0.0,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
        shuffle_turns=lambda turns: None,
    )
    data_manager.tracks = {0: None}
    data_manager.available_turns = list(range(12))
    data_manager.bunch_turns_by_file = _single_bunch_by_file(list(range(12)))
    data_manager.file_map = dict.fromkeys(range(12), 0)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1"], []))

    assert len(data_manager.turn_batches) == 6
    assert data_manager.validation_turn_batches == []
    # Every available (post-boundary) turn is used exactly once across the batches.
    assert _batch_turn_set(data_manager.turn_batches) == set(data_manager.available_turns)


def test_prepare_turn_batches_keeps_batches_within_their_file() -> None:
    """Each batch draws turns from a single file, spread across both files."""
    file0_turns = list(range(8))
    file1_turns = list(range(100, 108))
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=8,
            num_batches=1,
            n_run_turns=1,
            validation_fraction=0.0,
        ),
        measurement_files=["file0.parquet", "file1.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
        shuffle_turns=lambda turns: None,
    )
    data_manager.tracks = dict.fromkeys(range(2))
    data_manager.available_turns = file0_turns + file1_turns
    data_manager.bunch_turns_by_file = _single_bunch_by_file(file0_turns, file1_turns)
    data_manager.file_map = dict.fromkeys(file0_turns, 0) | dict.fromkeys(file1_turns, 1)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1"], []))

    # num_workers=8 / rspb=1 -> 8 turn batches split evenly across the two files.
    assert len(data_manager.turn_batches) == 8
    # Every batch draws from exactly one file.
    assert all(
        len({data_manager.file_map[turn] for turn in batch}) == 1
        for batch in data_manager.turn_batches
    )
    assert _batch_turn_set(data_manager.turn_batches) == set(data_manager.available_turns)


def test_prepare_turn_batches_caps_batches_at_num_workers() -> None:
    """When turns far exceed num_workers, exactly num_workers // rspb batches form."""
    total_turns = 302
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=60,
            num_batches=2,
            n_run_turns=1,
            validation_fraction=0.0,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
    )
    data_manager.tracks = {0: None}
    data_manager.available_turns = list(range(total_turns))
    data_manager.bunch_turns_by_file = _single_bunch_by_file(list(range(total_turns)))
    data_manager.file_map = dict.fromkeys(range(total_turns), 0)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1"], []))

    range_specs_per_batch, _ = _range_spec_plan(
        use_fixed_bpm=True, num_starts=1, num_ends=0
    )
    assert len(data_manager.turn_batches) == 60 // range_specs_per_batch
    # Batch sizes are trimmed to an even multiple of num_batches=2.
    assert all(len(batch) % 2 == 0 for batch in data_manager.turn_batches)


def test_prepare_turn_batches_num_batches_does_not_inflate_worker_groups() -> None:
    """num_batches (MAD sub-batching) must not change the number of turn batches."""
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=60,
            num_batches=40,
            use_fixed_bpm=True,
            n_run_turns=1,
            validation_fraction=0.0,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
    )
    data_manager.tracks = {0: None}
    data_manager.available_turns = list(range(400))
    data_manager.bunch_turns_by_file = _single_bunch_by_file(list(range(400)))
    data_manager.file_map = dict.fromkeys(range(400), 0)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1", "BPM.2"], []))

    range_specs_per_batch, _ = _range_spec_plan(
        use_fixed_bpm=True, num_starts=2, num_ends=0
    )
    assert len(data_manager.turn_batches) == 60 // range_specs_per_batch
    assert len(data_manager.turn_batches) * range_specs_per_batch == 60


def test_prepare_turn_batches_holds_out_disjoint_validation_turns() -> None:
    """Validation turns must be reserved from training, never overlapping it."""
    # 42 input turns -> 40 available after boundary removal; 25% held out = 10.
    turns = list(range(42))
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=4,
            num_batches=1,
            n_run_turns=1,
            validation_fraction=0.25,
            data_fraction=1.0,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
        shuffle_turns=lambda turns: None,
    )
    data_manager.tracks = {0: None}
    data_manager.available_turns = list(turns)
    data_manager.bunch_turns_by_file = _single_bunch_by_file(turns)
    data_manager.file_map = dict.fromkeys(turns, 0)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1"], []))

    training_turns = _batch_turn_set(data_manager.turn_batches)
    validation_turns = _batch_turn_set(data_manager.validation_turn_batches)
    available = set(data_manager.available_turns)

    assert data_manager.validation_turn_batches  # some data was held out
    assert len(validation_turns) == 10  # 25% of the 40 available turns
    assert len(training_turns) == 30
    # The whole point: training and validation never share a turn.
    assert training_turns.isdisjoint(validation_turns)
    assert training_turns | validation_turns == available


def test_prepare_turn_batches_data_fraction_samples_training_turns() -> None:
    """data_fraction keeps only that fraction of the (post-holdout) training turns."""
    # 22 input turns -> 20 available after boundary removal; data_fraction 0.5 -> 10.
    turns = list(range(22))
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=1,
            num_batches=1,
            n_run_turns=1,
            validation_fraction=0.0,
            data_fraction=0.5,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
        shuffle_turns=lambda turns: None,
    )
    data_manager.tracks = {0: None}
    data_manager.available_turns = list(turns)
    data_manager.bunch_turns_by_file = _single_bunch_by_file(turns)
    data_manager.file_map = dict.fromkeys(turns, 0)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1"], []))

    # Half of the 20 available turns are kept; num_batches=1 means no trimming.
    assert data_manager.get_total_turns() == 10
    assert data_manager.validation_turn_batches == []


def test_prepare_turn_batches_holds_out_disjoint_validation_across_files() -> None:
    """The disjoint train/validation invariant holds across multiple files.

    Each file is split independently (stratified), so every file contributes to
    both sets, the two sets never share a turn, and together they reconstruct the
    full set of available turns when data_fraction=1.0.
    """
    file0_turns = list(range(42))
    file1_turns = list(range(100, 142))
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=8,
            num_batches=1,
            n_run_turns=1,
            validation_fraction=0.25,
            data_fraction=1.0,
        ),
        measurement_files=["file0.parquet", "file1.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
        shuffle_turns=lambda turns: None,
    )
    data_manager.tracks = dict.fromkeys(range(2))
    data_manager.available_turns = file0_turns + file1_turns
    data_manager.bunch_turns_by_file = _single_bunch_by_file(file0_turns, file1_turns)
    data_manager.file_map = dict.fromkeys(file0_turns, 0) | dict.fromkeys(file1_turns, 1)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1"], []))

    training_turns = _batch_turn_set(data_manager.turn_batches)
    validation_turns = _batch_turn_set(data_manager.validation_turn_batches)
    available = set(data_manager.available_turns)

    # 40 available turns per file after boundary removal; 25% held out each.
    assert len(validation_turns) == 20
    assert len(training_turns) == 60
    assert training_turns.isdisjoint(validation_turns)
    assert training_turns | validation_turns == available
    # Both files contribute to validation (stratified holdout, one batch per file).
    assert len(data_manager.validation_turn_batches) == 2
    assert {data_manager.file_map[turn] for turn in validation_turns} == {0, 1}


def test_prepare_turn_batches_keeps_all_turns_when_too_little_to_hold_out() -> None:
    """A file with too few post-boundary turns holds out none and keeps them all.

    A 3-turn bunch loses its first and last turn to boundary filtering, leaving a
    single available turn. ``round(0.25 * 1) == 0`` clamped to keep >=1 training
    turn means nothing is held out: validation is empty, prepare_turn_batches does
    not crash, and the lone training turn survives.
    """
    turns = [0, 1, 2]
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=1,
            num_batches=1,
            n_run_turns=1,
            validation_fraction=0.25,
            data_fraction=1.0,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
        shuffle_turns=lambda turns: None,
    )
    data_manager.tracks = {0: None}
    data_manager.available_turns = list(turns)
    data_manager.bunch_turns_by_file = _single_bunch_by_file(turns)
    data_manager.file_map = dict.fromkeys(turns, 0)

    data_manager.prepare_turn_batches(_config_manager(["BPM.1"], []))

    assert data_manager.available_turns == [1]  # only the interior turn survives
    assert data_manager.validation_turn_batches == []
    assert _batch_turn_set(data_manager.turn_batches) == {1}


@pytest.mark.parametrize("bad_data_fraction", [0.0, -0.1, 1.5])
def test_simulation_config_rejects_invalid_data_fraction(bad_data_fraction: float) -> None:
    with pytest.raises(ValueError, match="data_fraction must be in"):
        SimulationConfig(num_workers=1, num_batches=1, data_fraction=bad_data_fraction)


@pytest.mark.parametrize("bad_validation_fraction", [-0.1, 1.0, 1.5])
def test_simulation_config_rejects_invalid_validation_fraction(
    bad_validation_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="validation_fraction must be in"):
        SimulationConfig(
            num_workers=1, num_batches=1, validation_fraction=bad_validation_fraction
        )


def test_get_total_turns_uses_real_batch_sizes() -> None:
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=2,
            num_batches=2,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
    )
    data_manager.turn_batches = [[1, 2, 3, 4, 5], [6, 7]]

    assert data_manager.get_total_turns() == 7


def test_get_total_turns_counts_multi_turn_tracking_samples() -> None:
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=1,
            num_batches=1,
            n_run_turns=256,
        ),
        measurement_files=["file0.parquet"],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
    )
    data_manager.turn_batches = [[0]]

    assert data_manager.get_total_turns() == 256


def test_marker_order_projects_a_missing_first_bpm_onto_the_next_recorded_one() -> None:
    """The requested boundary BPM is absent from this file, so its successor is used."""
    frame = pd.DataFrame({"name": ["BPM.D", "BPM.A", "BPM.C"]})

    order = _marker_order(frame, ["BPM.A", "BPM.BAD", "BPM.C", "BPM.D"], "BPM.BAD")

    assert order == ["BPM.C", "BPM.D", "BPM.A"]


def test_marker_order_places_a_non_ring_marker_before_the_first_bpm() -> None:
    """A kicker marker is recorded just before tracking starts, so it sorts first."""
    frame = pd.DataFrame({"name": ["KICKER", "BPM.A", "BPM.B", "BPM.C"]})

    order = _marker_order(frame, ["BPM.A", "BPM.B", "BPM.C"], "KICKER")

    assert order == ["KICKER", "BPM.A", "BPM.B", "BPM.C"]


def test_range_spec_plan_counts_workers_per_turn_batch() -> None:
    assert _range_spec_plan(use_fixed_bpm=True, num_starts=2, num_ends=4) == (
        6,
        "fixed pairs (2 starts + 4 ends)",
    )
    assert _range_spec_plan(use_fixed_bpm=False, num_starts=2, num_ends=3) == (
        12,
        "2 directions x 2 starts x 3 ends",
    )


def test_group_turns_by_file_partitions_turns() -> None:
    available_turns = [1, 2, 3, 10, 11]
    file_map = {1: 0, 2: 0, 3: 0, 10: 1, 11: 1}

    grouped = _group_turns_by_file(available_turns, file_map)

    assert grouped == {0: [1, 2, 3], 1: [10, 11]}


def test_allocate_batches_per_file_balances_equal_files() -> None:
    turns_by_file = {
        0: list(range(8)),
        1: list(range(100, 108)),
    }

    batches_per_file = _allocate_batches_per_file(turns_by_file, num_turn_batches=6)

    assert sum(batches_per_file.values()) == 6
    assert batches_per_file == {0: 3, 1: 3}


def test_boundary_turns_for_track_short_track() -> None:
    assert _boundary_turns_for_track([1, 2], margin=1) == [1, 2]


def test_boundary_turns_for_track_long_track() -> None:
    assert _boundary_turns_for_track([1, 2, 3, 4, 5], margin=1) == [1, 5]


def test_select_available_turns_removes_boundaries_per_bunch() -> None:
    available_turns = list(range(12))

    boundary_turns, selected = _DEFAULT_TRACKING_PLAN.select_available_turns(
        bunch_turns_by_file={0: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9, 10, 11]}},
        simulation_config=SimulationConfig(
            num_workers=1,
            num_batches=1,
        ),
        available_turns=available_turns,
    )

    assert boundary_turns == {0: {0, 3, 4, 11}}
    assert selected == [1, 2, 5, 6, 7, 8, 9, 10]


def test_load_track_data_requires_bunch_number_column(tmp_path) -> None:
    source = tmp_path / "track.parquet"
    _make_track_df([0, 1]).reset_index().to_parquet(source, index=False)
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            num_workers=1,
            num_batches=1,
        ),
        measurement_files=[str(source)],
        tracking_plan=_DEFAULT_TRACKING_PLAN,
    )

    with pytest.raises(ValueError, match="bunch_number"):
        data_manager.load_track_data()
