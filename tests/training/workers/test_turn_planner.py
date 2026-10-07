"""Turn planning must train every file, whatever the worker cap."""

from __future__ import annotations

from types import SimpleNamespace

from aba_optimiser.tracking.dispatch.turn_planner import WorkerTurnPlanner


class _TwoSpecPlan:
    """A tracking plan fanning each turn batch over 2 range specs (as ACD markers)."""

    def range_specs_per_batch(self, **_kwargs):
        return 2, "forward + backward"


def _plan(num_workers: int, num_files: int, turns_per_file: int = 50):
    file_map = {}
    for f in range(num_files):
        for t in range(turns_per_file):
            file_map[f * turns_per_file + t] = f
    planner = WorkerTurnPlanner(
        _TwoSpecPlan(),
        SimpleNamespace(num_workers=num_workers, num_batches=1, use_fixed_bpm=True),
        shuffle_turns=lambda _turns: None,
    )
    plan = planner.build_turn_batches(
        available_turns=list(file_map),
        file_map=file_map,
        num_files=num_files,
        num_starts=1,
        num_ends=1,
    )
    trained = {file_map[batch[0]] for batch in plan.turn_batches}
    return plan, trained


def test_worker_cap_below_file_count_still_trains_every_file():
    # psb_md's default: 8 workers, 2 range specs -> 4 batches for 15 files.
    plan, trained = _plan(num_workers=8, num_files=15)
    assert trained == set(range(15))
    assert len(plan.turn_batches) == 15
    assert sum(len(b) for b in plan.turn_batches) == 15 * 50


def test_worker_cap_above_file_count_is_unchanged():
    plan, trained = _plan(num_workers=60, num_files=10)
    assert trained == set(range(10))
    assert len(plan.turn_batches) == 30
