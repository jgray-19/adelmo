"""Fast sanity checks for data/worker utility logic."""

from __future__ import annotations

import numpy as np

from aba_optimiser.training.tracking.workers.payloads import WorkerPayloadBuilder
from aba_optimiser.training.tracking.workers.turn_planner import allocate_batches_per_file


def test_allocate_batches_spreads_across_files() -> None:
    """Batch allocator should not starve files when the budget allows both."""
    turns_by_file = {
        0: list(range(20)),
        1: list(range(20, 40)),
    }

    batches_per_file = allocate_batches_per_file(turns_by_file, num_turn_batches=4)

    assert sum(batches_per_file.values()) == 4
    # Equal-sized files should get an equal share.
    assert batches_per_file[0] == 2
    assert batches_per_file[1] == 2


def test_allocate_batches_never_exceeds_one_turn_per_batch() -> None:
    """A file cannot be split into more batches than it has turns."""
    turns_by_file = {0: [1, 2], 1: list(range(20, 30))}

    # Ask for more batches than file 0 can supply; the surplus must land on file 1.
    batches_per_file = allocate_batches_per_file(turns_by_file, num_turn_batches=8)

    assert batches_per_file[0] <= 2
    assert sum(batches_per_file.values()) == 8


def test_diagnostic_loss_per_bpm_reduces_multiturn() -> None:
    """Per-point losses should collapse to per-BPM sums over turns."""
    bpm_names = ["bpm1", "bpm2", "bpm3"]
    n_run_turns = 2
    # turn 0: [1,2,3], turn 1: [10,20,30]
    loss_per_point = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=np.float64)

    reduced = WorkerPayloadBuilder.diagnostic_loss_per_bpm(
        loss_per_point=loss_per_point,
        bpm_names=bpm_names,
        n_run_turns=n_run_turns,
        worker_id=0,
    )

    assert np.allclose(reduced, np.array([11.0, 22.0, 33.0], dtype=np.float64))
