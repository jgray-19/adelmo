"""A tracking worker sends exactly one reply per request, including after a failure."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from adelmo.config import SimulationConfig
from adelmo.fitting.worker import WorkerConfig
from adelmo.machine.accelerators import PSB
from adelmo.tracking.worker import PrecomputedTrackingWeights, TrackingData, TrackingWorker


def _worker(tmp_path, conn) -> TrackingWorker:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    shape = (2, 3)
    data = TrackingData(
        position_comparisons=np.zeros((*shape, 2)),
        momentum_comparisons=np.zeros((*shape, 2)),
        position_variances=np.ones((*shape, 2)),
        momentum_variances=np.ones((*shape, 2)),
        init_coords=np.zeros((2, 6)),
        init_pts=np.zeros(2),
        reading_ids=np.zeros(shape, dtype=np.int64),
        init_reading_ids=np.zeros(2, dtype=np.int64),
        init_variances=np.ones((2, 2)),
        precomputed_weights=PrecomputedTrackingWeights(
            x=np.ones(shape), y=np.ones(shape), px=np.ones(shape), py=np.ones(shape), scale=1.0
        ),
    )
    config = WorkerConfig(
        accelerator=PSB(ring=3, sequence_file=seq_file),
        tracking_start_bpm="BR3.BPM1L3",
        tracking_end_bpm="BR3.BPM3L3",
        magnet_range="$start/$end",
    )
    return TrackingWorker(conn, 0, data, config, SimulationConfig(num_workers=1, num_batches=1))


def test_failed_training_worker_sends_no_uncertainty_part(tmp_path) -> None:
    """After its error reply a failed worker must not also send an empty part."""
    parent, child = mp.Pipe()
    worker = _worker(tmp_path, child)
    worker.n_reply_knobs = 2  # set by ``on_start``, which needs MAD-NG
    worker.on_stop(None, failed=True)

    assert not parent.poll(0.1)
