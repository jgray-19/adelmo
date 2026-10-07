"""The loss normalisation must be one constant shared by every closed-twiss worker.

The fitter sums the workers' losses. Anything a worker divides its own loss by is
therefore a re-weighting of that momentum in the joint fit, and if the divisor is
derived from the worker's own data it re-weights by an accident of that data
rather than by any physics.

That is exactly what used to happen: each worker divided its weights by its own
largest weight and its loss by its own point count, so the optimiser minimised
``sum_w (1/(max_w * N_w)) * chi2_w`` while the reported 1-sigma - built from the
un-normalised ``JᵀWJ`` - described the pooled ``sum_w chi2_w``. The point estimate
and its error bar belonged to two different estimators.

These tests are pure arithmetic on the payloads and the worker's weight
alignment; no MAD-NG process is started.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training_closed_twiss.fitter import stamp_global_normalisation
from aba_optimiser.workers import ClosedTwissData, ClosedTwissWorker, Observable, WorkerConfig
from aba_optimiser.workers.closed_twiss import align_observables

if TYPE_CHECKING:
    from pathlib import Path

BPMS = ["BPM1", "BPM2", "BPM3", "BPM4"]


def _payload(delta: float, errors: np.ndarray) -> tuple[None, ClosedTwissData]:
    """One worker's data at *delta*, with per-BPM orbit errors *errors*."""
    return (
        None,
        ClosedTwissData(
            bpm_names=list(BPMS),
            observables=[
                Observable(
                    name="x",
                    targets=np.zeros(len(BPMS)),
                    variances=np.asarray(errors, dtype=float) ** 2,
                )
            ],
            pt=delta,
        ),
    )


def test_every_worker_gets_the_same_normalisation() -> None:
    precise = _payload(0.0, np.full(len(BPMS), 1e-5))
    coarse = _payload(3e-3, np.full(len(BPMS), 1e-3))

    stamp_global_normalisation([precise, coarse])

    assert precise[1].weight_scale == coarse[1].weight_scale
    assert precise[1].total_points == coarse[1].total_points
    # The scale is the largest weight anywhere in the fit, i.e. the most precise
    # point's, not each worker's own.
    assert precise[1].weight_scale == pytest.approx(1.0 / 1e-5**2)
    assert precise[1].total_points == 2 * len(BPMS)


def test_relative_weighting_between_momenta_survives_normalisation() -> None:
    """A 100x more precise momentum must stay 10000x more heavily weighted."""
    precise = _payload(0.0, np.full(len(BPMS), 1e-5))
    coarse = _payload(3e-3, np.full(len(BPMS), 1e-3))

    stamp_global_normalisation([precise, coarse])

    scale = precise[1].weight_scale
    precise_weight = (1.0 / 1e-5**2) / scale
    coarse_weight = (1.0 / 1e-3**2) / scale
    assert precise_weight / coarse_weight == pytest.approx(1e4)


def test_points_with_no_usable_error_do_not_count() -> None:
    """A dropped point must not inflate the denominator the loss is divided by."""
    errors = np.array([1e-4, np.nan, 1e-4, 0.0])
    (payload,) = [_payload(0.0, errors)]

    stamp_global_normalisation([payload])

    assert payload[1].total_points == 2


def test_worker_applies_the_stamped_normalisation(seq_psb: Path) -> None:
    """The worker must use the stamped constants, not recompute its own."""
    _config, data = _payload(0.0, np.array([1e-4, 1e-4, 2e-4, 2e-4]))
    data.weight_scale = 4.0e8
    data.total_points = 99

    config = WorkerConfig(
        accelerator=PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}}),
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range="$start/$end",
    )
    _parent, child = mp.Pipe()
    # The constructor calls prepare_data, which is what reads the stamped values.
    worker = ClosedTwissWorker(
        child, 0, data, config, SimulationConfig(num_workers=1, num_batches=1)
    )
    _targets, raw_weights, weights = align_observables(
        worker.observables, worker._measured_index, list(BPMS), worker.weight_scale, worker_id=0
    )

    assert worker.normalisation_points == 99
    # Raw inverse-variance weights are untouched; only the stepping copy is scaled.
    assert raw_weights[0][0] == pytest.approx(1.0 / 1e-4**2)
    assert weights[0][0] == pytest.approx((1.0 / 1e-4**2) / 4.0e8)
    assert weights[0][2] == pytest.approx((1.0 / 2e-4**2) / 4.0e8)


def test_worker_rejects_a_nonsensical_weight_scale(seq_psb: Path) -> None:
    _config, data = _payload(0.0, np.full(len(BPMS), 1e-4))
    data.weight_scale = 0.0

    config = WorkerConfig(
        accelerator=PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}}),
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range="$start/$end",
    )
    _parent, child = mp.Pipe()
    with pytest.raises(ValueError, match="weight_scale"):
        ClosedTwissWorker(
            child, 0, data, config, SimulationConfig(num_workers=1, num_batches=1)
        )
