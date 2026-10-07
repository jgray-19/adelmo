"""Screening decisions must reach the validation workers too.

The pre-optimisation outlier screen removes BPMs and whole workers from the
*fit*. Validation workers partition the same measurement files, so if the same
decisions are not replayed on them the held-out loss keeps scoring exactly the
data the optimiser was told to ignore -- and that loss can only rise as the fit
moves onto the data it kept.
"""

from __future__ import annotations

import numpy as np

from aba_optimiser.training.workers.screening import OutlierScreener
from aba_optimiser.training.workers.setup import WorkerRuntimeMetadata
from aba_optimiser.workers.common import KickPlane


def _meta(worker_id: int, file_idx: int, bpms: list[str], sdir: int = 1) -> WorkerRuntimeMetadata:
    return WorkerRuntimeMetadata(
        worker_id=worker_id,
        file_idx=file_idx,
        start_bpm="BPM.A",
        end_bpm="BPM.Z",
        sdir=sdir,
        kick_plane=KickPlane.XY,
        n_run_turns=1,
        bpm_names=bpms,
    )


def _screener() -> OutlierScreener:
    return OutlierScreener(payload_builder=None)  # ty:ignore[invalid-argument-type]


def test_masked_training_bpm_is_masked_in_validation() -> None:
    bpms = ["BPM.1", "BPM.2", "BPM.3"]
    training = [_meta(0, 0, bpms)]
    masks = [np.array([True, False, True])]

    validation_masks, disabled = _screener().build_validation_screening(
        training, masks, [False], [_meta(0, 0, bpms)]
    )

    np.testing.assert_array_equal(validation_masks[0], [True, False, True])
    assert disabled == [False]


def test_disabled_training_worker_disables_its_validation_counterpart() -> None:
    bpms = ["BPM.1", "BPM.2"]
    training = [_meta(0, 0, bpms)]

    _masks, disabled = _screener().build_validation_screening(
        training, [np.ones(2, dtype=bool)], [True], [_meta(0, 0, bpms)]
    )

    assert disabled == [True]


def test_one_surviving_training_worker_keeps_validation_enabled() -> None:
    """A range is only dead when every training worker sharing it was disabled."""
    bpms = ["BPM.1", "BPM.2"]
    training = [_meta(0, 0, bpms), _meta(1, 0, bpms)]

    _masks, disabled = _screener().build_validation_screening(
        training,
        [np.ones(2, dtype=bool), np.ones(2, dtype=bool)],
        [True, False],
        [_meta(0, 0, bpms)],
    )

    assert disabled == [False]


def test_pooled_file_without_a_training_counterpart_falls_back_to_the_range() -> None:
    """A pooled worker reports only its primary file; file 7 still gets screened."""
    bpms = ["BPM.1", "BPM.2"]
    training = [_meta(0, 0, bpms)]

    validation_masks, _disabled = _screener().build_validation_screening(
        training, [np.array([True, False])], [False], [_meta(0, 7, bpms)]
    )

    np.testing.assert_array_equal(validation_masks[0], [True, False])


def test_decisions_do_not_leak_across_directions() -> None:
    """Forward and backward workers are screened independently."""
    bpms = ["BPM.1", "BPM.2"]
    training = [_meta(0, 0, bpms, sdir=1)]

    validation_masks, _disabled = _screener().build_validation_screening(
        training, [np.array([True, False])], [False], [_meta(0, 0, bpms, sdir=-1)]
    )

    np.testing.assert_array_equal(validation_masks[0], [True, True])


def test_disabled_worker_does_not_donate_its_masked_bpms() -> None:
    """A disabled worker's BPM outliers must not mask a surviving sibling's data.

    Two workers cover the same range on the same file. One is disabled -- and a
    disabled worker is by construction a high-loss one, so it carries BPM
    outliers of its own. The other survives, so the fit is still constrained at
    every BPM of that range. Letting the disabled worker's drops through would
    blank those BPMs in the validation counterpart, which is the very defect
    this module exists to prevent, only inverted.
    """
    bpms = ["BPM.1", "BPM.2", "BPM.3"]
    training = [_meta(0, 0, bpms), _meta(1, 0, bpms)]
    # Worker 0 is disabled and would have dropped BPM.2; worker 1 survives clean.
    masks = [np.array([True, False, True]), np.array([True, True, True])]

    validation_masks, disabled = _screener().build_validation_screening(
        training, masks, [True, False], [_meta(0, 0, bpms)]
    )

    np.testing.assert_array_equal(validation_masks[0], [True, True, True])
    assert disabled == [False]


def test_validation_worker_has_worker_disabled_before_any_mask_arrives(tmp_path) -> None:
    """``worker_disabled`` must exist without ``APPLY_MASK`` ever being sent.

    Screening is optional (``enable_preloop_outlier_screening=False``), so the
    validation command loop reads this attribute on runs where no mask is ever
    pushed.
    """
    from aba_optimiser.accelerators import PSB
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers import TrackingData, WorkerConfig
    from aba_optimiser.workers.common import PrecomputedTrackingWeights
    from aba_optimiser.workers.tracking import TrackingWorker

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
    for validation in (False, True):
        worker = TrackingWorker(
            None, 0, data, config, SimulationConfig(num_workers=1, num_batches=1), validation
        )
        assert worker.worker_disabled is False
