"""The worker subtracts the reference orbit only in the planes whose data was differenced."""

from __future__ import annotations

import numpy as np
import pytest

from adelmo.poco.workers.closed_orbit import (
    ClosedOrbitMeasurementData,
    ClosedOrbitSeriesData,
    ClosedOrbitWorker,
)
from adelmo.poco.workers.closed_twiss import Observable


def _observable(name: str) -> Observable:
    return Observable(name=name, targets=np.zeros(2), variances=np.ones(2))


def _prepared(**overrides) -> ClosedOrbitWorker:
    worker = ClosedOrbitWorker.__new__(ClosedOrbitWorker)
    worker.worker_id = 0
    data = ClosedOrbitSeriesData(
        bpm_names=["BR3.BPM1L3", "BR3.BPM2L3"],
        measurements=[ClosedOrbitMeasurementData(observables=[_observable("x"), _observable("y")])],
        **{"machine_state": {"kbr3dhz8l1": 5e-5}, **overrides},
    )
    worker.prepare_data([data])
    return worker


#: Orbit rows (x, y) at two BPMs with one knob derivative each.
KICKED = (np.array([[3.0, 4.0], [5.0, 6.0]]), np.ones((2, 2, 1)) * 10.0)
NOMINAL = (np.array([[1.0, 1.0], [2.0, 2.0]]), np.ones((2, 2, 1)) * 4.0)


def _compare(worker: ClosedOrbitWorker):
    return ClosedOrbitWorker._compare_to_reference(worker.series[0], KICKED, NOMINAL)


def test_the_reference_is_subtracted_only_where_the_data_did() -> None:
    model, jacobian = _compare(_prepared(absolute_planes=("y",)))

    np.testing.assert_allclose(model[0], [2.0, 3.0])  # x: differenced
    np.testing.assert_allclose(model[1], [5.0, 6.0])  # y: absolute
    np.testing.assert_allclose(jacobian[0], 6.0)
    np.testing.assert_allclose(jacobian[1], 10.0)


def test_without_absolute_planes_both_planes_are_differenced() -> None:
    model, jacobian = _compare(_prepared())

    np.testing.assert_allclose(model, KICKED[0] - NOMINAL[0])
    np.testing.assert_allclose(jacobian, KICKED[1] - NOMINAL[1])


def test_an_untrimmed_series_is_signal_only_when_a_plane_is_absolute() -> None:
    with pytest.raises(ValueError, match="absolute plane"):
        _prepared(machine_state={})
    assert _prepared(machine_state={}, absolute_planes=("y",)).series[0].data.absolute_planes == ("y",)


def test_an_unknown_plane_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown absolute plane"):
        _prepared(absolute_planes=("z",))
