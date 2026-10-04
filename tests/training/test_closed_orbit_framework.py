"""Fast contracts for the reusable closed-orbit fitter framework."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.training_closed_twiss import (
    ClosedOrbitMeasurement,
    ClosedOrbitSeries,
)
from aba_optimiser.training_closed_twiss.fitter import (
    _apply_prior,
    _prior_alphas,
    _validate_prior_strengths,
)
from aba_optimiser.workers import Observable
from aba_optimiser.workers.closed_orbit import (
    ClosedOrbitMeasurementData,
    ClosedOrbitSeriesData,
)


def _frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"X": [value, value], "ERRX": [1.0, 1.0]},
        index=["BPM1", "BPM2"],
    )


def test_each_measurement_keeps_its_own_target_and_momenta() -> None:
    series = ClosedOrbitSeries(
        measurements=(
            ClosedOrbitMeasurement(_frame(1.0), pt=-1e-3, reference_pt=0.0),
            ClosedOrbitMeasurement(_frame(2.0), pt=2e-3, reference_pt=1e-3),
        ),
        control_knob="kick",
        control_delta=5e-5,
    )
    assert [item.orbit.iloc[0, 0] for item in series.measurements] == [1.0, 2.0]
    assert [item.pt for item in series.measurements] == [-1e-3, 2e-3]
    assert [item.reference_pt for item in series.measurements] == [0.0, 1e-3]


def test_nested_series_exposes_every_observable_for_global_normalisation() -> None:
    first = Observable("x", np.zeros(2), np.ones(2))
    second = Observable("x", np.ones(2), np.full(2, 4.0))
    data = ClosedOrbitSeriesData(
        bpm_names=["BPM1", "BPM2"],
        measurements=[
            ClosedOrbitMeasurementData([first]),
            ClosedOrbitMeasurementData([second], pt=1e-3),
        ],
    )
    assert data.all_observables == [first, second]


def test_prior_families_are_validated_and_negative_values_rejected() -> None:
    assert _validate_prior_strengths({"dk1l": 1e-4, "dy": 2e-4}) == {
        "dk1l": 1e-4,
        "dy": 2e-4,
    }
    with pytest.raises(ValueError, match=">= 0"):
        _validate_prior_strengths({"dy": -1.0})
    with pytest.raises(ValueError, match="terminal attribute"):
        _validate_prior_strengths({".dy": 1.0})


def test_suffix_priors_scale_each_unit_family_from_its_own_curvature() -> None:
    names = ["q1.dk1l", "q2.dk1l", "q1.dy", "q2.dy"]
    hessian = np.diag([10.0, 30.0, 1e8, 3e8])
    alphas = _prior_alphas(
        {"dk1l": 1e-4, "dy": 2e-4},
        hessian,
        names,
    )
    assert alphas == pytest.approx([2e-3, 2e-3, 4e4, 4e4])


def test_prior_families_must_exactly_cover_optimised_knobs() -> None:
    with pytest.raises(ValueError, match=r"missing=\['dy'\]"):
        _prior_alphas(
            {"dk1l": 1e-4},
            np.eye(2),
            ["q1.dk1l", "q1.dy"],
        )
    with pytest.raises(ValueError, match=r"unused=\['tilt'\]"):
        _prior_alphas(
            {"dk1l": 1e-4, "tilt": 1e-3},
            np.eye(1),
            ["q1.dk1l"],
        )


def test_vector_prior_is_applied_consistently() -> None:
    params = np.array([2.0, 3.0])
    mean = np.array([1.0, 1.0])
    loss, gradient, hessian = _apply_prior(
        5.0,
        np.zeros(2),
        np.eye(2),
        params,
        mean,
        np.array([2.0, 4.0]),
    )
    assert loss == pytest.approx(14.0)
    assert gradient == pytest.approx([2.0, 8.0])
    assert hessian == pytest.approx(np.diag([3.0, 5.0]))


def test_batched_measurements_keep_distinct_signal_orbits_and_share_only_reference() -> None:
    """Different pt targets are evaluated separately; only exact states are cached."""
    from aba_optimiser.workers.closed_orbit import ClosedOrbitWorker

    worker = ClosedOrbitWorker.__new__(ClosedOrbitWorker)
    worker.n_knobs = 1
    worker.knob_name_set = set()
    worker.control_nominal = 1.0
    worker.control_delta = 2.0
    worker.machine_state = {}
    worker._baseline = {}
    worker._subtract = np.ones(1)
    worker.weight_scale = 1.0  # the alignment below uses weights == raw weights
    worker.shared_reference = False
    worker.reference_only = False
    worker.series_measurements = [
        ClosedOrbitMeasurementData([Observable("x", np.array([11.0]), np.ones(1))], pt=1.0),
        ClosedOrbitMeasurementData([Observable("x", np.array([22.0]), np.ones(1))], pt=2.0),
    ]
    worker._measurement_alignment = [
        (item.observables, np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
        for item in worker.series_measurements
    ]
    states: list[tuple[float, float]] = []
    current = {"control": 0.0, "pt": 0.0}

    def set_control(self, _mad, value):
        current["control"] = value

    def set_pt(self, _mad, value):
        current["pt"] = value

    def model(self, _mad, *, context=""):
        state = (current["control"], current["pt"])
        states.append(state)
        value = state[0] + 10.0 * state[1]
        return np.array([[value]]), np.array([[[value]]])

    worker._set_control = MethodType(set_control, worker)
    worker._set_pt = MethodType(set_pt, worker)
    worker._model_and_jacobian = MethodType(model, worker)

    gradient, loss, hessian, normal_matrix = worker.compute_gradients_and_loss(
        object(), {}, 0
    )

    assert loss == pytest.approx(12.0**2 + 22.0**2)
    assert gradient == pytest.approx([2.0 * (12.0**2 + 22.0**2)])
    np.testing.assert_allclose(hessian, [[2.0 * (12.0**2 + 22.0**2)]])
    np.testing.assert_allclose(normal_matrix, [[12.0**2 + 22.0**2]])
    assert states == [(3.0, 1.0), (1.0, 0.0), (3.0, 2.0)]
    assert current["control"] == 1.0
