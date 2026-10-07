"""Fast contracts for the reusable closed-orbit fitter framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.config import SimulationConfig
from aba_optimiser.fitting.protocol import Evaluate, GradReply
from aba_optimiser.fitting.worker import WorkerConfig
from aba_optimiser.machine.accelerators import PSB
from aba_optimiser.machine.mad import GradientDescentMadInterface
from aba_optimiser.poco import (
    ClosedOrbitMeasurement,
    ClosedOrbitSeries,
)
from aba_optimiser.poco.fitter import (
    apply_prior,
    prior_alphas,
    validate_prior_strengths,
)
from aba_optimiser.poco.workers.closed_orbit import (
    ClosedOrbitMeasurementData,
    ClosedOrbitSeriesData,
    ClosedOrbitWorker,
)
from aba_optimiser.poco.workers.closed_twiss import Observable

if TYPE_CHECKING:
    from pathlib import Path


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
        machine_state={"kick": 5e-5},
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
    assert data.observables == [first, second]


def test_prior_families_are_validated_and_negative_values_rejected() -> None:
    assert validate_prior_strengths({"dk1l": 1e-4, "dy": 2e-4}) == {
        "dk1l": 1e-4,
        "dy": 2e-4,
    }
    with pytest.raises(ValueError, match=">= 0"):
        validate_prior_strengths({"dy": -1.0})
    with pytest.raises(ValueError, match="terminal attribute"):
        validate_prior_strengths({".dy": 1.0})


def test_suffix_priors_scale_each_unit_family_from_its_own_curvature() -> None:
    names = ["q1.dk1l", "q2.dk1l", "q1.dy", "q2.dy"]
    hessian = np.diag([10.0, 30.0, 1e8, 3e8])
    alphas = prior_alphas(
        {"dk1l": 1e-4, "dy": 2e-4},
        hessian,
        names,
    )
    assert alphas == pytest.approx([2e-3, 2e-3, 4e4, 4e4])


def test_prior_families_must_exactly_cover_optimised_knobs() -> None:
    with pytest.raises(ValueError, match=r"missing=\['dy'\]"):
        prior_alphas(
            {"dk1l": 1e-4},
            np.eye(2),
            ["q1.dk1l", "q1.dy"],
        )
    with pytest.raises(ValueError, match=r"unused=\['tilt'\]"):
        prior_alphas(
            {"dk1l": 1e-4, "tilt": 1e-3},
            np.eye(1),
            ["q1.dk1l"],
        )


def test_vector_prior_is_applied_consistently() -> None:
    params = np.array([2.0, 3.0])
    mean = np.array([1.0, 1.0])
    loss, gradient, hessian = apply_prior(
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


def test_measurements_in_one_series_add_up_like_separate_series(seq_psb: Path) -> None:
    """Measurements batched into one series are evaluated as independently as separate series.

    Two momenta measured against the same on-momentum reference: one series holding
    both must give the loss, gradient and normal matrix of the two one-measurement
    series summed. A signal orbit leaking between momenta, or a reference solved at
    the wrong momentum, breaks the equality.
    """
    accelerator = PSB(ring=3, sequence_file=seq_psb, misalignments={"quad": {"dx"}})
    iface = GradientDescentMadInterface(accelerator)
    bpms = list(iface.all_bpms)
    knobs = {name: 1e-5 * (i + 1) for i, name in enumerate(n for n in iface.knob_names if n != "pt")}
    iface.close()
    config = WorkerConfig(
        accelerator=accelerator,
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range="$start/$end",
        cycle_sequence=False,
    )

    def measurement(pt: float, target: float) -> ClosedOrbitMeasurementData:
        observable = Observable("x", np.full(len(bpms), target), np.full(len(bpms), 1e-8))
        return ClosedOrbitMeasurementData([observable], pt=pt, reference_pt=0.0)

    def evaluate(*measurements: ClosedOrbitMeasurementData) -> GradReply:
        series = ClosedOrbitSeriesData(bpm_names=bpms, measurements=list(measurements))
        worker = ClosedOrbitWorker(None, 0, [series], config, SimulationConfig(num_workers=1, num_batches=1))
        mad = worker.on_start(knobs)
        try:
            return worker.evaluate(mad, Evaluate(knobs))
        finally:
            mad.send("shush()")

    low, high = measurement(1e-3, 1e-4), measurement(2e-3, -2e-4)
    together = evaluate(low, high)
    apart = [evaluate(low), evaluate(high)]

    assert together.loss == pytest.approx(sum(r.loss for r in apart), rel=1e-12)
    np.testing.assert_allclose(together.grad, sum(r.grad for r in apart), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(together.normal, sum(r.normal for r in apart), rtol=1e-12, atol=0.0)
