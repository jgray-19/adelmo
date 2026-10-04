"""Quadrupole BBA through ``ClosedOrbitFitter``: measurements at different machine states.

A quadrupole offset ``dx`` kicks the beam by ``k1 * L * dx``, so changing the quadrupole
strengths between measurements, and the corrector settings with them, makes the offsets
observable from the closed orbit. Each ``ClosedOrbitSeries`` carries the ``machine_state``
(MAD-X globals) it was measured at; the fit sums the series' residual blocks.

``test_absolute_orbits_at_several_states_recover_quad_offsets``
    Absolute orbits at several quadrupole states are fitted jointly, and recover the
    injected ``dx``/``dy`` far better than the same orbit at a single state (relative
    error ~0.77 -> ~0.25). It plateaus there: the PSB only exposes two quadrupole
    globals, kbrqf and kbrqd, so every QF scales together and further states add little.
    Per-quadrupole globals would remove the plateau; the mechanism is the same.

``test_reference_subtracted_orbits_at_a_state``
    Orbit changes under a corrector trim, each taken against its own state's reference,
    reproduce the measured changes (the relative mode of the same fitter).

``test_mixed_absolute_and_relative_series``
    One absolute and one relative series, at different states, in one fit.

``test_machine_state_does_not_leak_between_series``
    More series than workers: a batch worker restores every global after each series.

``test_machine_state_validation``
    An unknown MAD-X name (reported by the worker) and a state that sets the control knob are refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.training.config.models import SequenceConfig
from aba_optimiser.training_closed_twiss import (
    ClosedOrbitFitter,
    ClosedOrbitMeasurement,
    ClosedOrbitSeries,
    LevenbergMarquardtConfig,
)

from .test_closed_twiss_fitter import _knob_names

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.serial, pytest.mark.slow]

KWARGS = {"misalignments": {"quad": {"dx", "dy"}}}
BPM_RESOLUTION = 5e-5  # m
CORRECTOR = "kbr3dhz2l4"
TRIM = 2e-4  # rad

#: Quadrupole settings as multiples of the model's own kbrqf/kbrqd.
STATES = {
    "nominal": (1.0, 1.0),
    "focusing-up": (1.08, 1.0),
    "defocusing-down": (1.0, 0.92),
    "both": (1.05, 0.95),
    "opposite": (0.94, 1.06),
}


def _quad_globals(seq_psb: Path, scales: tuple[float, float]) -> dict[str, float]:
    probe = GradientDescentMadInterface(PSB(ring=3, sequence_file=seq_psb, **KWARGS), py_name="py")
    values = {}
    for name in ("kbrqf", "kbrqd"):
        probe.mad.send(f"py:send(MADX['{name}'])")
        values[name] = float(probe.mad.recv())
    del probe
    return {name: values[name] * scale for name, scale in zip(("kbrqf", "kbrqd"), scales, strict=True)}


def _orbit(
    seq_psb: Path, truth: dict[str, float], machine_state: dict[str, float]
) -> pd.DataFrame:
    """The orbit of the misaligned machine at ``machine_state``, as a measurement."""
    iface = GradientDescentMadInterface(PSB(ring=3, sequence_file=seq_psb, **KWARGS), py_name="py")
    iface.update_knob_values(truth)
    for name, value in machine_state.items():
        iface.mad.send(f"MADX['{name}'] = {value:.15e}")
    iface.mad.send("motws = twiss{sequence=loaded_sequence, observe=1, coupling=true, method=6}")
    twiss = iface.mad.motws.to_df(columns=["name", "x", "y"]).set_index("name")
    del iface
    return pd.DataFrame(
        {"X": twiss["x"], "Y": twiss["y"], "ERRX": BPM_RESOLUTION, "ERRY": BPM_RESOLUTION}
    )


def _fit(seq_psb: Path, series: list[ClosedOrbitSeries], **kwargs) -> dict[str, float]:
    fitter = ClosedOrbitFitter(
        accelerator=PSB(ring=3, sequence_file=seq_psb, **KWARGS),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        series=series,
        lm_config=LevenbergMarquardtConfig(max_iterations=40, gradient_converged_value=1e-12),
        prior_strengths={"dx": 1e-6, "dy": 1e-6},
        **kwargs,
    )
    knobs, _ = fitter.run()
    return knobs


def _truth(seq_psb: Path, seed: int = 0) -> dict[str, float]:
    names = _knob_names(seq_psb, KWARGS)
    rng = np.random.default_rng(seed)
    return {name: float(value) for name, value in zip(names, rng.uniform(-5e-5, 5e-5, len(names)), strict=True)}


def _knob_error(fitted: dict[str, float], truth: dict[str, float]) -> float:
    names = list(truth)
    difference = np.array([fitted[n] - truth[n] for n in names])
    return float(np.linalg.norm(difference) / np.linalg.norm([truth[n] for n in names]))


def _absolute_series(seq_psb: Path, truth: dict[str, float], label: str) -> ClosedOrbitSeries:
    machine_state = _quad_globals(seq_psb, STATES[label])
    return ClosedOrbitSeries(
        measurements=(ClosedOrbitMeasurement(_orbit(seq_psb, truth, machine_state)),),
        machine_state=machine_state,
        absolute_planes=("x", "y"),
        label=label,
    )


def _relative_series(seq_psb: Path, truth: dict[str, float], label: str) -> ClosedOrbitSeries:
    """Orbit change under a corrector trim, at the quadrupole state ``label``."""
    machine_state = _quad_globals(seq_psb, STATES[label])
    trimmed = {**machine_state, CORRECTOR: TRIM}
    change = _orbit(seq_psb, truth, trimmed)
    reference = _orbit(seq_psb, truth, {**machine_state, CORRECTOR: 0.0})
    change[["X", "Y"]] = change[["X", "Y"]] - reference[["X", "Y"]]
    return ClosedOrbitSeries(
        measurements=(ClosedOrbitMeasurement(change),),
        machine_state=machine_state,
        control_knob=CORRECTOR,
        control_delta=TRIM,
        label=label,
    )


def test_absolute_orbits_at_several_states_recover_quad_offsets(seq_psb: Path) -> None:
    truth = _truth(seq_psb)
    series = [_absolute_series(seq_psb, truth, label) for label in STATES]

    single = _knob_error(_fit(seq_psb, series[:1]), truth)
    joint = _knob_error(_fit(seq_psb, series), truth)

    assert joint < 0.5 * single, f"five states ({joint:.2f}) should beat one ({single:.2f})"
    assert joint < 0.3, f"relative offset error {joint:.2f} after fitting five states"


def test_reference_subtracted_orbits_at_a_state(seq_psb: Path) -> None:
    truth = _truth(seq_psb, seed=1)
    series = [_relative_series(seq_psb, truth, label) for label in ("nominal", "both", "opposite")]

    fitted = _fit(seq_psb, series)

    for item in series:
        refit = _relative_series(seq_psb, fitted, item.label)
        measured = item.measurements[0].orbit
        residual = refit.measurements[0].orbit[["X", "Y"]] - measured[["X", "Y"]]
        assert float(np.abs(residual).to_numpy().max()) < 0.05 * float(np.abs(measured[["X", "Y"]]).to_numpy().max())


def test_mixed_absolute_and_relative_series(seq_psb: Path) -> None:
    truth = _truth(seq_psb, seed=2)
    series = [
        _absolute_series(seq_psb, truth, "focusing-up"),
        _relative_series(seq_psb, truth, "defocusing-down"),
    ]

    fitted = _fit(seq_psb, series)

    absolute = _absolute_series(seq_psb, fitted, "focusing-up").measurements[0].orbit
    measured = series[0].measurements[0].orbit
    scale = float(np.abs(measured[["X", "Y"]]).to_numpy().max())
    assert float(np.abs(absolute[["X", "Y"]] - measured[["X", "Y"]]).to_numpy().max()) < 0.05 * scale


def test_machine_state_does_not_leak_between_series(seq_psb: Path) -> None:
    """Five series on one worker give the fit of five series on five workers."""
    truth = _truth(seq_psb, seed=3)
    series = [_absolute_series(seq_psb, truth, label) for label in STATES]

    batched = _fit(seq_psb, series, max_workers=1)
    parallel = _fit(seq_psb, series, max_workers=len(series))

    assert max(abs(batched[name] - parallel[name]) for name in truth) < 1e-9


def test_machine_state_validation(seq_psb: Path) -> None:
    truth = _truth(seq_psb)
    orbit = _orbit(seq_psb, truth, {})
    measurement = (ClosedOrbitMeasurement(orbit),)

    with pytest.raises(RuntimeError, match="not a MAD-X variable"):
        _fit(seq_psb, [ClosedOrbitSeries(measurement, machine_state={"no_such_global": 1.0}, absolute_planes=("x", "y"))])
    with pytest.raises(ValueError, match="control knob"):
        ClosedOrbitSeries(measurement, machine_state={CORRECTOR: 1e-5}, control_knob=CORRECTOR, control_delta=TRIM)
