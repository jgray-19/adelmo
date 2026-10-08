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
    Orbit changes from the fitter's reference state to each series' state (quadrupole
    strengths and a corrector trim) reproduce the measured changes (the relative mode of the same fitter).

``test_mixed_absolute_and_relative_series``
    One absolute and one relative series, at different states, in one fit.

``test_machine_state_does_not_leak_between_series``
    More series than workers: a batch worker restores every global after each series.

``test_fitter_default_and_knobs_file_equal_per_series_state``
    The fitter-level ``machine_state`` (a dict or a knobs file) is the default every series
    inherits, and gives the same fit as spelling the state out on each series.

``test_series_state_is_the_change_from_the_fitter_state``
    A relative series is the change from the fitter's state, which may already stand away from
    the model's own; the fitter records its history.

``test_trim_is_added_to_the_models_own_value``
    A ``trim`` is added to the value the MAD environment holds for the global, with no state naming it,
    and fits like the explicit state of that sum.

``test_shared_reference_matches_per_series_references``
    Series at different states share the one reference solve (the fitter's state) and give the same fit.

``test_series_equal_to_the_reference_state_is_refused``
    A relative series whose state is the reference state measures nothing and is reported.

``test_machine_state_validation``
    An unknown MAD-X name is reported by the worker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pymadng_utils.io.utils import save_knobs

from adelmo.fitting.config import SequenceConfig
from adelmo.machine.accelerators import PSB
from adelmo.machine.mad import GradientDescentMadInterface
from adelmo.poco import (
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
#: The fitter's own state, which relative series are measured against.
REFERENCE = {CORRECTOR: 0.0}

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
    try:
        knobs = fitter.run().knobs
    finally:
        fitter.close()
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
    """Orbit change from the reference state to the quadrupole state ``label`` with a corrector trim."""
    machine_state = {**_quad_globals(seq_psb, STATES[label]), CORRECTOR: TRIM}
    change = _orbit(seq_psb, truth, machine_state)
    reference = _orbit(seq_psb, truth, REFERENCE)
    change[["X", "Y"]] = change[["X", "Y"]] - reference[["X", "Y"]]
    return ClosedOrbitSeries(
        measurements=(ClosedOrbitMeasurement(change),),
        machine_state=machine_state,
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

    fitted = _fit(seq_psb, series, machine_state=REFERENCE)

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

    fitted = _fit(seq_psb, series, machine_state=REFERENCE)

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


def test_fitter_default_and_knobs_file_equal_per_series_state(seq_psb: Path, tmp_path: Path) -> None:
    truth = _truth(seq_psb, seed=4)
    state = _quad_globals(seq_psb, STATES["both"])
    orbit = (ClosedOrbitMeasurement(_orbit(seq_psb, truth, state)),)
    explicit = _fit(seq_psb, [ClosedOrbitSeries(orbit, machine_state=state, absolute_planes=("x", "y"))])

    bare = ClosedOrbitSeries(orbit, absolute_planes=("x", "y"))
    knobs_file = tmp_path / "state.txt"
    save_knobs(state, knobs_file)
    from_dict = _fit(seq_psb, [bare], machine_state=state)
    from_file = _fit(seq_psb, [bare], machine_state=knobs_file)

    for fitted in (from_dict, from_file):
        assert max(abs(fitted[name] - explicit[name]) for name in truth) < 1e-9


def test_series_state_is_the_change_from_the_fitter_state(seq_psb: Path) -> None:
    """With the fitter standing at a corrector kick of 1e-4, a series trimmed to 1e-4 + TRIM is measured against that kick, not zero."""
    truth = _truth(seq_psb, seed=5)
    standing = {**_quad_globals(seq_psb, STATES["both"]), CORRECTOR: 1e-4}
    trimmed = {CORRECTOR: 1e-4 + TRIM}
    change = _orbit(seq_psb, truth, {**standing, **trimmed})
    reference = _orbit(seq_psb, truth, standing)
    change[["X", "Y"]] = change[["X", "Y"]] - reference[["X", "Y"]]
    series = ClosedOrbitSeries((ClosedOrbitMeasurement(change),), machine_state=trimmed)

    fitter = ClosedOrbitFitter(
        accelerator=PSB(ring=3, sequence_file=seq_psb, **KWARGS),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        series=[series],
        lm_config=LevenbergMarquardtConfig(max_iterations=40, gradient_converged_value=1e-12),
        prior_strengths={"dx": 1e-6, "dy": 1e-6},
        machine_state=standing,
    )
    try:
        fitted = fitter.run().knobs
    finally:
        fitter.close()

    assert fitter.history, "accepted iterations are recorded"
    losses = [loss for _, loss in fitter.history]
    assert losses == sorted(losses, reverse=True)
    # Tail check: the reproduced orbit change matches the measured one at the standing kick.
    refit = _orbit(seq_psb, fitted, {**standing, **trimmed})
    base = _orbit(seq_psb, fitted, standing)
    residual = (refit[["X", "Y"]] - base[["X", "Y"]]) - change[["X", "Y"]]
    assert float(np.abs(residual).to_numpy().max()) < 0.05 * float(np.abs(change[["X", "Y"]]).to_numpy().max())


def test_shared_reference_matches_per_series_references(seq_psb: Path) -> None:
    """Series at different states share the single reference solve and fit as if each solved its own."""
    truth = _truth(seq_psb, seed=6)
    state = _quad_globals(seq_psb, STATES["both"])
    reference = _orbit(seq_psb, truth, state)
    series = []
    for corrector in (CORRECTOR, "kbr3dhz8l1"):
        trimmed = _orbit(seq_psb, truth, {**state, corrector: TRIM})
        trimmed[["X", "Y"]] = trimmed[["X", "Y"]] - reference[["X", "Y"]]
        series.append(ClosedOrbitSeries((ClosedOrbitMeasurement(trimmed),), machine_state={corrector: TRIM}))

    shared = _fit(seq_psb, series, machine_state=state, shared_reference=True)
    own = _fit(seq_psb, series, machine_state=state, shared_reference=False)

    assert max(abs(shared[name] - own[name]) for name in truth) < 1e-9


def test_series_equal_to_the_reference_state_is_refused(seq_psb: Path) -> None:
    truth = _truth(seq_psb)
    measurement = (ClosedOrbitMeasurement(_orbit(seq_psb, truth, {})),)

    with pytest.raises(ValueError, match="differs from the reference"):
        _fit(seq_psb, [ClosedOrbitSeries(measurement)])


def test_machine_state_validation(seq_psb: Path) -> None:
    truth = _truth(seq_psb)
    measurement = (ClosedOrbitMeasurement(_orbit(seq_psb, truth, {})),)

    with pytest.raises(RuntimeError, match="neither a MAD-X variable nor an element attribute"):
        _fit(seq_psb, [ClosedOrbitSeries(measurement, machine_state={"no_such_global": 1.0}, absolute_planes=("x", "y"))])
