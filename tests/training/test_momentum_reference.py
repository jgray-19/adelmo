"""Physics tests for :mod:`adelmo.momentum_reference`.

The quantity under test is the closed-orbit *angle* ``px`` at the BPMs. It cannot
be measured -- BPMs read position -- so a momentum reconstruction has to take it
from a model, and a nominal model that does not carry the machine's magnet errors
gets it entirely wrong: its error equals the true angle.

``test_orbit_and_phase_recover_the_closed_orbit_angle`` is the headline claim:
closed orbit plus phase advance, the two observables that need no amplitude
calibration and no model-derived intermediate, are enough to fix that.

``test_orbit_alone_with_quad_knobs_warns`` pins the trap. The closed orbit is
blind to gradient errors (a quadrupole on a centred orbit produces no
deflection), so enabling quadrupole knobs against orbit-only observables adds
noise-absorbing freedom and measurably degrades the fit. Callers get warned.

``test_a_single_momentum_is_rejected`` pins the other precondition: at one
momentum the per-magnet Jacobians are degenerate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from adelmo.machine.accelerators import PSB
from adelmo.machine.mad import GradientDescentMadInterface
from adelmo.momentum_reference import (
    ORBIT_AND_PHASE,
    closed_orbit_at,
    fit_momentum_reference,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

DELTAS = (-3e-3, 0.0, 3e-3)
BEND_REL = 8e-4
QUAD_REL = 1e-3

MEASURED_FROM_TWISS = {
    "X": "x",
    "Y": "y",
    "BETX": "betx",
    "BETY": "bety",
    "ALFX": "alfx",
    "ALFY": "alfy",
    "DX": "dx",
    "DY": "dy",
    "MUX": "mu1",
    "MUY": "mu2",
}
RESOLUTIONS = {
    "ERRX": 5e-5,
    "ERRY": 5e-5,
    "ERRBETX": 1e-2,
    "ERRBETY": 1e-2,
    "ERRALFX": 1e-2,
    "ERRALFY": 1e-2,
    "ERRDX": 1e-3,
    "ERRDY": 1e-3,
}
PHASE_SIGMA = 1e-4


def _accelerator(seq: Path, **kwargs) -> PSB:
    return PSB(ring=3, sequence_file=seq, **kwargs)


def _both(seq: Path) -> PSB:
    return _accelerator(seq, errors={"bend": {"k0"}, "quad": {"k1"}})


def _truth(seq: Path, seed: int = 7) -> dict[str, float]:
    """Relative field errors on every bend and quad, expressed as knob values.

    The knobs are *integrated* perturbations while the accelerator reports
    un-integrated k0/k1, so element lengths turn a relative error into a knob.
    """
    iface = GradientDescentMadInterface(_both(seq))
    knobs = [k for k in iface.knob_names if k != "pt"]
    absolute = iface.get_absolute_knob_values(knobs)
    iface.mad.send("alltws = twiss{sequence=loaded_sequence, observe=0}")
    frame = iface.mad.alltws.to_df(columns=["name", "l"])
    lengths = dict(zip(frame["name"].tolist(), frame["l"].to_numpy(dtype=float), strict=True))
    del iface

    rng = np.random.default_rng(seed)
    truth = {}
    for knob in knobs:
        element, suffix = knob.rsplit(".dk", 1)
        integrated = float(absolute[f"{element}.k{suffix[0]}"]) * float(lengths[element])
        relative = BEND_REL if suffix[0] == "0" else QUAD_REL
        truth[knob] = float(rng.normal(0.0, relative) * integrated)
    return truth


def _measurement(
    seq: Path, truth: dict[str, float], delta: float, rng: np.random.Generator
) -> pd.DataFrame:
    """The perturbed machine's twiss, noised, as an omc3-style measurement."""
    iface = GradientDescentMadInterface(_both(seq))
    iface.update_knob_values(truth)
    iface.mad.send(
        f"mtws = twiss{{sequence=loaded_sequence, observe=1, "
        f"X0={{pt={delta:.15e}}}, coupling=true}}"
    )
    twiss = iface.mad.mtws.to_df(
        columns=["name", *sorted(set(MEASURED_FROM_TWISS.values()))]
    ).set_index("name")
    del iface

    measurement = pd.DataFrame(
        {measured: twiss[modelled] for measured, modelled in MEASURED_FROM_TWISS.items()},
        index=twiss.index,
    )
    for column, key in (("X", "ERRX"), ("Y", "ERRY"), ("DX", "ERRDX"), ("DY", "ERRDY")):
        measurement[column] += rng.normal(0.0, RESOLUTIONS[key], len(measurement))
    for column, key in (("BETX", "ERRBETX"), ("BETY", "ERRBETY")):
        measurement[column] *= 1.0 + rng.normal(0.0, RESOLUTIONS[key], len(measurement))
    for column in ("MUX", "MUY"):
        # Phase noise accumulates: the fit consumes advances, not absolute phase.
        measurement[column] += np.cumsum(rng.normal(0.0, PHASE_SIGMA, len(measurement)))
    for column, value in RESOLUTIONS.items():
        measurement[column] = value
    # Beta noise above is relative, so its declared error scales with beta.
    for column, key in (("BETX", "ERRBETX"), ("BETY", "ERRBETY")):
        measurement[key] = RESOLUTIONS[key] * measurement[column]
    measurement["mu1_var"] = np.arange(len(measurement)) * PHASE_SIGMA**2
    measurement["mu2_var"] = np.arange(len(measurement)) * PHASE_SIGMA**2
    return measurement


def _px_rms_error(reference: pd.DataFrame, machine: pd.DataFrame) -> float:
    common = reference.index.intersection(machine.index)
    residual = (reference.loc[common, "px"] - machine.loc[common, "px"]).to_numpy(dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


@pytest.fixture(scope="module")
def fitted(seq_psb: Path):
    """Truth, the machine's closed orbit, and the fit from orbit + phase."""
    truth = _truth(seq_psb)
    rng = np.random.default_rng(8)
    measurements = {d: _measurement(seq_psb, truth, d, rng) for d in DELTAS}
    reference = fit_momentum_reference(
        _both(seq_psb), measurements, observables=ORBIT_AND_PHASE
    )
    machine = closed_orbit_at(_both(seq_psb), truth, 0.0)
    return truth, machine, reference


@pytest.mark.slow
def test_orbit_and_phase_recover_the_closed_orbit_angle(seq_psb: Path, fitted) -> None:
    """Orbit + phase must beat the nominal model by a wide margin on ``px``."""
    _truth_knobs, machine, reference = fitted

    nominal = closed_orbit_at(_both(seq_psb), None, 0.0)
    nominal_error = _px_rms_error(nominal, machine)
    fitted_error = _px_rms_error(reference.closed_orbit, machine)

    true_rms = float(np.sqrt(np.mean(machine["px"].to_numpy(dtype=float) ** 2)))
    # The nominal model knows nothing, so its error *is* the whole true angle.
    # If that stops holding the test setup has lost its perturbation.
    assert nominal_error == pytest.approx(true_rms, rel=0.2)

    assert fitted_error < nominal_error / 10.0, (
        f"orbit+phase fit gave px error {fitted_error:.3e} against nominal "
        f"{nominal_error:.3e}; expected at least a 10x improvement"
    )


@pytest.mark.slow
def test_reference_carries_angles_the_measurement_cannot(fitted) -> None:
    """The point of the reference is px/py, which no BPM measures."""
    _truth_knobs, _machine, reference = fitted
    assert list(reference.closed_orbit.columns) == ["x", "y", "px", "py"]
    assert np.isfinite(reference.closed_orbit.to_numpy(dtype=float)).all()
    # A non-trivial angle, not an accidentally-zero column.
    assert np.abs(reference.closed_orbit["px"].to_numpy(dtype=float)).max() > 1e-6
    # Fitted strengths are returned so a downstream model can be rebuilt without
    # importing anything from this package.
    assert reference.magnet_strengths
    assert all(isinstance(v, float) for v in reference.magnet_strengths.values())


def test_a_single_momentum_is_rejected(seq_psb: Path) -> None:
    """One momentum leaves the per-magnet Jacobians degenerate."""
    with pytest.raises(ValueError, match="at least two momenta"):
        fit_momentum_reference(_both(seq_psb), {0.0: pd.DataFrame()})


def test_orbit_alone_with_quad_knobs_is_rejected(seq_psb: Path) -> None:
    """Gradient knobs with no gradient-sensitive observable is a measured mistake."""
    from adelmo.momentum_reference import _check_observable_knob_match

    with pytest.raises(ValueError, match="blind to gradients"):
        _check_observable_knob_match(_both(seq_psb), ("x", "y"))
    _check_observable_knob_match(_both(seq_psb), ORBIT_AND_PHASE)


def test_reference_records_plain_data_fit_metadata(fitted) -> None:
    _truth_knobs, _machine, reference = fitted
    assert reference.reference_pt == 0.0
    assert reference.momentum_points == DELTAS
    assert reference.uncertainties.keys() == reference.magnet_strengths.keys()
    assert set(reference.bpm_coverage) == set(DELTAS)
    assert reference.fit_settings["sequence_range"] == "$start/$end"
    assert isinstance(reference.diagnostics["converged"], bool)
    assert reference.diagnostics["iterations"] > 0


