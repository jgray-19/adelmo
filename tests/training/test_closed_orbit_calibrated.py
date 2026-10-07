"""``CalibratedClosedOrbitFitter`` end to end: BPM gains and corrector kick gains next to the magnet knobs.

The calibrated fitter names a corrector's gain after its ``k_<corrector>`` kick global, so the PSB
sequence is copied with two corrector variables renamed to that form. A series' own ``machine_state``
sets one such kick; its change from the fitter's ``machine_state`` (the reference) is the trim.

``test_unit_gains_are_recovered_as_zero``
    Orbit changes measured with perfect gains are reproduced, and every fitted gain stays small.

``test_series_must_set_exactly_one_corrector_kick``
    A series without a ``k_`` kick, or with two, is refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.training.config.models import SequenceConfig
from aba_optimiser.training_closed_twiss import (
    ClosedOrbitMeasurement,
    ClosedOrbitSeries,
    LevenbergMarquardtConfig,
)
from aba_optimiser.training_closed_twiss.calibrated import CalibratedClosedOrbitFitter

from .test_closed_orbit_machine_state import KWARGS, STATES, TRIM, _orbit, _quad_globals, _truth

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.serial, pytest.mark.slow]

#: Corrector variables of the PSB sequence, renamed ``k_<corrector>`` in the copy.
RENAMED = {"kbr3dhz2l4": "k_dhz2l4", "kbr3dhz8l1": "k_dhz8l1"}


@pytest.fixture(scope="module")
def seq_k(seq_psb: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    text = seq_psb.read_text()
    for old, new in RENAMED.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path_factory.mktemp("psb_k") / seq_psb.name
    path.write_text(text)
    return path


def _series(
    seq_k: Path, truth: dict[str, float], label: str, corrector: str, gain: float = 0.0
) -> ClosedOrbitSeries:
    """The orbit change from the model's own state to quadrupole state ``label`` with ``corrector`` trimmed.

    The machine puts ``(1 + gain) * TRIM`` into the lattice; the series' state records the requested ``TRIM``.
    """
    state = {**_quad_globals(seq_k, STATES[label]), corrector: TRIM}
    change = _orbit(seq_k, truth, {**state, corrector: (1.0 + gain) * TRIM})
    reference = _orbit(seq_k, truth, {})
    change[["X", "Y"]] = change[["X", "Y"]] - reference[["X", "Y"]]
    return ClosedOrbitSeries((ClosedOrbitMeasurement(change),), machine_state=state, label=label)


def _fitter(seq_k: Path, series: list[ClosedOrbitSeries], **kwargs) -> CalibratedClosedOrbitFitter:
    return CalibratedClosedOrbitFitter(
        accelerator=PSB(ring=3, sequence_file=seq_k, **KWARGS),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        series=series,
        lm_config=LevenbergMarquardtConfig(max_iterations=40, gradient_converged_value=1e-12),
        prior_strengths={"dx": 1e-6, "dy": 1e-6},
        **kwargs,
    )


def test_unit_gains_are_recovered_as_zero(seq_k: Path) -> None:
    truth = _truth(seq_k, seed=7)
    series = [
        _series(seq_k, truth, label, corrector)
        for label, corrector in (("nominal", "k_dhz2l4"), ("both", "k_dhz8l1"), ("opposite", "k_dhz2l4"))
    ]

    fitter = _fitter(seq_k, series)
    try:
        fitted = fitter.run().knobs
    finally:
        fitter.close()

    assert fitter.calibration_spec.correctors == ("DHZ2L4", "DHZ8L1")
    gains = fitter.calibration_result
    assert gains["corrgain.DHZ2L4"] == pytest.approx(0.0, abs=0.05)
    assert gains["corrgain.DHZ8L1"] == pytest.approx(0.0, abs=0.05)
    assert all(abs(value) < 0.05 for name, value in gains.items() if name.startswith("bpmgain."))
    for item in series:
        corrector = next(name for name in item.machine_state if name.startswith("k_"))
        refit = _series(seq_k, fitted, item.label, corrector).measurements[0].orbit
        measured = item.measurements[0].orbit
        residual = np.abs(refit[["X", "Y"]] - measured[["X", "Y"]]).to_numpy().max()
        assert residual < 0.1 * np.abs(measured[["X", "Y"]]).to_numpy().max()


def test_corrector_kick_gain_is_recovered(seq_k: Path) -> None:
    """A corrector that delivers 10% more kick than asked is fitted as a gain of ~0.1, not absorbed into the knobs."""
    truth = _truth(seq_k, seed=8)
    labels = ("nominal", "both", "opposite", "focusing-up", "defocusing-down")
    series = [_series(seq_k, truth, label, "k_dhz2l4", gain=0.1) for label in labels]
    series += [_series(seq_k, truth, label, "k_dhz8l1") for label in labels]

    fitter = _fitter(seq_k, series, sigma_corrector=0.5)  # the default 1e-2 prior would pull a true 0.1 to ~0
    try:
        fitter.run()
    finally:
        fitter.close()

    gains = fitter.calibration_result
    assert gains["corrgain.DHZ2L4"] - gains["corrgain.DHZ8L1"] == pytest.approx(0.1, abs=0.03)


def test_series_must_set_exactly_one_corrector_kick(seq_k: Path) -> None:
    truth = _truth(seq_k)
    orbit = _orbit(seq_k, truth, {})
    measurement = (ClosedOrbitMeasurement(orbit),)
    for state in ({}, {"k_dhz2l4": TRIM, "k_dhz8l1": TRIM}):
        with pytest.raises(ValueError, match="exactly one 'k_<corrector>' kick"):
            _fitter(seq_k, [ClosedOrbitSeries(measurement, machine_state=state)])
