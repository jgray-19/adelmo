"""``CalibratedClosedOrbitFitter`` end to end: BPM gains and corrector kick gains next to the magnet knobs.

The calibrated fitter names a corrector's gain after its ``k_<corrector>`` kick global, so the PSB
sequence is copied with two corrector variables renamed to that form. A series' own ``machine_state``
sets one such kick; its change from the fitter's ``machine_state`` (the reference) is the trim.

``test_unit_gains_are_recovered_as_zero``
    Orbit changes measured with perfect gains are reproduced, and every fitted gain stays small.

``test_off_momentum_series_without_a_kick_joins_a_gain_fit``
    A series that sets no kick (an off-momentum orbit) is fitted alongside trimmed ones; a series may also set several kicks.

``test_bpm_gains_alone_are_recovered_from_absolute_orbits``
    With ``fit_corrector_gains=False`` there is no corrector gain, a series may set several kicks, and every BPM's gain error is recovered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from adelmo.fitting.config import SequenceConfig
from adelmo.machine.accelerators import PSB
from adelmo.poco import (
    ClosedOrbitMeasurement,
    ClosedOrbitSeries,
    LevenbergMarquardtConfig,
)
from adelmo.poco.calibrated import CalibratedClosedOrbitFitter

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
    signal = _orbit(seq_k, truth, {**state, corrector: (1.0 + gain) * TRIM})
    reference = _orbit(seq_k, truth, {})
    return ClosedOrbitSeries((ClosedOrbitMeasurement(signal, reference=reference),), machine_state=state, label=label)


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
        refit = _series(seq_k, fitted, item.label, corrector).measurements[0]
        measured = item.measurements[0]
        refit_change = refit.orbit[["X", "Y"]] - refit.reference[["X", "Y"]]
        measured_change = measured.orbit[["X", "Y"]] - measured.reference[["X", "Y"]]
        residual = np.abs(refit_change - measured_change).to_numpy().max()
        assert residual < 0.1 * np.abs(measured_change).to_numpy().max()


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


def test_off_momentum_series_without_a_kick_joins_a_gain_fit(seq_k: Path) -> None:
    """The dispersion orbit has no corrector in its state; it constrains the magnets and leaves the gain fit working.

    One series sets two kicks (both correctors trimmed together), one sets none and is measured off momentum.
    """
    delta = 2e-4
    truth = _truth(seq_k, seed=8)
    labels = ("nominal", "both", "opposite", "focusing-up", "defocusing-down")
    series = [_series(seq_k, truth, label, "k_dhz2l4", gain=0.1) for label in labels]
    series += [_series(seq_k, truth, label, "k_dhz8l1") for label in labels]
    both = {"k_dhz2l4": TRIM, "k_dhz8l1": -TRIM}
    signal = _orbit(seq_k, truth, {k: (1.0 + (0.1 if k == "k_dhz2l4" else 0.0)) * v for k, v in both.items()})
    reference = _orbit(seq_k, truth, {})
    series.append(
        ClosedOrbitSeries((ClosedOrbitMeasurement(signal, reference=reference),), machine_state=both, label="two kicks")
    )
    dispersion = _orbit(seq_k, truth, {}, pt=delta)
    series.append(
        ClosedOrbitSeries(
            (ClosedOrbitMeasurement(dispersion, pt=delta, reference=reference),), machine_state={}, label="off momentum"
        )
    )

    fitter = _fitter(seq_k, series, sigma_corrector=0.5)
    try:
        fitter.run()
    finally:
        fitter.close()

    assert fitter.calibration_spec.correctors == ("DHZ2L4", "DHZ8L1")
    gains = fitter.calibration_result
    assert gains["corrgain.DHZ2L4"] - gains["corrgain.DHZ8L1"] == pytest.approx(0.1, abs=0.03)
    assert fitter.history, "accepted iterations are recorded"


def test_kick_prefix_names_the_corrector_globals(seq_psb: Path) -> None:
    """With ``kick_prefix="kbr3"`` the sequence's own corrector variables carry the gains; no renaming needed."""
    truth = _truth(seq_psb, seed=8)
    labels = ("nominal", "both", "opposite", "focusing-up", "defocusing-down")
    series = [_series(seq_psb, truth, label, "kbr3dhz2l4", gain=0.1) for label in labels]
    series += [_series(seq_psb, truth, label, "kbr3dhz8l1") for label in labels]

    fitter = _fitter(seq_psb, series, sigma_corrector=0.5, kick_prefix="kbr3", knob_sigmas={"dx": 1e-3, "dy": 1e-3})
    try:
        fitter.run()
    finally:
        fitter.close()

    assert fitter.calibration_spec.correctors == ("DHZ2L4", "DHZ8L1")
    gains = fitter.calibration_result
    assert gains["corrgain.DHZ2L4"] - gains["corrgain.DHZ8L1"] == pytest.approx(0.1, abs=0.03)


def test_bpm_gains_alone_are_recovered_from_absolute_orbits(seq_psb: Path) -> None:
    """Each BPM's reading error, both planes, is fitted from absolute orbits with several standing kicks and no corrector gain.

    Absolute orbits keep every corrector's standing kick, so a series sets three kicks (``fit_corrector_gains=False`` takes any
    number) and the quadrupole state differs from series to series. The machine is perfect, so the BPM gains are the only unknowns.
    """
    corrector_knobs = ("kbr3dhz2l4", "kbr3dhz8l1", "kbr3dvt2l4")  # the vertical kick makes the vertical gains observable
    standing_kicks = ((1e-3, -8e-4, 6e-4), (-6e-4, 1e-3, -9e-4))
    truth = {name: 0.0 for name in _truth(seq_psb)}
    rng = np.random.default_rng(3)
    errors = {plane: rng.uniform(-0.03, 0.03, 16) for plane in ("x", "y")}

    series = []
    for label in STATES:
        for kick in standing_kicks:
            state = {**_quad_globals(seq_psb, STATES[label]), **dict(zip(corrector_knobs, kick, strict=True))}
            orbit = _orbit(seq_psb, truth, state)
            orbit["X"] *= 1.0 + errors["x"]
            orbit["Y"] *= 1.0 + errors["y"]
            orbit[["ERRX", "ERRY"]] = 1e-6
            series.append(
                ClosedOrbitSeries(
                    (ClosedOrbitMeasurement(orbit),), machine_state=state, absolute_planes=("x", "y"), label=f"{label}{kick[0]:+g}"
                )
            )
    bpms = list(series[0].measurements[0].orbit.index)

    fitter = _fitter(
        seq_psb, series, fit_corrector_gains=False, sigma_bpm=0.1, knob_sigmas={"dx": 1e-9, "dy": 1e-9}, kick_prefix="kbr3"
    )
    try:
        fitter.run()
    finally:
        fitter.close()

    gains = fitter.calibration_result
    assert fitter.calibration_spec.correctors == ()
    assert not any(name.startswith("corrgain.") for name in gains)
    for plane in ("x", "y"):
        fitted = np.array([gains[f"bpmgain.{plane}.{bpm}"] for bpm in bpms])
        assert fitted == pytest.approx(errors[plane], abs=2e-3)


def test_bpm_gain_uncertainties_match_the_scatter_over_noise_seeds(seq_psb: Path) -> None:
    """Fitted BPM gains minus the truth, in units of the reported sigma, have unit spread over many noisy orbits."""
    corrector_knobs = ("kbr3dhz2l4", "kbr3dhz8l1", "kbr3dvt2l4")
    standing_kicks = ((1e-3, -8e-4, 6e-4), (-6e-4, 1e-3, -9e-4))
    noise = 3e-5
    truth = {name: 0.0 for name in _truth(seq_psb)}
    errors = {plane: np.random.default_rng(3).uniform(-0.03, 0.03, 16) for plane in ("x", "y")}

    clean = []
    for label in STATES:
        for kick in standing_kicks:
            state = {**_quad_globals(seq_psb, STATES[label]), **dict(zip(corrector_knobs, kick, strict=True))}
            orbit = _orbit(seq_psb, truth, state)
            orbit["X"] *= 1.0 + errors["x"]
            orbit["Y"] *= 1.0 + errors["y"]
            clean.append((state, orbit, f"{label}{kick[0]:+g}"))

    z = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        series = []
        for state, orbit, label in clean:
            noisy = orbit.copy()
            noisy[["X", "Y"]] += rng.normal(0.0, noise, (len(noisy), 2))
            noisy[["ERRX", "ERRY"]] = noise
            series.append(ClosedOrbitSeries((ClosedOrbitMeasurement(noisy),), machine_state=state, absolute_planes=("x", "y"), label=label))
        fitter = _fitter(
            seq_psb, series, fit_corrector_gains=False, sigma_bpm=0.1, knob_sigmas={"dx": 1e-9, "dy": 1e-9}, kick_prefix="kbr3"
        )
        try:
            result = fitter.run()
        finally:
            fitter.close()
        for plane in ("x", "y"):
            for index, bpm in enumerate(fitter.calibration_spec.bpms):
                name = f"bpmgain.{plane}.{bpm}"
                assert result.extra_uncertainties[name] > 0.0
                z.append((result.extra[name] - errors[plane][index]) / result.extra_uncertainties[name])

    assert 0.7 < float(np.std(z)) < 1.3, f"std of z = {np.std(z):.3f} over {len(z)} gains"
    assert abs(float(np.mean(z))) < 0.3, f"mean of z = {np.mean(z):.3f}"


def test_corrector_gain_uncertainties_match_the_scatter_over_noise_seeds(seq_k: Path) -> None:
    """Fitted corrector gains minus the truth, in units of the reported sigma, have unit spread over many noisy orbit changes."""
    noise = 2e-5
    truth = {name: 0.0 for name in _truth(seq_k)}  # a perfect machine: the gains are the only unknowns
    labels = ("nominal", "both", "opposite", "focusing-up", "defocusing-down")
    clean = [_series(seq_k, truth, label, "k_dhz2l4", gain=0.1) for label in labels]
    clean += [_series(seq_k, truth, label, "k_dhz8l1") for label in labels]

    z = []
    for seed in range(16):
        rng = np.random.default_rng(200 + seed)
        series = []
        for item in clean:
            measurement = item.measurements[0]
            noisy = measurement.orbit.copy()
            noisy[["X", "Y"]] += rng.normal(0.0, noise, (len(noisy), 2))
            noisy[["ERRX", "ERRY"]] = noise
            exact = measurement.reference.assign(ERRX=0.0, ERRY=0.0)  # only the signal carries noise
            series.append(
                ClosedOrbitSeries((ClosedOrbitMeasurement(noisy, reference=exact),), machine_state=item.machine_state, label=item.label)
            )
        # The BPM gains are pinned so no corrector-gain direction is set by a prior (the scatter would then fall below sigma).
        fitter = _fitter(seq_k, series, sigma_corrector=0.5, sigma_bpm=1e-6, knob_sigmas={"dx": 1e-9, "dy": 1e-9})
        try:
            result = fitter.run()
        finally:
            fitter.close()
        for name, expected in (("corrgain.DHZ2L4", 0.1), ("corrgain.DHZ8L1", 0.0)):
            assert result.extra_uncertainties[name] > 0.0
            z.append((result.extra[name] - expected) / result.extra_uncertainties[name])

    assert 0.6 < float(np.std(z)) < 1.5, f"std of z = {np.std(z):.3f}"
    assert abs(float(np.mean(z))) < 0.6, f"mean of z = {np.mean(z):.3f}"
