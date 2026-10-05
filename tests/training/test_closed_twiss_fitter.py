"""Physics integration tests for the closed-twiss fitter on PSB.

Each test perturbs real magnets, generates a *fake measurement* by twissing the
perturbed machine, and then recovers the perturbation through the real
``ClosedTwissFitter``. Because the synthetic measurement is produced by an
independent twiss of the same lattice, a successful recovery exercises the whole
chain end to end: the parametric ``mo=2`` map, the ``trkopt`` derivative columns,
the phase-advance differencing, the inverse-variance weighting across observables
in different units, and the Gauss-Newton solve.

The physics being asserted, test by test:

``test_quadrupole_gradients_recovered_from_beta_and_phase``
    Beta and phase respond to quadrupole gradients but the closed orbit does not
    (a gradient error steers nothing when the orbit through it is centred), so
    this isolates the newly added optical-function observables. A pure
    closed-orbit fit cannot see these errors at all, which the test asserts
    directly.

``test_vertical_dispersion_requires_a_vertical_source``
    The physics motivating the whole fitter: vertical dispersion is identically
    zero in an ideal flat machine, so a non-zero measured ``Dy`` can only be fitted
    by a knob that deflects vertically. Quadrupole ``dy`` feed-down is such a
    source; quadrupole gradients are not. The test asserts both directions.

``test_observables_are_consistent_with_an_independent_twiss``
    The values the worker reports through ``trkopt`` must equal an ordinary
    twiss of the same machine, and their knob derivatives must equal finite
    differences of it. This is the unit-level guard that the monomial encoding
    selects the derivative it claims to.

``test_orbit_jacobian_matches_finite_differences``
    The same guard for the closed orbit, which takes the *other* code path (saved
    map rather than ``trkopt``), over every steering family: ``dx``, ``dy`` and
    ``tilt``. A wrong Jacobian here does not error, it just quietly stops the
    orbit from being fitted.

``test_second_momentum_removes_the_null_space``
    What a second momentum actually buys, measured on the normal matrix rather
    than assumed.

``test_quadrupole_tilts_recovered_from_vertical_dispersion``
    An injected tilt pattern recovered from the vertical orbit and dispersion.

``test_all_observables_fitted_simultaneously``
    The whole point: orbit, beta, alpha, phase and dispersion driven to agreement
    in one solve, against a machine perturbed in gradient *and* alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.mad.scripts import CLOSED_TWISS_INIT, PYTHON_IN_MAD
from aba_optimiser.training.config.models import SequenceConfig
from aba_optimiser.training.workers.lifecycle import WorkerLifecycleManager
from aba_optimiser.training_closed_twiss import (
    DEFAULT_OBSERVABLES,
    ClosedTwissFitter,
    LevenbergMarquardtConfig,
)
from aba_optimiser.workers import ClosedTwissWorker
from aba_optimiser.workers.closed_twiss import read_orbit_only

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

DELTAS = (-3e-3, 0.0, 3e-3)

#: Model columns to copy into a fake measurement, as measurement column -> twiss column.
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


def _fake_measurement(
    seq_psb: Path,
    knob_values: dict[str, float],
    delta: float,
    accelerator_kwargs: dict,
) -> pd.DataFrame:
    """Twiss the perturbed machine and present the result as an omc3-style measurement.

    ``delta`` is pinned on the initial ``pt`` so twiss returns the off-momentum
    periodic solution - the same fixed-momentum mechanism the worker uses.

    The phase columns are the *cumulative* phase, matching what
    ``build_twiss_from_measurements`` produces; the fitter differences them into
    the per-interval advances it actually fits.
    """
    accel = PSB(ring=3, sequence_file=seq_psb, **accelerator_kwargs)
    iface = GradientDescentMadInterface(accel)
    iface.update_knob_values(knob_values)
    iface.mad.send(
        f"motws = twiss{{sequence=loaded_sequence, observe=1, "
        f"X0={{pt={delta:.15e}}}, coupling=true, method=6}}"
    )
    twiss = iface.mad.motws.to_df(
        columns=["name", *sorted(set(MEASURED_FROM_TWISS.values()))]
    ).set_index("name")
    del iface

    measurement = pd.DataFrame(
        {measured: twiss[modelled] for measured, modelled in MEASURED_FROM_TWISS.items()},
        index=twiss.index,
    )
    # A noiseless measurement still needs weights. Use a constant, realistic
    # resolution per family so the inverse-variance weighting is exercised
    # (rather than silently falling back to the no-errors branch).
    measurement["ERRX"] = 5e-5
    measurement["ERRY"] = 5e-5
    measurement["ERRBETX"] = 1e-2
    measurement["ERRBETY"] = 1e-2
    measurement["ERRALFX"] = 1e-2
    measurement["ERRALFY"] = 1e-2
    measurement["ERRDX"] = 1e-3
    measurement["ERRDY"] = 1e-3
    # The phase "error" columns are cumulative variances, as produced upstream.
    measurement["mu1_var"] = np.arange(len(measurement)) * (1e-4) ** 2
    measurement["mu2_var"] = np.arange(len(measurement)) * (1e-4) ** 2
    return measurement


def _knob_names(seq_psb: Path, accelerator_kwargs: dict) -> list[str]:
    """Optimisable knob names for a given PSB configuration (excluding momentum)."""
    probe = GradientDescentMadInterface(PSB(ring=3, sequence_file=seq_psb, **accelerator_kwargs))
    names = [k for k in probe.knob_names if k != "pt"]
    del probe
    return names


def _fit(
    seq_psb: Path,
    measurements: dict[float, pd.DataFrame],
    observables: tuple[str, ...],
    accelerator_kwargs: dict,
    prior_strength: float = 0.0,
) -> dict[str, float]:
    """Run the real ``ClosedTwissFitter`` and return the recovered knobs."""
    families = {
        family
        for selection, magnets, attr, family in (
            ("errors", "quad", "k1", "dk1l"),
            ("errors", "bend", "k0", "dk0l"),
            ("misalignments", "quad", "dx", "dx"),
            ("misalignments", "quad", "dy", "dy"),
            ("misalignments", "quad", "tilt", "tilt"),
        )
        if attr in accelerator_kwargs.get(selection, {}).get(magnets, ())
    }
    fitter = ClosedTwissFitter(
        accelerator=PSB(ring=3, sequence_file=seq_psb, **accelerator_kwargs),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        lm_config=LevenbergMarquardtConfig(max_iterations=40, gradient_converged_value=1e-12),
        measurements=measurements,
        observables=observables,
        prior_strengths=(
            dict.fromkeys(families, prior_strength) if prior_strength > 0.0 else None
        ),
    )
    final_knobs, _ = fitter.run()
    return final_knobs


@pytest.mark.slow
def test_quadrupole_gradients_recovered_from_beta_and_phase(seq_psb: Path) -> None:
    """Beta and phase identify quadrupole gradients that the closed orbit cannot see.

    A gradient error on a centred orbit produces no deflection, so the closed
    orbit is blind to it while the optics are not. Recovering the perturbation
    from ``betx/bety/mu1/mu2`` therefore proves the optical-function
    observables carry real, independent information - and the orbit-only control
    fit proves it was information the previous fitter did not have.
    """
    kwargs = {"errors": {"quad": {"k1"}}}
    quads = _knob_names(seq_psb, kwargs)
    assert len(quads) > 2

    rng = np.random.default_rng(0)
    truth = {k: float(v) for k, v in zip(quads, rng.uniform(-2e-3, 2e-3, len(quads)))}
    truth_vec = np.array([truth[k] for k in quads])
    truth_norm = float(np.linalg.norm(truth_vec))

    measurements = {d: _fake_measurement(seq_psb, truth, d, kwargs) for d in DELTAS}

    # The perturbation must actually move the optics, or the test is vacuous.
    nominal = _fake_measurement(seq_psb, dict.fromkeys(quads, 0.0), 0.0, kwargs)
    beta_beating = float(np.max(np.abs(measurements[0.0]["BETX"] / nominal["BETX"] - 1.0)))
    assert beta_beating > 0.01, f"perturbation only produced {beta_beating:.1%} beta beating"

    optics = _fit(seq_psb, measurements, ("betx", "bety", "mu1", "mu2"), kwargs)
    optics_err = float(np.linalg.norm(np.array([optics[k] for k in quads]) - truth_vec))
    assert optics_err < 0.05 * truth_norm, (
        f"beta+phase recovery error {optics_err:.3e} is not small against |k_true|={truth_norm:.3e}"
    )

    # Control: the closed orbit alone carries no gradient information, so an
    # orbit-only fit stays at the (zero) starting point instead of recovering.
    orbit_only = _fit(seq_psb, measurements, ("x", "y"), kwargs)
    orbit_err = float(np.linalg.norm(np.array([orbit_only[k] for k in quads]) - truth_vec))
    assert orbit_err > 0.5 * truth_norm, (
        "closed orbit alone should not identify quadrupole gradients, but recovered "
        f"to within {orbit_err:.3e} of |k_true|={truth_norm:.3e}"
    )


@pytest.mark.slow
def test_vertical_dispersion_requires_a_vertical_source(seq_psb: Path) -> None:
    """Measured ``Dy`` is fitted by quad ``dy`` feed-down, and not by gradients.

    An ideal flat lattice has ``Dy`` identically zero; the only way the model can
    reproduce a measured vertical dispersion is a knob that deflects vertically.
    Quadrupole vertical misalignment is such a knob (feed-down ``theta_y =
    -k1*L*dy``, whose deflection scales as ``1/(1+delta)`` and so is dispersive).
    Quadrupole *gradients* are not, no matter how they are set.

    The perturbation scale is chosen to reproduce the machine: 50 um of quad
    misalignment gives ~0.09 m peak ``Dy``, which is what PSB ring 3 actually
    measures. The assertion is on the *reproduced dispersion*, not on the knob
    vector, because 48 misalignments against 17 BPMs is genuinely
    under-determined - the null space is large, and which member of it the solver
    lands on is a property of the regularisation rather than of the physics.
    """
    dy_kwargs = {"misalignments": {"quad": {"dy"}}}
    dy_knobs = _knob_names(seq_psb, dy_kwargs)
    assert len(dy_knobs) > 2

    rng = np.random.default_rng(1)
    truth = {k: float(v) for k, v in zip(dy_knobs, rng.uniform(-5e-5, 5e-5, len(dy_knobs)))}
    truth_norm = float(np.linalg.norm(list(truth.values())))

    measurements = {d: _fake_measurement(seq_psb, truth, d, dy_kwargs) for d in DELTAS}

    # The premise: the perturbed machine has vertical dispersion, the ideal one
    # has exactly none. Without this the test would assert nothing.
    nominal = _fake_measurement(seq_psb, dict.fromkeys(dy_knobs, 0.0), 0.0, dy_kwargs)
    assert np.allclose(nominal["DY"], 0.0, atol=1e-12), "ideal PSB should have Dy == 0"
    measured_dy_rms = float(np.sqrt(np.mean(measurements[0.0]["DY"] ** 2)))
    measured_y_rms = float(np.sqrt(np.mean(measurements[0.0]["Y"] ** 2)))
    assert 0.03 < measured_dy_rms < 0.15, (
        f"Dy rms {measured_dy_rms:.3f} m is outside the measured PSB range"
    )

    # Fitting y and dy together recovers the misalignments themselves, not merely
    # some combination that reproduces the data: the vertical orbit and the
    # vertical dispersion at each BPM are two independent functionals of the same
    # knobs, which between them pin the individual magnets down.
    fitted = _fit(seq_psb, measurements, ("y", "dy"), dy_kwargs)
    fitted_vec = np.array([fitted[k] for k in dy_knobs])
    truth_vec = np.array([truth[k] for k in dy_knobs])
    knob_error = float(np.linalg.norm(fitted_vec - truth_vec))
    assert knob_error < 0.05 * truth_norm, (
        f"misalignment recovery error {knob_error:.3e} is not small against "
        f"|k_true|={truth_norm:.3e}"
    )

    # And the refitted machine reproduces both measured quantities.
    refit = _fake_measurement(seq_psb, fitted, 0.0, dy_kwargs)
    residual_dy = float(np.max(np.abs(refit["DY"] - measurements[0.0]["DY"])))
    residual_y = float(np.max(np.abs(refit["Y"] - measurements[0.0]["Y"])))
    assert residual_dy < 1e-3 * measured_dy_rms, (
        f"fitted Dy residual {residual_dy:.3e} m against a measured rms of {measured_dy_rms:.3e} m"
    )
    assert residual_y < 1e-3 * measured_y_rms, (
        f"fitted y residual {residual_y:.3e} m against a measured rms of {measured_y_rms:.3e} m"
    )

    # Quadrupole gradients cannot produce vertical dispersion at all: whatever the
    # fit does with them, the model's Dy stays at exactly zero.
    k1_kwargs = {"errors": {"quad": {"k1"}}}
    k1_fit = _fit(seq_psb, measurements, ("dy",), k1_kwargs, prior_strength=1e-6)
    k1_refit = _fake_measurement(seq_psb, k1_fit, 0.0, k1_kwargs)
    assert np.allclose(k1_refit["DY"], 0.0, atol=1e-12), (
        "quadrupole gradients must not generate vertical dispersion; the fit "
        "appears to have found a spurious vertical source"
    )


@pytest.mark.slow
def test_observables_are_consistent_with_an_independent_twiss(seq_psb: Path) -> None:
    """Worker observables match a plain twiss, and their derivatives match finite differences.

    ``trkopt`` names carry the knob monomial in the name string, so a mis-built
    monomial would silently return a different derivative (or the value itself).
    Checking against central differences of an ordinary twiss pins the encoding
    down independently of the fit.
    """
    kwargs = {"errors": {"quad": {"k1"}}}
    knobs = _knob_names(seq_psb, kwargs)[:3]

    # The init script talks to the MAD-side object under the name the workers
    # give it, so the interface has to be built the same way here.
    iface = GradientDescentMadInterface(
        PSB(ring=3, sequence_file=seq_psb, **kwargs), py_name=PYTHON_IN_MAD
    )
    mad = iface.mad
    mad["knob_names"] = knobs
    names = ["beta11_", "dx_", "mu1_"]
    columns = names + [
        f"{name}{'0' * i}1{'0' * (len(knobs) - i - 1)}" for name in names for i in range(len(knobs))
    ]
    mad["optics_columns"] = columns
    mad["orbit_coords"] = []
    mad.send(
        "\n".join(
            line
            for line in CLOSED_TWISS_INIT.read_text().splitlines()
            if line.strip() and not line.strip().startswith(("--", "!"))
        )
    )
    mad.send("compute_closed_twiss()")
    assert mad.recv(), "parametric closed twiss failed on the nominal machine"
    frame = mad.closed_tbl.to_df(columns=["name", *columns])

    # Values must equal an ordinary twiss of the same machine.
    reference = _fake_measurement(seq_psb, dict.fromkeys(knobs, 0.0), 0.0, kwargs)
    assert np.allclose(frame["beta11_"], reference["BETX"], rtol=1e-10)
    assert np.allclose(frame["dx_"], reference["DX"], rtol=1e-10)

    # Derivatives must equal central differences of that twiss.
    step = 1e-6
    for i, knob in enumerate(knobs):
        forward = _fake_measurement(seq_psb, {knob: step}, 0.0, kwargs)
        backward = _fake_measurement(seq_psb, {knob: -step}, 0.0, kwargs)
        column = f"beta11_{'0' * i}1{'0' * (len(knobs) - i - 1)}"
        analytic = frame[column].to_numpy(dtype=float)
        numeric = (forward["BETX"].to_numpy() - backward["BETX"].to_numpy()) / (2 * step)
        scale = float(np.max(np.abs(numeric)))
        assert np.max(np.abs(analytic - numeric)) < 1e-5 * scale, (
            f"d(beta11)/d({knob}) disagrees with finite differences"
        )
    del iface


def _normal_matrix(
    seq_psb: Path,
    measurements: dict[float, pd.DataFrame],
    observables: tuple[str, ...],
    accelerator_kwargs: dict,
) -> tuple[np.ndarray, list[str]]:
    """Physical normal matrix ``JᵀWJ`` at the nominal knobs, as the fitter builds it."""
    fitter = ClosedTwissFitter(
        accelerator=PSB(ring=3, sequence_file=seq_psb, **accelerator_kwargs),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        measurements=measurements,
        observables=observables,
    )
    manager = WorkerLifecycleManager(ClosedTwissWorker)
    try:
        manager.create_and_start_workers(
            [(data, config, fitter.simulation_config) for config, data in fitter.worker_payloads],
            send_handshake=False,
        )
        names = list(fitter.config_manager.knob_names)
        *_, normal_matrix, _ = fitter._collect_gn(
            manager.channels, dict.fromkeys(names, 0.0), names
        )
        return normal_matrix, names
    finally:
        manager.terminate_workers()
        iface = getattr(fitter.config_manager, "mad_iface", None)
        if iface is not None:
            iface.close()


def _identifiable_rank(normal_matrix: np.ndarray, tolerance: float) -> int:
    """Number of knob directions the data constrains, relative to the best one."""
    eigenvalues = np.clip(np.linalg.eigvalsh((normal_matrix + normal_matrix.T) / 2), 0.0, None)
    return int(np.sum(eigenvalues > eigenvalues.max() * tolerance))


@pytest.mark.slow
def test_second_momentum_removes_the_null_space(seq_psb: Path) -> None:
    """One momentum leaves an exact null space; a second one closes it.

    At a single momentum the 48 vertical misalignments are measured through two
    functionals per BPM (the orbit and its momentum derivative), which is 34
    numbers against 48 unknowns - so some combinations of magnets are invisible
    and the normal matrix is exactly singular. Measuring at a second momentum
    changes the lattice response itself (the feed-down deflection scales as
    ``1/(1+delta)``), which is what makes those combinations observable.

    The assertion is on the *singularity*, not on a rank count: with a condition
    number in the 1e12 range the number of eigenvalues above any fixed relative
    threshold wobbles, so counting them is not a stable thing to test.
    """
    kwargs = {"misalignments": {"quad": {"dy"}}}
    knobs = _knob_names(seq_psb, kwargs)

    rng = np.random.default_rng(1)
    truth = {k: float(v) for k, v in zip(knobs, rng.uniform(-5e-5, 5e-5, len(knobs)))}
    available = {d: _fake_measurement(seq_psb, truth, d, kwargs) for d in (-3e-3, 3e-3)}

    def smallest_relative_eigenvalue(deltas: tuple[float, ...]) -> float:
        matrix, names = _normal_matrix(
            seq_psb, {d: available[d] for d in deltas}, ("y", "dy"), kwargs
        )
        assert len(names) == len(knobs)
        eigenvalues = np.clip(np.linalg.eigvalsh((matrix + matrix.T) / 2), 0.0, None)
        return float(eigenvalues.min() / eigenvalues.max())

    single = smallest_relative_eigenvalue((-3e-3,))
    both = smallest_relative_eigenvalue((-3e-3, 3e-3))

    assert single < 1e-20, (
        f"a single momentum should leave an exactly singular normal matrix; "
        f"smallest relative eigenvalue is {single:.3e}"
    )
    assert both > 1e-16, (
        f"a second momentum should lift the null space; smallest relative "
        f"eigenvalue is still {both:.3e}"
    )


@pytest.mark.slow
def test_all_observables_fitted_simultaneously(seq_psb: Path) -> None:
    """Orbit, beta, alpha, phase and dispersion are all driven to agreement in one solve.

    The machine is perturbed in two physically distinct ways at once - quadrupole
    *gradients* (which move beta, alpha and phase but not the orbit) and
    quadrupole *vertical alignment* (which moves the vertical orbit and creates
    vertical dispersion) - so no single observable family can account for the
    measurement on its own. Every family must come down together, which is what
    fitting the closed twiss rather than the closed orbit buys.

    The residuals are judged against each family's own measured spread, since the
    families are in different units; that ratio is also exactly what the
    inverse-variance weighting is balancing internally.
    """
    kwargs = {"errors": {"quad": {"k1"}}, "misalignments": {"quad": {"dy"}}}
    knobs = _knob_names(seq_psb, kwargs)

    rng = np.random.default_rng(3)
    truth = {
        name: float(rng.uniform(-5e-5, 5e-5) if name.endswith(".dy") else rng.uniform(-2e-3, 2e-3))
        for name in knobs
    }
    measurements = {d: _fake_measurement(seq_psb, truth, d, kwargs) for d in DELTAS}

    fitted = _fit(seq_psb, measurements, DEFAULT_OBSERVABLES, kwargs, prior_strength=1e-6)
    refit = {d: _fake_measurement(seq_psb, fitted, d, kwargs) for d in DELTAS}

    # Below these the quantity is numerically zero and there is nothing to fit -
    # the horizontal plane is untouched by this perturbation, so its "mismatch" is
    # rounding noise and demanding a further reduction would assert nothing. The
    # requirement there is the opposite one: the fit must not spoil it.
    negligible = {"X": 1e-6, "Y": 1e-6, "DX": 1e-4, "DY": 1e-4}

    nominal = _fake_measurement(seq_psb, dict.fromkeys(knobs, 0.0), 0.0, kwargs)
    improved = []
    for column in ("X", "Y", "BETX", "BETY", "ALFX", "ALFY", "DX", "DY"):
        target = measurements[0.0][column].to_numpy(dtype=float)
        before = float(np.max(np.abs(nominal[column].to_numpy(dtype=float) - target)))
        after = float(np.max(np.abs(refit[0.0][column].to_numpy(dtype=float) - target)))
        floor = negligible.get(column, 1e-4 * float(np.max(np.abs(target))))
        if before <= floor:
            assert after <= floor, (
                f"{column}: started already matched ({before:.3e}) but the fit moved it "
                f"to {after:.3e}"
            )
            continue
        improved.append(column)
        assert after < 0.1 * before, (
            f"{column}: fit left {after:.3e} of an initial {before:.3e} mismatch"
        )

    # Guard the guard: the perturbation must genuinely disturb both the alignment
    # families and the gradient families, or this is not a simultaneous fit.
    assert {"Y", "DY"} <= set(improved), f"vertical families not exercised: {improved}"
    assert {"BETX", "BETY"} <= set(improved), f"gradient families not exercised: {improved}"

    # The phase advances (the observable actually fitted, not the cumulative
    # phase) must agree too, and at every momentum rather than just the middle one.
    for delta in DELTAS:
        for column in ("MUX", "MUY"):
            target = np.diff(measurements[delta][column].to_numpy(dtype=float))
            after = np.max(np.abs(np.diff(refit[delta][column].to_numpy(dtype=float)) - target))
            assert after < 1e-3, f"{column} advances at delta={delta} disagree by {after:.3e} turns"


def _analytic_orbit_jacobian(
    seq_psb: Path, kwargs: dict, delta: float, n_knobs: int = 3
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return the first ``n_knobs`` knobs and d(orbit)/d(knob) from ``compute_closed_orbit``.

    No optics columns are requested, so this is the orbit-only path: ``cofind`` and one
    parametric ``track`` on a first-order map, not ``twiss``.
    """
    iface = GradientDescentMadInterface(
        PSB(ring=3, sequence_file=seq_psb, **kwargs), py_name=PYTHON_IN_MAD
    )
    mad = iface.mad
    knobs = [k for k in iface.knob_names if k != "pt"][:n_knobs]
    mad["knob_names"] = knobs
    mad["optics_columns"] = []
    mad["orbit_coords"] = ["x", "y"]
    mad["nbpms"] = len(iface.all_bpms)
    mad.send(
        "\n".join(
            line
            for line in CLOSED_TWISS_INIT.read_text().splitlines()
            if line.strip() and not line.strip().startswith(("--", "!"))
        )
    )
    # Momentum is a pinned input on the parametric map, as in the worker; never a knob.
    mad.send(f"x0map.pt:set0({delta:.15e})")
    mad.send("compute_closed_orbit()")
    assert mad.recv(), "parametric closed orbit failed on the nominal machine"
    _, _, jac = read_orbit_only(mad, 2, len(knobs))
    jacobian = {"X": jac[0], "Y": jac[1]}
    del iface
    return knobs, jacobian


@pytest.mark.slow
@pytest.mark.parametrize(
    ("family", "driven", "delta", "step", "rtol", "null_is_exact"),
    [
        ("dx", "X", 0.0, 1e-7, 1e-6, True),
        ("dy", "Y", 0.0, 1e-7, 1e-6, True),
        ("tilt", "Y", 3e-3, 1e-6, 1e-5, False),
    ],
    ids=["dx", "dy", "tilt"],
)
def test_orbit_jacobian_matches_finite_differences(
    seq_psb: Path,
    family: str,
    driven: str,
    delta: float,
    step: float,
    rtol: float,
    null_is_exact: bool,
) -> None:
    """d(closed orbit)/d(knob) from the saved map equals central differences of a twiss.

    Covers every steering family: they reach the map by different routes
    (translation vs rotation) and a wrong Jacobian raises nothing. The off-plane
    column is checked too - exactly zero for a misalignment, second order in the
    angle for a tilt, which is why it needs ``pt != 0`` to steer at all.
    """
    kwargs = {"misalignments": {"quad": {family}}}
    null = "Y" if driven == "X" else "X"
    knobs, analytic = _analytic_orbit_jacobian(seq_psb, kwargs, delta)

    for i, knob in enumerate(knobs):
        forward = _fake_measurement(seq_psb, {knob: step}, delta, kwargs)
        backward = _fake_measurement(seq_psb, {knob: -step}, delta, kwargs)
        numeric = {
            plane: (forward[plane].to_numpy() - backward[plane].to_numpy()) / (2 * step)
            for plane in ("X", "Y")
        }

        scale = float(np.max(np.abs(numeric[driven])))
        assert scale > 1e-4, f"{knob} barely moves the {driven} orbit; test is vacuous"
        # Tolerance is the finite-difference floor: cancellation noise on a
        # derivative of order `scale`.
        assert np.max(np.abs(analytic[driven][:, i] - numeric[driven])) < rtol * scale, (
            f"d({driven})/d({knob}) disagrees with finite differences"
        )

        if null_is_exact:
            # A misalignment in one plane steers nothing in the other.
            assert np.allclose(numeric[null], 0.0, atol=1e-9)
            assert np.allclose(analytic[null][:, i], 0.0, atol=1e-12), (
                f"d({null})/d({knob}) should be identically zero for a misalignment"
            )
        else:
            # The off-plane response is second order in the angle, so at the 1e-9 rad
            # seed both derivatives are ~1e-10, below cofind's convergence noise.
            assert np.max(np.abs(numeric[null])) < 1e-6 * scale
            assert np.max(np.abs(analytic[null][:, i])) < 1e-6 * scale, (
                f"d({null})/d({knob}) is not second order in the tilt angle"
            )


def test_energy_knob_is_rejected(seq_psb: Path) -> None:
    """``pt`` is a per-measurement input, so it must not also be a fitted knob.

    The worker strips ``pt`` from the knob list it hands MAD-NG but the fitter
    aggregates over the unstripped list, so allowing this would silently
    mis-shape every Jacobian reshape by one column.
    """
    accel = PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}}, optimise_energy=True)
    with pytest.raises(ValueError, match="optimise_energy"):
        ClosedTwissFitter(
            accelerator=accel, sequence_config=None, measurements={0.0: pd.DataFrame()}
        )


@pytest.mark.slow
def test_quadrupole_tilts_recovered_from_vertical_dispersion(seq_psb: Path) -> None:
    """An injected tilt pattern is recovered from the vertical orbit and dispersion.

    The acceptance test for the family: the knobs are usable, not merely
    present. 48 knobs against 16 BPMs is under-determined, so the assertions are
    on the *shape* of the pattern and on reproducing the measurement - the
    fitted amplitude comes out at ~0.8 of truth, which is regularisation
    shrinkage. Every observable is needed: ``y`` and ``dy`` alone recover the
    pattern at only 0.66 correlation.
    """
    kwargs = {"misalignments": {"quad": {"tilt"}}}
    knobs = _knob_names(seq_psb, kwargs)
    assert len(knobs) > 2

    rng = np.random.default_rng(2)
    truth = {k: float(v) for k, v in zip(knobs, rng.normal(0.0, 1e-3, len(knobs)))}

    measurements = {d: _fake_measurement(seq_psb, truth, d, kwargs) for d in DELTAS}

    # The premise: tilts make Dy without a matching vertical orbit.
    nominal = _fake_measurement(seq_psb, dict.fromkeys(knobs, 0.0), 0.0, kwargs)
    assert np.allclose(nominal["DY"], 0.0, atol=1e-12), "ideal PSB should have Dy == 0"
    measured_dy_rms = float(np.sqrt(np.mean(measurements[0.0]["DY"] ** 2)))
    assert measured_dy_rms > 0.05, (
        f"injected tilts only produced {measured_dy_rms:.3e} m of Dy; test is vacuous"
    )

    fitted = _fit(seq_psb, measurements, DEFAULT_OBSERVABLES, kwargs, prior_strength=1e-6)
    fitted_vec = np.array([fitted[k] for k in knobs])
    truth_vec = np.array([truth[k] for k in knobs])

    correlation = float(
        np.corrcoef(fitted_vec - fitted_vec.mean(), truth_vec - truth_vec.mean())[0, 1]
    )
    assert correlation > 0.9, (
        f"recovered tilt pattern correlates only {correlation:.3f} with the injected one"
    )

    # And the refitted machine reproduces the measured Dy.
    refit = _fake_measurement(seq_psb, fitted, 0.0, kwargs)
    residual_dy = float(np.max(np.abs(refit["DY"] - measurements[0.0]["DY"])))
    assert residual_dy < 1e-2 * measured_dy_rms, (
        f"fitted Dy residual {residual_dy:.3e} m against a measured rms of {measured_dy_rms:.3e} m"
    )

def test_closed_twiss_worker_uses_method_6():
    """The worker bypasses ``run_twiss`` and must pin the integrator itself."""
    script = CLOSED_TWISS_INIT.read_text()
    assert "method   = 6" in script
