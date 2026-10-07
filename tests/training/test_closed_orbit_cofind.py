"""The orbit-only path: ``cofind`` plus one parametric ``track`` instead of ``twiss``.

``twiss`` also normalises the parametric one-turn map and fills every optical-function
column, none of which an x/y observable reads. When only orbit coordinates are observed
the worker calls ``compute_closed_orbit`` (``run_closed_twiss_init.mad``), which finds the
closed orbit with ``cofind`` and tracks once a first-order map already seeded with the
closed orbit's knob dependence ``dX0/dk = (I-R)^-1 dM/dk``.

What has to hold, test by test:

``test_orbit_only_path_matches_twiss``
    The whole point of the substitution: the orbit and its knob Jacobian equal what
    ``twiss`` returns, for every steering family and every orbit coordinate, at a non-zero
    momentum where the orbit is dispersive. (The finite-difference test in
    ``test_closed_twiss_fitter`` separately pins the Jacobian to an independent twiss.)

``test_orbit_only_map_is_first_order``
    ``d(orbit)/d(knob)`` is a pure-parameter monomial, order 1, so ``mo=1`` suffices; the
    optics path still needs ``mo=2``.

``test_worker_picks_the_solver_from_its_observables``
    Any optical function -- including phase advance -- keeps ``twiss``.

``test_lost_closed_orbit_is_reported_and_recovered_from``
    A knob setting with no closed orbit returns ``false`` (which the optimiser turns into a
    backtrack) and leaves no state behind: the next call at sane knobs succeeds.

``test_orbit_only_fit_reproduces_the_measured_orbit``
    End to end through the real fitter.
"""

from __future__ import annotations

import multiprocessing as mp
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.config import SimulationConfig
from aba_optimiser.fitting.worker import WorkerConfig
from aba_optimiser.machine.accelerators import PSB
from aba_optimiser.machine.mad import GradientDescentMadInterface
from aba_optimiser.machine.mad.scripts import CLOSED_TWISS_INIT, PYTHON_IN_MAD
from aba_optimiser.poco.workers.closed_orbit import (
    ClosedOrbitMeasurementData,
    ClosedOrbitSeriesData,
    ClosedOrbitWorker,
)
from aba_optimiser.poco.workers.closed_twiss import (
    ClosedTwissData,
    ClosedTwissWorker,
    Observable,
    read_orbit_only,
)

from .test_closed_twiss_fitter import DELTAS, _fake_measurement, _fit, _knob_names

if TYPE_CHECKING:
    from pathlib import Path

    from pymadng import MAD

pytestmark = pytest.mark.serial

COORDS = ["x", "px", "y", "py"]


def _send_init(iface: GradientDescentMadInterface, knobs: list[str], optics_columns: list[str], delta: float) -> None:
    """Install the closed-twiss init script exactly as a worker does."""
    mad = iface.mad
    mad["nbpms"] = iface.nbpms
    mad["knob_names"] = knobs
    mad["optics_columns"] = optics_columns
    mad["orbit_coords"] = COORDS
    mad.send(
        "\n".join(
            line
            for line in CLOSED_TWISS_INIT.read_text().splitlines()
            if line.strip() and not line.strip().startswith(("--", "!"))
        )
    )
    mad.send(f"x0map.pt:set0({delta:.15e})")


def _solve(mad: MAD, solver: str, n_knobs: int):
    """Run a solver and return (names, orbit (coord -> array), jacobian (coord -> matrix))."""
    mad.send(f"{solver}()")
    assert mad.recv(), f"{solver} failed"
    if solver == "compute_closed_orbit":
        names, orbit, jac = read_orbit_only(mad, len(COORDS), n_knobs)
        return names, dict(zip(COORDS, orbit)), dict(zip(COORDS, jac))
    frame = mad.closed_tbl.to_df(columns=["name", *COORDS])
    n_bpms = len(frame)
    mad.send("send_orbit_jacobian()")
    jacobian = {
        coord: np.asarray(mad.recv(), dtype=float).reshape(n_bpms, n_knobs) for coord in COORDS
    }
    return list(frame["name"]), {c: frame[c].to_numpy() for c in COORDS}, jacobian


@pytest.mark.slow
@pytest.mark.parametrize(
    ("family", "delta"),
    [("dx", 3e-3), ("dy", 3e-3), ("tilt", 3e-3), ("dx", 0.0)],
    ids=["dx", "dy", "tilt", "dx-on-momentum"],
)
def test_orbit_only_path_matches_twiss(seq_psb: Path, family: str, delta: float) -> None:
    """Orbit values and knob Jacobians from cofind+track equal those from twiss."""
    kwargs = {"misalignments": {"quad": {family}}}
    iface = GradientDescentMadInterface(
        PSB(ring=3, sequence_file=seq_psb, **kwargs), py_name=PYTHON_IN_MAD
    )
    knobs = [k for k in iface.knob_names if k != "pt"][:6]
    # Ask twiss for one optical function so the map is order 2 and both solvers can run
    # on the very same damap definition; the orbit path ignores the optics column.
    _send_init(iface, knobs, ["beta11_"], delta)

    names_orbit, orbit_orbit, jac_orbit = _solve(iface.mad, "compute_closed_orbit", len(knobs))
    names_twiss, orbit_twiss, jac_twiss = _solve(iface.mad, "compute_closed_twiss", len(knobs))

    assert names_orbit == names_twiss, "the two paths must observe the same BPMs in order"
    for coord in COORDS:
        orbit_scale = max(1e-12, float(np.max(np.abs(orbit_twiss[coord]))))
        jac_scale = max(1e-12, float(np.max(np.abs(jac_twiss[coord]))))
        assert np.max(np.abs(orbit_orbit[coord] - orbit_twiss[coord])) < 1e-8 * max(
            orbit_scale, 1e-3
        ), f"closed orbit {coord} differs between cofind+track and twiss"
        assert np.max(np.abs(jac_orbit[coord] - jac_twiss[coord])) < 1e-8 * max(jac_scale, 1e-3), (
            f"d({coord})/d(knob) differs between cofind+track and twiss"
        )

    # Not vacuous: the family must actually steer the plane it drives.
    driven = "x" if family == "dx" else "y"
    assert np.max(np.abs(jac_twiss[driven])) > 1e-2


def _map_order(mad: MAD) -> int:
    mad.send(f"{PYTHON_IN_MAD}:send(x0map:maxord())")
    return int(mad.recv())


def test_orbit_only_map_is_first_order(seq_psb: Path) -> None:
    """Orbit-only needs mo=1; observing an optical function still needs mo=2."""
    kwargs = {"misalignments": {"quad": {"dx"}}}
    for optics_columns, expected in (([], 1), (["beta11_"], 2)):
        iface = GradientDescentMadInterface(
            PSB(ring=3, sequence_file=seq_psb, **kwargs), py_name=PYTHON_IN_MAD
        )
        knobs = [k for k in iface.knob_names if k != "pt"][:3]
        _send_init(iface, knobs, optics_columns, 0.0)
        assert _map_order(iface.mad) == expected, optics_columns
        del iface


def test_every_solver_pins_the_integrator_to_method_6() -> None:
    """cofind, track and twiss each take their own ``method``; none may fall back to a default."""
    script = CLOSED_TWISS_INIT.read_text()
    for command in ("twiss", "cofind", "track"):
        blocks = re.findall(rf"\b{command}\s*\{{(.*?)\}}", script, flags=re.DOTALL)
        assert blocks, f"no {command}{{}} call found in the init script"
        for block in blocks:
            assert re.search(r"method\s*=\s*6", block), f"{command} call lacks method = 6"


def _worker_config(seq_psb: Path) -> WorkerConfig:
    return WorkerConfig(
        accelerator=PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}}),
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range="$start/$end",
    )


def _observables(names: tuple[str, ...], n_bpms: int = 4) -> list[Observable]:
    return [
        Observable(
            name=name,
            targets=np.zeros(n_bpms - 1 if name in {"mu1", "mu2"} else n_bpms),
            variances=np.full(n_bpms - 1 if name in {"mu1", "mu2"} else n_bpms, 1e-8),
        )
        for name in names
    ]


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("x",), "compute_closed_orbit()"),
        (("x", "y"), "compute_closed_orbit()"),
        (("x", "px", "y", "py"), "compute_closed_orbit()"),
        (("x", "betx"), "compute_closed_twiss()"),
        (("y", "dy"), "compute_closed_twiss()"),
        (("betx", "bety", "mu1", "mu2"), "compute_closed_twiss()"),
    ],
)
def test_worker_picks_the_solver_from_its_observables(
    seq_psb: Path, names: tuple[str, ...], expected: str
) -> None:
    """Only a fit with no optical function to normalise for skips twiss."""
    bpms = ["BPM1", "BPM2", "BPM3", "BPM4"]
    data = ClosedTwissData(bpm_names=bpms, observables=_observables(names), pt=0.0)
    _parent, child = mp.Pipe()
    worker = ClosedTwissWorker(
        child,
        0,
        data,
        _worker_config(seq_psb),
        SimulationConfig(num_workers=1, num_batches=1),
    )
    assert worker.closed_solver == expected


def test_phase_only_orbit_series_keeps_twiss(seq_psb: Path) -> None:
    """A ClosedOrbitWorker series is orbit-only or phase-only; the phase one needs twiss."""
    bpms = ["BPM1", "BPM2", "BPM3", "BPM4"]
    _parent, child = mp.Pipe()
    for names, expected in ((("x", "y"), "compute_closed_orbit()"), (("mu1", "mu2"), "compute_closed_twiss()")):
        series = ClosedOrbitSeriesData(
            bpm_names=bpms,
            measurements=[ClosedOrbitMeasurementData(observables=_observables(names))],
            absolute_planes=("x", "y") if names[0] == "x" else (),
        )
        worker = ClosedOrbitWorker(
            child, 0, [series], _worker_config(seq_psb), SimulationConfig(num_workers=1, num_batches=1)
        )
        assert worker.closed_solver == expected, names


@pytest.mark.slow
def test_lost_closed_orbit_is_reported_and_recovered_from(seq_psb: Path) -> None:
    """No closed orbit -> ``false`` from both solvers; the next sane call still works.

    The optimiser turns the ``false`` into a NaN loss and backtracks, so it must be
    reported, and it must not poison the call that follows.

    A linear lattice always has a closed orbit, even an optically unstable one, so knob
    values cannot force a loss short of crashing MAD-NG. A momentum deviation of
    ``pt = 10`` does. ``x0map.pt`` is the one lever both solvers read (``compute_closed_orbit``
    seeds its plain ``cofind`` from it alone), so it hits them identically.
    """
    kwargs = {"misalignments": {"quad": {"dx"}}}
    iface = GradientDescentMadInterface(
        PSB(ring=3, sequence_file=seq_psb, **kwargs), py_name=PYTHON_IN_MAD
    )
    mad = iface.mad
    _send_init(iface, [next(k for k in iface.knob_names if k != "pt")], ["beta11_"], 0.0)

    for solver in ("compute_closed_orbit", "compute_closed_twiss"):
        mad.send("x0map.pt:set0(10)")
        mad.send(f"{solver}()")
        assert not mad.recv(), f"{solver} should report a lost closed orbit from x0 = 10 m"

        mad.send("x0map.pt:set0(0)")
        mad.send(f"{solver}()")
        assert mad.recv(), f"{solver} did not recover after a lost closed orbit"
    del iface


@pytest.mark.slow
def test_orbit_only_fit_reproduces_the_measured_orbit(seq_psb: Path) -> None:
    """A real ``ClosedTwissFitter`` fit on x and y alone, run through cofind+track.

    48 horizontal misalignments against 3 x 16 orbit points is under-determined, so the
    assertion is on the *reproduced orbit*, not on the knob vector -- the same stance the
    vertical-dispersion test takes.
    """
    kwargs = {"misalignments": {"quad": {"dx"}}}
    knobs = _knob_names(seq_psb, kwargs)
    rng = np.random.default_rng(3)
    truth = {k: float(v) for k, v in zip(knobs, rng.uniform(-5e-5, 5e-5, len(knobs)))}

    measurements = {d: _fake_measurement(seq_psb, truth, d, kwargs) for d in DELTAS}
    measured_rms = float(np.sqrt(np.mean(measurements[0.0]["X"] ** 2)))
    assert measured_rms > 1e-4, f"injected misalignments only gave {measured_rms:.1e} m rms orbit"

    fitted = _fit(seq_psb, measurements, ("x",), kwargs, prior_strength=1e-6)
    for delta in DELTAS:
        refit = _fake_measurement(seq_psb, fitted, delta, kwargs)
        residual = float(np.max(np.abs(refit["X"] - measurements[delta]["X"])))
        assert residual < 1e-2 * measured_rms, (
            f"fitted x residual {residual:.3e} m at delta={delta:+.0e} against a "
            f"measured rms of {measured_rms:.3e} m"
        )
