"""MAD-NG vs xsuite: closed-orbit response to main-quadrupole dx / dy, and the MAD-NG gradients against xsuite finite differences.

The closed-orbit fitter (``run_closed_twiss_init.mad``, ``compute_closed_orbit``) takes dx / dy of every main quadrupole as TPSA
parameters of a MAD-NG damap. Two things have to hold for a fit against xsuite-simulated (or measured) orbits to work:

* the *orbit* MAD-NG predicts for a set of misalignments equals what xsuite tracks (``test_orbit_agrees_with_xsuite``);
* the *Jacobian* d(orbit)/d(dx, dy) MAD-NG returns equals the finite-difference slope of xsuite's closed orbit
  (``test_jacobian_matches_xsuite_finite_difference`` and ``test_directional_derivative_matches_xsuite``).

The orbit is compared through the BPMs, in the same order MAD-NG reports them. dx/dy are in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
xtt = pytest.importorskip("xtrack_tools")

from adelmo.machine.accelerators import LHC  # noqa: E402
from adelmo.machine.accelerators.base import MagnetFamily  # noqa: E402
from adelmo.machine.mad import GradientDescentMadInterface  # noqa: E402
from adelmo.machine.mad.scripts import CLOSED_TWISS_INIT, PYTHON_IN_MAD  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

KINETIC_ENERGY = 450.0
COORDS = ("x", "y")
ERROR_RMS = 100e-6  # rms of the random dx, dy [m]
STEP = 1e-5  # finite-difference step of a dx / dy [m]
N_FD_KNOBS = 8


class MainQuadDxDyLHC(LHC):
    """LHC whose main quadrupoles (``MQ.``) have free dx and dy."""

    FAMILIES = {
        **LHC.FAMILIES,
        "quad": MagnetFamily(
            {"quadrupole": LHC.PATTERN_MAIN_QUAD},
            errors=frozenset({"k1"}),
            misalignments=frozenset({"dx", "dy"}),
            nonzero_attr="k1",
            attr_patterns={"dx": LHC.PATTERN_MAIN_QUAD, "dy": LHC.PATTERN_MAIN_QUAD},
        ),
    }


@dataclass
class Orbit:
    """Closed orbit at the BPMs [m]: ``x`` / ``y`` values; ``jac_x`` / ``jac_y`` are (n_bpm, n_knob) when MAD-NG gives them."""

    bpms: list[str]
    x: np.ndarray
    y: np.ndarray
    jac_x: np.ndarray | None = None
    jac_y: np.ndarray | None = None


class MadngOrbit:
    """The fitter's orbit-only MAD-NG solve (``compute_closed_orbit``), driven directly."""

    def __init__(self, sequence: Path) -> None:
        accelerator = MainQuadDxDyLHC(
            beam=1, sequence_file=sequence, kinetic_energy=KINETIC_ENERGY, misalignments={"quad": {"dx", "dy"}}
        )
        self.iface = GradientDescentMadInterface(accelerator, py_name=PYTHON_IN_MAD)
        self.knobs = list(self.iface.knob_names)
        mad = self.iface.mad
        kept = [
            line
            for line in CLOSED_TWISS_INIT.read_text().splitlines()
            if line.strip() and not line.strip().startswith(("--", "!"))
        ]
        mad["optics_columns"] = []  # orbit only, as the fitter does for x / y observables
        mad["orbit_coords"] = list(COORDS)
        mad["nbpms"] = self.iface.nbpms
        mad.send("\n".join(kept))
        mad.send("x0map.pt:set0(0.0)")

    def orbit(self, values: dict[str, float]) -> Orbit:
        """Closed orbit and Jacobian with every knob not in ``values`` at zero."""
        mad = self.iface.mad
        full = dict.fromkeys(self.knobs, 0.0) | values
        mad.send("\n".join(f"loaded_sequence['{k}']:set0({v:.15e})" for k, v in full.items()))
        mad.send("compute_closed_orbit()")
        assert mad.recv(), "MAD-NG lost the closed orbit"
        mad.send("send_orbit_only()")
        bpms = [str(name).lower() for name in mad.recv()]
        out = {}
        for coord in COORDS:
            out[coord] = np.asarray(mad.recv(), dtype=float).ravel()
            out[f"jac_{coord}"] = np.asarray(mad.recv(), dtype=float).reshape(len(bpms), len(self.knobs))
        return Orbit(bpms, **out)

    def close(self) -> None:
        self.iface.close()


class XsuiteOrbit:
    """Closed 4D orbit of the xsuite line built from the same sequence, with dx / dy as element shifts."""

    def __init__(self, sequence: Path) -> None:
        env = xtt.create_xsuite_environment(sequence_file=sequence, kinetic_energy=KINETIC_ENERGY, seq_name="lhcb1")
        self.line = env["lhcb1"]

    def set_errors(self, values: dict[str, float], knobs: list[str]) -> None:
        """Set every knob (``MQ.11R2.B1.dx``) to its value in ``values``, zero if absent."""
        for knob in knobs:
            name, plane = knob.rsplit(".", 1)
            setattr(self.line.element_dict[name.lower()], "shift_x" if plane == "dx" else "shift_y", values.get(knob, 0.0))

    def orbit(self, bpms: list[str]) -> Orbit:
        rows = self.line.twiss(method="4d", only_orbit=True).rows[bpms]
        return Orbit(bpms, np.asarray(rows.x, dtype=float), np.asarray(rows.y, dtype=float))


@pytest.fixture(scope="module")
def models(seq_b1: Path):
    madng = MadngOrbit(seq_b1)
    xsuite = XsuiteOrbit(seq_b1)
    yield madng, xsuite
    madng.close()


@pytest.fixture(scope="module")
def errors(models) -> dict[str, float]:
    """Gaussian dx and dy of every main quadrupole."""
    madng, _ = models
    rng = np.random.default_rng(1)
    return {knob: float(rng.normal(0.0, ERROR_RMS)) for knob in madng.knobs}


def test_knobs_are_dx_dy_of_main_quads(models) -> None:
    madng, _ = models
    assert madng.knobs
    assert all(k.startswith("MQ.") and k.endswith((".dx", ".dy")) for k in madng.knobs)


@pytest.mark.parametrize("coord", COORDS)
def test_orbit_agrees_with_xsuite(models, errors, coord: str) -> None:
    """The orbit change caused by the errors, MAD-NG vs xsuite, BPM by BPM."""
    madng, xsuite = models
    reference = madng.orbit({})
    perturbed = madng.orbit(errors)
    xsuite.set_errors({}, madng.knobs)
    xs_reference = getattr(xsuite.orbit(perturbed.bpms), coord)
    xsuite.set_errors(errors, madng.knobs)
    xs_perturbed = getattr(xsuite.orbit(perturbed.bpms), coord)
    xsuite.set_errors({}, madng.knobs)

    madng_change = getattr(perturbed, coord) - getattr(reference, coord)
    xsuite_change = xs_perturbed - xs_reference
    signal = np.std(xsuite_change)
    assert signal > 1e-4, "the errors must move the orbit by far more than the tolerance"
    # Absolute orbits agree as well (the design orbit of both models is ~0)
    assert np.std(madng_change - xsuite_change) < 1e-3 * signal
    assert np.max(np.abs(getattr(perturbed, coord) - xs_perturbed)) < 1e-2 * signal


def _fd_knobs(madng: MadngOrbit) -> list[str]:
    """dx and dy of quadrupoles spread around the ring."""
    quads = sorted({k.rsplit(".", 1)[0] for k in madng.knobs})
    picked = quads[:: max(1, len(quads) // (N_FD_KNOBS // 2))][: N_FD_KNOBS // 2]
    return [f"{q}.{plane}" for q in picked for plane in ("dx", "dy")]


@pytest.mark.parametrize("about_errors", [False, True], ids=["at_design", "at_errors"])
def test_jacobian_matches_xsuite_finite_difference(models, errors, about_errors: bool) -> None:
    """Columns of MAD-NG's d(orbit)/d(dx, dy) against xsuite's central finite difference of the closed orbit."""
    madng, xsuite = models
    base = errors if about_errors else {}
    jacobian = madng.orbit(base)
    index = {k: i for i, k in enumerate(madng.knobs)}
    for knob in _fd_knobs(madng):
        orbits = []
        for sign in (+1.0, -1.0):
            xsuite.set_errors(base | {knob: base.get(knob, 0.0) + sign * STEP}, madng.knobs)
            orbits.append(xsuite.orbit(jacobian.bpms))
        xsuite.set_errors({}, madng.knobs)
        for coord in COORDS:
            fd = (getattr(orbits[0], coord) - getattr(orbits[1], coord)) / (2 * STEP)
            mad = getattr(jacobian, f"jac_{coord}")[:, index[knob]]
            scale = np.max(np.abs(fd))
            # A dx barely kicks y (and vice versa): the cross-plane column is zero in both
            assert scale > 0.0 or np.max(np.abs(mad)) < 1e-6
            assert np.max(np.abs(mad - fd)) <= 1e-3 * max(scale, 1.0), f"{knob} -> {coord}"


def test_directional_derivative_matches_xsuite(models, errors) -> None:
    """J v from MAD-NG for a random direction v over *all* dx / dy, against xsuite's central difference along v."""
    madng, xsuite = models
    rng = np.random.default_rng(2)
    direction = {k: float(rng.normal()) for k in madng.knobs}
    v = np.array([direction[k] for k in madng.knobs])
    jacobian = madng.orbit(errors)
    h = STEP
    orbits = []
    for sign in (+1.0, -1.0):
        xsuite.set_errors({k: errors[k] + sign * h * direction[k] for k in madng.knobs}, madng.knobs)
        orbits.append(xsuite.orbit(jacobian.bpms))
    xsuite.set_errors({}, madng.knobs)
    for coord in COORDS:
        fd = (getattr(orbits[0], coord) - getattr(orbits[1], coord)) / (2 * h)
        predicted = getattr(jacobian, f"jac_{coord}") @ v
        assert np.max(np.abs(predicted - fd)) <= 1e-3 * np.max(np.abs(fd)), coord
