"""A per-series machine state must act on the lattice underneath a live misalignment knob.

BBA of the quadrupoles fits ``dx``/``dy`` while the quadrupole (and corrector) settings
differ between measurements. That only works if assigning a MAD-X global (``kbrqf``,
``kbr3dhz2l4``, ...) after the TPSA knobs exist

* changes the element strengths, because the sequence defines them with ``:=``, and
* leaves ``d(orbit)/d(dx)`` correct at the new setting, with no re-``update()`` of the sequence.

``test_state_variable_drives_element_strength``
    The PSB quadrupole and corrector strengths are deferred expressions of the globals.

``test_jacobian_is_exact_after_a_state_change``
    The knob Jacobian matches central finite differences of the orbit in two quadrupole
    states, and differs between them -- the lever BBA relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from adelmo.machine.accelerators import PSB
from adelmo.machine.mad import GradientDescentMadInterface
from adelmo.machine.mad.scripts import PYTHON_IN_MAD

from .test_closed_orbit_cofind import COORDS, _send_init, _solve

if TYPE_CHECKING:
    from pathlib import Path

    from pymadng import MAD

pytestmark = [pytest.mark.serial, pytest.mark.slow]

QUAD_GLOBAL = "kbrqf"
QUAD_ELEMENT = "BR.QFO11"
CORRECTOR_GLOBAL = "kbr3dhz2l4"
CORRECTOR_ELEMENT = "BR3.DHZ2L4"
STRENGTH_CHANGE = 0.05  # relative change of kbrqf between the two states
STEP = 1e-6  # m, central difference step on the misalignment


def _element_attribute(mad: MAD, element: str, attribute: str) -> float:
    mad.send(f"{PYTHON_IN_MAD}:send(loaded_sequence['{element}'].{attribute})")
    return float(mad.recv())


def _global(mad: MAD, name: str) -> float:
    mad.send(f"{PYTHON_IN_MAD}:send(MADX['{name}'])")
    return float(mad.recv())


def _interface(seq_psb: Path) -> GradientDescentMadInterface:
    return GradientDescentMadInterface(
        PSB(ring=3, sequence_file=seq_psb, misalignments={"quad": {"dx"}}), py_name=PYTHON_IN_MAD
    )


def test_state_variable_drives_element_strength(seq_psb: Path) -> None:
    """Quadrupole k1 and corrector kick follow their MAD-X globals."""
    iface = _interface(seq_psb)
    mad = iface.mad

    k1 = _global(mad, QUAD_GLOBAL)
    assert _element_attribute(mad, QUAD_ELEMENT, "k1") == pytest.approx(k1)
    mad.send(f"MADX['{QUAD_GLOBAL}'] = {k1 * (1 + STRENGTH_CHANGE):.15e}")
    assert _element_attribute(mad, QUAD_ELEMENT, "k1") == pytest.approx(k1 * (1 + STRENGTH_CHANGE))

    mad.send(f"MADX['{CORRECTOR_GLOBAL}'] = 1e-5")
    assert _element_attribute(mad, CORRECTOR_ELEMENT, "kick") == pytest.approx(1e-5)
    del iface


def _orbit_x(mad: MAD, n_knobs: int) -> np.ndarray:
    return _solve(mad, "compute_closed_orbit", n_knobs)[1]["x"]


def test_jacobian_is_exact_after_a_state_change(seq_psb: Path) -> None:
    """Changing a global under live dx knobs keeps d(x)/d(dx) equal to finite differences."""
    iface = _interface(seq_psb)
    mad = iface.mad
    knobs = [k for k in iface.knob_names if k != "pt"][:4]
    _send_init(iface, knobs, [], 0.0)
    base = _global(mad, QUAD_GLOBAL)

    jacobians = []
    for factor in (1.0, 1.0 + STRENGTH_CHANGE):
        mad.send(f"MADX['{QUAD_GLOBAL}'] = {base * factor:.15e}")
        jacobian = _solve(mad, "compute_closed_orbit", len(knobs))[2]["x"]
        for column, knob in enumerate(knobs):
            numeric = []
            for sign in (+1, -1):
                mad.send(f"loaded_sequence['{knob}']:set0({sign * STEP:.15e})")
                numeric.append(_orbit_x(mad, len(knobs)))
            mad.send(f"loaded_sequence['{knob}']:set0(0)")
            central = (numeric[0] - numeric[1]) / (2 * STEP)
            scale = float(np.max(np.abs(central)))
            assert scale > 1e-2, f"{knob} does not steer x at kbrqf x {factor}"
            assert np.max(np.abs(jacobian[:, column] - central)) < 1e-4 * scale, (
                f"d(x)/d({knob}) disagrees with finite differences at kbrqf x {factor}"
            )
        jacobians.append(jacobian)

    change = np.max(np.abs(jacobians[1] - jacobians[0])) / np.max(np.abs(jacobians[0]))
    assert change > 1e-3, "the quadrupole state did not change the knob Jacobian"
    assert COORDS  # the helpers solve all four coordinates; only x is compared here
    del iface
