"""MAD-NG integration tests for PSB quadrupole tilt knobs.

A tilted quadrupole is a skew source: it couples the horizontal orbit and
dispersion into the vertical plane. The rotation is driven through the element's
own ``tilt``, the only route MAD-NG differentiates - see ``_TILT_SEED`` in
``accelerators/base.py``. The failure mode these tests guard is silence: a tilt
that never reaches the map raises nothing.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from adelmo.machine.accelerators import PSB
from adelmo.machine.accelerators.base import _TILT_SEED
from adelmo.machine.mad.optimising_mad_interface import (
    GenericMadInterface,
    GradientDescentMadInterface,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

pytestmark = pytest.mark.serial

#: Tilt angle used throughout: clear of numerical noise, still in the linear regime.
TILT = 3e-3

TWISS_COLUMNS = ("name", "x", "y", "dx", "dy")


def _twiss(iface: GenericMadInterface) -> pd.DataFrame:
    """Twiss the currently loaded machine and return the coupled solution."""
    iface.mad.send("tilttws = twiss{sequence=loaded_sequence, observe=1, coupling=true}")
    return iface.mad.tilttws.to_df(columns=list(TWISS_COLUMNS)).set_index("name")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _tilt_knobs(iface: GradientDescentMadInterface) -> list[str]:
    return [k for k in iface.knob_names if k.endswith(".tilt")]


def _matching_quadrupoles(iface: GenericMadInterface) -> dict[str, float]:
    """Return ``k1`` for every quadrupole matching the PSB quadrupole pattern."""
    iface.mad.send(f"""
    local k1_by_name = {{}}
    for i, e in loaded_sequence:siter(magnet_range) do
        if e.kind == "quadrupole" and e.name:match("{PSB.PATTERN_QUADRUPOLE}") then
            k1_by_name[e.name] = e.k1
        end
    end
    {iface.py_name}:send(k1_by_name, true)
    """)
    return {name: float(k1) for name, k1 in iface.mad.recv().items()}


@pytest.fixture(scope="function")
def tilt_iface(seq_psb: Path) -> GradientDescentMadInterface:
    """Interface for PSB ring 3 with quadrupole tilt knobs enabled."""
    accelerator = PSB(ring=3, sequence_file=seq_psb, misalignments={"quad": {"tilt"}})
    iface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
    yield iface
    with contextlib.suppress(Exception):
        del iface


def test_tilt_knob_is_equivalent_to_a_dpsi_misalignment(
    seq_psb: Path, tilt_iface: GradientDescentMadInterface
) -> None:
    """Driving a tilt knob reproduces the same rotation entered as ``misalign.dpsi``.

    ``misalign.dpsi`` is MAD-NG's independent implementation of the same
    rotation, so agreeing with it checks the knob does the right thing and not
    merely *something*.
    """
    knobs = _tilt_knobs(tilt_iface)
    assert len(knobs) > 2, f"expected a tilt knob per powered quadrupole, got {knobs}"

    rng = np.random.default_rng(0)
    tilts = {knob: float(v) for knob, v in zip(knobs, rng.normal(0.0, TILT, len(knobs)))}
    tilt_iface.update_knob_values(tilts)
    tilted = _twiss(tilt_iface)

    # The knob moved the machine, into the regime the measurement lives in.
    assert _rms(tilted["dy"]) > 0.05, (
        f"tilting {len(knobs)} quadrupoles by {TILT} rad rms produced only "
        f"{_rms(tilted['dy']):.3e} m of vertical dispersion; the knob looks inert"
    )

    # And it moved it the way the same rotation as a misalignment does.
    reference_iface = GenericMadInterface(
        accelerator=PSB(ring=3, sequence_file=seq_psb), discard_mad_output=True
    )
    try:
        for knob, value in tilts.items():
            element = knob.removesuffix(".tilt")
            reference_iface.mad.send(
                f"loaded_sequence['{element}'].misalign = {{dpsi = {value:.15e}}}"
            )
        misaligned = _twiss(reference_iface)
    finally:
        with contextlib.suppress(Exception):
            del reference_iface

    for column in ("x", "y", "dx", "dy"):
        assert np.allclose(tilted[column], misaligned[column], rtol=0, atol=1e-12), (
            f"'{column}' from the tilt knob differs from a dpsi misalignment by up to "
            f"{np.max(np.abs(tilted[column] - misaligned[column])):.3e}"
        )


def test_tilt_knobs_start_off_an_exactly_zero_angle(
    tilt_iface: GradientDescentMadInterface,
) -> None:
    """Tilt knobs start at ``_TILT_SEED``, above ``minang``, and below measurability.

    A knob at exactly zero would have an identically zero Jacobian column; one
    below ``minang`` (1e-10 rad) would be seen as zero by the scalar evaluation
    but not the parametric one.
    """
    knobs = _tilt_knobs(tilt_iface)
    values = tilt_iface.get_absolute_knob_values(knobs)

    assert set(values) == set(knobs)
    assert all(value == pytest.approx(_TILT_SEED) for value in values.values()), (
        f"tilt knobs are not seeded at {_TILT_SEED}: {values}"
    )
    assert _TILT_SEED > 1e-10, "the seed must clear MAD-NG's minang"

    # The seeded machine is still the ideal one, well below anything measurable.
    assert _rms(_twiss(tilt_iface)["dy"]) < 1e-6


def test_tilt_knobs_leave_vertical_offsets_working(seq_psb: Path) -> None:
    """Enabling tilts does not cost the quadrupoles their ``dy`` knobs.

    The two reach the element by different routes, and the ``misalign`` table
    ``dy`` uses is replaced wholesale whenever it is built.
    """
    accelerator = PSB(ring=3, sequence_file=seq_psb, misalignments={"quad": {"dy", "tilt"}})
    iface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
    try:
        dy_knobs = [k for k in iface.knob_names if k.endswith(".dy")]
        assert dy_knobs and _tilt_knobs(iface)

        nominal = _twiss(iface)
        assert _rms(nominal["y"]) < 1e-9, "the seeded machine should have no vertical orbit"

        iface.update_knob_values({dy_knobs[0]: 1e-4})
        offset = _twiss(iface)
        assert _rms(offset["y"]) > 1e-6, (
            f"100 um of dy produced only {_rms(offset['y']):.3e} m of vertical orbit "
            "alongside the tilt knobs"
        )
    finally:
        with contextlib.suppress(Exception):
            del iface


def test_tilt_knobs_are_off_by_default(seq_psb: Path) -> None:
    """Quadrupole ``tilt`` misalignments default off, and enabling it adds nothing but tilts.

    Compared on the full ordered knob list, so existing fits are unaffected.
    """

    def knob_names(**kwargs) -> list[str]:
        iface = GradientDescentMadInterface(
            accelerator=PSB(ring=3, sequence_file=seq_psb, **kwargs), discard_mad_output=True
        )
        try:
            return list(iface.knob_names)
        finally:
            with contextlib.suppress(Exception):
                del iface

    baseline = knob_names(errors={"quad": {"k1"}}, misalignments={"quad": {"dy"}})
    assert baseline, "the baseline configuration should produce knobs"
    assert not [k for k in baseline if k.endswith(".tilt")]

    with_tilts = knob_names(errors={"quad": {"k1"}}, misalignments={"quad": {"dy", "tilt"}})
    assert [k for k in with_tilts if not k.endswith(".tilt")] == baseline
    assert [k for k in with_tilts if k.endswith(".tilt")]


def test_unpowered_quadrupoles_get_no_tilt_knob(
    tilt_iface: GradientDescentMadInterface,
) -> None:
    """Tilting a quadrupole with no gradient does nothing, so it gets no knob."""
    k1_by_name = _matching_quadrupoles(tilt_iface)
    assert k1_by_name, "no quadrupoles matched the PSB pattern; test is vacuous"

    powered = {name for name, k1 in k1_by_name.items() if k1 != 0.0}
    knobbed = {knob.removesuffix(".tilt") for knob in _tilt_knobs(tilt_iface)}
    assert knobbed == powered, (
        f"tilt knobs disagree with the powered quadrupoles: {sorted(knobbed ^ powered)}"
    )
