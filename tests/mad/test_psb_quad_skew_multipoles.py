"""MAD-NG integration tests for PSB quadrupole skew-multipole error knobs.

``k0s``/``k1s`` are additive skew-multipole field errors on a quadrupole,
routed through the existing generic ``dksl`` deferred-table machinery (see
``MULTIPOLE_ATTRS`` in ``pymadng_utils``). Unlike ``dy``/``tilt`` they are not
misalignments and do not scale with the element's own ``k1`` in the knob's
Jacobian, which is why they exist alongside those knobs rather than
reparametrising them. See ``docs/studies/quadrupole-roll.md`` in ``psb_loco``
for why a ``k1*sin(2*tilt)`` skew-multipole *approximation of roll* was
rejected - these knobs are independent additive errors, not that.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.mad.optimising_mad_interface import (
    GenericMadInterface,
    GradientDescentMadInterface,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

pytestmark = pytest.mark.serial

TWISS_COLUMNS = ("name", "x", "y", "dx", "dy")


def _twiss(iface: GenericMadInterface) -> pd.DataFrame:
    iface.mad.send("skewtws = twiss{sequence=loaded_sequence, observe=1, coupling=true}")
    return iface.mad.skewtws.to_df(columns=list(TWISS_COLUMNS)).set_index("name")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


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


@pytest.mark.parametrize(
    ("suffix", "abs_attr"),
    [(".dk0sl", "k0s"), (".dk1sl", "k1s")],
)
def test_skew_knob_updates_use_dksl(
    seq_psb: Path, suffix: str, abs_attr: str
) -> None:
    """Skew-multipole dksl should follow the live knob value without mutating k1.

    Read through ``get_magnet_strengths``/``get_base_magnet_strengths`` (as the
    ``dk1l`` template test does for its primary assertion), not a raw
    ``interface.mad.loaded_sequence[...].dksl[index]`` chain: that chain read is
    a known pymadng client-side quirk on the ``dksl`` table specifically (it
    silently returns the pre-assignment value), unrelated to the underlying
    MAD-NG state, which the ``get_magnet_strengths`` round trip confirms is
    correct.
    """
    accelerator = PSB(ring=3, sequence_file=seq_psb, errors={"quad": {abs_attr}})
    interface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
    try:
        knob_name = next(knob for knob in interface.knob_names if knob.endswith(suffix))
        element_name = knob_name.removesuffix(suffix)
        absolute_name = f"{element_name}.{abs_attr}"
        length = float(interface.mad.loaded_sequence[element_name].l)
        initial_strength_base = interface.get_base_magnet_strengths([absolute_name])[absolute_name]
        initial_strength = interface.get_magnet_strengths([absolute_name])[absolute_name]
        assert np.isclose(initial_strength, initial_strength_base)
        initial_k1 = float(interface.mad.loaded_sequence[element_name].k1)

        step = 1e-4
        interface.mad.send(f"loaded_sequence['{knob_name}'] = {step}")

        updated_strength = interface.get_magnet_strengths([absolute_name])[absolute_name]
        updated_k1 = float(interface.mad.loaded_sequence[element_name].k1)

        assert np.isclose(updated_strength, initial_strength + step / length)
        assert np.isclose(updated_k1, initial_k1), "the skew knob must not mutate the base k1"

        interface.set_magnet_strengths({absolute_name: initial_strength_base})
        final_strength = interface.get_magnet_strengths([absolute_name])[absolute_name]
        assert np.isclose(final_strength, initial_strength_base)
    finally:
        with contextlib.suppress(Exception):
            del interface


def test_k1s_knob_moves_the_machine(seq_psb: Path) -> None:
    """A free k1s (skew gradient) knob is a genuine skew source: it couples orbit/dispersion."""
    accelerator = PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1s"}})
    interface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
    try:
        knobs = [k for k in interface.knob_names if k.endswith(".dk1sl")]
        assert len(knobs) > 2, f"expected a k1s knob per powered quadrupole, got {knobs}"

        rng = np.random.default_rng(0)
        values = {knob: float(v) for knob, v in zip(knobs, rng.normal(0.0, 5e-3, len(knobs)))}
        interface.update_knob_values(values)
        perturbed = _twiss(interface)

        assert _rms(perturbed["dy"]) > 1e-3, (
            f"perturbing {len(knobs)} k1s knobs produced only "
            f"{_rms(perturbed['dy']):.3e} m of vertical dispersion; the knob looks inert"
        )
    finally:
        with contextlib.suppress(Exception):
            del interface


def test_quad_k0s_k1s_knobs_are_off_by_default(seq_psb: Path) -> None:
    """Quadrupole ``k0s``/``k1s`` errors default off, and add nothing else.

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
    assert not [k for k in baseline if k.endswith((".dk0sl", ".dk1sl"))]

    with_skew = knob_names(
        errors={"quad": {"k0s", "k1", "k1s"}},
        misalignments={"quad": {"dy"}},
    )
    assert [k for k in with_skew if not k.endswith((".dk0sl", ".dk1sl"))] == baseline
    assert [k for k in with_skew if k.endswith(".dk0sl")]
    assert [k for k in with_skew if k.endswith(".dk1sl")]


def test_unpowered_quadrupoles_get_no_skew_knob(seq_psb: Path) -> None:
    """A quadrupole with no gradient gets no k0s/k1s knob (gated on nonzero k1)."""
    accelerator = PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k0s", "k1s"}})
    interface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
    try:
        k1_by_name = _matching_quadrupoles(interface)
        assert k1_by_name, "no quadrupoles matched the PSB pattern; test is vacuous"
        powered = {name for name, k1 in k1_by_name.items() if k1 != 0.0}

        for suffix in (".dk0sl", ".dk1sl"):
            knobbed = {
                knob.removesuffix(suffix)
                for knob in interface.knob_names
                if knob.endswith(suffix)
            }
            assert knobbed == powered, (
                f"{suffix} knobs disagree with the powered quadrupoles: "
                f"{sorted(knobbed ^ powered)}"
            )
    finally:
        with contextlib.suppress(Exception):
            del interface
