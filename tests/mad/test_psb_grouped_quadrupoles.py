"""Native PSB quadrupole grouping creates real shared optimisation knobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from adelmo.machine.accelerators import PSB
from adelmo.machine.mad import GradientDescentMadInterface

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial


@pytest.mark.slow
def test_grouping_changes_48_gradient_knobs_to_32(seq_psb: Path) -> None:
    counts = []
    positions = []
    for grouped in (False, True):
        interface = GradientDescentMadInterface(
            PSB(
                ring=3,
                sequence_file=seq_psb,
                errors={"quad": {"k1"}},
                group_quadrupoles_by_cell=grouped,
            )
        )
        try:
            counts.append(len(interface.knob_names))
            positions.append(dict(zip(interface.knob_names, interface.elem_spos, strict=True)))
        finally:
            interface.close()
    assert counts == [48, 32]
    assert positions[1]["BR.QFOCELL1.dk1l"] == pytest.approx(
        (positions[0]["BR.QFO11.dk1l"] + positions[0]["BR.QFO12.dk1l"]) / 2.0
    )


@pytest.mark.slow
def test_one_grouped_gradient_knob_moves_both_qfo_members(seq_psb: Path) -> None:
    interface = GradientDescentMadInterface(
        PSB(
            ring=3,
            sequence_file=seq_psb,
            errors={"quad": {"k1"}},
            group_quadrupoles_by_cell=True,
        )
    )
    try:
        interface.update_knob_values({"BR.QFOCELL1.dk1l": 1.25e-4})
        values = interface.mad.recv_vars(
            "loaded_sequence['BR.QFO11'].dknl[2]",
            "loaded_sequence['BR.QFO12'].dknl[2]",
        )
    finally:
        interface.close()
    assert values == pytest.approx((1.25e-4, 1.25e-4))


@pytest.mark.slow
@pytest.mark.parametrize("attribute", ["dx", "dy", "tilt"])
def test_direct_quadrupole_families_are_grouped_independently(
    seq_psb: Path, attribute: str
) -> None:
    kwargs = {"misalignments": {"quad": {attribute}}}
    interface = GradientDescentMadInterface(
        PSB(
            ring=3,
            sequence_file=seq_psb,
            group_quadrupoles_by_cell=True,
            **kwargs,
        )
    )
    try:
        assert len(interface.knob_names) == 32
        assert f"BR.QFOCELL1.{attribute}" in interface.knob_names
    finally:
        interface.close()
