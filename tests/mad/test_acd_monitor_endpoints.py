"""Regression tests for AC-dipole endpoint observation in MAD-NG."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from adelmo.machine.accelerators import PSB
from adelmo.machine.mad.optimising_mad_interface import GradientDescentMadInterface

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial


def _capture_slc_minus_4_names(iface: GradientDescentMadInterface) -> list[str]:
    """Return element names seen by the regular tracking atexit slice -4."""
    py = iface.py_name
    iface.mad.send(f"""
local names = {{}}
local function capture(elm, _, _, slc)
    if slc == -4 then
        names[#names+1] = elm.name
    end
end
MAD.track{{
    sequence = loaded_sequence,
    X0 = {{x=0, px=0, y=0, py=0, t=0, pt=0}},
    nturn = 1,
    save = false,
    atexit = capture,
}}
{py}:send(names, true)
    """)
    return [str(name) for name in iface.mad.recv()]


def test_acd_endpoints_are_monitors_and_visible_to_slc_minus_4(seq_psb: Path) -> None:
    accelerator = PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}})
    acd_after = accelerator.acd_marker_name("after")
    acd_before = accelerator.acd_marker_name("before")

    iface = GradientDescentMadInterface(
        accelerator=accelerator,
        bpm_range=f"{acd_after}/{acd_before}",
        tracking_anchor_mode="acd",
        discard_mad_output=True,
    )
    iface.cycle_to_start(acd_after)
    try:
        assert "monitor" in iface.mad.MADX[acd_after].kind
        assert "monitor" in iface.mad.MADX[acd_before].kind
        assert acd_after in iface.all_bpms
        assert acd_before in iface.all_bpms

        slc_minus_4_names = _capture_slc_minus_4_names(iface)

        assert slc_minus_4_names[0] == acd_after
        assert acd_before in slc_minus_4_names
    finally:
        with contextlib.suppress(Exception):
            del iface
