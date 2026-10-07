from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tfs

from aba_optimiser.machine.accelerators import PSB
from aba_optimiser.machine.mad.aba_mad_interface import AbaMadInterface

if TYPE_CHECKING:
    from pathlib import Path

NATURAL_TUNES = (0.17, 0.225)
DRIVEN_TUNES = (0.162, 0.232)


def test_psb_model_dir_contains_saved_sequence(psb_model_dir: Path) -> None:
    """The shared omc3 fixture produces a usable PSB model directory."""
    assert psb_model_dir.is_dir()

    sequence = psb_model_dir / "psb3_saved.seq"
    assert sequence.is_file()

    sequence_text = sequence.read_text()
    assert "hacmap" in sequence_text.lower()
    assert "vacmap" in sequence_text.lower()

    interface = AbaMadInterface(accelerator=PSB(ring=3, sequence_file=sequence))
    live_twiss = interface.run_twiss(observe=0)

    assert live_twiss.headers["q1"] % 1 == pytest.approx(NATURAL_TUNES[0], abs=2e-4)
    assert live_twiss.headers["q2"] % 1 == pytest.approx(NATURAL_TUNES[1], abs=2e-4)
    assert "HACMAP" in live_twiss.index
    assert "VACMAP" in live_twiss.index
    assert live_twiss.loc["HACMAP", "kind"] == "marker"
    assert live_twiss.loc["VACMAP", "kind"] == "marker"
    assert "hackicker" not in live_twiss.index
    assert "vackicker" not in live_twiss.index

    interface.install_ac_dipole(NATURAL_TUNES, DRIVEN_TUNES)
    live_twiss_ac = interface.run_twiss(observe=0)
    assert live_twiss_ac.headers["q1"] % 1 == pytest.approx(DRIVEN_TUNES[0], abs=2e-4)
    assert live_twiss_ac.headers["q2"] % 1 == pytest.approx(DRIVEN_TUNES[1], abs=2e-4)
    assert "hackicker" in live_twiss_ac.index
    assert "vackicker" in live_twiss_ac.index

    twiss = tfs.read(psb_model_dir / "twiss.dat")
    twiss_ac = tfs.read(psb_model_dir / "twiss_ac.dat")
    assert twiss.headers["Q1"] % 1 == pytest.approx(NATURAL_TUNES[0])
    assert twiss.headers["Q2"] % 1 == pytest.approx(NATURAL_TUNES[1])
    assert twiss_ac.headers["Q1"] % 1 == pytest.approx(DRIVEN_TUNES[0])
    assert twiss_ac.headers["Q2"] % 1 == pytest.approx(DRIVEN_TUNES[1])
