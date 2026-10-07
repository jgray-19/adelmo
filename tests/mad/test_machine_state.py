"""``machine_state``: one argument that puts a model in the state its measurements were taken at.

``test_resolve_accepts_dict_knobs_file_and_corrector_table``
    The three input forms give name/value pairs; a corrector table keeps only the kickers whose strength changed.

``test_merge_layers_later_states_over_earlier_ones``
    Several sources become one state, and nothing at all stays ``None``.

``test_interface_applies_variables_element_attributes_and_new_names``
    On a real PSB model a name is an existing MAD-X variable, else an element attribute, else a new variable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tfs
from pymadng_utils.io.utils import save_knobs

from aba_optimiser.machine.accelerators import PSB
from aba_optimiser.machine.mad import GenericMadInterface, merge_machine_states
from aba_optimiser.machine.mad.machine_state import resolve_machine_state

if TYPE_CHECKING:
    from pathlib import Path


def _corrector_table(path: Path) -> Path:
    table = tfs.TfsDataFrame(
        {
            "ename": ["BR3.DHZ2L4", "BR3.DVT2L4", "BR3.BPM1L3"],
            "kind": ["hkicker", "vkicker", "monitor"],
            "hkick": [2e-5, 0.0, 0.0],
            "hkick_old": [0.0, 0.0, 0.0],
            "vkick": [0.0, -3e-5, 0.0],
            "vkick_old": [0.0, 0.0, 0.0],
        }
    )
    tfs.write(path, table)
    return path


def test_resolve_accepts_dict_knobs_file_and_corrector_table(tmp_path: Path) -> None:
    save_knobs({"kbrqf": 0.73}, tmp_path / "tunes.txt")

    assert resolve_machine_state({"kbrqf": 1}) == {"kbrqf": 1.0}
    assert resolve_machine_state(tmp_path / "tunes.txt") == {"kbrqf": 0.73}
    assert resolve_machine_state(_corrector_table(tmp_path / "correctors.tfs")) == {
        "BR3.DHZ2L4.kick": 2e-5,
        "BR3.DVT2L4.kick": -3e-5,
    }


def test_merge_layers_later_states_over_earlier_ones(tmp_path: Path) -> None:
    save_knobs({"kbrqf": 0.73, "kbrqd": -0.74}, tmp_path / "tunes.txt")

    merged = merge_machine_states({"kbrqf": 1.0}, tmp_path / "tunes.txt", None)

    assert merged == {"kbrqf": 0.73, "kbrqd": -0.74}
    assert merge_machine_states(None, None) is None


def _value(interface: GenericMadInterface, expression: str) -> float:
    interface.mad.send(f"{interface.py_name}:send({expression})")
    return float(interface.mad.recv())


@pytest.mark.slow
def test_interface_applies_variables_element_attributes_and_new_names(seq_psb: Path, tmp_path: Path) -> None:
    state = {"kbrqf": 0.7, "br3.dhz2l4.kick": 2e-5, "not_in_the_model": 3.0}
    interface = GenericMadInterface(PSB(ring=3, sequence_file=seq_psb), machine_state=state)
    try:
        assert _value(interface, "MADX.kbrqf") == pytest.approx(0.7)
        assert _value(interface, "loaded_sequence['BR3.DHZ2L4'].kick") == pytest.approx(2e-5)
        assert _value(interface, "MADX.not_in_the_model") == pytest.approx(3.0)
    finally:
        interface.close()

    from_table = GenericMadInterface(
        PSB(ring=3, sequence_file=seq_psb), machine_state=_corrector_table(tmp_path / "correctors.tfs")
    )
    try:
        assert _value(from_table, "loaded_sequence['BR3.DVT2L4'].kick") == pytest.approx(-3e-5)
    finally:
        from_table.close()
