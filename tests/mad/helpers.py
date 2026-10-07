"""
Common helper functions for MAD interface tests.

This module contains utility functions used across MAD interface test modules.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tfs

import numpy as np

if TYPE_CHECKING:
    from aba_optimiser.machine.mad.aba_mad_interface import AbaMadInterface


def _recv_n(mad: Any, n: int) -> list[Any]:
    """Helper to receive n messages from a MAD instance."""
    results = []
    for _ in range(n):
        results.append(mad.recv())
    return results


def check_interface_basic_init(interface: Any, py_name: str = "py") -> None:
    """Check basic interface initialization properties."""
    assert interface.py_name == py_name
    assert hasattr(interface, "mad")


def cleanup_interface(interface: Any) -> None:
    """Clean up interface instance."""
    with contextlib.suppress(Exception):
        del interface


def check_sequence_loaded(interface: AbaMadInterface, seq_name: str = "lhcb1") -> None:
    """Check that sequence is loaded correctly."""
    assert seq_name == interface.mad.SEQ_NAME
    assert interface.mad.loaded_sequence is not None


def check_beam_setup(
    interface: AbaMadInterface,
    particle: str = "proton",
    kinetic_energy: float | None = None,
    charge: int = 1,
    spin: float = 0.5,
) -> None:
    """Check beam setup properties."""
    assert particle == interface.mad.loaded_sequence.beam.particle
    if kinetic_energy is not None:
        assert np.isclose(
            kinetic_energy,
            interface.mad.loaded_sequence.beam.energy - interface.mad.loaded_sequence.beam.mass,
            rtol=1e-10,
        )
    assert charge == interface.mad.loaded_sequence.beam.charge
    assert spin == interface.mad.loaded_sequence.beam.spin


def check_element_observations(
    interface: AbaMadInterface,
    condition: str,
) -> None:
    """Helper to check element observations in MAD."""
    interface.mad.send(f"""
for _, elm in loaded_sequence:iter() do
    if {condition} then
        assert(
            elm:is_observed(),
            "Found expected observation missing for element " .. elm.name
        )
    else
        assert(
            not elm:is_observed(),
            "Found unexpected observation for element " .. elm.name
        )
    end
end
py:send("PASS")""")
    assert interface.mad.recv() == "PASS"


def check_element_observations_by_names(
    interface: AbaMadInterface,
    observed_names: list[str],
) -> None:
    """Check that exactly the given elements are observed, no more, no less."""
    interface.mad.send("observed_set = {}")
    for name in observed_names:
        interface.mad.send(f"observed_set['{name}'] = true")
    interface.mad.send("""
for _, elm in loaded_sequence:iter() do
    local expected = observed_set[elm.name] == true
    if expected then
        assert(elm:is_observed(), "Missing expected observation: " .. elm.name)
    else
        assert(not elm:is_observed(), "Unexpected observation: " .. elm.name)
    end
end
py:send("PASS")""")
    assert interface.mad.recv() == "PASS"


def get_marker_and_element_positions(
    interface: AbaMadInterface, marker_name: str, element_name: str
) -> tuple[float, int, float, int]:
    """Helper to get positions and indices of marker and element."""
    interface.mad.send(f"""
local marker_pos, elm_pos, marker_idx, elm_idx
for i, elm, s, ds in loaded_sequence:siter() do
    if elm.name == "{marker_name}" then
        marker_idx = i
        marker_pos = s
    end
    if elm.name == "{element_name}" then
        elm_idx = i
        elm_pos = s
    end
end
py:send(marker_pos)
py:send(marker_idx)
py:send(elm_pos)
py:send(elm_idx)""")
    return _recv_n(interface.mad, 4)


def _corrector_targets(row) -> list[tuple[str, float]]:
    """Return the tracked attributes and expected values for one corrector row."""
    if row.kind == "hkicker":
        return [("kick", float(row.hkick))]
    if row.kind == "vkicker":
        return [("kick", float(row.vkick))]
    if row.kind == "tkicker":
        return [("hkick", float(row.hkick)), ("vkick", float(row.vkick))]
    raise ValueError(f"Unsupported corrector kind {row.kind!r}")


def _assert_corrector_strength(
    interface: AbaMadInterface, row_name: str, attr: str, expected: float
) -> None:
    """Assert one corrector strength in both MADX and loaded_sequence."""
    madx_strength = interface.mad[f"MADX['{row_name}'].{attr}"]
    seq_strength = interface.mad[f"loaded_sequence['{row_name}'].{attr}"]
    assert madx_strength == expected, (
        f"{row_name} MADX {attr} mismatch: {madx_strength} != {expected}"
    )
    assert seq_strength == expected, (
        f"{row_name} loaded_sequence {attr} mismatch: {seq_strength} != {expected}"
    )


def check_corrector_strengths_zero(
    interface: AbaMadInterface, corrector_table: tfs.TfsDataFrame
) -> None:
    """Check that all corrector strengths are initially zero in both MADX and loaded_sequence."""
    for row in corrector_table.itertuples():
        for attr, _ in _corrector_targets(row):
            _assert_corrector_strength(interface, row.name, attr, 0.0)


def check_corrector_strengths(
    interface: AbaMadInterface,
    corrector_table: tfs.TfsDataFrame,
) -> None:
    """Check that corrector strengths match expected values in both MADX and loaded_sequence."""
    for row in corrector_table.itertuples():
        for attr, expected in _corrector_targets(row):
            _assert_corrector_strength(interface, row.name, attr, expected)
