"""Tests for magnet perturbations using real loaded MAD sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from adelmo.machine.accelerators import PSB
from adelmo.machine.mad import GradientDescentMadInterface

if TYPE_CHECKING:
    from pathlib import Path

    from adelmo.machine.mad.aba_mad_interface import AbaMadInterface


def _get_element_attr(interface: AbaMadInterface, element_name: str, attr: str) -> float:
    """Read one element attribute from the currently loaded MAD sequence."""
    return float(interface.mad.loaded_sequence[element_name][attr])


def _get_effective_strength(interface: AbaMadInterface, element_name: str, attr: str) -> float:
    """Read the effective strength reported by the interface."""
    return interface.get_magnet_strengths([f"{element_name}.{attr}"])[f"{element_name}.{attr}"]


def _get_element_dknl(interface: AbaMadInterface, element_name: str, index: int) -> float:
    """Read one dknl component (0-based, Python indexing)."""
    try:
        return float(interface.mad.loaded_sequence[element_name].dknl[index])
    except IndexError:
        # If the dknl array is empty or shorter than expected, treat missing components as zero
        return 0


def test_effective_strength_matches_base_when_dknl_not_created(
    loaded_interface: AbaMadInterface,
    seq_b1: Path,
) -> None:
    """Effective strength lookup should fall back to the base strength before any perturbation."""
    del seq_b1
    quad_name = "MQY.B5L2.B1"

    assert len(loaded_interface.mad.loaded_sequence[quad_name].dknl) == 0
    assert np.isclose(
        _get_effective_strength(loaded_interface, quad_name, "k1"),
        _get_element_attr(loaded_interface, quad_name, "k1"),
    )


def test_perturbation_records_integrated_dknl_matching_readback(
    loaded_psb_interface: AbaMadInterface,
    seq_psb: Path,
) -> None:
    """The dict returned by apply_magnet_perturbations must be self-consistent.

    The ``dk*l`` knob name is an *integrated* strength, so the recorded value has
    to equal what ``get_magnet_strengths`` reads back for the same knob (which
    returns the integrated ``dknl`` component directly). A regression here means
    the returned "true" strengths are off by the element length.
    """
    del seq_psb
    magnet_strengths, _ = loaded_psb_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=42,
        magnet_type="q",
    )
    assert magnet_strengths, "expected at least one perturbed quadrupole"

    readback = loaded_psb_interface.get_magnet_strengths(list(magnet_strengths))
    for name, recorded in magnet_strengths.items():
        assert np.isclose(recorded, readback[name]), (
            f"{name}: apply recorded {recorded} but get_magnet_strengths "
            f"reads back {readback[name]}"
        )


@pytest.mark.parametrize(
    ("rel_error", "expect_non_table_changed"),
    [(None, False), (1e-2, True)],
    ids=["table_relative_errors", "global_relative_error"],
)
def test_lhc_quadrupole_perturbation_modes(
    loaded_interface: AbaMadInterface,
    seq_b1: Path,
    rel_error: float | None,
    expect_non_table_changed: bool,
) -> None:
    """LHC quadrupole perturbation should support table mode and global relative-error mode."""
    table_family_quad = "MQY.B5L2.B1"  # Covered by QUAD_ERROR_TABLE
    non_table_quad = "MQT.12R2.B1"  # Not covered by QUAD_ERROR_TABLE

    k1_table_before = _get_effective_strength(loaded_interface, table_family_quad, "k1")
    k1_non_table_before = _get_effective_strength(loaded_interface, non_table_quad, "k1")

    magnet_strengths, _ = loaded_interface.apply_magnet_perturbations(
        rel_error=rel_error,
        seed=42,
        magnet_type="q",
    )

    k1_table_after = _get_effective_strength(loaded_interface, table_family_quad, "k1")
    k1_non_table_after = _get_effective_strength(loaded_interface, non_table_quad, "k1")

    assert not np.isclose(k1_table_after, k1_table_before)
    non_table_rel_change = abs(k1_non_table_after - k1_non_table_before) / max(
        abs(k1_non_table_before), 1e-12
    )
    non_table_changed = non_table_rel_change > 1e-3
    assert non_table_changed == expect_non_table_changed
    assert f"{table_family_quad}.dk1l" in magnet_strengths
    if expect_non_table_changed:
        assert f"{non_table_quad}.dk1l" in magnet_strengths


def test_zero_strength_selected_magnet_is_not_perturbed(
    loaded_psb_interface: AbaMadInterface,
    seq_psb: Path,
) -> None:
    """Relative perturbations should skip selected magnets that are off."""
    del seq_psb
    off_quad = "BR.QFO11"
    loaded_psb_interface.mad.loaded_sequence[off_quad].k1 = 0.0

    magnet_strengths, true_strengths = loaded_psb_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=42,
        magnet_type="q",
    )

    assert f"{off_quad}.dk1l" not in magnet_strengths
    assert off_quad not in true_strengths
    assert np.isclose(_get_element_dknl(loaded_psb_interface, off_quad, 1), 0.0)


def test_psb_bend_and_qfo_qde_perturbation_families(
    loaded_psb_interface: AbaMadInterface,
    seq_psb: Path,
) -> None:
    """PSB perturbations should cover ring bends and QFO/QDE quadrupoles."""
    del seq_psb

    bend_strengths, _ = loaded_psb_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=42,
        magnet_type="d",
    )
    quad_strengths, _ = loaded_psb_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=24,
        magnet_type="q",
    )

    assert "BR.BHZ11.dk0l" in bend_strengths
    assert "BR.QFO11.dk1l" in quad_strengths
    assert "BR.QDE1.dk1l" in quad_strengths
    assert all(not name.startswith("BI3.BSW") for name in bend_strengths)


def test_perturbation_preserves_deferred_optimisation_knob_link(seq_psb: Path) -> None:
    """Perturbing a GradientDescentMadInterface must not break its dk*l knobs."""
    iface = GradientDescentMadInterface(
        accelerator=PSB(
            ring=3,
            sequence_file=seq_psb,
            errors={"bend": {"k0"}, "quad": {"k1"}},
        ),
        discard_mad_output=True,
    )
    try:
        knob = "BR.BHZ11.dk0l"
        magnet_strengths, _ = iface.apply_magnet_perturbations(
            rel_error=None,
            seed=20260811,
            magnet_type="d",
        )
        knob_values = dict(zip(iface.knob_names, iface.receive_knob_values(), strict=True))
        assert knob_values[knob] == pytest.approx(magnet_strengths[knob])
        assert iface.get_magnet_strengths([knob])[knob] == pytest.approx(magnet_strengths[knob])

        changed = 123e-6
        iface.update_knob_values({knob: changed})
        assert iface.get_magnet_strengths([knob])[knob] == pytest.approx(changed)
    finally:
        iface.close()


