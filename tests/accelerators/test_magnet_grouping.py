"""Tests for pure accelerator magnet grouping utilities."""

from __future__ import annotations

import pytest

from aba_optimiser.accelerators.magnet_grouping import (
    collapse_psb_grouped_quadrupole_knobs,
    expand_psb_grouped_quadrupole_knobs,
    normalise_lhcbend_magnets,
)


def test_psb_physical_qfo_pair_collapses_to_grouped_knob() -> None:
    assert collapse_psb_grouped_quadrupole_knobs(
        {"BR.QFO111.dk1l": 2e-4, "BR.QFO112.dk1l": 2e-4, "BR.QDE11.dk1l": 1e-4}
    ) == {"BR.QFOCELL11.dk1l": 2e-4, "BR.QDE11.dk1l": 1e-4}


@pytest.mark.parametrize(
    "values, message",
    [
        ({"BR.QFO111.dk1l": 2e-4}, "needs both"),
        ({"BR.QFO111.dk1l": 2e-4, "BR.QFO112.dk1l": 3e-4}, "unequal"),
    ],
)
def test_psb_invalid_physical_qfo_warm_start_is_rejected(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        collapse_psb_grouped_quadrupole_knobs(values)


def test_psb_grouped_qfo_value_expands_to_both_physical_magnets() -> None:
    expanded = expand_psb_grouped_quadrupole_knobs(
        {"BR.QFOCELL11.dk1l": 2e-4},
    )

    assert expanded == {
        "BR.QFO111.dk1l": 2e-4,
        "BR.QFO112.dk1l": 2e-4,
    }


def test_psb_grouped_qfo_value_can_expand_from_the_native_knob_name() -> None:
    expanded = expand_psb_grouped_quadrupole_knobs(
        {
            "BR.QFOCELL11.dk1l": 2e-4,
            "br.qfocell2.tilt": -3e-3,
            "BR.QDE11.dk1l": 1e-4,
        }
    )

    assert expanded == {
        "BR.QFO111.dk1l": 2e-4,
        "BR.QFO112.dk1l": 2e-4,
        "br.qfo21.tilt": -3e-3,
        "br.qfo22.tilt": -3e-3,
        "BR.QDE11.dk1l": 1e-4,
    }


def test_lhc_sbends_are_grouped_with_length_weighted_average() -> None:
    strengths = {
        "MB.A12L1.B1.dk0l": 1.0,
        "MB.B12L1.B1.dk0l": 3.0,
    }
    lengths = {
        "MB.A12L1.B1.dk0l": 1.0,
        "MB.B12L1.B1.dk0l": 3.0,
    }

    assert normalise_lhcbend_magnets(strengths, lengths) == {
        "MB.12L1.B1.dk0l": pytest.approx(2.5)
    }


def test_lhc_rbends_are_separated_by_strength_sign() -> None:
    strengths = {
        "MBR.A12L1.TRIM.dk0l": 1.0,
        "MBR.B12L1.TRIM.dk0l": -2.0,
        "MBR.C12L1.TRIM.dk0l": 3.0,
    }
    lengths = dict.fromkeys(strengths, 1.0)

    assert normalise_lhcbend_magnets(strengths, lengths) == {
        "MBR.12L1.TRIM_p.dk0l": pytest.approx(2.0),
        "MBR.12L1.TRIM_n.dk0l": pytest.approx(-2.0),
    }


def test_lhc_unmatched_magnet_strengths_are_preserved() -> None:
    strengths = {
        "MQ.12L1.dk1l": 0.25,
        "UNMATCHED.BEND.dk0l": -0.5,
    }

    assert normalise_lhcbend_magnets(strengths, {}) == strengths
