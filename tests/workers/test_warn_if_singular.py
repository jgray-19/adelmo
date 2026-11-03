"""``warn_if_singular`` flags normal matrices the data do not constrain, independent of the knobs' units."""

from __future__ import annotations

import logging

import numpy as np

from adelmo.fitting.uncertainty import warn_if_singular


def test_well_conditioned_matrix_is_silent(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert warn_if_singular(np.diag([1.0, 2.0, 3.0])) == 0
    assert not caplog.records


def test_degenerate_pair_is_reported_with_its_knobs(caplog) -> None:
    jac = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # knobs a and b only ever enter as a + b
    with caplog.at_level(logging.WARNING):
        assert warn_if_singular(jac.T @ jac, ["a", "b", "c"]) == 1
    assert "near-singular" in caplog.text
    assert "a (" in caplog.text
    assert "b (" in caplog.text


def test_unit_scaling_does_not_hide_or_fake_singularity(caplog) -> None:
    scale = np.diag([1e6, 1.0, 1e-6])  # knobs in wildly different units
    with caplog.at_level(logging.WARNING):
        assert warn_if_singular(scale @ np.diag([1.0, 2.0, 3.0]) @ scale) == 0
