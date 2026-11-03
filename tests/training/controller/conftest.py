"""Shared fixtures for controller integration tests."""

from __future__ import annotations

import pytest


@pytest.fixture(
    params=[
        pytest.param(
            "loss_regression",
            id="loss_regression",
            marks=pytest.mark.regression,
        ),
        pytest.param(
            "run_converges",
            id="run_converges",
            marks=[pytest.mark.convergence, pytest.mark.slow],
        ),
    ]
)
def controller_test_mode(request: pytest.FixtureRequest) -> str:
    """Run controller integration tests in fast loss and full optimisation modes."""
    return str(request.param)
