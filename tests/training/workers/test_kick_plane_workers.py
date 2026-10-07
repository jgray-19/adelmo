"""Kick-plane classification and the observables it selects.

A PSB kicker file is excited in one plane only; an AC-dipole file usually drives
both. The classification decides which observables a worker compares, so it has
to survive a quiet-but-not-silent off-plane signal.
"""

from __future__ import annotations

import pandas as pd

from aba_optimiser.training.tracking.data_manager import infer_kick_plane
from aba_optimiser.workers.tracking import active_observables


def _frame(x, px, y, py) -> pd.DataFrame:
    return pd.DataFrame({"x": x, "px": px, "y": y, "py": py})


def test_horizontal_kick_is_classified_as_single_plane() -> None:
    # A constant vertical offset is orbit, not excitation.
    frame = _frame([0.0, 3.0, -2.0], [0.0, 0.2, -0.1], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    assert infer_kick_plane(frame) == "x"


def test_vertical_kick_is_classified_as_single_plane() -> None:
    frame = _frame([2.0, 2.0, 2.0], [0.1, 0.1, 0.1], [0.0, 4.0, -3.0], [0.0, 0.3, -0.2])
    assert infer_kick_plane(frame) == "y"


def test_both_planes_driven_is_classified_as_dual_plane() -> None:
    frame = _frame([0.0, 3.0, -2.0], [0.0, 0.2, -0.1], [0.0, 4.0, -3.0], [0.0, 0.3, -0.2])
    assert infer_kick_plane(frame) == "xy"


def test_a_weakly_driven_second_plane_still_counts_as_dual_plane() -> None:
    """Only an order-of-magnitude difference makes a file single-plane."""
    frame = _frame([0.0, 3.0, -2.0], [0.0, 0.2, -0.1], [0.0, 1.0, -0.5], [0.0, 0.05, -0.02])
    assert infer_kick_plane(frame) == "xy"


def test_observables_follow_the_kick_plane_and_momentum_use() -> None:
    assert active_observables("xy", include_momentum=True) == ("x", "y", "px", "py")
    assert active_observables("xy", include_momentum=False) == ("x", "y")
    assert active_observables("x", include_momentum=True) == ("x", "px")
    assert active_observables("y", include_momentum=False) == ("y",)
