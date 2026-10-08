"""Analysis utilities that inspect a fit without changing the optimisation.

Currently this exposes the degeneracy checker, which examines the Gauss-Newton
normal matrix ``A = JᵀWJ`` to reveal knob combinations the measurement cannot
constrain - the flat directions that make an optimiser wander instead of
converging to a unique minimum.
"""

from __future__ import annotations

from adelmo.analysis.degeneracy_checker import (
    DegeneracyReport,
    DegenerateDirection,
    analyse_degeneracy,
)

__all__ = [
    "DegeneracyReport",
    "DegenerateDirection",
    "analyse_degeneracy",
]
