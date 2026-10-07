"""What a fit returns."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FitDiagnostics:
    """How the optimisation ended.

    ``iterations`` counts epochs (tracking) or Levenberg-Marquardt iterations.
    ``gradient_norm``, ``damping`` and ``accepted_evaluations`` describe the last
    LM update and are ``None`` for the tracking fit.
    """

    converged: bool
    reason: str | None
    iterations: int
    best_loss: float
    gradient_norm: float | None = None
    damping: float | None = None
    accepted_evaluations: int | None = None


@dataclass(frozen=True)
class FitResult:
    """Fitted knobs, their 1-sigma uncertainties (by knob name) and how the fit ended."""

    knobs: dict[str, float]
    uncertainties: dict[str, float]
    diagnostics: FitDiagnostics | None = None
    #: Extra fitted parameters that are not model knobs (e.g. BPM and corrector gains).
    extra: dict[str, float] = field(default_factory=dict)
