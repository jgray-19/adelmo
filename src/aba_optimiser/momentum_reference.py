"""Fit bends and quads to closed orbit + phase, and hand back a closed-orbit reference.

Why this exists
---------------
A transverse-momentum reconstruction needs the closed orbit's *angle*
``px``/``py`` at every BPM. BPMs measure position only, so the angle can come
from nothing but a model -- and a nominal model that does not carry the machine's
real magnet errors supplies an angle that is simply wrong. On PSB ring 3 with
realistic dipole errors the nominal model's angle error *equals* the true angle:
it contributes nothing.

Fitting the errors from a plain closed-orbit measurement fixes that, but only if
three things are right at once. Each was established by measurement, not
assumption (see NOTES_offmom_investigation, sections E and F, in the tmom-recon
repository):

1. **Regularise.** 80 knobs against 17 BPMs at realistic resolution, fitted with
   no prior, is *worse than not fitting at all* -- the recovered error vector
   comes out ~6x larger than the true one as the fit absorbs noise into
   compensating magnet errors. A modest Tikhonov prior swings the same data from
   2.7x worse than nominal to 22x better. This is the single most important
   setting here, which is why it has a non-zero default.
2. **Match the knob families to the observables.** The closed orbit is nearly
   blind to gradient errors: a quadrupole on a centred orbit produces no
   deflection. Giving an orbit-only fit free quadrupole knobs therefore makes it
   ~2.6x *worse*, because the knobs only add noise-absorbing freedom. Phase does
   respond to gradients, so with phase included the same quad knobs pay off ~4x.
3. **Use more than one momentum.** At a single momentum the per-magnet Jacobians
   are degenerate; see :class:`~aba_optimiser.workers.common.ClosedTwissData`.

Observables
-----------
The default is closed orbit plus phase advance, deliberately. Phase needs no
amplitude calibration, and unlike beta-from-amplitude or a model-derived
``DPX``/``DPY`` it does not feed a modelled quantity back into a model fit. Beta
and dispersion are available through *observables* for callers whose measurement
genuinely supports them.

Consuming the result
--------------------
:class:`MomentumReference` is plain data: fitted knob values and a per-BPM
closed-orbit frame carrying ``x``/``y``/``px``/``py``. A downstream
reconstruction takes those without importing anything from this package.

The reference is a *momentum origin*, not just an orbit: a reconstruction
expressed against :attr:`MomentumReference.closed_orbit` sees only what the
measurement has in excess of that orbit, so what it needs is the offset **from**
:attr:`MomentumReference.reference_pt`, never the measurement's absolute ``pt``.
The off-momentum study measured the difference at a reference sitting
:math:`3\\times10^{-3}` off the origin: using the absolute ``pt`` as if it were
the offset degrades the reconstructed ``px`` from 4.741e-4 to 7.702e-2, against
1.162e-3 for the offset -- a factor 66. First order cancels either way; the whole
penalty lands on the second-order dispersion term, so the mistake is invisible on
a linear lattice and ruinous on a real one.

Do not do that subtraction here. Pass the measured positions as
``closed_orbit_at_zero`` and ``orbit_mode="dynamic"`` to ``tmom_recon``, and
pass only the measurement momentum offset alongside them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.training.config.models import SequenceConfig
from aba_optimiser.training_closed_twiss import (
    ClosedTwissFitter,
    LevenbergMarquardtConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from aba_optimiser.accelerators.base import Accelerator
    from aba_optimiser.training.config.models import OutputConfig

LOGGER = logging.getLogger(__name__)

#: Closed orbit and phase advance: the observables that need no amplitude
#: calibration and no model-derived intermediate. See the module docstring.
ORBIT_AND_PHASE: tuple[str, ...] = ("x", "y", "mu1", "mu2")

#: Observables that carry quadrupole-gradient information. The closed orbit does
#: not: a gradient error on a centred orbit produces no deflection.
GRADIENT_SENSITIVE: frozenset[str] = frozenset({"mu1", "mu2", "betx", "bety", "dx", "dy"})

#: Dimensionless Tikhonov prior towards zero magnet error, scaled internally to
#: ``median(diag H)``. Non-zero by default because an unregularised fit of this
#: size is worse than no fit, but it must stay at the *low* end of the broad
#: useful range (1e-2 to 1e-1): the prior is a shrinkage, so a knob at the family
#: median curvature is fitted a factor ``1/(1 + strength)`` too small. At 1e-1
#: that is a 9.1% systematic under-estimate of every bend error, and since the
#: bends are what carry the closed-orbit *angle*, it biases the reconstructed
#: ``px`` by the same 9.1% -- 1.3e-4 rad against a 1.4e-3 rad true angle, which
#: is most of the residual and swamps the 9.2e-5 rad noise floor. A prior scan on
#: the PSB reference fixture puts the fit on that noise floor by 1e-2 with no
#: sign of ill-conditioning, so the extra regularisation bought nothing.
DEFAULT_PRIOR_STRENGTH: float = 1e-2

#: Columns of the returned closed-orbit reference.
REFERENCE_COLUMNS: tuple[str, ...] = ("x", "y", "px", "py")


@dataclass(frozen=True)
class MomentumReference:
    """A fitted lattice and the closed-orbit reference it implies.

    Attributes:
        magnet_strengths: Fitted knob values, keyed by MAD knob name
            (``BR.BHZ11.dk0l``, ``BR.QFO11.dk1l``, ...). Apply these to a model
            of the same sequence to reproduce the fitted machine.
        closed_orbit: Per-BPM ``x``/``y``/``px``/``py`` of the fitted machine's
            closed orbit at :attr:`reference_pt`, indexed by BPM name. ``x``/``y`` are
            a cross-check against the measurement; ``px``/``py`` are the part
            that cannot be measured and is the reason this class exists.
        reference_pt: MAD-NG ``pt`` at which the reference was evaluated. This is
            the momentum origin of :attr:`closed_orbit`; momenta handed to a
            reconstruction against this reference are offsets from it, not
            absolute ``pt`` values; see the module docstring for how to hand
            both to ``tmom_recon`` without doing the subtraction by hand.
        observables: Observables the fit actually used.
        prior_strength: Tikhonov prior used.
    """

    magnet_strengths: dict[str, float]
    uncertainties: dict[str, float]
    closed_orbit: pd.DataFrame
    reference_pt: float
    observables: tuple[str, ...]
    prior_strength: float
    momentum_points: tuple[float, ...]
    bpm_coverage: dict[float, tuple[str, ...]]
    fit_settings: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _knobs_provenance(knobs: Any) -> dict[str, float] | str | None:
    """Knob provenance for a saved fit: the values themselves where possible."""
    if knobs is None:
        return None
    if isinstance(knobs, str | Path):
        return str(knobs)
    return {str(name): float(value) for name, value in sorted(dict(knobs).items())}


def closed_orbit_at(
    accelerator: Accelerator,
    magnet_strengths: Mapping[str, float] | None,
    pt: float = 0.0,
    *,
    corrector_knobs: Path | None = None,
    tune_knobs: Path | None = None,
) -> pd.DataFrame:
    """Closed orbit and its angles at the BPMs, for a model carrying *magnet_strengths*.

    ``pt`` is pinned on the initial condition, so this is the periodic solution at
    that momentum rather than an on-momentum orbit with a momentum offset added.
    """
    iface = GradientDescentMadInterface(
        accelerator,
        corrector_knobs=corrector_knobs,
        tune_knobs=tune_knobs,
    )
    try:
        if magnet_strengths:
            iface.update_knob_values(dict(magnet_strengths))
        iface.mad.send(
            f"moref = twiss{{sequence=loaded_sequence, observe=1, "
            f"X0={{pt={float(pt):.15e}}}, coupling=true}}"
        )
        frame = iface.mad.moref.to_df(columns=["name", *REFERENCE_COLUMNS])
    finally:
        iface.close()
    return frame.set_index("name")[list(REFERENCE_COLUMNS)]


def _check_observable_knob_match(
    accelerator: Accelerator, observables: Sequence[str]
) -> None:
    """Reject knob families which the requested observables cannot constrain.

    Fitting gradients against the closed orbit alone is not merely uninformative,
    it is harmful: the extra knobs absorb measurement noise into compensating
    errors, which measured ~2.6x worse than leaving them out.
    """
    if not any(
        spec.kind == "quadrupole" and spec.attribute == "k1" for spec in accelerator.get_knob_specs()
    ):
        return
    if GRADIENT_SENSITIVE.intersection(observables):
        return
    raise ValueError(
        "Quadrupole knobs are enabled but none of the observables "
        f"{tuple(observables)} responds to a gradient error; the closed orbit is "
        "blind to gradients on a centred orbit. Add mu1/mu2 (or beta/dispersion), "
        "or stop fitting quadrupole k1."
    )


def fit_momentum_reference(
    accelerator: Accelerator,
    measurements: Mapping[float, pd.DataFrame],
    *,
    observables: Sequence[str] = ORBIT_AND_PHASE,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    reference_pt: float = 0.0,
    sequence_config: SequenceConfig | None = None,
    lm_config: LevenbergMarquardtConfig | None = None,
    initial_knob_strengths: Mapping[str, float] | None = None,
    corrector_knobs: Path | None = None,
    tune_knobs: Path | None = None,
    output_config: OutputConfig | None = None,
) -> MomentumReference:
    """Fit magnet errors to measured optics, and return the closed-orbit reference.

    Args:
        accelerator: Configured accelerator. Enable the knob families you intend
            to fit (bend ``k0`` and, if the observables can constrain them,
            quadrupole ``k1``) via its ``errors`` selection.
        measurements: Measured optics keyed by the *known MAD-NG ``pt``* the
            measurement was taken at. At least two momenta are required: at one
            momentum the per-magnet Jacobians are degenerate.
        observables: What to fit. Defaults to closed orbit plus phase advance.
        prior_strength: Tikhonov prior towards zero magnet error. Do not set this
            to zero without a specific reason; see the module docstring.
        reference_pt: MAD-NG ``pt`` at which to evaluate the returned closed orbit.
        sequence_config: Sequence range; defaults to the full ring.
        lm_config: Optimiser settings.
        initial_knob_strengths: Starting point, if resuming a previous fit.

    Returns:
        A :class:`MomentumReference`.

    Raises:
        ValueError: If fewer than two momenta are supplied, or *prior_strength*
            is negative.
    """
    momentum_points = tuple(sorted(float(pt) for pt in measurements))
    if len(set(momentum_points)) < 2:
        raise ValueError(
            "fit_momentum_reference needs measurements at at least two momenta; "
            f"got {list(momentum_points)}. At a single momentum the per-magnet "
            "Jacobians are degenerate and the fit cannot separate the magnets."
        )
    if prior_strength < 0.0:
        raise ValueError("prior_strength must be >= 0")
    if prior_strength == 0.0:
        LOGGER.warning(
            "prior_strength=0 requested. An unregularised fit of this size was "
            "measured to be worse than not fitting at all; prefer %g.",
            DEFAULT_PRIOR_STRENGTH,
        )

    observables = tuple(observables)
    _check_observable_knob_match(accelerator, observables)

    fitter = ClosedTwissFitter(
        accelerator=accelerator,
        sequence_config=sequence_config or SequenceConfig(magnet_range="$start/$end"),
        measurements=dict(measurements),
        observables=observables,
        lm_config=lm_config,
        initial_knob_strengths=dict(initial_knob_strengths) if initial_knob_strengths else None,
        prior_strengths=(
            {
                {
                    "k0": "dk0l",
                    "k1": "dk1l",
                    "k2": "dk2l",
                }.get(spec.attribute, spec.attribute): prior_strength
                for spec in accelerator.get_knob_specs()
            }
            if prior_strength > 0.0
            else None
        ),
        corrector_knobs=corrector_knobs,
        tune_knobs=tune_knobs,
        output_config=output_config,
    )
    try:
        magnet_strengths, uncertainties = fitter.run()
    finally:
        # The workers are stopped by ``run``; this is the independent setup MAD
        # interface owned by the configuration manager.
        fitter.config_manager.mad_iface.close()

    LOGGER.info(
        "Fitted %d knobs against %s at momenta %s (prior %g)",
        len(magnet_strengths),
        observables,
        momentum_points,
        prior_strength,
    )

    reference = closed_orbit_at(
        accelerator,
        magnet_strengths,
        reference_pt,
        corrector_knobs=corrector_knobs,
        tune_knobs=tune_knobs,
    )
    uncertainty_map = {
        str(name): float(value) for name, value in (uncertainties or {}).items()
    }
    coverage = {
        float(pt): tuple(str(name) for name in frame.index)
        for pt, frame in measurements.items()
    }
    return MomentumReference(
        magnet_strengths=dict(magnet_strengths),
        uncertainties=uncertainty_map,
        closed_orbit=reference,
        reference_pt=float(reference_pt),
        observables=observables,
        prior_strength=float(prior_strength),
        momentum_points=momentum_points,
        bpm_coverage=coverage,
        fit_settings={
            "sequence_range": (sequence_config or SequenceConfig(magnet_range="$start/$end")).magnet_range,
            # The knobs themselves, not a path: they are tens of values, and a
            # path names a file that later runs rewrite -- which makes a saved
            # fit unreproducible from its own metadata.
            "corrector_knobs": _knobs_provenance(corrector_knobs),
            "tune_knobs": _knobs_provenance(tune_knobs),
            "lm_config": repr(lm_config or fitter.lm_config),
        },
        diagnostics={**fitter.diagnostics, "n_knobs": len(magnet_strengths)},
    )


__all__ = [
    "DEFAULT_PRIOR_STRENGTH",
    "GRADIENT_SENSITIVE",
    "ORBIT_AND_PHASE",
    "REFERENCE_COLUMNS",
    "MomentumReference",
    "closed_orbit_at",
    "fit_momentum_reference",
]
