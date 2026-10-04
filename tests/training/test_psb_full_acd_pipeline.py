"""End-to-end PSB closed-orbit, ACD reconstruction, and quadrupole fit.

This deliberately stays in the ordinary serial suite: it is the executable
contract between the PSB measurement pipeline, tmom-recon, omc3 and the
ACD-marker optimiser. Each excitation method (ACD-driven and single-kick) has
4 cases: noise on/off crossed with a fast/slow optimisation budget.

pytest /afs/cern.ch/work/j/jmgray/private/sgd-magnet-tuner/tests/training/test_psb_full_acd_pipeline.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import tfs
from omc3.model_creator import create_instance_and_model
from pymadng_utils.io.utils import read_knobs, save_knobs
from pymadng_utils.mad.model_creator_mad_interface import ModelCreatorMadInterface
from pymadng_utils.madx.tfs_utils import convert_multiple_tfs_files
from xtrack_tools.acd import run_ac_dipole_tracking
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.mad import GradientDescentMadInterface, merge_machine_states
from aba_optimiser.measurements.acd_pipeline import (
    ACDOpticsAnalysisConfig,
    build_mixed_closed_orbit_reference,
    run_driven_and_compensated_optics,
)
from aba_optimiser.measurements.reconstruction import _scale_position_variances_after_svd
from aba_optimiser.measurements.reference import reconstruction_frame
from aba_optimiser.measurements.variances import assign_known_noise_variances
from aba_optimiser.momentum_reference import ORBIT_AND_PHASE, fit_momentum_reference
from aba_optimiser.noise.noise import load_bpm_noise_table
from aba_optimiser.training.config.models import (
    KickerConfig,
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.data_manager import FileTracks
from aba_optimiser.training.tracking_fitter import ACDMarkerFitter, KickerFitter
from aba_optimiser.training_closed_twiss import LevenbergMarquardtConfig
from tests.training.controller_test_utils import (
    _generate_kicker_track,
    evaluate_controller_worker_loss,
    evaluate_controller_worker_losses,
)

pytest.importorskip("tmom_recon")

from tmom_recon import (  # noqa: E402
    ACDipoleConfig,
    ModelDetails,
    OpticsInput,
    calculate_acd_pz,
    calculate_pz,
)
from tmom_recon.acd.integration import (  # noqa: E402
    apply_precomputed_ac_dipole_bpm_overrides,
    resolve_ac_dipole_config,
)
from tmom_recon.acd.reconstruction import ACDipoleStateConsistencyError  # noqa: E402
from tmom_recon.measurements.twiss_from_measurement import (  # noqa: E402
    build_twiss_from_measurements,
)
from tmom_recon.reconstruction import ACDipolePzGenerator  # noqa: E402
from tmom_recon.svd import weighted_svd_clean_measurements  # noqa: E402

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    import xtrack as xt

    from aba_optimiser.analysis.degeneracy_checker import DegeneracyReport

    pytestmark = pytest.mark.serial

MAIN_STRENGTHS = {
    "kbrqf": 7.289003149414066e-01,
    "kbrqd": -7.442765966796879e-01,
}
RAMP_TURNS = 1_000
FLATTOP_TURNS = 10_000

#: Consecutive turn windows the flat-top is cut into before Harpy sees it.
#: omc3 needs more than one acquisition to estimate any measurement error at
#: all; three keeps a usable spread without shortening the windows to the
#: point where the tune resolution suffers.
ACQUISITIONS = 3
OPTIMISATION_DATA_FRACTION = 0.250
BPM_PATTERN = r"(?i)^br3\.bpm\d+l3$"
BEND_REL_RMS = 8e-4
QUAD_REL_RMS = 2e-3
ACD_NAME = "HACMAP"
ACD_HORIZONTAL_EXCITATION = 0.0158188853291
ACD_VERTICAL_EXCITATION = 0.01078784
MAX_ACD_BPM_PEAK_TO_PEAK = 2.1e-3
# Measured on the 2026-08-14 campaign, 0 mm orbit, 24 acquisitions: the driven
# amplitude is 351.6 um (x) and 329.3 um (y) once the pre-AC-dipole variance is
# subtracted, against the ~995 um this test's default excitation produces. The BPM
# resolution over the same files' first 1000 turns is 77.6/76.7 um median, so the
# real per-turn SNR is 3.8 -- and 2.3 at BR3.BPM2L3, whose 149.7 um resolution is a
# standing instrument problem. The default profile above runs at about 16.
CAMPAIGN_EXCITATION_SCALE = 0.353
CAMPAIGN_NOISE_FILE = Path(__file__).resolve().parents[1] / "data" / "psb_bpm_noise_2026_08_14.txt"
# At the campaign SNR the state-consistency guard legitimately rejects some
# acquisitions -- psb_md measures 8 of 23 passing at 0 mm with the full chain, and
# the guard tolerance must never be widened to improve that. This test's chain is
# simpler than the campaign's, so it asks only that a majority survive.
MIN_CAMPAIGN_GUARD_PASS_FRACTION = 0.6
CAMPAIGN_SEEDS = 3
PSB_ORBIT_BPM_NAMES = [f"BR3.BPM{idx}L3" for idx in range(1, 17)]
PSB_X_ORBIT_CORRECTOR_NAMES = [
    "BR3.DHZ8L1",
    "BR3.DHZ9L1",
    "BR3.DHZ11L4",
    "BR3.DHZ12L4",
    "BR3.DHZ13L4",
    "BR3.DHZ14L1",
]
# Carrier-preservation limits. psb_md measures, over 23 real acquisitions and every
# chain it tried, a worst-case driven-amplitude change of 0.27% and a worst-case
# AC-dipole window relative-phase change of 0.27 mrad. These are set an order of
# magnitude looser because the simulated chain here also carries reconstruction
# error, not only preprocessing: tighten them if a measurement supports it.
MAX_CARRIER_AMPLITUDE_ERROR = 0.03
MAX_WINDOW_PHASE_ERROR = 3e-3  # rad
# The action is quadratic and reconstructed through the model, so a per-mille
# amplitude change and a few tenths of a mrad of phase compound into per-cent.
# psb_md measures preprocessing alone moving it 1.4% (x) / 2.9% (y), worst file 3.8%.
MAX_ACTION_ERROR = 0.10
# psb_md refuses to reconstruct in the dynamic-part frame against a model carrying
# more than 5e-5 m of closed orbit; the model there is meant to be flat, not nearly
# flat, because the data has had its orbit removed outright.
# Per-BPM constant px offset, as a fraction of the px signal std. A turn-independent
# offset is not averageable and displaces the loss minimum, so it must stay well under
# the signal. Measured 0.162 with the closed orbit correct; it was 1.28 while the ACD
# model silently dropped its correctors (tmom-recon set_knobs could not read a TFS table).
MAX_BPM_PX_BIAS_FRACTION = 0.3
# The truth knobs are the noise floor of the ACD objective.  Descending below
# that floor is by definition fitting the measurement noise, so the depth of the
# excursion is bounded relative to the genuine signal (initial - truth) that the
# fit had available.  Clean runs sit near -0.15 (fast) and 0.07 (slow); noisy
# runs that consume noise reach 1.3-1.9.
MAX_OVERFIT_DEPTH = 0.5
# The ACD fit starts from the momentum-reference fit, so beating the untouched
# nominal machine proves nothing; it has to improve on the point it started at.
# Clean runs reach 0.70-0.90 of the reference error, noise-limited runs 0.96+.
MAX_QUAD_RATIO_VS_REFERENCE = 0.93
# Final quadrupole error per (noise_factor, profile), so the slow profile can be
# held to the answer the cheap profile already reached.
_FINAL_QUAD_BY_CASE: dict[tuple[float, str], float] = {}

DPP_STEP = 1.2e-3
DRIVEN_TUNE_OFFSETS = (-0.0027, 0.0027)
REFERENCE_DPP = 0.0
DPP_VALUES = (-DPP_STEP, REFERENCE_DPP, DPP_STEP)


def _log_elapsed(stage: str, start: float, **details: Any) -> None:
    if details:
        detail_text = ", ".join(f"{key}={value}" for key, value in details.items())
        LOGGER.info("%s completed in %.1fs; %s", stage, time.perf_counter() - start, detail_text)
        return
    LOGGER.info("%s completed in %.1fs", stage, time.perf_counter() - start)


@dataclass(frozen=True)
class PipelineCase:
    """Accelerator-specific hooks; an LHC case can be added without branching."""

    name: str
    accelerator_factory: Callable[..., PSB]
    ring: int
    kinetic_energy: float
    bpm_pattern: str
    acd_name: str


@dataclass(frozen=True)
class MeasurementScenario:
    """One coherent closed-orbit configuration for the whole analysis chain.

    The closed orbit is what constrains the bends (a quadrupole on a centred orbit
    produces no deflection -- see the ``momentum_reference`` module docstring), and
    the orbit correctors exist only to close that orbit. So removing the orbit from
    the data forces bends and correctors out of the analysis at the same time, and
    these four settings cannot be chosen independently. Bundling them means an
    inconsistent combination cannot be assembled at a call site.

    ``dynamic-part`` is what production runs as ``--optimise-dynamic-part``.
    """

    name: str
    dynamic_planes: tuple[str, ...]
    #: Reference-fit observables. Phase alone cannot constrain bends.
    observables: tuple[str, ...]
    fit_bends: bool
    #: Whether the analysis models the orbit correctors at all.
    use_correctors: bool


#: Declared PSB BPM position resolution (m). The fixture's closed orbits are
#: exact model truth, so this is not their scatter -- it is the uncertainty a
#: real measurement of the same orbit would carry, and it is what sets the
#: orbit family's weight against phase in the reference fit.
PSB_BPM_RESOLUTION: float = 1e-4

FULL_ORBIT = MeasurementScenario(
    name="full-orbit",
    dynamic_planes=(),
    observables=ORBIT_AND_PHASE,
    fit_bends=True,
    use_correctors=True,
)
DYNAMIC_PART = MeasurementScenario(
    name="dynamic-part",
    dynamic_planes=("x", "y"),
    observables=("mu1", "mu2"),
    fit_bends=False,
    use_correctors=False,
)


def _reference_fit_errors(scenario: MeasurementScenario) -> dict[str, set[str]]:
    """Return the reference fit's error selection: quad k1, plus bend k0 if fitted."""
    return {"bend": {"k0"}, "quad": {"k1"}} if scenario.fit_bends else {"quad": {"k1"}}


SCENARIOS = [
    pytest.param(FULL_ORBIT, id="full-orbit"),
    pytest.param(DYNAMIC_PART, id="dynamic-part"),
]


@dataclass
class MachineArtifacts:
    case: PipelineCase
    accelerator: PSB
    truth: dict[str, float]
    tune_knobs: Path
    corrector_file: Path
    corrector_knobs: dict[str, float]
    closed_orbits: dict[float, pd.DataFrame]
    free_twiss: dict[float, pd.DataFrame]
    measured_tunes: dict[float, tuple[float, float]]
    tracking: dict[float, pd.DataFrame]


def _rms(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def _degenerate_summary(report: DegeneracyReport, limit: int = 3) -> str:
    """One-line, name-only summary of the weakest degenerate directions."""
    parts = []
    for direction in report.degenerate_directions[:limit]:
        top_knob = direction.contributions[0][0] if direction.contributions else "?"
        parts.append(f"{top_knob}(rel_eig={direction.relative_eigenvalue:.1e})")
    return ", ".join(parts) if parts else "none"


def _strength_fingerprint(strengths: dict[str, float]) -> str:
    payload = "\n".join(f"{name}={float(value):.17e}" for name, value in sorted(strengths.items()))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def _twiss_by_lower_name(twiss: pd.DataFrame) -> pd.DataFrame:
    frame = twiss.copy()
    columns = {str(column).lower(): column for column in frame.columns}
    names = frame[columns["name"]] if "name" in columns else frame.index
    frame.index = pd.Index([str(name).lower() for name in names])
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


# Orbit/momenta agreement between MAD-NG's `method=6` twiss (its exact
# thick-element integrator, matching xsuite's `_configure_line_models`
# drift-kick-drift-exact + yoshida4 setup) and the xsuite turn-averaged
# AC-dipole orbit. Measured residuals with method=6 sit at a dpp-independent
# few-1e-8 to 1e-7 floor (confirmed by direct convergence testing -- more xsuite
# slicing does not shrink it further, so this is not a discretization
# artifact to tune away). Without method=6 the same comparison degrades to
# ~1e-6 relative off-momentum -- that was the actual bug: MAD-NG's default
# twiss integrator is too low-order, not a real xsuite/MAD-NG model
# disagreement. Optics functions (beta/alfa/mu) are computed by genuinely
# different methods on each side (closed-solution twiss vs 4D tracking) and
# are intentionally NOT checked here -- some disagreement there is expected
# and not indicative of a bug; only orbit and momenta, which both sides
# should reproduce identically for the same lattice, are gated.
XSUITE_MADNG_ORBIT_TOL = 1.5e-7
XSUITE_MADNG_MOMENTA_TOL = 5e-8


def _assert_xsuite_matches_madng_closed_orbit(
    free_twiss: dict[float, pd.DataFrame], tracking: dict[float, pd.DataFrame]
) -> None:
    """Guard against xsuite/MAD-NG model divergence: both are built from the
    same truth magnet strengths and correctors, so their closed orbit and
    momenta must agree to within a tight tolerance -- any real disagreement
    here is a modelling bug, not physics, and should fail fast rather than
    surface many hours later as an unexplained fit-quality regression.
    """
    for dpp, twiss in free_twiss.items():
        madng = twiss[["x", "y", "px", "py"]].copy()
        madng.index = pd.Index([str(name).upper() for name in madng.index])
        bpm_rows = tracking[dpp].loc[tracking[dpp]["name"].str.match(BPM_PATTERN)]
        xsuite = bpm_rows.groupby("name", observed=True)[["x", "y", "px", "py"]].mean()
        common = madng.index.intersection(xsuite.index)
        assert len(common) == 16, f"dpp={dpp:+.4e}: expected 16 common BPMs, got {len(common)}"
        orbit_rms = _rms(
            madng.loc[common, ["x", "y"]].to_numpy(dtype=float)
            - xsuite.loc[common, ["x", "y"]].to_numpy(dtype=float)
        )
        momenta_rms = _rms(
            madng.loc[common, ["px", "py"]].to_numpy(dtype=float)
            - xsuite.loc[common, ["px", "py"]].to_numpy(dtype=float)
        )
        assert orbit_rms < XSUITE_MADNG_ORBIT_TOL, (
            f"xsuite/MAD-NG orbit mismatch at dpp={dpp:+.4e}: "
            f"rms={orbit_rms:.3e} >= {XSUITE_MADNG_ORBIT_TOL:.1e}"
        )
        assert momenta_rms < XSUITE_MADNG_MOMENTA_TOL, (
            f"xsuite/MAD-NG momenta mismatch at dpp={dpp:+.4e}: "
            f"rms={momenta_rms:.3e} >= {XSUITE_MADNG_MOMENTA_TOL:.1e}"
        )


def _pt_by_dpp(accelerator: PSB) -> dict[float, float]:
    return {dpp: float(accelerator.dp2pt(dpp)) for dpp in DPP_VALUES}


def _natural_tunes(machine: MachineArtifacts, dpp: float) -> tuple[float, float]:
    try:
        return machine.measured_tunes[dpp]
    except KeyError as exc:
        raise KeyError(f"No measured tunes stored for dpp={dpp:.17g}") from exc


def _driven_tunes(machine: MachineArtifacts, dpp: float) -> tuple[float, float]:
    qx, qy = _natural_tunes(machine, dpp)
    return _driven_tunes_from_natural(qx, qy)


def _driven_tunes_from_natural(qx: float, qy: float) -> tuple[float, float]:
    return ((qx + DRIVEN_TUNE_OFFSETS[0]) % 1.0, (qy + DRIVEN_TUNE_OFFSETS[1]) % 1.0)


def _fitted_optics_quality(
    machine: MachineArtifacts, strengths: dict[str, float]
) -> tuple[float, float]:
    """Return (phase-advance RMS, beta-beating RMS) of a knob set against the machine.

    Individual quadrupole strengths are not separately observable here: 16 BPMs
    constrain far fewer degrees of freedom than there are gradient knobs, and phase
    advance fixes only the integrated gradient between pickups. A fit is therefore
    free to move along the degenerate directions without changing anything it can
    see, which makes a per-magnet comparison against truth a poor measure of whether
    the fit worked. What the reconstruction actually consumes is the *optics* the
    combination of quads produces, so measure that.
    """
    accelerator = machine.case.accelerator_factory(
        ring=machine.case.ring,
        kinetic_energy=machine.case.kinetic_energy,
        sequence_file=machine.accelerator.sequence_file,
        errors={"bend": {"k0"}, "quad": {"k1"}},
    )
    # The reference fit is given the machine's correctors and tune knobs, and the
    # tune knobs in particular move the phase directly (kbrqf/kbrqd rematch the
    # model onto the measured tunes). Comparing its output against a bare lattice
    # would measure that missing rematch rather than the fit.
    iface = GradientDescentMadInterface(
        accelerator,
        machine_state=merge_machine_states(machine.corrector_file, machine.tune_knobs),
    )
    if strengths:
        iface.update_knob_values(strengths)
    twiss = _twiss_by_lower_name(_direct_twiss(iface, REFERENCE_DPP))
    del iface
    truth = _twiss_by_lower_name(machine.free_twiss[REFERENCE_DPP])
    phase = _twiss_phase_advance_rms(twiss, truth)
    common = twiss.index.intersection(truth.index)
    beating = []
    for column in ("beta11", "beta22"):
        if column in twiss and column in truth:
            model = twiss.loc[common, column].to_numpy(dtype=float)
            true = truth.loc[common, column].to_numpy(dtype=float)
            beating.append((model - true) / true)
    return phase, (_rms(np.concatenate(beating)) if beating else float("nan"))


def _relative_error_rms(estimate: dict[str, float], truth: dict[str, float], suffix: str) -> float:
    keys = [name for name in truth if name.lower().endswith(suffix)]
    return _rms([estimate.get(name, 0.0) - truth[name] for name in keys])


def _twiss_orbit_rms(
    twiss: pd.DataFrame, reference: pd.DataFrame, columns: tuple[str, ...]
) -> float:
    common = twiss.index.intersection(reference.index)
    residuals = []
    for column in columns:
        if column in twiss and column in reference:
            residuals.append(
                twiss.loc[common, column].to_numpy(dtype=float)
                - reference.loc[common, column].to_numpy(dtype=float)
            )
    if not residuals:
        return float("nan")
    return _rms(np.concatenate(residuals))


def _twiss_phase_advance_rms(model: pd.DataFrame, truth: pd.DataFrame) -> float:
    common = [
        name.lower()
        for name in PSB_ORBIT_BPM_NAMES
        if name.lower() in model.index and name.lower() in truth.index
    ]
    residuals = []
    for column in ("mu1", "mu2"):
        if column not in model or column not in truth:
            continue
        for left, right in zip(common, common[1:], strict=False):
            model_advance = float((model.loc[right, column] - model.loc[left, column]) % 1.0)
            truth_advance = float((truth.loc[right, column] - truth.loc[left, column]) % 1.0)
            residuals.append((model_advance - truth_advance + 0.5) % 1.0 - 0.5)
    if not residuals:
        return float("nan")
    return _rms(residuals)


def _truth_errors(iface: GradientDescentMadInterface) -> dict[str, float]:
    """Seed every bend and individual QFO/QDE gradient."""
    bends, _ = iface.apply_magnet_perturbations(
        rel_error=BEND_REL_RMS,
        seed=37,
        magnet_type="d",
    )
    quads, _ = iface.apply_magnet_perturbations(
        rel_error=QUAD_REL_RMS,
        seed=24,
        magnet_type="q",
    )
    return bends | quads


def _corrector_strengths(path: Path) -> dict[str, float]:
    frame = tfs.read(path)
    strengths: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        name = str(getattr(row, "name", getattr(row, "NAME", "")))
        kind = str(getattr(row, "kind", getattr(row, "KIND", ""))).lower()
        if "monitor" in kind or not name:
            continue
        for column in ("hkick", "vkick"):
            value = getattr(row, column, getattr(row, column.upper(), 0.0))
            if np.isfinite(value) and float(value) != 0.0:
                strengths[f"{name}.{column}"] = float(value)
    return strengths


def _direct_twiss(iface: GradientDescentMadInterface, deltap: float) -> pd.DataFrame:
    start = time.perf_counter()
    LOGGER.info("Running direct MAD-NG Twiss for dpp=%+.4e", deltap)
    # method=6 matches the exact thick-element integrator xsuite is configured
    # with (see xtrack_tools.env._configure_line_models); without it MAD-NG's
    # default integrator disagrees with xsuite by ~1e-6-1.7e-5 relative
    # off-momentum, vs ~1e-8 with method=6 -- confirmed by direct convergence
    # testing (more nslice beyond this does not tighten it further).
    frame = iface.run_twiss(observe=1, deltap=deltap, coupling=True, method=6)
    tunes = (float(frame.headers["q1"]), float(frame.headers["q2"]))
    bpms = frame.loc[frame.index.to_series().str.match(BPM_PATTERN)].copy()
    bpms.attrs["q1"], bpms.attrs["q2"] = tunes
    assert len(bpms) == 16, f"direct MAD-NG Twiss found {len(bpms)} PSB BPMs"
    _log_elapsed(
        "Direct MAD-NG Twiss",
        start,
        dpp=f"{deltap:+.4e}",
        q1=f"{tunes[0]:.6f}",
        q2=f"{tunes[1]:.6f}",
        bpms=len(bpms),
    )
    return bpms


def _track_one_momentum(
    env: xt.Environment,
    dpp: float,
    accelerator: PSB,
    driven_tunes: tuple[float, float],
    excitation_scale: float = 1.0,
) -> pd.DataFrame:
    start = time.perf_counter()
    LOGGER.info(
        "Tracking AC-dipole data for dpp=%+.4e with driven tunes qx=%.6f qy=%.6f, "
        "excitation_scale=%.4f",
        dpp,
        driven_tunes[0],
        driven_tunes[1],
        excitation_scale,
    )
    line = env[accelerator.seq_name.lower()].copy()
    twiss = line.twiss(method="4d", delta0=dpp)
    monitored = run_ac_dipole_tracking(
        line=line,
        acd_marker=ACD_NAME,
        sequence_name=accelerator.seq_name,
        tws=twiss,
        ramp_turns=RAMP_TURNS,
        flattop_turns=FLATTOP_TURNS,
        driven_tunes=list(driven_tunes),
        bpm_pattern=BPM_PATTERN,
        deltap=dpp,
        state_markers=True,
        horizontal_excitation=excitation_scale * ACD_HORIZONTAL_EXCITATION,
        vertical_excitation=excitation_scale * ACD_VERTICAL_EXCITATION,
    )
    tracked = process_tracking_data(monitored, ramp_turns=RAMP_TURNS, flattop_turns=FLATTOP_TURNS)
    tracked["name"] = tracked["name"].astype(str).str.upper()
    tracked["bunch_number"] = 0
    bpm_rows = tracked[tracked["name"].str.match(BPM_PATTERN)]
    assert bpm_rows["name"].nunique() == 16
    assert bpm_rows.groupby("name", observed=True)["turn"].nunique().eq(FLATTOP_TURNS).all()
    peak_to_peak = bpm_rows.groupby("name", observed=True)[["x", "y"]].agg(np.ptp)
    max_peak_to_peak = {plane: float(peak_to_peak[plane].max()) for plane in ("x", "y")}
    LOGGER.info(
        "ACD tracking amplitude; dpp=%+.4e, max_p2p_x=%.3e m, max_p2p_y=%.3e m",
        dpp,
        max_peak_to_peak["x"],
        max_peak_to_peak["y"],
    )
    peak_to_peak_limit = excitation_scale * MAX_ACD_BPM_PEAK_TO_PEAK
    assert max(max_peak_to_peak.values()) <= peak_to_peak_limit, (
        f"dpp={dpp:+.4e}: ACD BPM peak-to-peak amplitude exceeds "
        f"{peak_to_peak_limit:.1e} m: {max_peak_to_peak}"
    )
    _log_elapsed(
        "AC-dipole tracking",
        start,
        dpp=f"{dpp:+.4e}",
        rows=len(tracked),
        bpm_rows=len(bpm_rows),
        bpm_count=bpm_rows["name"].nunique(),
        turns=bpm_rows["turn"].nunique(),
    )
    return tracked


def _inject_bpm_noise(
    tracked: pd.DataFrame,
    noise_factor: float,
    rng: np.random.Generator,
    noise_file: Path | None = None,
) -> pd.DataFrame:
    """Inject the BPM resolution, declare it as production does, then clean.

    A real acquisition carries the full per-turn BPM resolution and only ever
    reaches the reconstruction through ``weighted_svd_clean_measurements``
    (``measurements/reconstruction.py``). Both halves belong in the simulation:
    injecting a pre-reduced residual instead would assume the cleaning gain that
    this path exists to measure.

    The declared variances come from ``assign_known_noise_variances`` -- the same
    production call, reading the same table the noise was drawn from -- rather than
    from the injected sample. Writing the drawn values back would let the test
    declare an uncertainty no real analysis can know.
    """
    LOGGER.info(
        "Injecting BPM resolution noise; noise_factor=%s, input_rows=%d, table=%s",
        noise_factor,
        len(tracked),
        noise_file or "packaged",
    )
    bpm = tracked.loc[tracked["name"].str.match(BPM_PATTERN)].copy()
    resolution = load_bpm_noise_table("psb", noise_file=noise_file).set_index("name")
    std_x = noise_factor * bpm["name"].map(resolution["Horizontal_STD"]).to_numpy(dtype=float)
    std_y = noise_factor * bpm["name"].map(resolution["Vertical_STD"]).to_numpy(dtype=float)
    bpm["x"] += rng.normal(0.0, std_x)
    bpm["y"] += rng.normal(0.0, std_y)
    bpm["bunch_number"] = 0
    if noise_factor:
        # Production indexes by name across this call (see _assign_variances).
        bpm = assign_known_noise_variances(
            bpm.set_index("name"), bad_bpms=[], accelerator_type="psb", noise_file=noise_file
        ).reset_index()
    else:
        # An exact-position run still needs finite weights to divide by.
        bpm["var_x"] = 1e-30
        bpm["var_y"] = 1e-30
    # calculate_pz fills the momentum variances from the propagated position
    # variances and the optics errors; these are only placeholders until then.
    bpm["var_px"] = 1.0
    bpm["var_py"] = 1.0
    if noise_factor:
        noisy = bpm[["name", "turn", "x", "y"]].copy()
        cleaned = weighted_svd_clean_measurements(bpm)
        _log_svd_cleaning_gain(truth=tracked, noisy=noisy, cleaned=cleaned)
        # Production reduces the declared position variance by the gain the SVD
        # actually achieved; mirror it so the weights match the cleaned data.
        bpm = _scale_position_variances_after_svd(
            cleaned,
            n_bpms=int(cleaned["name"].nunique()),
            svd_ranks=(cleaned.attrs.get("svd_rank_x"), cleaned.attrs.get("svd_rank_y")),
        )
    LOGGER.info(
        "Injected BPM noise; rows=%d, bpms=%d, std_x_rms=%.3e m, std_y_rms=%.3e m",
        len(bpm),
        bpm["name"].nunique(),
        _rms(std_x),
        _rms(std_y),
    )
    return bpm


def _log_svd_cleaning_gain(
    *, truth: pd.DataFrame, noisy: pd.DataFrame, cleaned: pd.DataFrame
) -> dict[str, float]:
    """Report how much position error the SVD cleaning actually removes.

    Gains are RMS(noisy - truth) / RMS(cleaned - truth), so the residual the
    reconstruction actually sees is measured rather than assumed.
    """
    reference = truth.set_index(["name", "turn"])
    noisy_indexed = noisy.set_index(["name", "turn"])
    cleaned_indexed = cleaned.set_index(["name", "turn"])
    gains: dict[str, float] = {}
    for plane in ("x", "y"):
        exact = reference.loc[noisy_indexed.index, plane].to_numpy(dtype=float)
        before = _rms(noisy_indexed[plane].to_numpy(dtype=float) - exact)
        after = _rms(cleaned_indexed.loc[noisy_indexed.index, plane].to_numpy(dtype=float) - exact)
        gains[plane] = before / after if after > 0.0 else float("inf")
        LOGGER.info(
            "SVD cleaning gain; plane=%s, rank=%s, err_before=%.3e m, err_after=%.3e m, gain=%.2fx",
            plane,
            cleaned.attrs.get(f"svd_rank_{plane}"),
            before,
            after,
            gains[plane],
        )
    return gains


def _create_psb_omc3_model(
    *,
    root: Path,
    dpp: float,
    machine: MachineArtifacts,
) -> Path:
    """Create the PSB-specific model consumed by the neutral HIO helper."""
    start = time.perf_counter()
    natural_tunes = _natural_tunes(machine, dpp)
    driven_tunes = _driven_tunes(machine, dpp)
    model_dir = root / "model"
    LOGGER.info(
        "Creating PSB OMC3 model for dpp=%+.4e at %s; nat=(%.6f, %.6f), drv=(%.6f, %.6f)",
        dpp,
        model_dir,
        natural_tunes[0],
        natural_tunes[1],
        driven_tunes[0],
        driven_tunes[1],
    )
    create_instance_and_model(
        outputdir=model_dir,
        type="nominal",
        logfile=model_dir / "madx.log",
        fetch="path",
        path=Path(__file__).resolve().parents[1] / "data" / "acc-models-psb",
        list_choices=False,
        show_help=False,
        accel="psbooster",
        nat_tunes=list(natural_tunes),
        dpp=dpp,
        scenario="lhc_indiv",
        year="2026",
        cycle_point="1_flat_bottom",
        str_file="psb_fb_lhcindiv.str",
        ring=machine.case.ring,
        driven_excitation="acd",
        drv_tunes=list(driven_tunes),
    )
    with ModelCreatorMadInterface(
        accelerator=machine.accelerator,
        model_dir=model_dir,
        tunes=list(natural_tunes),
        drv_tunes=list(driven_tunes),
        deltap=dpp,
    ) as model:
        model.set_madx_variables(**read_knobs(machine.tune_knobs))
        model.set_magnet_strengths(machine.truth)
        correctors = tfs.read(machine.corrector_file)
        for row in correctors.itertuples(index=False):
            kind = str(row.kind).lower()
            if kind == "monitor":
                continue
            element_name = str(row.ename)
            if kind == "hkicker":
                model.mad.send(f"loaded_sequence['{element_name}'].kick = {float(row.hkick):.17g}")
            elif kind == "vkicker":
                model.mad.send(f"loaded_sequence['{element_name}'].kick = {float(row.vkick):.17g}")
            elif kind == "tkicker":
                model.mad.send(
                    f"loaded_sequence['{element_name}'].hkick = {float(row.hkick):.17g}; "
                    f"loaded_sequence['{element_name}'].vkick = {float(row.vkick):.17g}"
                )
        model.compute_and_export_twiss_tables(export_madx_names=True)
    convert_multiple_tfs_files(
        [
            model_dir / "twiss.dat",
            model_dir / "twiss_ac.dat",
            model_dir / "twiss_elements.dat",
        ]
    )
    _log_elapsed("PSB OMC3 model creation", start, dpp=f"{dpp:+.4e}", model_dir=model_dir)
    return model_dir


def _split_into_acquisitions(
    bpm_data: pd.DataFrame, label: str, count: int
) -> list[tuple[Path, pd.DataFrame]]:
    """Cut one long acquisition into ``count`` consecutive ones.

    omc3 takes its measurement errors from the spread *between* acquisitions --
    ``optics_measurements.phase._get_phases`` writes a matrix of exact zeros when
    handed a single file -- so a one-file analysis reports no phase error at all,
    and the reference fit, which weights by ``1/variance``, cannot use the phase.

    Splitting the flat-top into consecutive turn windows is how the same estimate
    is obtained from one real acquisition: each window is an independent sample of
    the same driven motion, and their scatter measures everything that actually
    limits the phase -- BPM noise, decoherence, tune drift and spectral leakage --
    rather than a resolution figure asserted up front. Turns are renumbered from
    zero in each window so Harpy sees them as separate acquisitions.

    Args:
        bpm_data: Long-form ``name/turn/x/y`` turn-by-turn data.
        label: Stem for the per-acquisition source file names.
        count: Number of windows to cut.

    Returns:
        ``(source_file, frame)`` pairs, one per window.

    Raises:
        ValueError: If ``count`` is below two, or the windows would be empty.
    """
    if count < 2:
        raise ValueError(f"At least two acquisitions are needed for a spread, got {count}")
    turns = np.sort(bpm_data["turn"].unique())
    window = len(turns) // count
    if window < 1:
        raise ValueError(f"{len(turns)} turns cannot be split into {count} acquisitions")

    acquisitions: list[tuple[Path, pd.DataFrame]] = []
    for index in range(count):
        wanted = turns[index * window : (index + 1) * window]
        frame = bpm_data[bpm_data["turn"].isin(wanted)].copy()
        frame["turn"] = frame["turn"] - wanted[0]
        acquisitions.append((Path(f"{label}_acq{index}.sdds"), frame))
    LOGGER.info(
        "Split %d turns into %d acquisitions of %d turns each for %s",
        len(turns),
        count,
        window,
        label,
    )
    return acquisitions


def _run_phase_analysis(
    *,
    root: Path,
    dpp: float,
    bpm_data: pd.DataFrame,
    machine: MachineArtifacts,
    clean: bool = False,
) -> tuple[Path, Path]:
    start = time.perf_counter()
    nat_tunes = _natural_tunes(machine, dpp)
    driven_tunes = _driven_tunes(machine, dpp)
    label = f"dpp_{dpp:+.4f}".replace("+", "p").replace("-", "m")
    stage_root = root / label
    LOGGER.info(
        "Running driven/compensated optics for dpp=%+.4e; bpm_rows=%d, stage_root=%s",
        dpp,
        len(bpm_data),
        stage_root,
    )
    model_dir = _create_psb_omc3_model(
        root=stage_root,
        dpp=dpp,
        machine=machine,
    )
    acquisitions = _split_into_acquisitions(bpm_data, label, ACQUISITIONS)
    driven, compensated = run_driven_and_compensated_optics(
        acquisitions,
        output_dir=stage_root,
        config=ACDOpticsAnalysisConfig(
            model_dir=model_dir,
            harpy_options={
                "unit": "m",
                "turns": [0, FLATTOP_TURNS // ACQUISITIONS],
                "clean": clean,
                "keep_exact_zeros": True,
                "peak_to_peak": 1e-10,
                "max_peak": 0.02,
                "tunes": [*driven_tunes, 0.0],
                "nattunes": [*nat_tunes, 0.0],
                "output_bits": 12,
                "turn_bits": 16,
                "tolerance": 1e-3,
                "tune_clean_limit": 1e-3,
                "to_write": ["full_spectra", "lin"],
            },
            optics_options={
                "three_bpm_method": True,
                "accel": "psbooster",
                "ring": machine.case.ring,
            },
        ),
    )
    _log_elapsed(
        "Driven/compensated optics",
        start,
        dpp=f"{dpp:+.4e}",
        driven=driven,
        compensated=compensated,
    )
    return driven, compensated


def _phase_rms(optics_dir: Path, truth: pd.DataFrame) -> float:
    errors: list[float] = []
    for plane, mu_column in (("x", "mu1"), ("y", "mu2")):
        phase = tfs.read(optics_dir / f"phase_{plane}.tfs")
        names = phase["NAME"].astype(str).str.upper()
        names2 = phase["NAME2"].astype(str).str.upper()
        measured_column = f"PHASE{plane.upper()}"
        for name, name2, measured in zip(names, names2, phase[measured_column], strict=True):
            if name not in truth.index or name2 not in truth.index:
                continue
            expected = float((truth.loc[name2, mu_column] - truth.loc[name, mu_column]) % 1.0)
            residual = (float(measured) - expected + 0.5) % 1.0 - 0.5
            errors.append(residual)
    rms = _rms(errors)
    LOGGER.info(
        "Phase RMS for %s; pairs=%d, rms=%.3e",
        optics_dir,
        len(errors),
        rms,
    )
    return rms


def _fit_reference(
    machine: MachineArtifacts,
    compensated_dirs: dict[float, Path],
    root: Path,
    scenario: MeasurementScenario = FULL_ORBIT,
    noise_factor: float | None = None,
):
    start = time.perf_counter()
    LOGGER.info(
        "Fitting momentum reference; scenario=%s, observables=%s, bends=%s, "
        "correctors=%s, pt_values=%s, compensated_dirs=%s",
        scenario.name,
        scenario.observables,
        scenario.fit_bends,
        scenario.use_correctors,
        [f"{pt:+.6e}" for pt in compensated_dirs],
        [str(path) for path in compensated_dirs.values()],
    )
    pt_by_dpp = _pt_by_dpp(machine.accelerator)
    measurements: dict[float, pd.DataFrame] = {}
    for dpp, orbit in machine.closed_orbits.items():
        pt = pt_by_dpp[dpp]
        LOGGER.info(
            "Building measured Twiss for reference fit; dpp=%+.4e, pt=%+.6e, orbit_bpms=%d",
            dpp,
            pt,
            len(orbit),
        )
        optics, _ = build_twiss_from_measurements(
            compensated_dirs[pt], include_errors=True, use_amplitude_beta=True
        )
        if noise_factor == 0.0:
            # No noise to scatter, so use the declared error like orbit does.
            expected_phase_error = _phase_rms(compensated_dirs[pt], machine.free_twiss[dpp])
            if np.isfinite(expected_phase_error) and expected_phase_error > 0.0:
                LOGGER.info(
                    "Overriding phase variance for dpp=%+.4e (noise_factor=0) with "
                    "expected error %.3e turns (was scatter-derived)",
                    dpp,
                    expected_phase_error,
                )
                # mu1_var/mu2_var are cumulative (diffed downstream), so a flat
                # value would zero out every per-interval variance.
                cumulative = np.arange(1, len(optics) + 1, dtype=float) * expected_phase_error**2
                optics["mu1_var"] = cumulative
                optics["mu2_var"] = cumulative
        common = optics.index.intersection(orbit.index)
        frame = optics.loc[common].copy()
        for column in ("X", "Y", "ERRX", "ERRY"):
            frame[column] = orbit.loc[common, column]
        measurements[pt] = frame
        LOGGER.info(
            "Reference-fit measurement assembled; dpp=%+.4e, pt=%+.6e, optics_rows=%d, common_bpms=%d",
            dpp,
            pt,
            len(optics),
            len(common),
        )
    accelerator = machine.case.accelerator_factory(
        ring=machine.case.ring,
        kinetic_energy=machine.case.kinetic_energy,
        sequence_file=machine.accelerator.sequence_file,
        errors=_reference_fit_errors(scenario),
    )
    reference = fit_momentum_reference(
        accelerator,
        measurements,
        observables=scenario.observables,
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        lm_config=LevenbergMarquardtConfig(max_iterations=50),
        reference_pt=_pt_by_dpp(machine.accelerator)[REFERENCE_DPP],
        machine_state=merge_machine_states(machine.corrector_file if scenario.use_correctors else None, machine.tune_knobs),
        output_config=OutputConfig(
            write_tensorboard_logs=False,
            include_uncertainty=False,
            mad_logfile=root / "reference_fit_mad.log",
        ),
    )
    _log_elapsed(
        "Momentum reference fit",
        start,
        strengths=len(reference.magnet_strengths),
        uncertainties=len(reference.uncertainties),
    )
    return reference, measurements


def _mixed_reference(
    machine: MachineArtifacts, fitted, scenario: MeasurementScenario = FULL_ORBIT
) -> pd.DataFrame:
    """Build the closed-orbit reference in the frame the scenario reconstructs in.

    ``full-orbit`` mixes the measured positions with the fitted model momenta, which
    is what the production PSB reconstruction does.

    ``dynamic-part`` uses the simulated setting-zero acquisition as its measured
    orbit zero. No model-orbit or per-file-mean fallback is permitted.
    """
    if not scenario.dynamic_planes:
        LOGGER.info("Building mixed closed-orbit reference")
        measured = machine.free_twiss[REFERENCE_DPP]
        reference = build_mixed_closed_orbit_reference(measured[["x", "y"]], fitted.closed_orbit)
    else:
        reference = machine.free_twiss[REFERENCE_DPP][["x", "px", "y", "py"]].copy()
    fitted_angles = fitted.closed_orbit[["px", "py"]].to_numpy(dtype=float)
    reference_angles = reference[["px", "py"]].to_numpy(dtype=float)
    LOGGER.warning(
        "Fitted closed-orbit angle check; scenario=%s, fitted_px_rms=%.3e rad, "
        "fitted_py_rms=%.3e rad, reference_px_rms=%.3e rad, "
        "reference_py_rms=%.3e rad, fitted_max_abs_px=%.3e rad, "
        "fitted_max_abs_py=%.3e rad",
        scenario.name,
        _rms(fitted_angles[:, 0]),
        _rms(fitted_angles[:, 1]),
        _rms(reference_angles[:, 0]),
        _rms(reference_angles[:, 1]),
        float(np.max(np.abs(fitted_angles[:, 0]))),
        float(np.max(np.abs(fitted_angles[:, 1]))),
    )
    assert len(reference) == 16
    LOGGER.info("Closed-orbit reference built; rows=%d, scenario=%s", len(reference), scenario.name)
    return reference


def _marker_rmse(estimate: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    estimate = estimate.assign(name=estimate["name"].astype(str).str.upper())
    metrics: dict[str, float] = {}
    for side in ("before", "after"):
        marker = f"{ACD_NAME}_{side}".upper()
        expected = truth.loc[truth["name"] == marker, ["turn", "px", "py"]]
        actual = estimate.loc[estimate["name"] == marker, ["turn", "px", "py"]]
        merged = actual.merge(expected, on="turn", suffixes=("_fit", "_true"))
        assert len(merged) == FLATTOP_TURNS, f"{marker}: retained {len(merged)} turns"
        px_residual = merged["px_fit"].to_numpy() - merged["px_true"].to_numpy()
        py_residual = merged["py_fit"].to_numpy() - merged["py_true"].to_numpy()
        metrics[side] = _rms(
            np.concatenate(
                [
                    px_residual - px_residual.mean(),
                    py_residual - py_residual.mean(),
                ]
            )
        )
        LOGGER.info(
            "ACD marker RMSE; marker=%s, turns=%d, rmse=%.3e",
            marker,
            len(merged),
            metrics[side],
        )
    return metrics


def _bpm_momentum_bias(
    estimate: pd.DataFrame,
    truth: pd.DataFrame,
    dpp: float,
    scenario: MeasurementScenario = FULL_ORBIT,
    closed_orbit: pd.DataFrame | None = None,
) -> float:
    """Return the per-BPM constant momentum offset, as a fraction of the px signal.

    The *residual* is deliberately never mean-subtracted, unlike :func:`_marker_rmse`.
    A turn-independent offset survives any amount of turn averaging and shifts the
    loss minimum away from the truth knobs, so it is the one reconstruction error the
    fit cannot absorb -- and mean-subtracting is exactly what hid a bias larger than
    the signal itself.

    What *is* frame-dependent is the truth it is compared against. A dynamic-part
    reconstruction subtracts the measured closed orbit at *zero momentum*, so it
    carries no on-momentum closed-orbit angle -- but off momentum it does still
    carry the dispersive part, ``D'x.delta``, because that is real machine motion
    and the zero-momentum orbit does not contain it. So the frame match is to
    remove the machine's on-momentum closed-orbit angle from the truth and nothing
    else.

    Removing the truth's per-BPM *mean* instead is wrong: that also strips the
    dispersive angle, which the estimate legitimately keeps, and then reports the
    difference as bias. Measured on the PSB fixture at dpp=-1.2e-3, the estimate's
    constant px is 5.104e-4 and the machine's is 5.292e-4, and their difference is
    4.228e-4 -- matching the on-momentum closed-orbit angle of 4.222e-4 to 0.14%.
    The reconstruction is right; only the comparison frame was wrong.
    """
    estimate = estimate.assign(name=estimate["name"].astype(str).str.upper())
    truth = truth.assign(name=truth["name"].astype(str).str.upper())
    # The machine's own per-BPM constant px, before any frame subtraction. Off
    # momentum this is the dispersive closed-orbit angle D'x.delta, which is real
    # motion the machine has -- so it is the number the reconstruction's own
    # constant px has to be compared against to tell a dispersion mismatch from a
    # frame error.
    raw_true_mean = truth.groupby("name", observed=True)["px"].mean()
    if scenario.dynamic_planes:
        if closed_orbit is None:
            raise ValueError(
                "A dynamic-part comparison needs the machine's on-momentum closed "
                "orbit to put the truth in the reconstruction's frame"
            )
        orbit = closed_orbit.copy()
        orbit.index = orbit.index.astype(str).str.upper()
        truth = truth.copy()
        # The truth table also carries the AC-dipole marker rows, which have no
        # entry in the 16-BPM closed orbit and are filtered out below anyway. Only
        # the rows that actually reach the comparison need the frame shift.
        is_bpm = truth["name"].str.match(BPM_PATTERN)
        for column in ("px", "py"):
            offsets = truth["name"].map(orbit[column])
            missing = sorted(set(truth.loc[is_bpm & offsets.isna(), "name"]))
            if missing:
                raise ValueError(f"closed orbit is missing {column} for BPM(s) {missing}")
            truth[column] = truth[column] - offsets.fillna(0.0).to_numpy(dtype=float)
    merged = estimate.merge(
        truth[["name", "turn", "px", "py"]], on=["name", "turn"], suffixes=("_fit", "_true")
    )
    merged = merged[merged["name"].str.match(BPM_PATTERN)]
    assert merged["name"].nunique() == 16, f"dpp={dpp:+.4e}: {merged['name'].nunique()} BPMs"
    per_bpm = (
        merged.assign(
            px_res=merged["px_fit"] - merged["px_true"],
            py_res=merged["py_fit"] - merged["py_true"],
        )
        .groupby("name", observed=True)[["px_res", "py_res"]]
        .mean()
    )
    px_bias, py_bias = _rms(per_bpm["px_res"]), _rms(per_bpm["py_res"])
    signal = float(merged["px_true"].std())
    fraction = px_bias / signal
    # Split the residual across its two possible sources. In a dynamic-part
    # frame both sides are supposed to carry no closed-orbit momentum, so a
    # large ``fit`` mean means the reconstruction kept an orbit it should have
    # dropped, whereas a large ``true`` mean means the subtraction above did not
    # put the truth in the frame the estimate is actually in.
    fit_mean = merged.groupby("name", observed=True)["px_fit"].mean()
    true_mean = merged.groupby("name", observed=True)["px_true"].mean()
    LOGGER.info(
        "BPM momentum bias; dpp=%+.4e, px_bias_rms=%.3e, py_bias_rms=%.3e, "
        "px_signal_std=%.3e, bias/signal=%.3f, worst_bpm=%.3e, "
        "px_fit_mean_rms=%.3e, px_true_mean_rms=%.3e, "
        "px_true_raw_mean_rms=%.3e, px_fit_minus_raw_true_rms=%.3e",
        dpp,
        px_bias,
        py_bias,
        signal,
        fraction,
        per_bpm["px_res"].abs().max(),
        _rms(fit_mean),
        _rms(true_mean),
        _rms(raw_true_mean),
        _rms(fit_mean.subtract(raw_true_mean, fill_value=np.nan).dropna()),
    )
    return fraction


def _reconstruct_one(
    *,
    root: Path,
    dpp: float,
    bpm_data: pd.DataFrame,
    compensated_dir: Path,
    machine: MachineArtifacts,
    fitted,
    reference_co: pd.DataFrame,
    scenario: MeasurementScenario = FULL_ORBIT,
) -> tuple[Path, Any, Any, dict[str, float], dict[str, float]]:
    start = time.perf_counter()
    pt = _pt_by_dpp(machine.accelerator)[dpp]
    LOGGER.info(
        "Reconstructing ACD momenta for dpp=%+.4e, pt=%+.6e; scenario=%s, "
        "bpm_rows=%d, compensated_dir=%s",
        dpp,
        pt,
        scenario.name,
        len(bpm_data),
        compensated_dir,
    )
    model_details = ModelDetails(
        accelerator=machine.case.accelerator_factory(
            ring=machine.case.ring,
            kinetic_energy=machine.case.kinetic_energy,
            sequence_file=machine.accelerator.sequence_file,
            errors=_reference_fit_errors(scenario),
        ),
        pt=pt,
        magnet_strengths=fitted.magnet_strengths,
        machine_state=merge_machine_states(machine.corrector_file if scenario.use_correctors else None, machine.tune_knobs),
    )
    LOGGER.info(
        "Reconstruction model strengths; dpp=%+.4e, count=%d, bends=%d, quads=%d, fingerprint=%s",
        dpp,
        len(fitted.magnet_strengths),
        sum(name.lower().endswith(".dk0l") for name in fitted.magnet_strengths),
        sum(name.lower().endswith(".dk1l") for name in fitted.magnet_strengths),
        _strength_fingerprint(fitted.magnet_strengths),
    )
    acd_config = ACDipoleConfig(
        ac_dipole_marker=machine.case.acd_name,
        driven_tunes=_driven_tunes(machine, dpp),
    )
    resolved_acd = resolve_ac_dipole_config(model_details, acd_config)
    model_closed_orbit = _twiss_by_lower_name(resolved_acd.closed_orbit_tws)
    model_optics = _twiss_by_lower_name(resolved_acd.optics_tws)
    reference = _twiss_by_lower_name(reference_co)
    truth = _twiss_by_lower_name(machine.free_twiss[dpp])
    common_bpms = model_optics.index.intersection(truth.index)
    required = ("x", "px", "y", "py", "mu1", "mu2", "beta11", "beta22", "alfa11", "alfa22")
    missing = [column for column in required if column not in model_optics]
    phase_advance_rms = _twiss_phase_advance_rms(model_optics, truth)
    orbit_xy_rms = _twiss_orbit_rms(model_closed_orbit, reference, ("x", "y"))
    orbit_angle_rms = _twiss_orbit_rms(model_closed_orbit, reference, ("px", "py"))
    LOGGER.info(
        "ModelDetails Twiss diagnostics; dpp=%+.4e, bpm_overlap=%d, missing=%s, "
        "q_model=(%.6f, %.6f), q_truth=(%.6f, %.6f), orbit_xy_rms=%.3e, "
        "orbit_angle_rms=%.3e, phase_advance_rms=%.3e",
        dpp,
        len(common_bpms),
        missing,
        float(resolved_acd.optics_tws.headers["q1"]) % 1.0,
        float(resolved_acd.optics_tws.headers["q2"]) % 1.0,
        float(truth.attrs["q1"]) % 1.0,
        float(truth.attrs["q2"]) % 1.0,
        orbit_xy_rms,
        orbit_angle_rms,
        phase_advance_rms,
    )
    assert len(common_bpms) == 16
    assert not missing
    # Free-lattice optics at the exciter, needed to turn the marker states into a
    # Courant-Snyder action. Taken from the undriven closed-orbit twiss, not the
    # driven optics: install_ac_dipole replaces and removes the exciter element, so
    # it does not survive into optics_tws. The exciter is a thin kick, so beta/alfa
    # are equal on both sides of it and one row serves both marker states. Looked up
    # loudly -- psb_md section 8.1 records a silent NaN here costing a whole survey.
    marker_key = machine.case.acd_name.lower()
    assert marker_key in model_closed_orbit.index, (
        f"AC-dipole marker {machine.case.acd_name} missing from the closed-orbit "
        f"twiss; cannot compute the marker action. Nearest names: "
        f"{[n for n in model_closed_orbit.index if 'des' in str(n)][:5]}"
    )
    marker_optics = model_closed_orbit.loc[marker_key, ["beta11", "alfa11", "beta22", "alfa22"]]
    # The all-BPM neighbour-pair reconstruction must not transport across the
    # exciter: the free model optics contain no kick, so a pair that brackets it is
    # propagated through a deflection the model does not know about. That is exactly
    # the two window pickups, and BR3.BPM2L3 is where the state guard then fails.
    # Production always supplies this (psb_md passes barrier_s=acdipole_window.ac_s);
    # it defaults to None, so omitting it fails silently.
    # The barrier belongs to the reconstruction call, not to the AC-dipole model.
    barrier_s = float(model_closed_orbit.loc[marker_key, "s"])
    LOGGER.info(
        "AC-dipole barrier set; dpp=%+.4e, marker=%s, barrier_s=%.6f m",
        dpp,
        machine.case.acd_name,
        barrier_s,
    )
    assert np.isfinite(model_closed_orbit[["x", "px", "y", "py"]].to_numpy(dtype=float)).all()
    assert np.isfinite(
        model_optics[["mu1", "mu2", "beta11", "beta22", "alfa11", "alfa22"]].to_numpy(dtype=float)
    ).all()
    # Production reaches calculate_pz through process_single_dataframe, which
    # preprocesses and filters first. Mirror that order here; skipping it is what
    # let this path drift from the pipeline it is supposed to represent.
    model_twiss = model_closed_orbit.loc[:, ~model_closed_orbit.columns.duplicated()].copy()
    model_twiss.index = model_twiss.index.astype(str).str.upper()
    prepared = bpm_data[bpm_data["name"].isin(model_twiss.index)]
    frame = reconstruction_frame(reference_co, dynamic_planes=scenario.dynamic_planes)
    LOGGER.info(
        "Measurement framed; dpp=%+.4e, scenario=%s, dynamic_planes=%s, rows=%d -> %d",
        dpp,
        scenario.name,
        scenario.dynamic_planes,
        len(bpm_data),
        len(prepared),
    )
    # ``acd=`` puts the all-BPM reconstruction on the driven optics; the kick fit
    # is now its own call rather than an attrs side effect.
    #
    # Dispersion stays on the model, as it does in production
    # (aba_optimiser/measurements/reconstruction.py): a single AC-dipole
    # acquisition measures no dispersion, so the compensated omc3 directory has
    # none. OpticsInput refuses to substitute the model silently, so requesting
    # "dispersion": "measurement" here raises rather than warning.
    result = calculate_pz(
        prepared,
        model_details,
        closed_orbit_at_zero=frame.closed_orbit_at_zero,
        orbit_mode=frame.orbit_mode,
        optics=OpticsInput(
            measurement_dir=compensated_dir,
            sources={"phase": "measurement"},
        ),
        acd=acd_config,
        info=False,
        barrier_s=barrier_s,
    )
    assert isinstance(result, pd.DataFrame)
    acd_result = calculate_acd_pz(
        prepared,
        model_details,
        acd_config,
        closed_orbit_at_zero=frame.closed_orbit_at_zero,
        orbit_mode=frame.orbit_mode,
        optics=OpticsInput(
            measurement_dir=compensated_dir,
            sources={"phase": "measurement"},
        ),
    )
    result.attrs["acd_result"] = acd_result
    acd_result.attrs["marker_optics"] = marker_optics
    requested_driven = _driven_tunes(machine, dpp)
    LOGGER.info(
        "ACD model inputs; dpp=%+.4e, scenario=%s, driven_requested=(%.6f, %.6f), "
        "driven_model=(%.6f, %.6f), driven_fitted=(%.6f, %.6f), pt=%+.6e, "
        "pt_used=%+.6e, bends_in_strengths=%d, correctors=%s",
        dpp,
        scenario.name,
        requested_driven[0],
        requested_driven[1],
        float(resolved_acd.optics_tws.headers["q1"]) % 1.0,
        float(resolved_acd.optics_tws.headers["q2"]) % 1.0,
        float(acd_result.attrs["dpx_tune"]),
        float(acd_result.attrs["dpy_tune"]),
        pt,
        float(acd_result.attrs["pt_used"]),
        sum(name.lower().endswith(".dk0l") for name in fitted.magnet_strengths),
        scenario.use_correctors,
    )
    for record in acd_result.attrs.get("acd_state_consistency", []):
        LOGGER.info(
            "ACD state guard; dpp=%+.4e, scenario=%s, bpm=%s, coord=%s, "
            "max_residual=%.3e, state=%.3e, fraction=%.1f%%, tolerance=%.3e, passed=%s",
            dpp,
            scenario.name,
            record.get("bpm"),
            record.get("coord"),
            record["max_residual"],
            record["state_amplitude"],
            100 * record["max_residual"] / record["state_amplitude"],
            record["tolerance"],
            record["passed"],
        )
    LOGGER.info(
        "calculate_pz completed for dpp=%+.4e; result_rows=%d, acd_rows=%d, result_columns=%d",
        dpp,
        len(result),
        len(acd_result),
        len(result.columns),
    )
    raw_metrics = _marker_rmse(acd_result.attrs["raw_marker_states"], machine.tracking[dpp])
    cleaned_metrics = _marker_rmse(acd_result, machine.tracking[dpp])

    reconstructed = apply_precomputed_ac_dipole_bpm_overrides(result, acd_result)
    marker_rows = acd_result.loc[
        acd_result["name"].astype(str).str.lower().str.endswith(("_before", "_after"))
    ].reindex(columns=reconstructed.columns)
    marker_rows["name"] = marker_rows["name"].map(
        lambda name: (
            f"{str(name).rsplit('_', 1)[0].upper()}_{str(name).rsplit('_', 1)[1].lower()}"
        )
    )
    reconstructed = pd.concat([reconstructed, marker_rows], ignore_index=True)
    reconstructed["bunch_number"] = 0
    for column in ("var_x", "var_y", "var_px", "var_py"):
        if column not in reconstructed:
            reconstructed[column] = 1e-30
        reconstructed[column] = reconstructed[column].fillna(1e-30)

    bias_fraction = _bpm_momentum_bias(
        reconstructed, machine.tracking[dpp], dpp, scenario, closed_orbit=reference_co
    )
    assert bias_fraction < MAX_BPM_PX_BIAS_FRACTION, (
        f"dpp={dpp:+.4e}: per-BPM constant px offset is {bias_fraction:.3f} of the px "
        f"signal (limit {MAX_BPM_PX_BIAS_FRACTION}); the reconstruction closed orbit is wrong"
    )

    bpm_rows = reconstructed[reconstructed["name"].astype(str).str.match(BPM_PATTERN)]
    assert bpm_rows["name"].nunique() == 16
    assert bpm_rows.groupby("name", observed=True)["turn"].nunique().eq(FLATTOP_TURNS).all()
    assert np.isfinite(bpm_rows[["px", "py"]].to_numpy(dtype=float)).all()

    output = root / f"reconstructed_{dpp:+.4f}.parquet"
    reconstructed.to_parquet(output, index=False)
    LOGGER.info(
        "Reconstructed ACD data written; dpp=%+.4e, output=%s, rows=%d, bpm_rows=%d",
        dpp,
        output,
        len(reconstructed),
        len(bpm_rows),
    )
    generator = ACDipolePzGenerator.build(
        data=prepared,
        model_details=model_details,
        config=acd_config,
        closed_orbit_at_zero=frame.closed_orbit_at_zero,
        orbit_mode=frame.orbit_mode,
        optics=OpticsInput(
            measurement_dir=compensated_dir,
            sources={"phase": "measurement"},
        ),
    )
    _log_elapsed(
        "ACD momentum reconstruction",
        start,
        dpp=f"{dpp:+.4e}",
        output=output,
        raw_before=f"{raw_metrics['before']:.3e}",
        clean_before=f"{cleaned_metrics['before']:.3e}",
        raw_after=f"{raw_metrics['after']:.3e}",
        clean_after=f"{cleaned_metrics['after']:.3e}",
    )
    return output, result, generator, raw_metrics, cleaned_metrics


def _complex_amplitude(values: np.ndarray, tune: float) -> complex:
    """Project a turn series onto the driven line, returning its complex amplitude.

    A least-squares projection rather than an FFT bin, so the driven tune does not
    have to be commensurate with the record length.
    """
    turns = np.arange(len(values), dtype=float)
    phase = 2.0 * np.pi * tune * turns
    design = np.column_stack([np.cos(phase), np.sin(phase), np.ones_like(turns)])
    cosine, sine, _ = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)[0]
    return complex(cosine, -sine)


def _marker_states(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the ``_before``/``_after`` marker rows, turn-sorted, by side."""
    named = frame.assign(name=frame["name"].astype(str).str.upper())
    out: dict[str, pd.DataFrame] = {}
    for side in ("before", "after"):
        rows = named.loc[named["name"] == f"{ACD_NAME}_{side}".upper()]
        out[side] = rows.sort_values("turn").reset_index(drop=True)
    return out


def _assert_carrier_preserved(
    acd_result: pd.DataFrame, truth: pd.DataFrame, label: str
) -> dict[str, float]:
    """Check the noise and cleaning chain preserved the driven carrier.

    psb_md judges a preprocessing chain on the complex driven amplitude and the
    AC-dipole window relative phase, never on R^2 -- because preprocessing removes
    exactly the content R^2 measures, so a chain that removes more raises it whether
    or not the physics improved. A per-BPM operation could also bias the relative
    phase without moving either amplitude, which is why both are checked.
    """
    tunes = {"x": float(acd_result.attrs["dpx_tune"]), "y": float(acd_result.attrs["dpy_tune"])}
    estimate = _marker_states(acd_result)
    expected = _marker_states(truth)
    metrics: dict[str, float] = {}
    for plane, tune in tunes.items():
        window: dict[str, complex] = {}
        for source, frame in (("fit", estimate), ("true", expected)):
            amplitudes = {
                side: _complex_amplitude(frame[side][plane].to_numpy(dtype=float), tune)
                for side in ("before", "after")
            }
            window[source] = amplitudes["after"] / amplitudes["before"]
            metrics[f"{plane}_{source}_amplitude"] = abs(amplitudes["before"])
        ratio = metrics[f"{plane}_fit_amplitude"] / metrics[f"{plane}_true_amplitude"]
        phase_error = float(np.angle(window["fit"] / window["true"]))
        metrics[f"{plane}_amplitude_ratio"] = ratio
        metrics[f"{plane}_window_phase_error"] = phase_error
        LOGGER.info(
            "ACD carrier preservation; %s, plane=%s, amplitude_ratio=%.6f, "
            "window_phase_error=%.3f mrad",
            label,
            plane,
            ratio,
            1e3 * phase_error,
        )
        assert abs(ratio - 1.0) < MAX_CARRIER_AMPLITUDE_ERROR, (
            f"{label}, plane={plane}: driven amplitude changed by "
            f"{100 * (ratio - 1.0):.3f}% (limit {100 * MAX_CARRIER_AMPLITUDE_ERROR:.1f}%)"
        )
        assert abs(phase_error) < MAX_WINDOW_PHASE_ERROR, (
            f"{label}, plane={plane}: AC-dipole window relative phase moved by "
            f"{1e3 * phase_error:.3f} mrad (limit {1e3 * MAX_WINDOW_PHASE_ERROR:.1f} mrad)"
        )
    return metrics


def _marker_action(states: dict[str, pd.DataFrame], optics: pd.Series) -> dict[str, float]:
    """Courant-Snyder action of each marker state, each centred on its own mean.

    The two states differ by the AC-dipole kick itself -- ``px`` jumps across the
    exciter, ``x`` does not -- so pooling them onto one mean produces a number that
    is neither state's action. psb_md section 8.1 records that as a real defect.
    """
    actions: dict[str, float] = {}
    for side, frame in states.items():
        for plane, (position, momentum, beta, alfa) in (
            ("x", ("x", "px", "beta11", "alfa11")),
            ("y", ("y", "py", "beta22", "alfa22")),
        ):
            centred_position = frame[position].to_numpy(dtype=float)
            centred_momentum = frame[momentum].to_numpy(dtype=float)
            centred_position = centred_position - centred_position.mean()
            centred_momentum = centred_momentum - centred_momentum.mean()
            beta_value = float(optics[beta])
            alfa_value = float(optics[alfa])
            invariant = (
                centred_position**2
                + (alfa_value * centred_position + beta_value * centred_momentum) ** 2
            ) / beta_value
            actions[f"{plane}_{side}"] = float(np.mean(invariant))
    for plane in ("x", "y"):
        actions[f"{plane}_kick"] = actions[f"{plane}_after"] - actions[f"{plane}_before"]
    return actions


def _assert_marker_action(
    acd_result: pd.DataFrame, machine: MachineArtifacts, dpp: float, label: str
) -> dict[str, float]:
    """Check the reconstructed action at the marker matches the tracked one."""
    optics = acd_result.attrs["marker_optics"]
    estimate = _marker_action(_marker_states(acd_result), optics)
    expected = _marker_action(_marker_states(machine.tracking[dpp]), optics)
    for plane in ("x", "y"):
        for side in ("before", "after"):
            key = f"{plane}_{side}"
            ratio = estimate[key] / expected[key]
            LOGGER.info(
                "ACD marker action; %s, %s, fit=%.4e m rad, true=%.4e m rad, ratio=%.5f",
                label,
                key,
                estimate[key],
                expected[key],
                ratio,
            )
            assert abs(ratio - 1.0) < MAX_ACTION_ERROR, (
                f"{label}, {key}: reconstructed action is {100 * (ratio - 1.0):+.2f}% "
                f"off the tracked value (limit {100 * MAX_ACTION_ERROR:.0f}%)"
            )
    return estimate


def _marker_initial_condition_diagnostics(
    estimate: pd.DataFrame, truth: pd.DataFrame, dpp: float
) -> dict[str, float]:
    """Measure the uncentred launch-state error retained by the fitter."""
    estimate = estimate.assign(name=estimate["name"].astype(str).str.upper())
    truth = truth.assign(name=truth["name"].astype(str).str.upper())
    diagnostics: dict[str, float] = {}
    for side in ("before", "after"):
        marker = f"{ACD_NAME}_{side}".upper()
        actual = estimate.loc[estimate["name"] == marker]
        expected = truth.loc[truth["name"] == marker]
        merged = actual.merge(expected, on="turn", suffixes=("_fit", "_true"))
        assert len(merged) == FLATTOP_TURNS, (
            f"dpp={dpp:+.4e}, marker={marker}: retained {len(merged)} turns"
        )
        values = merged[["x_fit", "px_fit", "y_fit", "py_fit"]].to_numpy(dtype=float)
        expected_values = merged[["x_true", "px_true", "y_true", "py_true"]].to_numpy(dtype=float)
        residual = values - expected_values
        diagnostics[side] = _rms(residual)
        assert np.isfinite(values).all(), f"dpp={dpp:+.4e}, marker={marker}: non-finite launch"
        assert np.isfinite(residual).all(), (
            f"dpp={dpp:+.4e}, marker={marker}: non-finite launch residual"
        )
        LOGGER.info(
            "ACD marker initial condition; dpp=%+.4e, marker=%s, residual_rms=%.3e",
            dpp,
            marker,
            diagnostics[side],
        )
    return diagnostics


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize(
    "noise_factor",
    [pytest.param(0.0, id="clean"), pytest.param(1.0, id="noisy")],
)
def test_psb_acd_initial_conditions_and_fit_r2(
    noise_factor: float,
    scenario: MeasurementScenario,
    tmp_path: Path,
    psb_pipeline_machine: MachineArtifacts,
) -> None:
    """Check reconstruction inputs and harmonic-fit quality before SGD starts."""
    test_start = time.perf_counter()
    machine = psb_pipeline_machine
    rng = np.random.default_rng(10_000 + int(100 * noise_factor))
    pt_values = _pt_by_dpp(machine.accelerator)
    bpm_data: dict[float, pd.DataFrame] = {}
    compensated_dirs: dict[float, Path] = {}
    driven_phase_errors: dict[float, float] = {}
    compensated_phase_errors: dict[float, float] = {}
    for dpp in DPP_VALUES:
        bpm_data[dpp] = _inject_bpm_noise(machine.tracking[dpp], noise_factor, rng)
        driven, compensated = _run_phase_analysis(
            root=tmp_path / "hio",
            dpp=dpp,
            bpm_data=bpm_data[dpp],
            machine=machine,
        )
        compensated_dirs[pt_values[dpp]] = compensated
        assert _phase_coverage(driven) == (16, 16)
        assert _phase_coverage(compensated) == (16, 16)
        driven_phase_errors[dpp] = _phase_rms(driven, machine.free_twiss[dpp])
        compensated_phase_errors[dpp] = _phase_rms(compensated, machine.free_twiss[dpp])

    assert _rms(list(compensated_phase_errors.values())) < _rms(list(driven_phase_errors.values()))
    fitted, _ = _fit_reference(
        machine, compensated_dirs, tmp_path, scenario, noise_factor=noise_factor
    )
    mixed_reference = _mixed_reference(machine, fitted, scenario)

    for dpp in DPP_VALUES:
        _, result, _, _, _ = _reconstruct_one(
            root=tmp_path,
            dpp=dpp,
            bpm_data=bpm_data[dpp],
            compensated_dir=compensated_dirs[pt_values[dpp]],
            machine=machine,
            fitted=fitted,
            reference_co=mixed_reference,
            scenario=scenario,
        )
        acd_result = result.attrs["acd_result"]
        marker_metrics = _marker_initial_condition_diagnostics(
            acd_result, machine.tracking[dpp], dpp
        )
        fit_r2 = {
            "x": float(acd_result.attrs["dpx_r2"]),
            "y": float(acd_result.attrs["dpy_r2"]),
        }
        label = f"scenario={scenario.name}, noise={noise_factor:g}, dpp={dpp:+.4e}"
        LOGGER.info(
            "ACD fit quality; %s, dpx_r2=%.6f, dpy_r2=%.6f, marker_residuals=%s",
            label,
            fit_r2["x"],
            fit_r2["y"],
            marker_metrics,
        )
        _assert_carrier_preserved(acd_result, machine.tracking[dpp], label)
        _assert_marker_action(acd_result, machine, dpp, label)
        assert fit_r2["x"] > 0.9, f"{label}: horizontal ACD fit R^2={fit_r2['x']:.6f}"
        assert fit_r2["y"] > 0.9, f"{label}: vertical ACD fit R^2={fit_r2['y']:.6f}"

    _log_elapsed(
        "PSB ACD initial-condition and R2 test",
        test_start,
        noise_factor=noise_factor,
    )


def test_psb_acd_campaign_snr_guard_pass_rate(
    tmp_path: Path,
    psb_campaign_machine: MachineArtifacts,
) -> None:
    """Reconstruct at the SNR the machine actually delivers, over several seeds.

    The other tests run at a comfortable driven amplitude. This one drives at the
    2026-08-14 campaign's measured 351.6/329.3 um against that campaign's measured
    per-BPM resolution, giving a per-turn SNR of about 3.8. At that SNR the
    AC-dipole state-consistency guard is expected to reject some acquisitions, so
    the assertion is a pass fraction over noise seeds rather than a pass. The guard
    tolerance is never touched: the correct response to a rejected acquisition is to
    drop it.
    """
    test_start = time.perf_counter()
    machine = psb_campaign_machine
    scenario = FULL_ORBIT
    pt_values = _pt_by_dpp(machine.accelerator)
    outcomes: list[bool] = []
    residual_fractions: list[float] = []

    for seed in range(CAMPAIGN_SEEDS):
        rng = np.random.default_rng(20_000 + seed)
        seed_root = tmp_path / f"seed_{seed}"
        # The reference fit needs every momentum, so all three are analysed even
        # though only the on-momentum one is reconstructed below.
        bpm_data: dict[float, pd.DataFrame] = {}
        compensated_dirs: dict[float, Path] = {}
        for dpp in DPP_VALUES:
            bpm_data[dpp] = _inject_bpm_noise(
                machine.tracking[dpp], 1.0, rng, noise_file=CAMPAIGN_NOISE_FILE
            )
            _, compensated = _run_phase_analysis(
                root=seed_root / "hio",
                dpp=dpp,
                bpm_data=bpm_data[dpp],
                machine=machine,
            )
            compensated_dirs[pt_values[dpp]] = compensated
        fitted, _ = _fit_reference(machine, compensated_dirs, seed_root, scenario)
        try:
            _, result, _, _, _ = _reconstruct_one(
                root=seed_root,
                dpp=REFERENCE_DPP,
                bpm_data=bpm_data[REFERENCE_DPP],
                compensated_dir=compensated_dirs[pt_values[REFERENCE_DPP]],
                machine=machine,
                fitted=fitted,
                reference_co=_mixed_reference(machine, fitted, scenario),
                scenario=scenario,
            )
        except ACDipoleStateConsistencyError as error:
            outcomes.append(False)
            residual_fractions.append(error.max_residual / error.state_amplitude)
            LOGGER.warning(
                "Campaign-SNR acquisition rejected; seed=%d, bpm=%s, coord=%s, "
                "residual=%.1f%% of state",
                seed,
                error.bpm_name,
                error.coord,
                100 * error.max_residual / error.state_amplitude,
            )
            continue
        outcomes.append(True)
        acd_result = result.attrs["acd_result"]
        records = acd_result.attrs["acd_state_consistency"]
        worst = max(record["max_residual"] / record["state_amplitude"] for record in records)
        residual_fractions.append(worst)
        LOGGER.info(
            "Campaign-SNR acquisition accepted; seed=%d, dpx_r2=%.4f, dpy_r2=%.4f, "
            "worst_guard_residual=%.1f%% of state",
            seed,
            float(acd_result.attrs["dpx_r2"]),
            float(acd_result.attrs["dpy_r2"]),
            100 * worst,
        )

    pass_fraction = sum(outcomes) / len(outcomes)
    LOGGER.warning(
        "Campaign-SNR guard pass rate; %d/%d seeds, guard residual %.1f-%.1f%% of state",
        sum(outcomes),
        len(outcomes),
        100 * min(residual_fractions),
        100 * max(residual_fractions),
    )
    _log_elapsed("PSB ACD campaign-SNR test", test_start, pass_fraction=pass_fraction)
    assert pass_fraction >= MIN_CAMPAIGN_GUARD_PASS_FRACTION, (
        f"only {sum(outcomes)}/{len(outcomes)} campaign-SNR acquisitions survived the "
        f"AC-dipole state guard (limit {MIN_CAMPAIGN_GUARD_PASS_FRACTION:.0%}); "
        f"guard residuals were {[f'{100 * f:.1f}%' for f in residual_fractions]}. "
        f"Do not widen the guard tolerance to make this pass."
    )


@pytest.mark.parametrize(
    ("factor", "noise_factor"),
    [
        pytest.param("baseline", 0.0, id="clean-baseline"),
        pytest.param("truth_quads", 0.0, id="clean-no-quad-errors"),
        pytest.param("truth_bends", 0.0, id="clean-no-bend-errors"),
        pytest.param("truth_magnets", 0.0, id="clean-no-magnet-errors"),
        pytest.param("truth_closed_orbit", 0.0, id="clean-no-closed-orbit-error"),
        pytest.param("no_correctors", 0.0, id="clean-no-corrector-closure"),
    ],
)
def test_psb_acd_r2_factor_case_study(
    factor: str,
    noise_factor: float,
    tmp_path: Path,
    psb_pipeline_machine: MachineArtifacts,
) -> None:
    """Run one controlled noisy reconstruction with one suspected factor removed.

    Pinned to ``FULL_ORBIT``: every factor here is a closed-orbit quantity -- bend
    errors, corrector closure, the closed-orbit source -- and none of them exists in
    the dynamic-part frame, where the orbit is subtracted before reconstruction.
    """
    scenario = FULL_ORBIT
    machine = psb_pipeline_machine
    if factor == "no_correctors":
        correctors = tfs.read(machine.corrector_file).copy()
        correctors["hkick"] = 0.0
        correctors["vkick"] = 0.0
        zero_corrector_file = tmp_path / "zero_correctors.tfs"
        tfs.write(zero_corrector_file, correctors, save_index=False)
        machine = replace(machine, corrector_file=zero_corrector_file)

    rng = np.random.default_rng(10_100)
    bpm_data: dict[float, pd.DataFrame] = {}
    compensated_dirs: dict[float, Path] = {}
    pt_values = _pt_by_dpp(machine.accelerator)
    for dpp in DPP_VALUES:
        bpm_data[dpp] = _inject_bpm_noise(machine.tracking[dpp], noise_factor, rng)
        _, compensated = _run_phase_analysis(
            root=tmp_path / "hio",
            dpp=dpp,
            bpm_data=bpm_data[dpp],
            machine=machine,
        )
        compensated_dirs[pt_values[dpp]] = compensated
    fitted, _ = _fit_reference(
        machine, compensated_dirs, tmp_path, scenario, noise_factor=noise_factor
    )

    if factor in {"truth_quads", "truth_bends", "truth_magnets"}:
        if factor == "truth_magnets":
            strengths = dict(machine.truth)
        else:
            suffix = ".dk1l" if factor == "truth_quads" else ".dk0l"
            strengths = {
                name: machine.truth[name] if name.lower().endswith(suffix) else value
                for name, value in fitted.magnet_strengths.items()
            }
        fitted = replace(fitted, magnet_strengths=strengths)

    reference_co = (
        machine.free_twiss[REFERENCE_DPP]
        if factor == "truth_closed_orbit"
        else _mixed_reference(machine, fitted)
    )
    case_root = tmp_path / factor
    case_root.mkdir(parents=True, exist_ok=True)
    _, result, _, _, _ = _reconstruct_one(
        root=case_root,
        dpp=REFERENCE_DPP,
        bpm_data=bpm_data[REFERENCE_DPP],
        compensated_dir=compensated_dirs[pt_values[REFERENCE_DPP]],
        machine=machine,
        fitted=fitted,
        reference_co=reference_co,
        scenario=scenario,
    )
    acd_result = result.attrs["acd_result"]
    LOGGER.warning(
        "ACD R2 factor case; factor=%s, dpx_r2=%.6f, dpy_r2=%.6f, dpx_amp=%.6e, dpy_amp=%.6e",
        factor,
        float(acd_result.attrs["dpx_r2"]),
        float(acd_result.attrs["dpy_r2"]),
        float(acd_result.attrs["dpx_amplitude"]),
        float(acd_result.attrs["dpy_amplitude"]),
    )
    assert np.isfinite(float(acd_result.attrs["dpx_r2"]))
    assert np.isfinite(float(acd_result.attrs["dpy_r2"]))
    assert float(acd_result.attrs["dpx_r2"]) > 0.9, (
        f"factor={factor}, noise={noise_factor:g}: "
        f"horizontal R^2={float(acd_result.attrs['dpx_r2']):.6f}"
    )
    assert float(acd_result.attrs["dpy_r2"]) > 0.9, (
        f"factor={factor}, noise={noise_factor:g}: "
        f"vertical R^2={float(acd_result.attrs['dpy_r2']):.6f}"
    )


def _run_acd_fit(
    *,
    root: Path,
    label: str,
    machine: MachineArtifacts,
    files: dict[float, Path],
    generators: dict[float, Any],
    initial: dict[str, float],
    max_epochs: int = 500,
    num_workers: int = 20,
    num_batches: int = 20,
    data_fraction: float = OPTIMISATION_DATA_FRACTION,
    validation_fraction: float = 0.1,
) -> tuple[dict[str, float], float, ACDMarkerFitter]:
    controller_ref: dict[str, ACDMarkerFitter] = {}
    track_data_ref: dict[int, FileTracks] = {}

    def refresh_marker_initial_conditions(
        current_knobs: dict[str, float], _best_knobs: dict[str, float]
    ) -> np.ndarray:
        """Refresh marker launches after each optimiser epoch.

        The ACD marker states depend on the fitted lattice. Keeping the initial
        states from the reference fit makes the tracking objective include a
        stale launch error, which is particularly damaging once measurement
        noise is present.
        """
        controller = controller_ref.get("controller")
        if controller is None:
            raise RuntimeError("ACD controller was not initialised before refreshing markers")

        full_strengths = {**initial, **current_knobs}
        updated_track_data: dict[int, FileTracks] = {}
        for file_idx, dpp in enumerate(files):
            refreshed = generators[dpp].update(magnet_strengths=full_strengths)
            tracks = track_data_ref[file_idx]
            values = {name: array.copy() for name, array in tracks.values.items()}
            column_by_name = {name.upper(): col for name, col in tracks.marker_col.items()}
            marker_rows = refreshed.loc[
                refreshed["name"].astype(str).str.lower().str.endswith(("_before", "_after")),
                ["name", "turn", "x", "px", "y", "py"],
            ]
            for marker_name, marker in marker_rows.groupby("name", sort=False):
                col = column_by_name[str(marker_name).upper()]
                marker = marker.sort_values("turn")
                if len(marker) != len(tracks.turns):
                    raise ValueError(
                        f"Marker refresh length mismatch for {marker_name}: "
                        f"{len(tracks.turns)} != {len(marker)}"
                    )
                for coordinate in ("x", "px", "y", "py"):
                    values[coordinate][:, col] = marker[coordinate].to_numpy()
            updated_track_data[file_idx] = FileTracks(
                turns=tracks.turns,
                markers=tracks.markers,
                values=values,
                kick_plane=tracks.kick_plane,
            )

        return controller.worker_manager.build_update_coords(updated_track_data)

    start = time.perf_counter()
    ctrl = _build_acd_marker_fitter(
        root=root,
        label=label,
        machine=machine,
        files=files,
        initial=initial,
        max_epochs=max_epochs,
        num_workers=num_workers,
        num_batches=num_batches,
        data_fraction=data_fraction,
        validation_fraction=validation_fraction,
        initial_conditions_callback=refresh_marker_initial_conditions,
    )
    controller_ref["controller"] = ctrl
    # The refresh copies each grid before writing, so the stored tracks stay intact.
    track_data_ref.update(ctrl.data_manager.tracks)
    assert ctrl.optimisation_loop.use_true_strengths
    quad_truth = {name for name in machine.truth if name.lower().endswith(".dk1l")}
    assert quad_truth <= set(ctrl.optimisation_loop.true_strengths)

    report = ctrl.check_degeneracy(rel_tol=1e-12)
    LOGGER.info(
        "ACD marker fit '%s' degeneracy: knobs=%d, rank=%d, condition_number=%.3e",
        label,
        len(report.knob_names),
        report.numerical_rank,
        report.condition_number,
    )
    assert report.n_degenerate == 0, (
        f"{len(report.knob_names)}-momentum ACD marker problem is rank deficient: "
        f"rank={report.numerical_rank}/{len(report.knob_names)}, "
        f"degenerate directions={_degenerate_summary(report)}"
    )
    assert report.condition_number < 1e10

    LOGGER.info("Starting optimiser run for ACD marker fit '%s'", label)
    estimate, _ = ctrl.run()
    _log_acd_objective_scan(ctrl, machine, estimate, label)
    combined = {**initial, **estimate}
    best_loss = float(ctrl.optimisation_loop.best_loss)
    _log_elapsed(
        f"ACD marker fit '{label}'",
        start,
        estimated_strengths=len(estimate),
        combined_strengths=len(combined),
        best_loss=f"{best_loss:.3e}",
    )
    return combined, best_loss, ctrl


def _build_acd_marker_fitter(
    *,
    root: Path,
    label: str,
    machine: MachineArtifacts,
    files: dict[float, Path],
    initial: dict[str, float],
    max_epochs: int = 500,
    optimise_momenta: bool = False,
    num_workers: int = 20,
    num_batches: int = 20,
    data_fraction: float = OPTIMISATION_DATA_FRACTION,
    validation_fraction: float = 0.1,
    initial_conditions_callback: Callable[[dict[str, float], dict[str, float]], np.ndarray]
    | None = None,
    scenario: MeasurementScenario = FULL_ORBIT,
) -> ACDMarkerFitter:
    selected = list(files)
    LOGGER.info(
        "Building ACD marker fitter '%s'; scenario=%s, dpp_values=%s, files=%s, "
        "initial_strengths=%d",
        label,
        scenario.name,
        [f"{dpp:+.4e}" for dpp in selected],
        [str(files[dpp]) for dpp in selected],
        len(initial),
    )
    interface_options: dict[str, Any] = {
        "machine_state": merge_machine_states(
            machine.corrector_file if scenario.use_correctors else None, machine.tune_knobs
        )
    }
    details = {
        files[dpp]: MeasurementDetails(
            interface_options=dict(interface_options),
            machine_deltap=dpp,
        )
        for dpp in selected
    }
    return ACDMarkerFitter(
        accelerator=machine.case.accelerator_factory(
            ring=machine.case.ring,
            kinetic_energy=machine.case.kinetic_energy,
            sequence_file=machine.accelerator.sequence_file,
            # The ACD fit never moves bends in either scenario -- they stay at the
            # reference-fit value -- so this does not follow scenario.fit_bends.
            errors={"quad": {"k1"}},
        ),
        optimiser_config=OptimiserConfig(
            max_epochs=max_epochs,
            warmup_epochs=15,
            warmup_lr_start=1e-7,
            max_lr=2e-6,
            min_lr=1e-6,
            gradient_converged_value=1e-11,
        ),
        simulation_config=SimulationConfig(
            num_workers=num_workers,
            num_batches=num_batches,
            data_fraction=data_fraction,
            validation_fraction=validation_fraction,
            optimise_momenta=optimise_momenta,
            use_fixed_bpm=True,
            enable_preloop_outlier_screening=False,
        ),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        measurement_config=MeasurementConfig(details),
        initial_knob_strengths=initial,
        true_strengths=machine.truth,
        initial_conditions_callback=initial_conditions_callback,
        output_config=OutputConfig(
            write_tensorboard_logs=False,
            include_uncertainty=False,
            mad_logfile=root / f"{label}_mad.log",
        ),
    )


def _log_acd_objective_scan(
    ctrl: ACDMarkerFitter,
    machine: MachineArtifacts,
    estimate: dict[str, float],
    label: str,
) -> None:
    """Log objective values at diagnostic knob points without gating the test."""
    knob_names = list(ctrl.initial_knobs)
    truth = {name: machine.truth[name] for name in knob_names}

    scan: list[tuple[str, dict[str, float]]] = [
        ("initial", dict(ctrl.initial_knobs)),
        ("estimate", {name: estimate[name] for name in knob_names}),
        ("truth", truth),
    ]
    for alpha in (0.25, 0.50, 0.75):
        scan.append(
            (
                f"line_alpha={alpha:.2f}",
                {
                    name: ctrl.initial_knobs[name]
                    + alpha * (truth[name] - ctrl.initial_knobs[name])
                    for name in knob_names
                },
            )
        )

    losses = evaluate_controller_worker_losses(ctrl, [knobs for _, knobs in scan])
    best_loss = min(losses)
    diag = []
    for (point_label, knobs), loss in zip(scan, losses, strict=True):
        truth_l1 = sum(abs(knobs[name] - truth[name]) for name in knob_names)
        diag.append(f"{point_label}:loss={loss:.3e},ratio={loss / best_loss:.3f},td={truth_l1:.3e}")
    LOGGER.info("ACD marker fit '%s' objective scan; %s", label, "; ".join(diag))


def _phase_coverage(path: Path) -> tuple[int, int]:
    coverage = []
    for plane in ("x", "y"):
        frame = tfs.read(path / f"phase_{plane}.tfs")
        names = set(frame["NAME"].astype(str).str.upper())
        names.update(frame["NAME2"].astype(str).str.upper())
        coverage.append(len(names))
    result = tuple(coverage)
    LOGGER.info("Phase coverage for %s; x=%d, y=%d", path, result[0], result[1])
    return result


@dataclass(frozen=True)
class AcdFitProfile:
    max_epochs: int
    num_workers: int
    num_batches: int
    data_fraction: float
    validation_fraction: float


# Each worker is a separate MAD-NG tracking subprocess; leave headroom for
# validation workers and the main pytest process rather than oversubscribing
# whatever host this runs on.
_MAX_FIT_WORKERS = max(1, min(54, (os.cpu_count() or 4) - 3))

ACD_FIT_PROFILES = {
    "fast": AcdFitProfile(
        max_epochs=15,
        num_workers=_MAX_FIT_WORKERS,
        num_batches=4,
        data_fraction=1.0,
        validation_fraction=0.1,
    ),
    "slow": AcdFitProfile(
        max_epochs=100,
        num_workers=_MAX_FIT_WORKERS,
        num_batches=20,
        data_fraction=1.0,
        validation_fraction=0.1,
    ),
}


@pytest.mark.parametrize(
    ("noise_factor", "profile"),
    [
        pytest.param(0.0, "fast", id="noise-0-fast"),
        pytest.param(0.0, "slow", id="noise-0-slow"),
        pytest.param(1.0, "fast", id="noise-1-fast"),
        pytest.param(1.0, "slow", id="noise-1-slow"),
    ],
)
def test_psb_full_acd_reconstruction_and_optimisation(
    noise_factor: float,
    profile: str,
    tmp_path: Path,
    psb_pipeline_machine: MachineArtifacts,
) -> None:
    """Exercise tracking -> HIO -> reference -> reconstruction -> ACD fit.

    Two independent axes: noise on/off and fit budget fast/slow. "fast" is a
    cheap pipeline-breakage regression check; "slow" proves the workers
    converge under a realistic optimisation budget.
    """
    test_start = time.perf_counter()
    LOGGER.info(
        "Starting PSB full ACD pipeline test; noise_factor=%s, tmp_path=%s",
        noise_factor,
        tmp_path,
    )
    machine = psb_pipeline_machine
    scenario = FULL_ORBIT
    correctors_before = dict(machine.corrector_knobs)
    rng = np.random.default_rng(10_000 + int(100 * noise_factor))
    pt_values = _pt_by_dpp(machine.accelerator)

    bpm_data: dict[float, pd.DataFrame] = {}
    driven_dirs: dict[float, Path] = {}
    compensated_dirs: dict[float, Path] = {}
    driven_phase_errors: dict[float, float] = {}
    compensated_phase_errors: dict[float, float] = {}
    for dpp in DPP_VALUES:
        momentum_start = time.perf_counter()
        LOGGER.info("Starting optics stage for dpp=%+.4e, pt=%+.6e", dpp, pt_values[dpp])
        bpm_data[dpp] = _inject_bpm_noise(machine.tracking[dpp], noise_factor, rng)
        driven, compensated = _run_phase_analysis(
            root=tmp_path / "hio",
            dpp=dpp,
            bpm_data=bpm_data[dpp],
            machine=machine,
        )
        driven_dirs[pt_values[dpp]] = driven
        compensated_dirs[pt_values[dpp]] = compensated
        assert _phase_coverage(driven) == (16, 16)
        assert _phase_coverage(compensated) == (16, 16)
        driven_phase_errors[dpp] = _phase_rms(driven, machine.free_twiss[dpp])
        compensated_phase_errors[dpp] = _phase_rms(compensated, machine.free_twiss[dpp])
        _log_elapsed(
            "Per-momentum optics stage",
            momentum_start,
            dpp=f"{dpp:+.4e}",
            driven_phase_rms=f"{driven_phase_errors[dpp]:.3e}",
            compensated_phase_rms=f"{compensated_phase_errors[dpp]:.3e}",
        )

    phase_driven = _rms(list(driven_phase_errors.values()))
    phase_compensated = _rms(list(compensated_phase_errors.values()))
    phase_diag = (
        f"noise={noise_factor:g}, driven phase RMS={phase_driven:.3e}, "
        f"compensated phase RMS={phase_compensated:.3e}"
    )
    LOGGER.info("Phase compensation summary: %s", phase_diag)
    assert phase_compensated < phase_driven, phase_diag

    fitted, _measurements = _fit_reference(
        machine, compensated_dirs, tmp_path, noise_factor=noise_factor
    )
    nominal_bend = _relative_error_rms({}, machine.truth, ".dk0l")
    nominal_quad = _relative_error_rms({}, machine.truth, ".dk1l")
    fitted_bend = _relative_error_rms(fitted.magnet_strengths, machine.truth, ".dk0l")
    fitted_quad = _relative_error_rms(fitted.magnet_strengths, machine.truth, ".dk1l")
    fitted_bend_resolution = _rms(
        [
            uncertainty
            for name, uncertainty in fitted.uncertainties.items()
            if name.lower().endswith(".dk0l")
        ]
    )
    fitted_quad_resolution = _rms(
        [
            uncertainty
            for name, uncertainty in fitted.uncertainties.items()
            if name.lower().endswith(".dk1l")
        ]
    )
    fit_diag = (
        f"noise={noise_factor:g}, bend RMS nominal/ref="
        f"{nominal_bend:.3e}/{fitted_bend:.3e}, quad RMS nominal/ref="
        f"{nominal_quad:.3e}/{fitted_quad:.3e}, bend/quad resolution="
        f"{fitted_bend_resolution:.3e}/{fitted_quad_resolution:.3e}; {phase_diag}"
    )
    LOGGER.info("Reference fit summary: %s", fit_diag)
    LOGGER.info(
        "Reference fit strengths; count=%d, bends=%d, quads=%d, fingerprint=%s",
        len(fitted.magnet_strengths),
        sum(name.lower().endswith(".dk0l") for name in fitted.magnet_strengths),
        sum(name.lower().endswith(".dk1l") for name in fitted.magnet_strengths),
        _strength_fingerprint(fitted.magnet_strengths),
    )
    # Per-magnet quad error is not the quantity the pipeline depends on -- see
    # _fitted_optics_quality. Log the observable combination next to it so the two
    # can be compared before deciding which the assertion should use.
    fitted_phase, fitted_beating = _fitted_optics_quality(machine, fitted.magnet_strengths)
    nominal_phase, nominal_beating = _fitted_optics_quality(machine, {})
    LOGGER.info(
        "Reference fit optics quality; phase_advance_rms nominal/ref=%.3e/%.3e, "
        "beta_beating_rms nominal/ref=%.3e/%.3e",
        nominal_phase,
        fitted_phase,
        nominal_beating,
        fitted_beating,
    )
    assert fitted_bend < nominal_bend, fit_diag
    assert fitted_quad < nominal_quad, fit_diag

    mixed_reference = _mixed_reference(machine, fitted, scenario)
    reconstructed_files: dict[float, Path] = {}
    generators: dict[float, Any] = {}
    reconstruction_results: dict[float, Any] = {}
    raw_rmse: dict[float, dict[str, float]] = {}
    cleaned_rmse: dict[float, dict[str, float]] = {}
    for dpp in DPP_VALUES:
        reconstruction_start = time.perf_counter()
        (
            reconstructed_files[dpp],
            reconstruction_results[dpp],
            generators[dpp],
            raw_rmse[dpp],
            cleaned_rmse[dpp],
        ) = _reconstruct_one(
            root=tmp_path,
            dpp=dpp,
            bpm_data=bpm_data[dpp],
            compensated_dir=compensated_dirs[pt_values[dpp]],
            machine=machine,
            fitted=fitted,
            reference_co=mixed_reference,
        )
        guard_records = (
            reconstruction_results[dpp].attrs["acd_result"].attrs["acd_state_consistency"]
        )
        LOGGER.info(
            "ACD state guard records; dpp=%+.4e, records=%s",
            dpp,
            [
                {
                    "plane": record.get("plane"),
                    "side": record.get("side"),
                    "rms_residual": record.get("rms_residual"),
                    "threshold": record.get("threshold"),
                    "passed": record.get("passed"),
                }
                for record in guard_records
            ],
        )
        assert len(guard_records) == 4
        assert all(np.isfinite(record["rms_residual"]) for record in guard_records)
        _log_elapsed(
            "Per-momentum reconstruction stage",
            reconstruction_start,
            dpp=f"{dpp:+.4e}",
            raw_before=f"{raw_rmse[dpp]['before']:.3e}",
            clean_before=f"{cleaned_rmse[dpp]['before']:.3e}",
            raw_after=f"{raw_rmse[dpp]['after']:.3e}",
            clean_after=f"{cleaned_rmse[dpp]['after']:.3e}",
        )
    marker_diag = "; ".join(
        f"dpp={dpp:+.1e} {side} raw/clean={raw_rmse[dpp][side]:.3e}/{cleaned_rmse[dpp][side]:.3e}"
        for dpp in DPP_VALUES
        for side in ("before", "after")
    )
    LOGGER.info("Marker cleaning summary: noise=%s; %s", noise_factor, marker_diag)
    for dpp in DPP_VALUES:
        for side in ("before", "after"):
            raw = raw_rmse[dpp][side]
            cleaned = cleaned_rmse[dpp][side]
            LOGGER.info(
                "Checking marker cleaning; noise=%s, dpp=%+.4e, side=%s, raw=%.3e, cleaned=%.3e, delta=%.3e",
                noise_factor,
                dpp,
                side,
                raw,
                cleaned,
                cleaned - raw,
            )
            if noise_factor:
                # The BPM positions have already been SVD-cleaned upstream, so
                # this asks the marker cleaner to improve on that residual --
                # not to claim the larger gain available against raw BPM noise.
                assert cleaned < raw, f"noise={noise_factor:g}; {marker_diag}; {fit_diag}"
            else:
                assert cleaned <= raw + 1e-9, (
                    f"zero-noise cleaning regression; {marker_diag}; {fit_diag}"
                )

    fit_profile = ACD_FIT_PROFILES[profile]
    all_strengths, final_loss, all_ctrl = _run_acd_fit(
        root=tmp_path,
        label="three_dpp",
        machine=machine,
        files=reconstructed_files,
        generators=generators,
        initial=fitted.magnet_strengths,
        max_epochs=fit_profile.max_epochs,
        num_workers=fit_profile.num_workers,
        num_batches=fit_profile.num_batches,
        data_fraction=fit_profile.data_fraction,
        validation_fraction=fit_profile.validation_fraction,
    )
    final_quad = _relative_error_rms(all_strengths, machine.truth, ".dk1l")
    final_bend = _relative_error_rms(all_strengths, machine.truth, ".dk0l")
    estimated_knobs = {name: all_strengths[name] for name in all_ctrl.initial_knobs}
    truth_knobs = {name: machine.truth[name] for name in all_ctrl.initial_knobs}
    initial_objective, final_objective, truth_objective = evaluate_controller_worker_losses(
        all_ctrl, [dict(all_ctrl.initial_knobs), estimated_knobs, truth_knobs]
    )
    overfit_depth = (truth_objective - final_objective) / (initial_objective - truth_objective)
    final_diag = (
        f"noise={noise_factor:g}, profile={profile}, bend RMS nominal/ref/all="
        f"{nominal_bend:.3e}/{fitted_bend:.3e}/{final_bend:.3e}, "
        f"quad RMS nominal/ref/all="
        f"{nominal_quad:.3e}/{fitted_quad:.3e}/{final_quad:.3e}, "
        f"phase driven/comp={phase_driven:.3e}/{phase_compensated:.3e}, "
        f"final loss={final_loss:.3e}, objective initial/final/truth="
        f"{initial_objective:.3e}/{final_objective:.3e}/{truth_objective:.3e}, "
        f"overfit depth={overfit_depth:.3f}; {marker_diag}"
    )
    LOGGER.info("Three-dpp ACD fit summary: %s", final_diag)
    _FINAL_QUAD_BY_CASE[noise_factor, profile] = final_quad
    for name in machine.truth:
        if name.lower().endswith(".dk0l"):
            assert all_strengths[name] == fitted.magnet_strengths[name], final_diag
    assert final_objective < initial_objective, final_diag
    # The truth point has to be the better explanation of the data, otherwise the
    # objective is dominated by noise and nothing below is meaningful.
    assert truth_objective < initial_objective, final_diag
    assert overfit_depth < MAX_OVERFIT_DEPTH, final_diag
    assert final_quad < nominal_quad, final_diag
    assert final_quad < MAX_QUAD_RATIO_VS_REFERENCE * fitted_quad, final_diag
    fast_quad = _FINAL_QUAD_BY_CASE.get((noise_factor, "fast"))
    if profile == "slow" and fast_quad is not None:
        assert final_quad <= fast_quad, final_diag

    tune_knobs = read_knobs(machine.tune_knobs)
    initial_model = all_ctrl.config_manager.initial_model_values
    for name, value in tune_knobs.items():
        assert initial_model.get(name, value) == pytest.approx(value)
    assert _corrector_strengths(machine.corrector_file) == pytest.approx(correctors_before), (
        final_diag
    )
    _log_elapsed("PSB full ACD pipeline test", test_start, noise_factor=noise_factor)


@pytest.fixture(scope="module")
def psb_pipeline_machine(
    tmp_path_factory: pytest.TempPathFactory, seq_psb: Path
) -> MachineArtifacts:
    """The default machine: a comfortable, deliberately optimistic driven amplitude."""
    return _build_machine(
        tmp_path_factory.mktemp("psb_full_acd_machine"), seq_psb, excitation_scale=1.0
    )


@pytest.fixture(scope="module")
def psb_campaign_machine(
    tmp_path_factory: pytest.TempPathFactory, seq_psb: Path
) -> MachineArtifacts:
    """A machine driven at the amplitude the real campaign actually achieved.

    Separate from :func:`psb_pipeline_machine` rather than a parameter on it: the
    excitation is baked into the tracking, so the two need different tracking runs,
    and parametrising the shared fixture would double the cost of every test that
    uses it including the factor study and the SGD fit.
    """
    return _build_machine(
        tmp_path_factory.mktemp("psb_campaign_machine"),
        seq_psb,
        excitation_scale=CAMPAIGN_EXCITATION_SCALE,
    )


def _build_machine(root: Path, seq_psb: Path, *, excitation_scale: float) -> MachineArtifacts:
    fixture_start = time.perf_counter()
    LOGGER.info(
        "Building PSB pipeline machine fixture; root=%s, sequence=%s, excitation_scale=%.4f",
        root,
        seq_psb,
        excitation_scale,
    )
    case = PipelineCase(
        name="psb-ring3",
        accelerator_factory=PSB,
        ring=3,
        kinetic_energy=0.160,
        bpm_pattern=BPM_PATTERN,
        acd_name=ACD_NAME,
    )
    accelerator = case.accelerator_factory(
        ring=case.ring,
        kinetic_energy=case.kinetic_energy,
        sequence_file=seq_psb,
        errors={"bend": {"k0"}, "quad": {"k1"}},
    )
    iface = GradientDescentMadInterface(accelerator)
    try:
        LOGGER.info("Applying main PSB strengths: %s", MAIN_STRENGTHS)
        iface.set_madx_variables(**MAIN_STRENGTHS)
        tune_knobs = root / "main_strengths.txt"
        save_knobs(MAIN_STRENGTHS, tune_knobs)

        truth = _truth_errors(iface)
        LOGGER.info(
            "Injected truth errors; total=%d, bends=%d, quadrupoles=%d",
            len(truth),
            sum(name.lower().endswith(".dk0l") for name in truth),
            sum(name.lower().endswith(".dk1l") for name in truth),
        )
        corrector_file = root / "known_correctors.tfs"
        LOGGER.info("Performing closed-orbit correction into %s", corrector_file)
        iface.perform_orbit_correction(
            machine_deltap=REFERENCE_DPP,
            corrector_file=corrector_file,
            twiss_name="nil",
            correct_tunes=False,
            bpms=PSB_ORBIT_BPM_NAMES,
            correctors=PSB_X_ORBIT_CORRECTOR_NAMES,
            plane="x",
        )

        free_twiss = {dpp: _direct_twiss(iface, dpp) for dpp in DPP_VALUES}
        measured_tunes = {
            dpp: (float(frame.attrs["q1"]) % 1.0, float(frame.attrs["q2"]) % 1.0)
            for dpp, frame in free_twiss.items()
        }
        LOGGER.info(
            "Measured natural tunes by dpp: %s",
            {
                f"{dpp:+.4e}": (f"{tunes[0]:.6f}", f"{tunes[1]:.6f}")
                for dpp, tunes in measured_tunes.items()
            },
        )
        for frame in free_twiss.values():
            assert int(frame.attrs["q1"]) == 4
            assert int(frame.attrs["q2"]) == 4

        closed_orbits = {
            dpp: pd.DataFrame(
                {
                    "X": frame["x"],
                    "Y": frame["y"],
                    "ERRX": PSB_BPM_RESOLUTION,
                    "ERRY": PSB_BPM_RESOLUTION,
                },
                index=frame.index,
            )
            for dpp, frame in free_twiss.items()
        }

        correctors = _corrector_strengths(corrector_file)
        LOGGER.info(
            "Loaded non-zero corrector strengths; count=%d, values=%s", len(correctors), correctors
        )
        corrector_table = tfs.read(corrector_file)
        corrector_table = corrector_table.loc[
            ~corrector_table["kind"].astype(str).str.contains("monitor")  # ty:ignore[unresolved-attribute]
        ]
        LOGGER.info("Initialising xtrack environment")
        env = initialise_env(
            matched_tunes=MAIN_STRENGTHS,
            magnet_strengths=truth,
            corrector_table=corrector_table,
            sequence_file=seq_psb,
            seq_name=accelerator.seq_name,
            kinetic_energy=case.kinetic_energy,
            strict_set=False,
        )
        tracking = {
            dpp: _track_one_momentum(
                env,
                dpp,
                accelerator,
                _driven_tunes_from_natural(*measured_tunes[dpp]),
                excitation_scale=excitation_scale,
            )
            for dpp in DPP_VALUES
        }
    finally:
        iface.close()

    _assert_xsuite_matches_madng_closed_orbit(free_twiss, tracking)

    _log_elapsed(
        "PSB pipeline machine fixture",
        fixture_start,
        truth_strengths=len(truth),
        correctors=len(correctors),
        tracking_sets=len(tracking),
    )
    return MachineArtifacts(
        case=case,
        accelerator=accelerator,
        truth=truth,
        tune_knobs=tune_knobs,
        corrector_file=corrector_file,
        corrector_knobs=correctors,
        closed_orbits=closed_orbits,
        free_twiss=free_twiss,
        measured_tunes=measured_tunes,
        tracking=tracking,
    )


KICKER_FLATTOP_TURNS = 5


@pytest.mark.parametrize(
    "noise_factor",
    [
        pytest.param(0.0, id="noise-0"),
    ],
)
def test_psb_kicker_measurement_and_optimisation(
    noise_factor: float,
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: GradientDescentMadInterface,
) -> None:
    """Simulate a single-kick measurement and run a full optimisation.

    A single-turn kicker measurement differs from the driven ACD pipeline
    above: the kicker marker directly supplies the initial x/px/y/py state
    (no tmom_recon momentum reconstruction is needed) and only the
    downstream BPM x/y positions are corrupted with packaged BPM
    resolution noise, matching what a real kicker measurement provides.

    Kicker mode has only one multi-turn training sample and validation is
    deliberately disabled in the tracking plan, so this asserts a large
    training-objective reduction rather than using held-out loss.
    """
    test_start = time.perf_counter()
    rng = np.random.default_rng(20260812 + int(100 * noise_factor))
    resolution = load_bpm_noise_table("psb").set_index("name") if noise_factor else None
    kicker_name: str | None = None
    magnet_strengths: dict[str, float] | None = None
    measurement_details: dict[Path, MeasurementDetails] = {}
    for dpp in DPP_VALUES:
        track_path = tmp_path / f"kicker_track_{dpp:+.4f}.parquet"
        corrector_file, magnet_strengths, tune_knobs, kicker_name = _generate_kicker_track(
            loaded_psb_interface,
            KICKER_FLATTOP_TURNS,
            track_path,
            dpp_value=dpp,
            bpm_pattern=BPM_PATTERN,
            magnet_strengths=magnet_strengths,
            bpms=PSB_ORBIT_BPM_NAMES,
            correctors=PSB_X_ORBIT_CORRECTOR_NAMES,
        )

        tracking = pd.read_parquet(track_path)
        bpm_mask = tracking["name"].astype(str).str.match(BPM_PATTERN) & (
            tracking["name"] != kicker_name
        )
        if noise_factor:
            assert resolution is not None
            names = tracking.loc[bpm_mask, "name"]
            std_x = noise_factor * names.map(resolution["Horizontal_STD"]).to_numpy(dtype=float)
            std_y = noise_factor * names.map(resolution["Vertical_STD"]).to_numpy(dtype=float)
            tracking.loc[bpm_mask, "x"] += rng.normal(0.0, std_x)
            tracking.loc[bpm_mask, "y"] += rng.normal(0.0, std_y)
            tracking.loc[bpm_mask, "var_x"] = std_x**2
            tracking.loc[bpm_mask, "var_y"] = std_y**2
            tracking.to_parquet(track_path, index=False)
        measurement_details[track_path] = MeasurementDetails(
            interface_options={
                "machine_state": merge_machine_states(corrector_file, tune_knobs),
            },
            machine_deltap=dpp,
        )
        LOGGER.info(
            "Prepared kicker measurement; noise_factor=%s, dpp=%+.4e, rows=%d, bpm_rows=%d",
            noise_factor,
            dpp,
            len(tracking),
            int(bpm_mask.sum()),
        )
    assert kicker_name is not None
    assert magnet_strengths is not None

    accelerator = PSB(
        ring=3,
        kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
        sequence_file=seq_psb,
        errors={"quad": {"k1"}},
    )
    ctrl = KickerFitter(
        accelerator,
        OptimiserConfig(
            max_epochs=100,
            warmup_epochs=20,
            warmup_lr_start=1e-6,
            max_lr=2e-5,
            min_lr=1e-6,
            gradient_converged_value=5e-20,
        ),
        SimulationConfig(num_workers=1, num_batches=1, optimise_momenta=False),
        SequenceConfig(magnet_range="$start/$end"),
        MeasurementConfig(measurement_details),
        KickerConfig(kicker_name=kicker_name, turns_after_kicker=KICKER_FLATTOP_TURNS),
        output_config=OutputConfig(
            mad_logfile=tmp_path / "kicker_mad.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths.copy(),
    )

    degeneracy = ctrl.check_degeneracy()
    LOGGER.info(
        "Kicker degeneracy: knobs=%d, rank=%d, condition_number=%.3e, degenerate=%s",
        len(degeneracy.knob_names),
        degeneracy.numerical_rank,
        degeneracy.condition_number,
        _degenerate_summary(degeneracy),
    )

    initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.initial_knobs)
    initial_diff = _rms(
        [ctrl.initial_knobs[name] - magnet_strengths[name] for name in magnet_strengths]
    )
    estimated_strengths, _ = ctrl.run()
    estimated_knobs = {name: estimated_strengths[name] for name in ctrl.initial_knobs}
    final_loss = evaluate_controller_worker_loss(ctrl, estimated_knobs)
    final_diff = _rms([estimated_knobs[name] - magnet_strengths[name] for name in estimated_knobs])
    truth_knobs = {name: magnet_strengths[name] for name in ctrl.initial_knobs}
    truth_loss = evaluate_controller_worker_loss(ctrl, truth_knobs)
    LOGGER.info(
        "Kicker optimisation summary: noise=%s, loss initial/final/truth="
        "%.3e/%.3e/%.3e, quad diff initial/final=%.3e/%.3e",
        noise_factor,
        initial_loss,
        final_loss,
        truth_loss,
        initial_diff,
        final_diff,
    )
    diag = (
        f"noise={noise_factor:g}, loss initial/final/truth="
        f"{initial_loss:.3e}/{final_loss:.3e}/{truth_loss:.3e}, "
        f"quad diff initial/final={initial_diff:.3e}/{final_diff:.3e}"
    )
    LOGGER.info("Kicker objective summary: %s", diag)
    assert truth_loss < initial_loss, diag
    assert final_loss < 0.25 * initial_loss, diag
    _log_elapsed("PSB kicker measurement test", test_start, noise_factor=noise_factor)
