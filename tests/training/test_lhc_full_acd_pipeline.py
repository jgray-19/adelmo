"""End-to-end LHC closed-orbit-free ACD reconstruction, stopping short of the fit.

The LHC counterpart of ``test_psb_full_acd_pipeline.py``: real MAD-NG truth
quadrupole errors, real xsuite AC-dipole tracking, real Harpy/OMC3 driven and
compensated optics, real ``tmom_recon`` momentum reconstruction, and a real
``ACDMarkerFitter`` construction/wiring check -- no mocks anywhere. It does
NOT call ``fitter.run()``: at LHC's full-ring scale the optimiser settings
ported from the PSB test did not converge in a reasonable epoch budget (see
the test's own docstring below), and tuning that is separate follow-up work.

Deliberately narrower than the PSB test: quadrupoles only (no bends, so no
orbit correction or multi-dpp momentum-reference fit is needed -- a pure
quadrupole-gradient error does not kick a particle sitting on a centred
orbit), a single dpp=0, and no injected BPM noise, so the reconstruction can
be held to tight truth-recovery tolerances instead of noise-dominated ones.

Reuses the pre-built ``models/lhcb1_12cm`` OMC3 model directory (real
twiss.dat/twiss_ac.dat from a saved, matched LHC sequence) as the Harpy/optics
reference model, rather than regenerating one: OMC3's own LHC model creator
needs AFS-only production modifier files that are not portable to this test.

pytest /afs/cern.ch/work/j/jmgray/private/sgd-magnet-tuner/tests/training/test_lhc_full_acd_pipeline.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs
from xtrack_tools.acd import run_ac_dipole_tracking
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.measurements.acd_pipeline import (
    ACDOpticsAnalysisConfig,
    run_driven_and_compensated_optics,
)
from aba_optimiser.measurements.reference import reconstruction_frame
from aba_optimiser.training.config.models import (
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.tracking.fitter import ACDMarkerFitter, FitterOptions

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

LOGGER = logging.getLogger(__name__)
pytestmark = pytest.mark.serial

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "lhcb1_12cm"
SEQ_B1 = MODEL_DIR / "lhcb1_saved.seq"

# Real values, not fabricated: the AC-dipole marker and excitation amplitudes
# used by xtrack-tools' own LHC ACD tests (xtrack-tools/tests/test_acd.py),
# and the general BPM/corrector patterns from LHC(BaseLHC) class constants.
ACD_NAME = "MKQA.6L4.B1"
HORIZONTAL_EXCITATION = 0.000371554879506
VERTICAL_EXCITATION = 0.000415765635123
BPM_PATTERN = r"(?i)^bpm.*$"
QUAD_REL_RMS = 5e-4
RAMP_TURNS = 100
FLATTOP_TURNS = 2000
DRIVEN_TUNE_OFFSETS = (-0.01, 0.01)
REFERENCE_DPP = 0.0

# Each worker is a separate MAD-NG tracking subprocess; leave headroom for the
# 2 validation workers and the main pytest process rather than oversubscribing
# whatever host this runs on.
NUM_TRAINING_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 3))

# Reconstruction quality measured directly against the real xsuite truth
# momenta in this test (no noise injected): px/py RMS errors of 1.35e-6 and
# 3.55e-6 rad against signal RMS of 1.81e-5/4.57e-5 rad (~7-8% relative).
# This is a real optics-model/harpy systematic (nominal reference model vs.
# perturbed truth), not something a tighter fit will remove -- see the module
# docstring's rationale for the noise-free tolerance regime.
MAX_RECON_RELATIVE_ERROR = 0.15


def _rms(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def _twiss_by_lower_name(twiss: pd.DataFrame) -> pd.DataFrame:
    frame = twiss.copy()
    columns = {str(column).lower(): column for column in frame.columns}
    names = frame[columns["name"]] if "name" in columns else frame.index
    frame.index = pd.Index([str(name).lower() for name in names])
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


@dataclass
class LhcMachine:
    accelerator: LHC
    truth: dict[str, float]
    free_twiss: pd.DataFrame
    reference_co: pd.DataFrame
    tracking: pd.DataFrame
    driven_tunes: tuple[float, float]


def _build_machine(root: Path) -> LhcMachine:
    start = time.perf_counter()
    accelerator = LHC(
        beam=1,
        sequence_file=SEQ_B1,
        kinetic_energy=6800.0,
        errors={"quad": {"k1"}},
    )
    iface = GradientDescentMadInterface(accelerator)
    try:
        truth, _ = iface.apply_magnet_perturbations(
            rel_error=QUAD_REL_RMS, seed=24, magnet_type="q"
        )
        free_twiss = iface.run_twiss(observe=1, deltap=REFERENCE_DPP, coupling=True, method=6)
    finally:
        iface.close()
    q1, q2 = float(free_twiss.headers["q1"]), float(free_twiss.headers["q2"])
    bpms = free_twiss.loc[free_twiss.index.to_series().str.match(BPM_PATTERN)]
    assert len(bpms) > 500, f"expected >500 LHC BPMs, got {len(bpms)}"
    reference_co = pd.DataFrame(
        {"x": bpms["x"], "y": bpms["y"], "errx": 1e-8, "erry": 1e-8}, index=bpms.index
    )
    driven_tunes = ((q1 % 1.0) + DRIVEN_TUNE_OFFSETS[0], (q2 % 1.0) + DRIVEN_TUNE_OFFSETS[1])

    env = initialise_env(
        matched_tunes={},
        magnet_strengths=truth,
        corrector_table=tfs.TfsDataFrame({"name": [], "kind": [], "hkick": [], "vkick": []}),
        sequence_file=SEQ_B1,
        seq_name=accelerator.seq_name,
        kinetic_energy=6800.0,
        strict_set=False,
    )
    line = env[accelerator.seq_name.lower()].copy()
    xt_twiss = line.twiss(method="4d")
    monitored = run_ac_dipole_tracking(
        line=line,
        acd_marker=ACD_NAME,
        sequence_name=accelerator.seq_name,
        tws=xt_twiss,
        ramp_turns=RAMP_TURNS,
        flattop_turns=FLATTOP_TURNS,
        driven_tunes=list(driven_tunes),
        bpm_pattern=BPM_PATTERN,
        deltap=REFERENCE_DPP,
        state_markers=True,
        horizontal_excitation=HORIZONTAL_EXCITATION,
        vertical_excitation=VERTICAL_EXCITATION,
    )
    tracked = process_tracking_data(monitored, ramp_turns=RAMP_TURNS, flattop_turns=FLATTOP_TURNS)
    tracked["name"] = tracked["name"].astype(str).str.upper()
    tracked["bunch_number"] = 0
    bpm_rows = tracked[tracked["name"].str.match(r"(?i)^BPM.*$")]
    assert bpm_rows.groupby("name", observed=True)["turn"].nunique().eq(FLATTOP_TURNS).all()
    LOGGER.info(
        "LHC pipeline machine built in %.1fs: %d truth quads, %d tracked rows",
        time.perf_counter() - start,
        len(truth),
        len(tracked),
    )
    return LhcMachine(
        accelerator=accelerator,
        truth=truth,
        free_twiss=free_twiss,
        reference_co=reference_co,
        tracking=tracked,
        driven_tunes=driven_tunes,
    )


@pytest.fixture(scope="module")
def lhc_pipeline_machine(tmp_path_factory: pytest.TempPathFactory) -> LhcMachine:
    return _build_machine(tmp_path_factory.mktemp("lhc_full_acd_machine"))


def _run_phase_analysis(*, root: Path, machine: LhcMachine) -> Path:
    """Run Harpy then driven/compensated optics against the pre-built nominal model."""
    start = time.perf_counter()
    bpm_rows = machine.tracking[machine.tracking["name"].str.match(r"(?i)^BPM.*$")]
    q1_nat = float(machine.free_twiss.headers["q1"]) % 1.0
    q2_nat = float(machine.free_twiss.headers["q2"]) % 1.0
    _driven_dir, compensated_dir = run_driven_and_compensated_optics(
        [(Path("lhc_acd.sdds"), bpm_rows)],
        output_dir=root,
        config=ACDOpticsAnalysisConfig(
            model_dir=MODEL_DIR,
            harpy_options={
                "unit": "m",
                "turns": [0, FLATTOP_TURNS],
                "clean": False,
                "keep_exact_zeros": True,
                "peak_to_peak": 1e-10,
                "max_peak": 0.02,
                "tunes": [*machine.driven_tunes, 0.0],
                "nattunes": [q1_nat, q2_nat, 0.0],
                "output_bits": 12,
                "turn_bits": 18,
                "tolerance": 1e-3,
                "tune_clean_limit": 1e-3,
                "to_write": ["full_spectra", "lin"],
            },
            optics_options={
                "three_bpm_method": True,
                "accel": "lhc",
                "beam": 1,
                "year": "2025",
            },
        ),
    )
    LOGGER.info(
        "Driven/compensated optics done in %.1fs: %s", time.perf_counter() - start, compensated_dir
    )
    return compensated_dir


def _reconstruct(*, root: Path, machine: LhcMachine, compensated_dir: Path) -> pd.DataFrame:
    start = time.perf_counter()
    model_details = ModelDetails(
        accelerator=LHC(
            beam=1,
            sequence_file=SEQ_B1,
            kinetic_energy=6800.0,
            errors={"quad": {"k1"}},
        ),
        pt=0.0,
        # The reference model carries no strengths: the ACD marker fitter that
        # consumes this reconstruction is what solves for the quadrupole
        # errors, so the reconstruction itself must start from nominal.
        magnet_strengths={},
        machine_state=None,
    )
    acd_config = ACDipoleConfig(ac_dipole_marker=ACD_NAME, driven_tunes=machine.driven_tunes)
    resolved = resolve_ac_dipole_config(model_details, acd_config)
    model_closed_orbit = _twiss_by_lower_name(resolved.closed_orbit_tws)
    marker_key = ACD_NAME.lower()
    assert marker_key in model_closed_orbit.index, (
        f"AC-dipole marker {ACD_NAME} missing from closed-orbit twiss"
    )
    # The barrier belongs to the reconstruction call, not to the AC-dipole model.
    barrier_s = float(model_closed_orbit.loc[marker_key, "s"])

    model_twiss = model_closed_orbit.loc[:, ~model_closed_orbit.columns.duplicated()].copy()
    model_twiss.index = model_twiss.index.astype(str).str.upper()
    bpm_rows = machine.tracking[machine.tracking["name"].str.match(r"(?i)^BPM.*$")]
    prepared = bpm_rows[bpm_rows["name"].isin(model_twiss.index)]
    frame = reconstruction_frame(machine.reference_co)

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

    # Compare against the real xsuite truth before packaging the fitter input:
    # a noise-free reconstruction has to track the actual momenta it was
    # generated from, not merely "look reasonable".
    truth_indexed = machine.tracking.set_index(["name", "turn"])
    recon_indexed = result.assign(name=result["name"].astype(str).str.upper()).set_index(
        ["name", "turn"]
    )
    common = recon_indexed.index.intersection(truth_indexed.index)
    for plane in ("px", "py"):
        recon_vals = recon_indexed.loc[common, plane].to_numpy(dtype=float)
        truth_vals = truth_indexed.loc[common, plane].to_numpy(dtype=float)
        finite = np.isfinite(recon_vals) & np.isfinite(truth_vals)
        relative_error = _rms(recon_vals[finite] - truth_vals[finite]) / _rms(truth_vals[finite])
        LOGGER.info("%s reconstruction relative RMS error: %.4f", plane, relative_error)
        assert relative_error < MAX_RECON_RELATIVE_ERROR, (
            f"{plane} reconstruction relative RMS error {relative_error:.4f} exceeds "
            f"{MAX_RECON_RELATIVE_ERROR}"
        )

    reconstructed = apply_precomputed_ac_dipole_bpm_overrides(result, acd_result)
    marker_rows = acd_result.loc[
        acd_result["name"].astype(str).str.lower().str.endswith(("_before", "_after"))
    ].reindex(columns=reconstructed.columns)
    marker_rows["name"] = marker_rows["name"].map(
        lambda name: f"{str(name).rsplit('_', 1)[0].upper()}_{str(name).rsplit('_', 1)[1].lower()}"
    )
    reconstructed = pd.concat([reconstructed, marker_rows], ignore_index=True)
    reconstructed["bunch_number"] = 0
    for column in ("var_x", "var_y", "var_px", "var_py"):
        if column not in reconstructed:
            reconstructed[column] = 1e-30
        reconstructed[column] = reconstructed[column].fillna(1e-30)

    LOGGER.info(
        "Reconstruction done in %.1fs: %d rows", time.perf_counter() - start, len(reconstructed)
    )
    return reconstructed


def test_lhc_acd_reconstruction_and_fitter_setup(
    tmp_path: Path, lhc_pipeline_machine: LhcMachine
) -> None:
    """Track real LHC ACD data, reconstruct momenta, and wire up the ACD marker fitter.

    Deliberately stops short of calling ``fitter.run()``: the ACDMarkerFitter
    optimiser settings ported from the PSB test do not converge at LHC's
    full-ring scale within a reasonable epoch budget (verified: loss dropped
    but gradients were tiny and the quadrupole-error residual barely moved
    over 15 epochs / ~50 minutes), and that needs real tuning work of its own.
    This test covers everything up to that point -- truth injection, ACD
    tracking, Harpy/optics, tmom_recon reconstruction, and fitter
    construction/wiring -- which is the expensive-to-get-right "preamble" and
    is independently verified here against the real tracked truth momenta.
    """
    machine = lhc_pipeline_machine
    compensated_dir = _run_phase_analysis(root=tmp_path / "hio", machine=machine)
    reconstructed = _reconstruct(root=tmp_path, machine=machine, compensated_dir=compensated_dir)

    output = tmp_path / "reconstructed.parquet"
    reconstructed.to_parquet(output, index=False)

    fitter = ACDMarkerFitter(
        accelerator=LHC(
            beam=1,
            sequence_file=SEQ_B1,
            kinetic_energy=6800.0,
            errors={"quad": {"k1"}},
        ),
        optimiser_config=OptimiserConfig(
            max_epochs=15,
            warmup_epochs=3,
            warmup_lr_start=1e-7,
            max_lr=2e-5,
            min_lr=1e-6,
            gradient_converged_value=1e-11,
        ),
        simulation_config=SimulationConfig(
            num_workers=NUM_TRAINING_WORKERS,
            num_batches=4,
            data_fraction=1.0,
            validation_fraction=0.1,
            use_fixed_bpm=True,
            enable_preloop_outlier_screening=False,
        ),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        measurement_config=MeasurementConfig(
            {output: MeasurementDetails(interface_options={}, machine_deltap=REFERENCE_DPP)}
        ),
        options=FitterOptions(
            initial_knob_strengths={},
            true_strengths=machine.truth,
            output_config=OutputConfig(
                write_tensorboard_logs=False,
                include_uncertainty=False,
                mad_logfile=tmp_path / "acd_mad.log",
            ),
        ),
    )

    # The accelerator only optimises a subset of quad families (see
    # LHC.PATTERN_MAIN_QUAD); some injected truth knobs legitimately fall
    # outside that range ("Ignoring N true strengths outside the
    # optimisation range" in the logs), so the fitter's filtered true
    # strengths are a subset of the full injected truth, not equal to it.
    quad_truth = {name for name in machine.truth if name.lower().endswith(".dk1l")}
    assert quad_truth, "expected at least one truth quadrupole knob"
    fitted_truth = set(fitter.machine.true_strengths)
    assert fitted_truth, "fitter found no true strengths within its optimisation range"
    assert fitted_truth <= quad_truth
    assert set(fitter.machine.initial_knobs) == fitted_truth
