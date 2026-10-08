"""SPS AC-dipole study: both planes driven together vs each plane separately.

The SPS BPMs measure one plane each (BPH horizontal, BPV vertical). This study
compares two ways of exciting the beam for a quadrupole-error fit:

- ``diagonal``: both AC-dipole planes driven in one acquisition;
- ``separate``: one acquisition driven in x only and one driven in y only.

Both cases track the same perturbed SPS lattice with xsuite, record the true
AC-dipole ``before``/``after`` states, and fit the quadrupole errors with
:class:`ACDMarkerFitter`. The AC dipole sits at ZKHA.21991 (``SPS.ac_dipole_name``).

Outputs: ``summary.csv`` / ``summary.json`` (knob recovery and residual beta-beating
per case), ``beta_beating.png`` and ``quadrupole_errors.png``.

Example:

```bash
../accpy/bin/python examples/study_sps_acd_diagonal_vs_separate.py \\
  --output-dir analysis/sps_acd_diagonal_vs_separate
```
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs
from pymadng_utils.io.utils import save_knobs
from xtrack_tools.acd import run_ac_dipole_tracking
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from aba_optimiser.accelerators import SPS
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
from aba_optimiser.training.config.models import (
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.tracking_fitter import ACDMarkerFitter

LOGGER = logging.getLogger(__name__)

SEQUENCE_FILE = Path(__file__).resolve().parents[1] / "tests" / "data" / "sequences" / "sps.seq"
KINETIC_ENERGY = 450.0
NATURAL_TUNES = (0.13, 0.18)
DRIVEN_TUNES = (0.12, 0.19)
# Gives ~1e-4 m driven amplitude at the SPS BPMs.
EXCITATION = 4e-4
RAMP_TURNS = 100
BPM_PATTERN = r"bp[hv]\..*"
TRACK_COLUMNS = ["turn", "bunch_number", "name", "x", "px", "y", "py", "var_x", "var_y", "var_px", "var_py"]
# (horizontal, vertical) excitation per acquisition of each case.
CASES = {
    "diagonal": {"xy": (EXCITATION, EXCITATION)},
    "separate": {"x": (EXCITATION, 0.0), "y": (0.0, EXCITATION)},
}


@dataclass(frozen=True)
class StudyData:
    """Tracked acquisitions of every case, plus the lattice they were tracked in."""

    files: dict[str, list[Path]]
    magnet_strengths: dict[str, float]
    tune_knobs: dict[str, float]
    tune_knobs_file: Path


def generate_cases(output_dir: Path, *, flattop_turns: int, seed: int = 42) -> StudyData:
    """Track every case's acquisitions through one quadrupole-perturbed SPS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    interface = AbaMadInterface(accelerator=SPS(sequence_file=SEQUENCE_FILE, kinetic_energy=KINETIC_ENERGY))
    errors, _ = interface.apply_magnet_perturbations(rel_error=None, seed=seed, magnet_type="q")
    tune_knobs = {
        name: float(value)
        for name, value in interface.match_tunes(
            target_qx=NATURAL_TUNES[0], target_qy=NATURAL_TUNES[1], deltap=0.0
        ).items()
    }
    magnet_strengths = interface.get_magnet_strengths(list(errors))
    tune_knobs_file = output_dir / "tune_knobs.txt"
    save_knobs(tune_knobs, tune_knobs_file)

    env = initialise_env(
        matched_tunes=tune_knobs,
        magnet_strengths=magnet_strengths,
        corrector_table=tfs.TfsDataFrame(columns=["kind", "hkick", "hkick_old", "vkick", "vkick_old"]),
        sequence_file=SEQUENCE_FILE,
        seq_name="sps",
        kinetic_energy=KINETIC_ENERGY,
        strict_set=False,
    )
    line = env["sps"]
    tws = line.twiss4d()

    files: dict[str, list[Path]] = {}
    for case, acquisitions in CASES.items():
        files[case] = [
            _track_acquisition(
                line,
                tws,
                output_dir / f"{case}_{label}.parquet",
                flattop_turns=flattop_turns,
                excitation=excitation,
            )
            for label, excitation in acquisitions.items()
        ]
    return StudyData(files, magnet_strengths, tune_knobs, tune_knobs_file)


def _track_acquisition(
    line, tws, path: Path, *, flattop_turns: int, excitation: tuple[float, float]
) -> Path:
    """Track one noise-free AC-dipole acquisition and write it in the fitter's schema."""
    accelerator = SPS(sequence_file=SEQUENCE_FILE)
    monitored = run_ac_dipole_tracking(
        line,
        accelerator.ac_dipole_name.lower(),
        "sps",
        driven_tunes=list(DRIVEN_TUNES),
        tws=tws,
        ramp_turns=RAMP_TURNS,
        flattop_turns=flattop_turns,
        bpm_pattern=BPM_PATTERN,
        state_markers=True,
        horizontal_excitation=excitation[0],
        vertical_excitation=excitation[1],
    )
    frame = process_tracking_data(monitored, RAMP_TURNS, flattop_turns, add_variance_columns=True)
    frame["bunch_number"] = 0
    # xsuite upper-cases every monitor; the MAD-NG marker names keep a lower-case side.
    frame["name"] = frame["name"].astype(str).replace(
        {accelerator.acd_marker_name(side).upper(): accelerator.acd_marker_name(side) for side in ("before", "after")}
    )
    frame.loc[:, TRACK_COLUMNS].to_parquet(path, index=False)
    return path


def build_fitter(data: StudyData, case: str, output_dir: Path, *, max_epochs: int) -> ACDMarkerFitter:
    """Return the quadrupole fit of one case's acquisitions."""
    accelerator = SPS(sequence_file=SEQUENCE_FILE, kinetic_energy=KINETIC_ENERGY, errors={"quad": {"k1"}})
    fit_dir = output_dir / case
    fit_dir.mkdir(parents=True, exist_ok=True)
    measurements = {
        path: MeasurementDetails(interface_options={"machine_state": data.tune_knobs_file})
        for path in data.files[case]
    }
    return ACDMarkerFitter(
        accelerator,
        OptimiserConfig(
            max_epochs=max_epochs,
            # The ~0.1 mm ACD orbit puts the loss near 1e-17 and the per-knob gradients
            # near 1e-12, far below Adam's default eps, so eps is dropped and each knob
            # moves by about the learning rate: a fraction of the ~5e-6 quadrupole errors.
            warmup_epochs=min(10, max_epochs),
            warmup_lr_start=5e-8,
            max_lr=1e-6,
            min_lr=1e-7,
            gradient_converged_value=1e-30,
            optimiser_type="adam",
            adam_eps=1e-30,
        ),
        # Noise-free tracking needs no held-out turns, so the loss-change stop follows
        # the training loss.
        SimulationConfig(num_workers=8, num_batches=2, validation_fraction=0.0),
        SequenceConfig(magnet_range="$start/$end"),
        MeasurementConfig(measurements),
        output_config=OutputConfig(mad_logfile=fit_dir / "mad.log", write_tensorboard_logs=False),
        true_strengths=data.magnet_strengths.copy(),
    )


def knob_error(knobs: dict[str, float], truth: dict[str, float]) -> float:
    """Summed absolute quadrupole error, ``sum |knob - truth|``."""
    return float(sum(abs(knobs[name] - truth[name]) for name in truth))


def bpm_betas(data: StudyData, strengths: dict[str, float] | None) -> pd.DataFrame:
    """BPM ``s`` and betas of the tune-matched SPS with the given quadrupole errors."""
    interface = AbaMadInterface(accelerator=SPS(sequence_file=SEQUENCE_FILE, kinetic_energy=KINETIC_ENERGY))
    interface.set_variables(**data.tune_knobs)
    if strengths:
        interface.set_magnet_strengths(strengths)
    interface.observe(interface.accelerator.bpm_pattern)
    tws = interface.run_twiss()
    interface.close()
    return tws[["s", "beta11", "beta22"]]


def beta_beating_rms(betas: pd.DataFrame, truth: pd.DataFrame) -> tuple[float, float]:
    """RMS relative beta difference from the true lattice, per plane."""
    return tuple(
        float(np.sqrt(np.mean((betas[col] / truth[col] - 1.0) ** 2))) for col in ("beta11", "beta22")
    )


def run_study(output_dir: Path, *, flattop_turns: int, max_epochs: int, seed: int) -> pd.DataFrame:
    """Generate both cases, fit them and write the summary and plots."""
    data = generate_cases(output_dir, flattop_turns=flattop_turns, seed=seed)
    truth = data.magnet_strengths
    true_betas = bpm_betas(data, truth)
    nominal_betas = bpm_betas(data, None)

    rows = []
    estimates: dict[str, dict[str, float]] = {}
    fit_betas: dict[str, pd.DataFrame] = {}
    for case in CASES:
        fitter = build_fitter(data, case, output_dir, max_epochs=max_epochs)
        initial_error = knob_error(fitter.initial_knobs, truth)
        estimate, _uncertainties = fitter.run()
        estimates[case] = estimate
        fit_betas[case] = bpm_betas(data, {name: estimate[name] for name in truth})
        beat_x, beat_y = beta_beating_rms(fit_betas[case], true_betas)
        rows.append(
            {
                "case": case,
                "acquisitions": len(data.files[case]),
                "turns_per_acquisition": flattop_turns,
                "initial_knob_error": initial_error,
                "final_knob_error": knob_error(estimate, truth),
                "residual_beta_beating_x": beat_x,
                "residual_beta_beating_y": beat_y,
            }
        )
    initial_x, initial_y = beta_beating_rms(nominal_betas, true_betas)
    summary = pd.DataFrame(rows).assign(
        initial_beta_beating_x=initial_x, initial_beta_beating_y=initial_y
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))

    _plot_beta_beating(true_betas, nominal_betas, fit_betas, output_dir / "beta_beating.png")
    _plot_quadrupole_errors(truth, estimates, output_dir / "quadrupole_errors.png")
    return summary


def _plot_beta_beating(
    truth: pd.DataFrame, nominal: pd.DataFrame, fits: dict[str, pd.DataFrame], path: Path
) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6), constrained_layout=True)
    for ax, col, plane in zip(axes, ("beta11", "beta22"), ("x", "y"), strict=True):
        ax.plot(truth["s"], 100 * (nominal[col] / truth[col] - 1), color="0.6", lw=1, label="nominal model")
        for case, betas in fits.items():
            ax.plot(truth["s"], 100 * (betas[col] / truth[col] - 1), lw=1.2, label=f"{case} fit")
        ax.set_ylabel(rf"$\Delta\beta_{plane}/\beta_{plane}$ [%]")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("s [m]")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_quadrupole_errors(
    truth: dict[str, float], estimates: dict[str, dict[str, float]], path: Path
) -> None:
    names = list(truth)
    index = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(index, [truth[name] for name in names], "k.", ms=4, label="true")
    for case, estimate in estimates.items():
        ax.plot(index, [estimate[name] for name in names], ".", ms=3, label=f"{case} fit")
    ax.set_xlabel("quadrupole (sequence order)")
    ax.set_ylabel(r"$\Delta k_1 L$ [m$^{-1}$]")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis") / "sps_acd_diagonal_vs_separate")
    parser.add_argument("--flattop-turns", type=int, default=200, help="Turns per acquisition.")
    parser.add_argument("--epochs", type=int, default=300, help="Optimiser epochs per fit.")
    parser.add_argument("--seed", type=int, default=42, help="Quadrupole error seed.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    summary = run_study(
        args.output_dir.resolve(), flattop_turns=args.flattop_turns, max_epochs=args.epochs, seed=args.seed
    )
    LOGGER.info("SPS ACD study summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
