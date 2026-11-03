"""Compare closed orbit before/after energy optimisation and after corrector optimisation.

This script generates closed orbit (x, y) from MAD-NG twiss for three cases:
1) Baseline (deltap=0, baseline correctors)
2) After energy optimisation (deltap=average from results file, baseline correctors)
3) After corrector optimisation (same average deltap, optimised correctors)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymadng_utils.io.utils import read_knobs

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.mad import GenericMadInterface, merge_machine_states

PC = 6800  # Beam energy in GeV
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare closed orbit before/after energy and corrector optimisation."
    )
    parser.add_argument("--beam", type=int, choices=[1, 2], default=1, help="Beam number")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Folder containing closed orbit optimisation outputs (e.g. b{beam}co_results)",
    )
    parser.add_argument(
        "--point",
        type=str,
        default="0",
        help="Point label to load (default: 0)",
    )
    parser.add_argument(
        "--sequence-file",
        type=Path,
        required=True,
        help="Path to the sequence file (e.g. models/lhcb{beam}_12cm/lhcb{beam}_saved.seq)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path (default: plots/closed_orbit_comparison_beam{beam}.png)",
    )
    return parser.parse_args()


def read_mean_deltap(results_file: Path) -> float:
    """Read MeanArcs deltap from a results file."""
    with results_file.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] == "MeanArcs":
                return float(parts[1])
    raise ValueError(f"MeanArcs not found in results file: {results_file}")

def read_real_closed_orbit(measurement_file: Path) -> pd.DataFrame:
    """Read parquet data and compute closed orbit (mean x/y per BPM)."""
    df = pd.read_parquet(measurement_file)
    required_cols = {"name", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet data: {sorted(missing)}")
    return df.groupby("name", observed=False)[["x", "y"]].mean().reset_index()


def generate_closed_orbit(
    sequence_file: Path,
    beam: int,
    deltap: float,
    corrector_file: Path | None,
    tune_knobs: Path | None,
    new_magnet_strengths: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate closed orbit (name, s, x, y) from MAD-NG twiss."""
    accelerator = LHC(beam=beam, kinetic_energy=PC, sequence_file=sequence_file)
    mad_iface = GenericMadInterface(
        accelerator,
        machine_state=merge_machine_states(corrector_file, tune_knobs),
    )
    if new_magnet_strengths:
        mad_iface.set_magnet_strengths(new_magnet_strengths)
    tws = mad_iface.run_twiss(deltap=deltap, observe=1).reset_index()
    bpm_mask = tws["name"].str.startswith("BPM")
    tws = tws.loc[bpm_mask, ["name", "s", "x", "y"]]
    return tws.sort_values("s")


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(values**2)))


def plot_closed_orbits(
    before: pd.DataFrame,
    after_energy: pd.DataFrame,
    after_corrector: pd.DataFrame,
    real_data: pd.DataFrame | None,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax_x.plot(before["s"], before["x"] * 1e3, label="Before", color="tab:blue")
    ax_x.plot(after_energy["s"], after_energy["x"] * 1e3, label="After energy", color="tab:orange")
    ax_x.plot(
        after_corrector["s"],
        after_corrector["x"] * 1e3,
        label="After correctors",
        color="tab:green",
    )
    if real_data is not None and not real_data.empty:
        ax_x.plot(
            real_data["s"],
            real_data["x"] * 1e3,
            label="Measurement",
            color="tab:red",
            marker="1",
            markersize=3,
            alpha=0.7,
        )
    ax_x.set_ylabel("x [mm]")
    ax_x.legend(loc="upper right")
    ax_x.grid(visible=True, alpha=0.3)

    ax_y.plot(before["s"], before["y"] * 1e3, label="Before", color="tab:blue")
    ax_y.plot(after_energy["s"], after_energy["y"] * 1e3, label="After energy", color="tab:orange")
    ax_y.plot(
        after_corrector["s"],
        after_corrector["y"] * 1e3,
        label="After correctors",
        color="tab:green",
    )
    if real_data is not None and not real_data.empty:
        ax_y.plot(
            real_data["s"],
            real_data["y"] * 1e3,
            label="Measurement",
            color="tab:red",
            # linestyle="none",
            marker="1",
            markersize=3,
            alpha=0.7,
        )
    ax_y.set_ylabel("y [mm]")
    ax_y.set_xlabel("s [m]")
    ax_y.grid(visible=True, alpha=0.3)

    fig.suptitle("Closed orbit comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    logger.info("Saved plot to %s", output_path)


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir
    point = args.point
    deltap_results_file = analysis_dir / f"{point}.txt"
    tune_knobs = analysis_dir / f"tune_knobs_{point}.txt"
    corrector_knobs = analysis_dir / f"corrector_knobs_{point}.txt"
    corrector_optimised_file = analysis_dir / f"corrector_knobs_{point}_optimised.txt"
    measurement_file = PROJECT_ROOT / f"temp_analysis_co_{args.beam}" / "pz_data.parquet"

    if not deltap_results_file.exists():
        raise FileNotFoundError(f"Missing deltap results file: {deltap_results_file}")
    if not tune_knobs.exists():
        raise FileNotFoundError(f"Missing tune knobs file: {tune_knobs}")
    if not corrector_knobs.exists():
        raise FileNotFoundError(f"Missing corrector knobs file: {corrector_knobs}")
    if not corrector_optimised_file.exists():
        raise FileNotFoundError(f"Missing optimised corrector file: {corrector_optimised_file}")

    deltap_avg = read_mean_deltap(deltap_results_file)
    logger.info("Using average deltap (MeanArcs): %.6e", deltap_avg)

    before = generate_closed_orbit(
        args.sequence_file, args.beam, 0.0, None, None
    )
    after_energy = generate_closed_orbit(
        args.sequence_file,
        args.beam,
        deltap_avg,
        None,
        None,
    )

    corrector_knobs = read_knobs(corrector_optimised_file)
    for k in corrector_knobs:
        corrector_knobs[k] *= 1

    after_corrector = generate_closed_orbit(
        args.sequence_file,
        args.beam,
        deltap_avg,
        None,
        None,
        new_magnet_strengths=corrector_knobs,
    )

    real_data = None
    if measurement_file.exists():
        real_data = read_real_closed_orbit(measurement_file)
        real_data = before[["name", "s"]].merge(real_data, on="name", how="inner")
        real_data = real_data.sort_values("s")
        logger.info("Loaded measurement closed orbit from %s", measurement_file)
    else:
        logger.warning("Measurement parquet not found, skipping real data plot: %s", measurement_file)

    merged = before.merge(after_energy, on="name", suffixes=("_before", "_energy"))
    merged = merged.merge(after_corrector, on="name", suffixes=("", "_corrector"))

    dx_energy = merged["x_energy"] - merged["x_before"]
    dy_energy = merged["y_energy"] - merged["y_before"]
    dx_corrector = merged["x"] - merged["x_energy"]
    dy_corrector = merged["y"] - merged["y_energy"]

    logger.info("RMS delta (energy - before): x=%.3e m, y=%.3e m", rms(dx_energy), rms(dy_energy))
    logger.info(
        "RMS delta (corrector - energy): x=%.3e m, y=%.3e m",
        rms(dx_corrector),
        rms(dy_corrector),
    )

    output_path = args.output
    if output_path is None:
        output_path = analysis_dir / f"closed_orbit_comparison_beam{args.beam}_{point}.png"
    plot_closed_orbits(before, after_energy, after_corrector, real_data, output_path)
    plt.show()


if __name__ == "__main__":
    main()
