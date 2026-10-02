from __future__ import annotations

import argparse
import logging
import math
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from pymadng_utils.io.utils import save_knobs
from xobjects import ContextCpu as Context
from xtrack_tools.acd import prepare_acd_line_with_monitors
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT, OptimiserConfig, SimulationConfig
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
from aba_optimiser.training.tracking_fitter import ArcByArcFitter
from aba_optimiser.training.controller_config import SequenceConfig
from aba_optimiser.training.controller_helpers import (
    create_arc_measurement_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    import xtrack as xt

logger = logging.getLogger(__name__)

PC_GEV = 6800.0
TRACK_COLUMNS = (
    "turn",
    "name",
    "x",
    "px",
    "y",
    "py",
    "var_x",
    "var_y",
    "var_px",
    "var_py",
)
DEFAULT_DPP_VALUES = [-1e-4, 0.0, 1e-4, 2e-4, -2e-4]


@dataclass
class RangeConfig:
    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


def weighted_mean(values: list[float], uncertainties: list[float]) -> float:
    """Compute weighted mean where weights are 1/sigma^2."""
    finite_pairs = [(v, u) for v, u in zip(values, uncertainties) if u > 0]
    if not finite_pairs:
        raise ValueError("Cannot compute weighted mean without positive uncertainties")
    weights = [1 / (u**2) for _, u in finite_pairs]
    numerator = sum(v * w for (v, _), w in zip(finite_pairs, weights))
    return numerator / sum(weights)


def _variance_of_mean(values: np.ndarray) -> float:
    """Return variance of the mean using sample variance / N; zero for <2 samples."""
    if values.size <= 1:
        return 0.0
    return float(np.var(values, ddof=1) / values.size)


def average_tracking_dataframe(tracking_df: pd.DataFrame, turns: int = 3) -> pd.DataFrame:
    """Average tracking over turns per BPM and replicate a fixed number of turns.

    Mirrors the measurement pipeline where per-BPM data are averaged and the
    variance of the mean is propagated, then expanded to a small number of
    turns for downstream consumers.
    """
    rows = []
    grouped = tracking_df.groupby("name")
    for name, sub in grouped:
        xs = sub["x"].to_numpy()
        ys = sub["y"].to_numpy()
        pxs = sub["px"].to_numpy()
        pys = sub["py"].to_numpy()

        rows.append(
            {
                "name": name,
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "px": float(np.mean(pxs)),
                "py": float(np.mean(pys)),
                "var_x": 1e-4**2,
                "var_y": 1e-4**2,
                "var_px": 3e-6**2,
                "var_py": 3e-6**2,
            }
        )

    averaged = pd.DataFrame(rows)
    expanded_rows = []
    for turn in range(1, turns + 1):
        for _, row in averaged.iterrows():
            expanded_rows.append(
                {
                    "name": row["name"],
                    "turn": turn,
                    "x": row["x"],
                    "y": row["y"],
                    "px": row["px"],
                    "py": row["py"],
                    "var_x": row["var_x"],
                    "var_y": row["var_y"],
                    "var_px": row["var_px"],
                    "var_py": row["var_py"],
                }
            )

    result = pd.DataFrame(expanded_rows)
    result["name"] = result["name"].astype("category")
    result["turn"] = result["turn"].astype("int32")
    return result


def format_dpp_label(dpp: float) -> str:
    if dpp == 0:
        return "0"
    sign = "" if dpp > 0 else "m"
    dpp = dpp * 1000  # Scale to match typical notation
    dpp_str = f"{dpp}".replace(".", "p").strip("-")
    return f"{sign}{dpp_str}"


def get_sequence_file(beam: int) -> Path:
    """Get the default sequence file path for a given beam."""
    return PROJECT_ROOT / f"tests/data/sequences/lhcb{beam}.seq"


def get_results_dir(beam: int) -> Path:
    """Get the results directory for a given beam."""
    return PROJECT_ROOT / f"b{beam}_sim_results"


def get_temp_dir(beam: int) -> Path:
    """Get the temporary analysis directory for a given beam."""
    return PROJECT_ROOT / f"temp_analysis_sim_b{beam}"


def create_arc_config(beam: int) -> RangeConfig:
    """Create arc range configuration for the given beam."""
    if beam == 1:
        arc_magnet_ranges = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
        arc_bpm_starts = [[f"BPM.{i}R{s}.B1" for i in range(9, 35, 3)] for s in range(1, 9)]
        arc_bpm_end_points = [
            [f"BPM.{i}L{s % 8 + 1}.B1" for i in range(9, 34, 3)] for s in range(1, 9)
        ]
    elif beam == 2:
        arc_magnet_ranges = [f"BPM.9L{s}.B2/BPM.9R{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)]
        arc_bpm_starts = [[f"BPM.{i}L{s}.B2" for i in range(9, 34, 3)] for s in range(8, 0, -1)]
        arc_bpm_end_points = [
            [f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in range(9, 35, 3)] for s in range(8, 0, -1)
        ]
    else:
        raise ValueError(f"Unsupported beam: {beam}. Must be 1 or 2.")
    return RangeConfig(
        magnet_ranges=arc_magnet_ranges,
        bpm_starts=arc_bpm_starts,
        bpm_end_points=arc_bpm_end_points,
    )


def generate_track_with_errors(
    sequence_path: Path,
    destination_dir: Path,
    dpp_value: float,
    flattop_turns: int,
    corrector_path: Path,
    tune_knobs_path: Path,
    beam: int,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    measurement_file = destination_dir / "pz_data.parquet"

    iface = AbaMadInterface(
        accelerator=LHC(beam=beam, sequence_file=sequence_path, kinetic_energy=PC_GEV)
    )
    iface.mad["zero_twiss", "_"] = iface.mad.twiss(sequence="loaded_sequence")
    # tws = iface.mad.zero_twiss.to_df().set_index("name")

    magnet_strengths, _ = iface.apply_magnet_perturbations(
        rel_error=1e-4,
        seed=42,
        magnet_type="qsd",
    )
    matched_tunes = iface.perform_orbit_correction(
        machine_deltap=0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_path,
    )
    matched_tunes = iface.perform_orbit_correction(
        machine_deltap=dpp_value,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_path,
    )
    corrector_table = tfs.read(corrector_path)
    corrector_table = corrector_table.loc[corrector_table.loc[:, "kind"] != "monitor"]
    iface.observe_elements()
    # after_tws = iface.run_twiss(delta0=dpp_value)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(tws["s"], tws["dx"], label="x before correction")
    # plt.plot(after_tws["s"], after_tws["dx"], label="x after correction")
    # plt.xlabel("s (m)")
    # plt.ylabel("x (m)")
    # plt.legend()
    # plt.show()

    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        sequence_file=sequence_path,
        kinetic_energy=PC_GEV,
        strict_set=False,
    )

    save_knobs(matched_tunes, tune_knobs_path)

    line: xt.Line = env[iface.accelerator.seq_name]
    ramp_turns=1000
    monitored_line, total_turns = prepare_acd_line_with_monitors(
        line=line,
        beam=beam,
        tws=line.twiss4d(delta0=dpp_value),
        flattop_turns=flattop_turns,
        ramp_turns=ramp_turns,
        bpm_pattern="bpm.*[^k]",
        driven_tunes=[0.27, 0.322],
        lag=0.0,
        num_particles=1,
    )

    ctx = Context()
    particles: xt.Particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
        delta=dpp_value,
    )

    logger.info(f"Tracking {total_turns} turns with AC dipole")
    monitored_line.track(particles, num_turns=total_turns, with_progress=True)

    tracking_df = process_tracking_data(
        monitored_line, ramp_turns, flattop_turns, add_variance_columns=True
    )

    averaged_df = average_tracking_dataframe(tracking_df, turns=3)
    averaged_df.to_parquet(measurement_file, index=False)
    return measurement_file


def optimise_ranges(
    range_config: RangeConfig,
    beam: int,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    corrector_knobs: Path,
    tune_knobs: Path,
    measurement_file: Path,
    bad_bpms: list[str],
    title: str,
    sequence_path: Path,
    flattop_turns: int,
) -> tuple[list[float], list[float]]:
    results: list[float] = []
    uncertainties: list[float] = []
    num_ranges = len(range_config.magnet_ranges)
    for i in range(num_ranges):
        logger.info("Starting optimisation for arc %s/%s (%s)", i + 1, num_ranges, title)

        measurement_config = create_arc_measurement_config(
            measurement_file,
            machine_deltap=0.0,
            corrector_knobs=corrector_knobs,
            tune_knobs=tune_knobs,
        )

        accelerator = LHC(
            beam=beam,
            sequence_file=sequence_path,
            kinetic_energy=PC_GEV,
            optimise_energy=True,
        )
        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
        )

        controller = ArcByArcFitter(
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_start_points=range_config.bpm_starts[i],
            bpm_end_points=range_config.bpm_end_points[i],
            show_plots=False,
            initial_knob_strengths=None,
            true_strengths=None,
        )
        final_knobs, uncs = controller.run()
        results.append(final_knobs["deltap"])
        uncertainties.append(uncs["deltap"])
        logger.info("Arc %s: deltap = %s", i + 1, results[-1])
        logger.info("Finished optimisation for arc %s/%s (%s)", i + 1, num_ranges, title)
    return results, uncertainties


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beam",
        type=int,
        choices=[1, 2],
        default=1,
        help="Beam number (1 or 2)",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve temporary directory",
    )
    parser.add_argument(
        "--dpp-values",
        type=str,
        default=None,
        help="Comma-separated dp/p values (e.g. 0,3e-4,-3e-4)",
    )
    parser.add_argument(
        "--no-fixed-bpm",
        action="store_true",
        help="Use full Cartesian BPM pairing",
    )
    args = parser.parse_args()

    # Derive paths from beam number
    sequence_file = get_sequence_file(args.beam)
    results_dir = get_results_dir(args.beam)
    temp_dir = get_temp_dir(args.beam)

    dpp_values = (
        DEFAULT_DPP_VALUES
        if args.dpp_values is None
        else [float(item) for item in args.dpp_values.split(",") if item.strip()]
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    arc_config = create_arc_config(args.beam)

    optimiser_config = OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=3,
        warmup_lr_start=5e-7,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=1e-6,
        optimiser_type="lbfgs",
        expected_rel_error=0,
    )
    simulation_config = SimulationConfig(
        num_workers=1,
        num_batches=1,
        optimise_momenta=False,
        use_fixed_bpm=not args.no_fixed_bpm,
    )

    for dpp_value in dpp_values:
        label = format_dpp_label(dpp_value)
        logger.info("Running simulation for dpp=%s (%s)", dpp_value, label)
        sub_temp = temp_dir / label
        measurement_file = generate_track_with_errors(
            sequence_path=sequence_file,
            destination_dir=sub_temp,
            dpp_value=dpp_value,
            flattop_turns=2_000,
            corrector_path=results_dir / f"corrector_knobs_{label}.txt",
            tune_knobs_path=results_dir / f"tune_knobs_{label}.txt",
            beam=args.beam,
        )

        results, uncs = optimise_ranges(
            range_config=arc_config,
            beam=args.beam,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            corrector_knobs=results_dir / f"corrector_knobs_{label}.txt",
            tune_knobs=results_dir / f"tune_knobs_{label}.txt",
            measurement_file=measurement_file,
            bad_bpms=[],
            title=label,
            sequence_path=sequence_file,
            flattop_turns=2_000,
        )

        logger.info("All arc optimisations complete for %s", label)
        if not results:
            logger.warning("No arc results produced for %s", label)
            continue

        try:
            mean_arcs = weighted_mean(results, uncs)
        except ValueError:
            mean_arcs = float(np.mean(results))
            logger.warning(
                "Falling back to unweighted mean for %s due to non-positive uncertainties",
                label,
            )

        std_arcs = float(np.std(results))
        stderr = std_arcs / math.sqrt(len(results)) if results else 0.0
        weighted_unc = None
        if uncs and all(u > 0 for u in uncs):
            weights = [1 / (u**2) for u in uncs]
            weighted_unc = math.sqrt(1 / sum(weights))

        results_file = results_dir / f"{label}.txt"
        with results_file.open("w") as f:
            f.write("range\tdeltap\tuncertainty\n")
            for idx, (dp, unc) in enumerate(zip(results, uncs)):
                f.write(f"arc{idx + 1}\t{dp}\t{unc}\n")
            f.write(f"MeanArcs\t{mean_arcs}\t\n")
            f.write(f"StdDevArcs\t{std_arcs}\t\n")
            f.write(f"StdErrArcs\t{stderr}\t\n")
            if weighted_unc is not None:
                f.write(f"WeightedUncArcs\t{weighted_unc}\t\n")

    if not args.keep_temp:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
