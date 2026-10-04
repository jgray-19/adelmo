#!/usr/bin/env python3
"""Plot arc-by-arc dpp variation comparison between simulation and measurement.

This script compares the arc-by-arc dpp results for the 0 dpp case from:
1. Simulation with guessed errors (from run_dpp_scan.py results)
2. Actual measurement (from optimise_closed_orbit.py results)

Currently supports Beam 1 only.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from aba_optimiser.config import PROJECT_ROOT

logger = logging.getLogger("plot_arc_comparison")

FONT_SIZE = 14
SCALE_FACTOR = 1e5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot arc-by-arc dpp variation comparison for measurement point"
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=1,
        choices=[1, 2],
        help="Beam number (default: 1)",
    )
    parser.add_argument(
        "--point",
        type=str,
        default="0",
        choices=["0", "0p1", "0p2", "m0p1", "m0p2"],
        help="dpp point label (default: 0)",
    )
    parser.add_argument(
        "--sim-results",
        type=Path,
        help="Path to simulation results file (overrides auto-discovery)",
    )
    parser.add_argument(
        "--meas-results",
        type=Path,
        help="Path to measurement results file (default: b{beam}co_results/{point}.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the plot (default: plots/arc_comparison_beam{beam}_{point}.png)",
    )
    parser.add_argument(
        "--label-sim",
        type=str,
        default="Simulation (guessed errors)",
        help="Label for simulation data in legend",
    )
    parser.add_argument(
        "--label-meas",
        type=str,
        default="Measurement",
        help="Label for measurement data in legend",
    )
    return parser.parse_args()


def read_arc_results(filepath: Path, point_label: str | None = None) -> dict[str, float]:
    """Read arc results from a results file and return a dict of arc_index -> deltap value.

    Handles two formats:
    1. Measurement format: range\tdeltap (with arc1, arc2, ... MeanArcs, etc.)
    2. Simulation format: Label\tExpected_dpp\tMean_dpp\tStd_dpp\tArcs (with comma-separated arc values)
    """
    results = {}
    try:
        with filepath.open("r") as f:
            lines = f.readlines()

        # Detect format from first non-empty, non-header line
        header = lines[0].strip() if lines else ""

        if "Label\tExpected_dpp" in header:
            # Simulation results.txt format
            if point_label is None:
                logger.error("point_label required for simulation results format")
                return results

            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 5 and parts[0] == point_label:
                    # Found the point line, extract arc values
                    arc_values_str = parts[4]  # Last column has comma-separated arc values
                    arc_values = [float(v) for v in arc_values_str.split(",")]
                    for idx, val in enumerate(arc_values, start=1):
                        results[f"arc{idx}"] = val
                    break
        else:
            # Measurement format: range\tdeltap
            for line in lines:
                line = line.strip()
                if not line or line.startswith("range"):
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    key, value = parts
                    if key.startswith("arc"):
                        try:
                            results[key] = float(value)
                        except ValueError:
                            logger.warning(f"Could not parse value {value} for {key}")
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise

    return results


def find_simulation_results(point_label: str = "0", beam: int = 1) -> Path | None:
    """Find the simulation results file for a given point label (0, 0p1, etc.)."""
    dpp_map = {
        "0": "0",
        "0p1": "0p1",
        "0p2": "0p2",
        "m0p1": "m0p1",
        "m0p2": "m0p2",
    }

    if point_label not in dpp_map:
        logger.warning(f"Unknown point label: {point_label}")
        return None

    sim_dir = PROJECT_ROOT / "runs" / "dpp_scan_sim" / f"beam{beam}_results"
    # Look for results.txt in the results directory
    if sim_dir.exists():
        results_file = sim_dir / "results.txt"
        if results_file.exists():
            return results_file
        # Also check for point-specific subdirectories for backward compatibility
        point_dir = sim_dir / f"beam{beam}_{point_label}"
        if point_dir.exists() and (point_dir / "results.txt").exists():
            return point_dir / "results.txt"

    logger.warning(f"Could not find simulation results for Beam{beam} point {point_label}")
    return None


def plot_comparison(
    sim_results: dict[str, float],
    meas_results: dict[str, float],
    output_path: Path,
    label_sim: str,
    label_meas: str,
    beam: int = 1,
    point: str = "0",
) -> None:
    """Create comparison plots of arc-by-arc dpp variation."""
    # Extract arc indices and values, ensuring they're in order
    arc_indices = sorted(
        {int(k.replace("arc", "")) for k in sim_results.keys() | meas_results.keys()}
    )

    sim_values = np.array([sim_results.get(f"arc{i}", np.nan) for i in arc_indices])
    meas_values = np.array([meas_results.get(f"arc{i}", np.nan) for i in arc_indices])

    x = np.arange(len(arc_indices))
    width = 0.35

    with plt.rc_context({"font.size": FONT_SIZE}):
        # --- Plot 1: Bar chart comparison ---
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot bars
        ax.bar(x - width / 2, sim_values * SCALE_FACTOR, width, label=label_sim)
        ax.bar(x + width / 2, meas_values * SCALE_FACTOR, width, label=label_meas)

        # Labels and formatting
        ax.set_xlabel("Arc")
        ax.set_ylabel(r"$\Delta p/p$ ($\times 10^{-5}$)")
        ax.set_title(
            rf"Arc-by-Arc $\Delta p/p$ Variation: Simulation vs Measurement (Beam {beam}, {point} dpp)"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Arc {i}" for i in arc_indices])
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.grid(visible=True, alpha=0.3, axis="y")
        ax.legend(fontsize=FONT_SIZE - 1)

        # Format y-axis
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved bar comparison plot to {output_path}")
        plt.close(fig)

        # --- Plot 2: Point/Scatter chart ---
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(arc_indices, sim_values * SCALE_FACTOR, s=100, marker="o",
                   label=label_sim, alpha=0.7, edgecolors="black", linewidth=1.5)
        ax.scatter(arc_indices, meas_values * SCALE_FACTOR, s=100, marker="s",
                   label=label_meas, alpha=0.7, edgecolors="black", linewidth=1.5)

        ax.set_xlabel("Arc")
        ax.set_ylabel(r"$\Delta p/p$ ($\times 10^{-5}$)")
        ax.set_title(
            rf"Arc-by-Arc $\Delta p/p$: Point Chart Comparison (Beam {beam}, {point} dpp)"
        )
        ax.set_xticks(arc_indices)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.grid(visible=True, alpha=0.3)
        ax.legend(fontsize=FONT_SIZE - 1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

        fig.tight_layout()
        point_chart_path = output_path.parent / output_path.name.replace(".png", "_points.png")
        fig.savefig(point_chart_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved point chart to {point_chart_path}")
        plt.close(fig)

        # --- Plot 3: Consecutive arc differences plot ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate differences between consecutive arcs
        sim_consecutive_diffs = np.diff(sim_values)
        meas_consecutive_diffs = np.diff(meas_values)
        arc_pair_labels = [f"Arc{i}-Arc{i-1}" for i in range(2, len(arc_indices) + 1)]
        arc_pair_indices = np.arange(1, len(arc_indices))

        ax.scatter(arc_pair_indices, sim_consecutive_diffs * SCALE_FACTOR, s=100, marker="o",
                   label=label_sim, alpha=0.7, edgecolors="black", linewidth=1.5)
        ax.scatter(arc_pair_indices, meas_consecutive_diffs * SCALE_FACTOR, s=100, marker="s",
                   label=label_meas, alpha=0.7, edgecolors="black", linewidth=1.5)

        ax.set_xlabel("Arc Pair")
        ax.set_ylabel(r"$\Delta p/p$ difference ($\times 10^{-5}$)")
        ax.set_title(
            rf"Consecutive Arc Differences: Arc(n) - Arc(n-1) (Beam {beam}, {point} dpp)"
        )
        ax.set_xticks(arc_pair_indices)
        ax.set_xticklabels(arc_pair_labels)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=1, alpha=0.5)
        ax.grid(visible=True, alpha=0.3)
        ax.legend(fontsize=FONT_SIZE - 1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

        fig.tight_layout()
        arc_diff_path = output_path.parent / output_path.name.replace(".png", "_arc_diff.png")
        fig.savefig(arc_diff_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved arc differences plot to {arc_diff_path}")
        plt.close(fig)


def print_statistics(
    sim_results: dict[str, float],
    meas_results: dict[str, float],
    label_sim: str,
    label_meas: str,
) -> None:
    """Print statistics for both datasets."""
    arc_indices = sorted(
        {int(k.replace("arc", "")) for k in sim_results.keys() | meas_results.keys()}
    )

    sim_values = np.array([sim_results.get(f"arc{i}", np.nan) for i in arc_indices])
    meas_values = np.array([meas_results.get(f"arc{i}", np.nan) for i in arc_indices])

    print(f"\n{'='*60}")
    print("Arc-by-Arc dpp Comparison Statistics")
    print(f"{'='*60}")

    print(f"\n{label_sim}:")
    print(f"  Mean:     {np.nanmean(sim_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵")
    print(f"  Std Dev:  {np.nanstd(sim_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵")
    print(f"  Min:      {np.nanmin(sim_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵  (Arc {arc_indices[np.nanargmin(sim_values)]})")
    print(f"  Max:      {np.nanmax(sim_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵  (Arc {arc_indices[np.nanargmax(sim_values)]})")

    print(f"\n{label_meas}:")
    print(f"  Mean:     {np.nanmean(meas_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵")
    print(f"  Std Dev:  {np.nanstd(meas_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵")
    print(f"  Min:      {np.nanmin(meas_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵  (Arc {arc_indices[np.nanargmin(meas_values)]})")
    print(f"  Max:      {np.nanmax(meas_values) * SCALE_FACTOR:>12.3f} x 10⁻⁵  (Arc {arc_indices[np.nanargmax(meas_values)]})")

    # Print arc-by-arc comparison
    print("\nArc-by-Arc Values:")
    print(f"{'Arc':<8} {label_sim:<25} {label_meas:<25} {'Diff':<15}")
    print("-" * 75)
    for i in arc_indices:
        sim_val = sim_results.get(f"arc{i}", np.nan)
        meas_val = meas_results.get(f"arc{i}", np.nan)
        diff = sim_val - meas_val if not (np.isnan(sim_val) or np.isnan(meas_val)) else np.nan
        print(
            f"{i:<8} {sim_val * SCALE_FACTOR:>24.3f}   {meas_val * SCALE_FACTOR:>24.3f}   "
            f"{diff * SCALE_FACTOR:>14.3f}"
        )

    print(f"{'='*60}\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Set default paths based on beam and point
    beam = args.beam
    point = args.point

    meas_results_path = args.meas_results or (PROJECT_ROOT / f"b{beam}co_results" / f"{point}.txt")
    output_path = args.output or (PROJECT_ROOT / "plots" / f"arc_comparison_beam{beam}_{point}.png")

    # Try to find simulation results if not provided
    sim_path = args.sim_results
    if sim_path is None:
        sim_path = find_simulation_results(point_label=point, beam=beam)
        if sim_path is None:
            logger.error("Could not locate simulation results. Please provide --sim-results.")
            return

    logger.info(f"Reading simulation results from {sim_path}")
    logger.info(f"Reading measurement results from {meas_results_path}")

    try:
        sim_results = read_arc_results(sim_path, point_label=point)
        meas_results = read_arc_results(meas_results_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to read results: {e}")
        return

    if not sim_results or not meas_results:
        logger.error("No valid arc results found in one or both files")
        return

    logger.info(f"Loaded {len(sim_results)} simulation arcs and {len(meas_results)} measurement arcs")

    # Print statistics
    print_statistics(sim_results, meas_results, args.label_sim, args.label_meas)

    # Create plots
    plot_comparison(sim_results, meas_results, output_path, args.label_sim, args.label_meas, beam, point)


if __name__ == "__main__":
    main()
