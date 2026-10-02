"""Visualize Arc1 optimisation performance by comparing actual vs estimated tracking.

This script:
1. Reads particle data from a parquet file
2. Tracks particles through Arc1 using the base model (no optimized strengths)
3. Tracks particles through Arc1 using optimized magnet strengths
4. Creates three plots:
   - Plot 1: Initial phase space at Arc1 start (x vs px, y vs py)
   - Plot 2: Final phase space at Arc1 end with three overlays:
     * Actual (from parquet data)
     * Estimated simple (base model)
     * Estimated optimised (with Arc1 magnet corrections)
   - Plot 3: Closed orbit along Arc1 (parquet averaged vs twiss design)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.mad import GenericMadInterface, GradientDescentMadInterface
from aba_optimiser.measurements.sequence import get_or_make_sequence
from aba_optimiser.measurements.squeeze.config import (
    ANALYSIS_DIRS,
    get_model_dir,
    get_results_dir,
)
from aba_optimiser.measurements.squeeze.io import load_estimates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_BEAM_ENERGY = 450.0
# Arc 1 BPM boundaries (Beam 1)
ARC1_START_BPM = "BPM.13R3.B1"
# ARC1_START_BPM = "BPM.7L3.B1"
# ARC1_START_BPM = "BPM.13R6.B1"
# ARC1_START_BPM = "BPM.20R1.B1"
# ARC1_END_BPM = "BPM_A.7R3.B1"
# ARC1_END_BPM = "BPM.12L4.B1"
ARC1_END_BPM = "BPM.30R3.B1"
# ARC1_END_BPM = "BPM.15R3.B1"
# ARC1_END_BPM = "BPM.12L7.B1"
# ARC1_END_BPM = "BPM.27R1.B1"


def resolve_file_paths(
    squeeze_step: str, frequency: str, beam: int
) -> tuple[dict[str, Path | None], float, float]:
    """Resolve all file paths from squeeze_step and frequency.

    Args:
        squeeze_step: Squeeze step (e.g., "1.2m", "0.6m")
        frequency: Frequency (e.g., "0Hz", "+50Hz", "-50Hz")
        beam: Beam number (1 or 2)

    Returns:
        Dictionary with keys: parquet_file, sequence_file, estimates_file,
                             tune_knobs, analysis_dir, model_dir
    """
    # Model directory
    model_dir = get_model_dir(beam, squeeze_step)

    # Sequence file
    sequence_file = get_or_make_sequence(beam, model_dir)
    logger.info(f"Using sequence file: {sequence_file}")

    # Analysis directory
    if squeeze_step not in ANALYSIS_DIRS.get(beam, {}):
        raise ValueError(f"No analysis directory defined for beam {beam}, squeeze_step {squeeze_step}")
    analysis_dir = PROJECT_ROOT / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
    if not analysis_dir.exists():
        raise ValueError(f"Analysis directory not found: {analysis_dir}")
    logger.info(f"Using analysis directory: {analysis_dir}")

    # Beam energy from optimisation metadata (same source used during processing)
    metadata_file = analysis_dir / "metadata.json"
    if metadata_file.exists():
        with metadata_file.open("r") as f:
            metadata = json.load(f)
        beam_energy = float(metadata.get("energy", DEFAULT_BEAM_ENERGY))
        deltap = float(metadata.get("machine_deltap", 0.0))
    else:
        logger.warning(
            "Metadata file not found at %s, falling back to %.1f GeV",
            metadata_file,
            DEFAULT_BEAM_ENERGY,
        )
        beam_energy = DEFAULT_BEAM_ENERGY
        deltap = 0.0

    # Parquet file - find the first matching file with the given frequency
    parquet_file = None
    for pz_file in sorted(analysis_dir.glob(f"pz_data_{frequency}_*.parquet")):
        parquet_file = pz_file
        break
    if parquet_file is None:
        raise ValueError(f"No parquet file found in {analysis_dir} with pattern pz_data_{frequency}_*.parquet")
    logger.info(f"Using parquet file: {parquet_file}")

    # Estimates file
    results_dir = get_results_dir(beam)
    estimates_file = results_dir / f"quad_estimates_{squeeze_step}.txt"
    if not estimates_file.exists():
        raise ValueError(f"Estimates file not found: {estimates_file}")
    logger.info(f"Using estimates file: {estimates_file}")

    # Tune knobs file
    tune_knobs = results_dir / f"tune_knobs_{squeeze_step}_{frequency}.txt"
    if not tune_knobs.exists():
        raise ValueError(f"Tune knobs file not found: {tune_knobs}")
    logger.info(f"Using tune knobs file: {tune_knobs}")

    # Corrector file (optional)
    correctors_file = results_dir / f"corrector_strengths_{squeeze_step}_{frequency}.txt"

    return (
        {
            "parquet_file": parquet_file,
            "sequence_file": sequence_file,
            "estimates_file": estimates_file,
            "tune_knobs": tune_knobs,
            "correctors_file": correctors_file if correctors_file.exists() else None,
            "analysis_dir": analysis_dir,
            "model_dir": model_dir,
        },
        beam_energy,
        deltap,
    )


def read_parquet_data(parquet_file: Path) -> pd.DataFrame:
    """Read particle tracking data from parquet file.

    Args:
        parquet_file: Path to parquet file

    Returns:
        DataFrame with particle data
    """
    logger.info(f"Reading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    # Check required columns exist
    required_cols = ['name', 'x', 'px', 'y', 'py', 'turn', 'var_x', 'var_y', 'var_px', 'var_py']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    logger.info(f"Loaded {len(df)} rows, {len(df['name'].unique())} unique BPMs")
    return df


def get_arc1_data(df: pd.DataFrame, bpm_name: str) -> pd.DataFrame:
    """Extract data for a specific BPM.

    Args:
        df: Full tracking DataFrame
        bpm_name: BPM name to extract

    Returns:
        DataFrame filtered to specified BPM
    """
    bpm_data = df[df['name'] == bpm_name].copy()
    if bpm_data.empty:
        available = df['name'].unique()
        raise ValueError(
            f"BPM {bpm_name} not found. Available BPMs: {available}"
        )
    logger.info(f"Found {len(bpm_data)} data points for {bpm_name}")
    return bpm_data


def load_twiss_data(parquet_file: Path) -> pd.DataFrame:
    """Load twiss.dat file from the same directory as parquet file.

    Args:
        parquet_file: Path to parquet file (used to determine twiss location)

    Returns:
        DataFrame with twiss data (must have columns: name, s, x, y)
    """
    twiss_file = parquet_file.parent / "twiss.dat"
    if not twiss_file.exists():
        raise FileNotFoundError(f"Twiss file not found: {twiss_file}")

    logger.info(f"Loading twiss data from: {twiss_file}")

    # Read twiss file - typically space-separated with header
    twiss_df = tfs.read(twiss_file)

    # Check required columns
    required_cols = ['name', 's', 'x', 'y']
    missing = set(required_cols) - set(twiss_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in twiss file: {missing}. Available: {twiss_df.columns.tolist()}")

    logger.info(f"Loaded twiss data for {len(twiss_df)} elements")
    return twiss_df

def generate_twiss_data(accelerator: LHC, sequence_file: Path, deltap: float) -> pd.DataFrame:
    """Generate twiss data using MAD-NG for the given sequence file.

    Args:
        sequence_file: Path to sequence file
        accelerator: Shared accelerator instance
    Returns:
        DataFrame with twiss data (columns: name, s, x, y)
    """

    logger.info(f"Generating twiss data using MAD-NG for sequence: {sequence_file}")

    mad_iface = GenericMadInterface(accelerator)


    # Generate twiss table
    return mad_iface.run_twiss(deltap=deltap, observe=1).reset_index()

def calculate_closed_orbit_from_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate closed orbit by averaging parquet data at each BPM.

    Args:
        df: Full tracking DataFrame with particle data

    Returns:
        DataFrame with columns: name, x, y (averaged values)
    """
    # Group by BPM name and calculate mean positions
    closed_orbit = df.groupby('name', observed=False)[['x', 'y']].mean().reset_index()

    # Log turn statistics
    measurement_counts = df.groupby('name', observed=False).size()
    logger.info(f"Calculated closed orbit for {len(closed_orbit)} BPMs")
    logger.info(f"Measurement statistics per BPM: min={measurement_counts.min()}, max={measurement_counts.max()}, mean={measurement_counts.mean():.1f}")
    logger.info(f"Turn numbers in dataset: min={df['turn'].min()}, max={df['turn'].max()}")

    return closed_orbit


def plot_closed_orbit(
    parquet_co: pd.DataFrame,
    twiss_df: pd.DataFrame,
    output_dir: Path,
):
    """Create closed orbit comparison plot.

    Args:
        parquet_co: Closed orbit from parquet data (columns: name, x, y)
        twiss_df: Twiss data (columns: name, s, x, y)
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge parquet closed orbit with twiss s positions
    merged = twiss_df.merge(parquet_co, on='name', suffixes=('_twiss', '_parquet'))

    if merged.empty:
        logger.warning("No common BPMs found between parquet and twiss data")
        return

    # Sort by s position
    merged = merged.sort_values('s')

    # Get s positions for Arc1 BPMs
    arc1_start_s = twiss_df[twiss_df['name'] == ARC1_START_BPM]['s'].iloc[0] if not twiss_df[twiss_df['name'] == ARC1_START_BPM].empty else None
    arc1_end_s = twiss_df[twiss_df['name'] == ARC1_END_BPM]['s'].iloc[0] if not twiss_df[twiss_df['name'] == ARC1_END_BPM].empty else None

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # X vs S
    ax1.plot(merged['s'], merged['x_twiss'] * 1e3, 'b-', linewidth=2, label='Twiss CO', marker='o', markersize=4, alpha=0.7)
    ax1.plot(merged['s'], merged['x_parquet'] * 1e3, 'r--', linewidth=2, label='Parquet CO', marker='s', markersize=4, alpha=0.7)

    # Add vertical lines for Arc1 BPMs
    if arc1_start_s is not None:
        ax1.axvline(x=arc1_start_s, color='k', linestyle='--', alpha=0.7, label=f'{ARC1_START_BPM}')
    if arc1_end_s is not None:
        ax1.axvline(x=arc1_end_s, color='k', linestyle='--', alpha=0.7, label=f'{ARC1_END_BPM}')

    ax1.set_xlabel('s [m]')
    ax1.set_ylabel('x [mm]')
    ax1.set_title('Closed Orbit: X plane (All BPMs with data)')
    ax1.grid(visible=True, alpha=0.3)
    ax1.legend()

    # Y vs S
    ax2.plot(merged['s'], merged['y_twiss'] * 1e3, 'b-', linewidth=2, label='Twiss CO', marker='o', markersize=4, alpha=0.7)
    ax2.plot(merged['s'], merged['y_parquet'] * 1e3, 'r--', linewidth=2, label='Parquet CO', marker='s', markersize=4, alpha=0.7)

    # Add vertical lines for Arc1 BPMs
    if arc1_start_s is not None:
        ax2.axvline(x=arc1_start_s, color='k', linestyle='--', alpha=0.7, label=f'{ARC1_START_BPM}')
    if arc1_end_s is not None:
        ax2.axvline(x=arc1_end_s, color='k', linestyle='--', alpha=0.7, label=f'{ARC1_END_BPM}')

    ax2.set_xlabel('s [m]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title('Closed Orbit: Y plane (All BPMs with data)')
    ax2.grid(visible=True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = output_dir / "arc1_closed_orbit_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved closed orbit comparison plot: {plot_path}")
    plt.show()

    # Compute RMS differences
    rms_x = np.sqrt(np.mean((merged['x_twiss'].values - merged['x_parquet'].values)**2))
    rms_y = np.sqrt(np.mean((merged['y_twiss'].values - merged['y_parquet'].values)**2))

    logger.info("Closed orbit RMS differences (Twiss vs Parquet):")
    logger.info(f"  x: {rms_x*1e3:.4f} mm")
    logger.info(f"  y: {rms_y*1e3:.4f} mm")


def track_particles_through_arc1(
    accelerator: LHC,
    initial_coords: pd.DataFrame,
    magnet_strengths: dict[str, float] | None = None,
    tune_knobs: Path | None = None,
    corrector_file: Path | None = None,
    deltap: float = 0.0,
) -> pd.DataFrame:
    """Track particles through Arc1 using MAD-NG.

    Args:
        accelerator: Shared accelerator instance
        initial_coords: DataFrame with initial particle coordinates (x, px, y, py)
        magnet_strengths: Optional dictionary of magnet strengths to apply
        tune_knobs: Optional path to tune knobs file
        corrector_file: Optional path to corrector strengths file
        deltap: Relative momentum deviation (delta p/p)
    Returns:
        DataFrame with tracking results at Arc1 end
    """
    logger.info(f"Setting up tracking through Arc1 ({'with' if magnet_strengths else 'without'} optimized magnets)")
    print("tune knobs file:", tune_knobs)

    # Setup MAD interface for Arc1
    mad_iface = GradientDescentMadInterface(
        accelerator,
        magnet_range=f"{ARC1_START_BPM}/{ARC1_END_BPM}",
        bpm_range=f"{ARC1_START_BPM}/{ARC1_END_BPM}",
        corrector_knobs=corrector_file,
        tune_knobs=tune_knobs,
        py_name="py",
        # debug=True,
        # mad_logfile=Path("mad_arc1_tracking.log"),
    )
    mad_iface.cycle_to_start(ARC1_START_BPM)

    # Apply optimized magnet strengths if provided
    if magnet_strengths is not None:
        # mad_iface.set_magnet_strengths(magnet_strengths)
        mad_iface.set_madx_variables(**magnet_strengths)
    mad = mad_iface.mad

    # Track each particle and collect results
    results = []
    n_particles = len(initial_coords)
    logger.info(f"Tracking {n_particles} particles through Arc1")
    logger.info(f"Using deltap: {deltap}")
    logger.debug(f"Initial coordinates sample:\n{initial_coords.head()}")
    for idx, row in initial_coords.iterrows():
        # Run tracking
        mad.send(f"""
mtbl, _ = track {{ sequence = loaded_sequence, X0 = {{py:recv(), py:recv(), py:recv(), py:recv()}}, deltap=py:recv(), nturn=1, range = "{ARC1_START_BPM}/{ARC1_END_BPM}", observe=1 }};
""")
        mad.send(row['x'])
        mad.send(row['px'])
        mad.send(row['y'])
        mad.send(row['py'])
        mad.send(deltap)  # delta p/p

        # Get tracking data
        trk_df = mad.mtbl.to_df()
        assert trk_df.headers["deltap"] == deltap, "Mismatch in deltap after tracking"

        # Get final point (Arc1 end)
        final = trk_df[trk_df['name'] == ARC1_END_BPM]
        if not final.empty:
            results.append({
                'x': final['x'].iloc[0],
                'px': final['px'].iloc[0],
                'y': final['y'].iloc[0],
                'py': final['py'].iloc[0],
                'particle_id': idx,
            })

    logger.info(f"Successfully tracked {len(results)} particles")
    return pd.DataFrame(results)


def plot_phase_space_comparison(
    actual_start: pd.DataFrame,
    actual_end: pd.DataFrame,
    estimated_simple: pd.DataFrame,
    estimated_opt: pd.DataFrame,
    output_dir: Path,
):
    """Create phase space comparison plots.

    Args:
        actual_start: Actual data at Arc1 start
        actual_end: Actual data at Arc1 end
        estimated_simple: Base model tracking results
        estimated_opt: Optimized model tracking results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Initial phase space at Arc1 start
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.errorbar(actual_start['x'] * 1e3, actual_start['px'] * 1e3,
                 xerr=np.sqrt(actual_start['var_x']) * 1e3, yerr=np.sqrt(actual_start['var_px']) * 1e3,
                 fmt='o', markersize=5, alpha=0.6, label='Actual', ecolor='blue', capsize=2)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('px [mrad]')
    ax1.set_title(f'Initial Phase Space at {ARC1_START_BPM}')
    ax1.grid(visible=True, alpha=0.3)
    ax1.legend()

    ax2.errorbar(actual_start['y'] * 1e3, actual_start['py'] * 1e3,
                 xerr=np.sqrt(actual_start['var_y']) * 1e3, yerr=np.sqrt(actual_start['var_py']) * 1e3,
                 fmt='o', markersize=5, alpha=0.6, label='Actual', ecolor='blue', capsize=2)
    ax2.set_xlabel('y [mm]')
    ax2.set_ylabel('py [mrad]')
    ax2.set_title(f'Initial Phase Space at {ARC1_START_BPM}')
    ax2.grid(visible=True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot1_path = output_dir / "arc1_initial_phase_space.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved initial phase space plot: {plot1_path}")

    # Plot 2: Final phase space comparison
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # X plane
    ax3.errorbar(actual_end['x'] * 1e3, actual_end['px'] * 1e3,
                 xerr=np.sqrt(actual_end['var_x']) * 1e3, yerr=np.sqrt(actual_end['var_px']) * 1e3,
                 fmt='o', markersize=5, alpha=0.7, label='Actual', ecolor='blue', capsize=2)
    ax3.scatter(estimated_simple['x'] * 1e3, estimated_simple['px'] * 1e3,
                s=15, alpha=0.5, label='Base Model', color='orange', marker='x')
    ax3.scatter(estimated_opt['x'] * 1e3, estimated_opt['px'] * 1e3,
                s=15, alpha=0.5, label='Optimized Model', color='green', marker='^')
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('px [mrad]')
    ax3.set_title(f'Final Phase Space at {ARC1_END_BPM}')
    ax3.grid(visible=True, alpha=0.3)
    ax3.legend()

    # Y plane
    ax4.errorbar(actual_end['y'] * 1e3, actual_end['py'] * 1e3,
                 xerr=np.sqrt(actual_end['var_y']) * 1e3, yerr=np.sqrt(actual_end['var_py']) * 1e3,
                 fmt='o', markersize=5, alpha=0.7, label='Actual', ecolor='blue', capsize=2)
    ax4.scatter(estimated_simple['y'] * 1e3, estimated_simple['py'] * 1e3,
                s=15, alpha=0.5, label='Base Model', color='orange', marker='x')
    ax4.scatter(estimated_opt['y'] * 1e3, estimated_opt['py'] * 1e3,
                s=15, alpha=0.5, label='Optimized Model', color='green', marker='^')
    ax4.set_xlabel('y [mm]')
    ax4.set_ylabel('py [mrad]')
    ax4.set_title(f'Final Phase Space at {ARC1_END_BPM}')
    ax4.grid(visible=True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plot2_path = output_dir / "arc1_final_phase_space_comparison.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved final phase space comparison: {plot2_path}")
    plt.show()

    # Compute RMS differences
    rms_x_simple = np.sqrt(np.mean((actual_end['x'].values - estimated_simple['x'].values)**2))
    rms_x_opt = np.sqrt(np.mean((actual_end['x'].values - estimated_opt['x'].values)**2))
    rms_px_simple = np.sqrt(np.mean((actual_end['px'].values - estimated_simple['px'].values)**2))
    rms_px_opt = np.sqrt(np.mean((actual_end['px'].values - estimated_opt['px'].values)**2))

    rms_y_simple = np.sqrt(np.mean((actual_end['y'].values - estimated_simple['y'].values)**2))
    rms_y_opt = np.sqrt(np.mean((actual_end['y'].values - estimated_opt['y'].values)**2))
    rms_py_simple = np.sqrt(np.mean((actual_end['py'].values - estimated_simple['py'].values)**2))
    rms_py_opt = np.sqrt(np.mean((actual_end['py'].values - estimated_opt['py'].values)**2))

    logger.info("RMS differences (Base model):")
    logger.info(f"  x: {rms_x_simple*1e3:.4f} mm, px: {rms_px_simple*1e3:.4f} mrad")
    logger.info(f"  y: {rms_y_simple*1e3:.4f} mm, py: {rms_py_simple*1e3:.4f} mrad")

    logger.info("RMS differences (Optimized model):")
    logger.info(f"  x: {rms_x_opt*1e3:.4f} mm, px: {rms_px_opt*1e3:.4f} mrad")
    logger.info(f"  y: {rms_y_opt*1e3:.4f} mm, py: {rms_py_opt*1e3:.4f} mrad")

    improvement_x = (rms_x_simple - rms_x_opt) / rms_x_simple * 100
    improvement_px = (rms_px_simple - rms_px_opt) / rms_px_simple * 100
    improvement_y = (rms_y_simple - rms_y_opt) / rms_y_simple * 100
    improvement_py = (rms_py_simple - rms_py_opt) / rms_py_simple * 100

    logger.info("Improvement from optimisation:")
    logger.info(f"  x: {improvement_x:.2f}%, px: {improvement_px:.2f}%")
    logger.info(f"  y: {improvement_y:.2f}%, py: {improvement_py:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Arc1 optimisation performance"
    )
    parser.add_argument(
        "--squeeze-step",
        type=str,
        required=True,
        help="Squeeze step (e.g., '1.2m', '0.6m')"
    )
    parser.add_argument(
        "--beam", type=int, default=1, choices=[1, 2], help="Beam number (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: arc1_optimisation_plots)"
    )
    parser.add_argument(
        "--include-correctors",
        action="store_true",
        help="Include corrector strengths in tracking if available"
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=100,
        help="Maximum number of particles to track (default: 100)",
    )

    args = parser.parse_args()

    # Always use 0Hz frequency
    frequency = "0Hz"

    # Resolve file paths from squeeze_step and frequency
    try:
        file_paths, beam_energy, deltap = resolve_file_paths(
            args.squeeze_step, frequency, args.beam
        )
    except ValueError as e:
        logger.error(f"Error resolving file paths: {e}")
        raise

    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = (
            Path("arc1_optimisation_plots") / f"b{args.beam}_{args.squeeze_step}_{frequency}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Type assertions for required paths (resolve_file_paths ensures these are not None)
    assert file_paths["parquet_file"] is not None
    assert file_paths["sequence_file"] is not None
    assert file_paths["estimates_file"] is not None
    assert file_paths["tune_knobs"] is not None

    parquet_file = file_paths["parquet_file"]
    sequence_file = file_paths["sequence_file"]
    estimates_file = file_paths["estimates_file"]
    tune_knobs = file_paths["tune_knobs"]

    accelerator = LHC(
        beam=args.beam,
        beam_energy=beam_energy,
        sequence_file=sequence_file,
        errors={"quad": {"k1"}, "bend": {"k0"}, "other_quad": {"k1"}, "corrector": {"kick"}},
    )

    # Load data
    df = read_parquet_data(parquet_file)

    # Extract Arc1 start and end data
    start_data = get_arc1_data(df, ARC1_START_BPM)
    end_data = get_arc1_data(df, ARC1_END_BPM)

    # Limit to max particles
    if len(start_data) > args.max_particles:
        logger.info(f"Limiting to {args.max_particles} particles")
        # Select same turns for both
        selected_turns = start_data['turn'].unique()[:args.max_particles]
        start_data = start_data[start_data['turn'].isin(selected_turns)]
        end_data = end_data[end_data['turn'].isin(selected_turns)]

    # Load optimized magnet estimates
    all_estimates = load_estimates(estimates_file)
    # Extract Arc 1 magnets only - handle both nested dict and flat dict formats
    estimates: dict[str, float] = {}
    for key, estimate_dict in all_estimates.items():
        if isinstance(estimate_dict, dict):
            estimates.update({k: float(v) for k, v in estimate_dict.items()})
    initial_coords = start_data[['x', 'px', 'y', 'py']].copy()

    # Determine if correctors should be used
    correctors_file = file_paths["correctors_file"] if args.include_correctors else None
    if args.include_correctors and correctors_file is None:
        raise FileNotFoundError(
            f"Requested --include-correctors but no corrector file was found for {args.squeeze_step} {args.frequency}."
        )

    # Track with base model (no optimized strengths)
    logger.info("=" * 60)
    logger.info("Tracking with BASE MODEL (no optimisations)")
    logger.info("=" * 60)
    estimated_simple = track_particles_through_arc1(
        accelerator,
        initial_coords,
        magnet_strengths=None,
        tune_knobs=tune_knobs,
        corrector_file=correctors_file,
        deltap=deltap,
    )

    # Track with optimized magnets
    logger.info("=" * 60)
    logger.info("Tracking with OPTIMIZED MODEL")
    logger.info("=" * 60)
    estimated_opt = track_particles_through_arc1(
        accelerator,
        initial_coords,
        magnet_strengths=estimates if estimates else None,
        tune_knobs=tune_knobs,
        corrector_file=correctors_file,
        deltap=deltap,
    )

    # Create closed orbit plot first
    try:
        # twiss_df = load_twiss_data(parquet_file)
        ng_twiss = generate_twiss_data(accelerator, sequence_file, deltap=0)
        print(ng_twiss.columns, ng_twiss.index)

        parquet_co = calculate_closed_orbit_from_parquet(df)
        plot_closed_orbit(
            parquet_co=parquet_co,
            twiss_df=ng_twiss,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as e:
        logger.warning(f"Could not create closed orbit plot: {e}")

    # Create phase space plots
    plot_phase_space_comparison(
        actual_start=start_data,
        actual_end=end_data,
        estimated_simple=estimated_simple,
        estimated_opt=estimated_opt,
        output_dir=args.output_dir,
    )

    logger.info("=" * 60)
    logger.info("Arc1 optimisation check complete!")
    logger.info(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
