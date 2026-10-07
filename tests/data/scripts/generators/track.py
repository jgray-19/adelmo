"""Integration tests for transverse momentum reconstruction using xtrack data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import tfs
import xtrack as xt
from dateutil import tz

pytest.importorskip("xtrack")
pytest.importorskip("xpart")
pytest.importorskip("xobjects")

from omc3.hole_in_one import hole_in_one_entrypoint
from turn_by_turn import write_tbt
from turn_by_turn.structures import TbtData, TransverseData
from xobjects import ContextCpu  # noqa: E402
from xtrack_tools import (
    create_xsuite_environment,
    get_monitor_names_at_pattern,
    initialise_env,
    insert_ac_dipole,
    line_to_dataframes,
    run_tracking,
)

from aba_optimiser.machine.accelerators import LHC
from aba_optimiser.machine.mad.aba_mad_interface import AbaMadInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

LOGGER = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).parent.parent
SEQUENCE_FILE = DATA_DIR / "lhcb1.seq"
NATURAL_TUNES = [0.28, 0.31]
DRIVEN_TUNES = [0.27, 0.322]
PC = 6800
DELTA_P = 2e-4
CORRECTOR_FILE_PATH = DATA_DIR / "corrector_table.tfs"
RAMP_TURNS = 1000
FLATTOP_TURNS = 6600
NTURNS = RAMP_TURNS + FLATTOP_TURNS
NOISE_LEVEL = 1e-4
SEED = 42
NUM_FILES = 3

COL_LIST = [
    "name",
    "s",
    "betx",
    "alfx",
    "bety",
    "alfy",
    "mux",
    "muy",
    "dx",
    "dy",
    "dpx",
    "dpy",
    "x",
    "y",
    "ddx",
    "ddy",
    # "wx",
    # "wy",
    # "phix",
    # "phiy",
    "dmux",
    "dmuy",
]


def run_acd_twiss(
    line: xt.Line, beam: int, dpp: float, driven_tunes: list[float]
) -> xt.TwissTable:
    """
    Run twiss calculation with AC dipole elements.

    Args:
        line: xsuite Line object
        beam: Beam number
        dpp: Delta p/p value
        driven_tunes: List of driven tunes [qx, qy, qz]

    Returns:
        Twiss table with AC dipole
    """
    line_acd = line.copy()
    before_acd_tws = line_acd.twiss(method="4d", delta0=dpp)
    acd_marker = f"mkqa.6l4.b{beam}"
    bet_at_acdipole = before_acd_tws.rows[acd_marker]

    line_acd.env.elements[f"mkach.6l4.b{beam}"] = xt.ACDipole(
        plane="x",
        natural_q=before_acd_tws["qx"] % 1,
        freq=driven_tunes[0],
        beta_at_acdipole=bet_at_acdipole["betx"],
        twiss_mode=True,
    )
    line_acd.env.elements[f"mkacv.6l4.b{beam}"] = xt.ACDipole(
        plane="y",
        natural_q=before_acd_tws["qy"] % 1,
        freq=driven_tunes[1],
        beta_at_acdipole=bet_at_acdipole["bety"],
        twiss_mode=True,
    )

    # Insert the ACDipole elements at the correct position
    placement = line_acd.get_s_position(acd_marker)
    line_acd.insert(f"mkach.6l4.b{beam}", at=placement)
    line_acd.insert(f"mkacv.6l4.b{beam}", at=placement)
    return line_acd.twiss(method="4d", delta0=dpp)


def write_tfs(
    filepath: Path,
    twiss: xt.TwissTable,
    ng_tws: tfs.TfsDataFrame | None = None,
):
    """Write twiss table to TFS file."""
    twiss_df = twiss.cols[COL_LIST].to_pandas()
    header = {
        "Q1": twiss.qx,
        "Q2": twiss.qy,
    }
    # Convert all the column names to uppercase
    twiss_df.columns = [col.upper() for col in twiss_df.columns]

    # Convert all the names to uppercase
    twiss_df["NAME"] = twiss_df["NAME"].str.upper()

    if ng_tws is not None:
        # Merge ng_tws k1l and k2l columns into twiss_df
        ng_tws.columns = [col.upper() for col in ng_tws.columns]
        ng_tws_df = ng_tws[
            ["NAME", "K1L", "K2L", "K1SL", "K3L", "K4L", "R11", "R12", "R21", "R22"]
        ]
        twiss_df = twiss_df.merge(ng_tws_df, on="NAME", how="left").fillna(0.0)

    # Write to TFS file
    tfs.write(filepath, twiss_df, headers_dict=header)


def generate_model_data():
    """Generate the model data without writing files."""
    LOGGER.info("Generating model data...")

    # Create xsuite environment
    env = create_xsuite_environment(
        sequence_file=SEQUENCE_FILE,
        seq_name="lhcb1",
    )

    line = env["lhcb1"].copy()
    tws: xt.TwissTable = line.twiss(method="4d")

    # Verify natural tunes
    qx = float(tws.qx % 1)
    qy = float(tws.qy % 1)
    assert np.isclose(qx, NATURAL_TUNES[0], atol=1e-6, rtol=1e-6)
    assert np.isclose(qy, NATURAL_TUNES[1], atol=1e-6, rtol=1e-6)

    # Run twiss with AC dipole
    tws_acd = run_acd_twiss(
        line=line,
        beam=1,
        dpp=0.0,
        driven_tunes=DRIVEN_TUNES,
    )

    # Create MAD interface and load sequence
    accel = LHC(beam=1, sequence_file=SEQUENCE_FILE)
    mad = AbaMadInterface(accel)
    mad.run_twiss(observe=0, coupling=True)
    mad.mad.send("""
local melmcol in MAD.gphys
melmcol(tws, {'k1l' , 'k2l', 'k1sl', 'k3l', 'k4l'})
    """)
    ng_tws = mad.mad.tws.convert_to_dataframe()

    # Perform orbit correction for off-momentum beam
    matched_tunes = mad.perform_orbit_correction(
        machine_deltap=DELTA_P,
        target_qx=NATURAL_TUNES[0],
        target_qy=NATURAL_TUNES[1],
        corrector_file=CORRECTOR_FILE_PATH,
    )
    # Read corrector table
    corrector_table = tfs.read(CORRECTOR_FILE_PATH)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    magnet_strengths, _ = mad.apply_magnet_perturbations(
        rel_error=1e-4,
        seed=123,
    )
    assert magnet_strengths, "Expected magnet perturbations to update strengths"

    # Create xsuite environment with orbit correction applied
    corrected_env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,  # ty:ignore[invalid-argument-type]
        sequence_file=SEQUENCE_FILE,
        seq_name="lhcb1",
    )

    return corrected_env["lhcb1"], tws, tws_acd, ng_tws
    # return base_env["lhcb1"], tws, tws_acd, ng_tws


def write_model_files(
    model_dir: Path,
    tws: xt.TwissTable,
    tws_acd: xt.TwissTable,
    ng_tws: tfs.TfsDataFrame,
):
    """Write model files to the specified directory."""
    LOGGER.info(f"Writing model files to {model_dir}")

    # Create the model directory
    model_dir.mkdir(exist_ok=True)

    # Write twiss files
    write_tfs(filepath=model_dir / "twiss_elements.dat", twiss=tws, ng_tws=ng_tws)
    write_tfs(
        filepath=model_dir / "twiss.dat",
        twiss=tws.rows["bpm.*[^k]"],
        ng_tws=ng_tws,
    )

    write_tfs(
        filepath=model_dir / "twiss_ac.dat",
        twiss=tws_acd.rows["bpm.*[^k]"],
        ng_tws=ng_tws,
    )


def perform_tracking(line, tws, files):
    """Perform particle tracking with AC dipole."""
    LOGGER.info("Performing tracking...")

    # Insert AC dipole
    trk_line = insert_ac_dipole(
        line=line,
        tws=tws,
        beam=1,
        acd_ramp=RAMP_TURNS,
        total_turns=NTURNS,
        driven_tunes=DRIVEN_TUNES,
    )

    monitor_names = get_monitor_names_at_pattern(trk_line, r"^bpm.*\.b1$")

    # Build and track particles
    ctx = ContextCpu()
    particles = trk_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
        delta=DELTA_P,
    )

    run_tracking(trk_line, particles, NTURNS, monitor_names=monitor_names)
    tracking_df = line_to_dataframes(trk_line)[0]

    # Write the true tracking data without noise as parquet for reference
    true_tracking_path = DATA_DIR / "tracking" / "true_data.parquet"
    tracking_df.to_parquet(true_tracking_path)

    # Convert to TbtData
    bpm_names = tracking_df["name"].unique().tolist()

    for i, file in enumerate(files):
        rng = np.random.default_rng(SEED + i)
        transverse_data = TransverseData(
            X=tracking_df.pivot_table(
                index="name",
                columns="turn",
                values="x",
                fill_value=0,
            ).reindex(index=bpm_names),
            Y=tracking_df.pivot_table(
                index="name",
                columns="turn",
                values="y",
                fill_value=0,
            ).reindex(index=bpm_names),
        )

        # Add noise manually
        for bpm in bpm_names:
            scale = NOISE_LEVEL / 10 if bpm.endswith("_DOROS") else NOISE_LEVEL
            transverse_data.X.loc[bpm] += rng.normal(
                0, scale, len(transverse_data.X.columns)
            )
            transverse_data.Y.loc[bpm] += rng.normal(
                0, scale, len(transverse_data.Y.columns)
            )

        # Convert from m to mm
        transverse_data.X *= 1e3
        transverse_data.Y *= 1e3

        # Create TbtData
        tbt_data = TbtData(
            matrices=[transverse_data],
            bunch_ids=[0],
            nturns=NTURNS,
            meta={
                "file": file,
                "source_datatype": "xsuite",
                "date": datetime.now(tz=tz.tzutc()),
            },
        )
        write_tbt(
            file,
            tbt_data,
            # noise=0,  # Noise already added manually
            # seed=SEED + i,
            datatype="lhc",
        )

    return files


def run_analysis(files, model_dir):
    """Run analysis on the tracking data."""
    LOGGER.info("Running analysis...")

    # Run Harpy analysis
    hole_in_one_entrypoint(  # harpy
        harpy=True,
        files=files,
        outputdir=DATA_DIR / "lin_files",
        to_write=["lin", "spectra", "bpm_summary"],
        unit="mm",
        tolerance=5e-3,
        turn_bits=18,
        opposite_direction=False,  # Beam 1
        turns=[RAMP_TURNS, NTURNS],
        accel="lhc",
        beam=1,
        year="2025",
        energy=PC,
        model_dir=model_dir,
        tunes=DRIVEN_TUNES + [0.0],
        nattunes=NATURAL_TUNES + [0.0],
        driven_excitation="acd",
        compensation="model",
        first_bpm="BPMYB.5L2.B1",
    )

    # Run optics analysis
    hole_in_one_entrypoint(  # optics
        optics=True,
        files=[DATA_DIR / "lin_files" / f.name for f in files],
        outputdir=DATA_DIR / "optics",
        energy=PC,
        model_dir=model_dir,
        compensation="model",
        accel="lhc",
        beam=1,
        year="2025",
        driven_excitation="acd",
    )


if __name__ == "__main__":
    # Generate model data
    line, tws, tws_acd, ng_tws = generate_model_data()

    # Write model files
    model_dir = DATA_DIR / "model"
    write_model_files(model_dir, tws, tws_acd, ng_tws)

    # Tracking
    tracking_dir = DATA_DIR / "tracking"
    tracking_dir.mkdir(exist_ok=True)
    files = [tracking_dir / f"acd_errs_noisy_{i}.sdds" for i in range(NUM_FILES)]
    perform_tracking(line, tws, files)

    # Analysis
    run_analysis(files, model_dir)
