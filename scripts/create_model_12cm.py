#!/usr/bin/env python3
"""
Script to create LHC model for beam 2 with 12cm optics based on the provided ini file.
"""

import pathlib
import sys

# Add the src directory to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from omc3.hole_in_one import hole_in_one_entrypoint
from pymadng_utils.model_creator.create_models import create_lhc_model


def model_creator(beam: int = 2) -> None:
    # Parameters from the ini file
    nat_tunes = [0.28, 0.31]
    drv_tunes = [0.27, 0.322]
    energy = 6800
    year = "2025"
    if beam == 2:
        modifiers = [
            "R2025aRP_A12cmC12cmA10mL200cm_Flat.madx",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Models/2025-11-07_B2_12cm/knobs.madx",
        ]
    else:
        modifiers = [
            "R2025aRP_A12cmC12cmA10mL200cm_Flat.madx",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/knobs.madx",
        ]

    # Local output directory
    output_dir = pathlib.Path("models") / f"lhcb{beam}_12cm"

    create_lhc_model(
        beam=beam,
        output_dir=output_dir,
        nat_tunes=nat_tunes,
        drv_tunes=drv_tunes,
        energy=energy,
        year=year,
        modifiers=modifiers,
        matching_knob="",
    )


def harpy_and_optics(beam: int = 2) -> None:
    analysis_folder = pathlib.Path(f"analysis_b{beam}_12cm")
    linfile_folder = analysis_folder / "linfiles"
    linfile_folder.mkdir(parents=True, exist_ok=True)
    model_dir = pathlib.Path("models") / f"lhcb{beam}_12cm"

    if beam == 2:
        meas_files = [
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Measurements/Beam2@BunchTurn@2025_11_07@03_47_23_140/Beam2@BunchTurn@2025_11_07@03_47_23_140.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Measurements/Beam2@BunchTurn@2025_11_07@03_48_41_820/Beam2@BunchTurn@2025_11_07@03_48_41_820.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Measurements/Beam2@BunchTurn@2025_11_07@04_04_50_100/Beam2@BunchTurn@2025_11_07@04_04_50_100.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Measurements/Beam2@BunchTurn@2025_11_07@04_05_55_920/Beam2@BunchTurn@2025_11_07@04_05_55_920.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Measurements/Beam2@BunchTurn@2025_11_07@04_13_34_020/Beam2@BunchTurn@2025_11_07@04_13_34_020.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Measurements/Beam2@BunchTurn@2025_11_07@04_14_47_519/Beam2@BunchTurn@2025_11_07@04_14_47_519.sdds",
        ]
        bad_bpms = [
            "BPM.32L2.B2",
            "BPM.9R1.B2",
            "BPMWE.4L3.B2",
            "BPM.28R3.B2",
            "BPM.15R3.B2",
            "BPM.33R4.B2",
            "BPMS.2L5.B2",
            "BPMWB.4L5.B2",
            "BPM.17L7.B2",
            "BPM.32R7.B2",
            "BPM.6L8.B2",
            "BPMYB.5R8.B2",
        ]
        first_bpm = "BPM.34R8.B2"
    else:
        first_bpm = "BPM.33L2.B1"
        bad_bpms = []
        meas_files = [
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@03_41_17_540/Beam1@BunchTurn@2025_11_07@03_41_17_540.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@03_40_08_940/Beam1@BunchTurn@2025_11_07@03_40_08_940.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@03_43_30_334/Beam1@BunchTurn@2025_11_07@03_43_30_334.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@03_42_25_013/Beam1@BunchTurn@2025_11_07@03_42_25_013.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@03_44_37_636/Beam1@BunchTurn@2025_11_07@03_44_37_636.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@04_03_06_778/Beam1@BunchTurn@2025_11_07@04_03_06_778.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@04_02_00_900/Beam1@BunchTurn@2025_11_07@04_02_00_900.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@04_00_51_749/Beam1@BunchTurn@2025_11_07@04_00_51_749.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@04_04_20_860/Beam1@BunchTurn@2025_11_07@04_04_20_860.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@04_12_33_860/Beam1@BunchTurn@2025_11_07@04_12_33_860.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@04_13_41_832/Beam1@BunchTurn@2025_11_07@04_13_41_832.sdds",
            "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Measurements/Beam1@BunchTurn@2025_11_07@04_14_49_860/Beam1@BunchTurn@2025_11_07@04_14_49_860.sdds",
        ]

    hole_in_one_entrypoint(
        harpy=True,
        bad_bpms=bad_bpms,
        clean=True,
        files=[pathlib.Path(f) for f in meas_files],
        first_bpm=first_bpm,
        is_free_kick=False,
        keep_exact_zeros=False,
        max_peak=0.02,
        model=model_dir / "twiss.dat",
        nattunes=[0.28, 0.31, 0.0],
        num_svd_iterations=3,
        opposite_direction=False,
        output_bits=12,
        outputdir=linfile_folder,
        peak_to_peak=1e-08,
        resonances=4,
        sing_val=12,
        svd_dominance_limit=0.925,
        tbt_datatype="lhc",
        to_write=["lin", "spectra", "bpm_summary"],
        tolerance=0.005,
        tune_clean_limit=1e-05,
        tunes=[0.27, 0.322, 0.0],
        turn_bits=18,
        turns=[0, 50000],
        unit="mm",
    )

    # Now loop through the generated linfiles and store in list
    linfiles = list(linfile_folder.glob("*.linx"))

    # Strip the .linx extension for harpy input
    linfile_paths = [str(f.with_suffix("")) for f in linfiles]

    hole_in_one_entrypoint(
        optics=True,
        analyse_dpp=0,
        calibrationdir=f"/afs/cern.ch/eng/sl/lintrack/LHC_commissioning2017/Calibration_factors_2017/Calibration_factors_2017_beam{beam}",
        chromatic_beating=False,
        compensation="equation",
        coupling_method=2,
        coupling_pairing=0,
        files=linfile_paths,
        isolation_forest=False,
        only_coupling=False,
        outputdir=analysis_folder,
        range_of_bpms=11,
        rdt_magnet_order=4,
        second_order_dispersion=False,
        three_bpm_method=False,
        three_d_excitation=False,
        union=False,
        accel="lhc",
        ats=False,
        beam=beam,
        dpp=0.0,
        model_dir=model_dir,
        xing=False,
        year="2025",
    )


if __name__ == "__main__":
    model_creator(beam=2)
    # harpy_and_optics(beam=1)
