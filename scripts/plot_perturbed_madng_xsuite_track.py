"""Track the same perturbed machine in xsuite and MAD-NG between two BPMs.

This diagnostic script:
1. Builds one perturbed machine state using the same helpers as the training tests.
2. Tracks one particle with xsuite to create the reference measurement.
3. Replays the same segment through the MAD-NG validation worker path.
4. Saves an overlay plot and a parquet table with aligned BPM-by-BPM coordinates.

The goal is to make backend mismatches visible without running a full optimisation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pymadng_utils.io.utils import save_knobs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aba_optimiser.accelerators import LHC, SPS  # noqa: E402
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface  # noqa: E402
from tests.training.controller_test_utils import _run_track_with_model  # noqa: E402
from tests.training.helpers import generate_xsuite_env_with_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=("lhc", "sps"), default="lhc")
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--start-bpm", required=True)
    parser.add_argument("--end-bpm", required=True)
    parser.add_argument("--plane", choices=("xy", "x", "y"), default="xy")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--beam-energy", type=float, default=None)
    parser.add_argument("--dpp", type=float, default=0.0)
    parser.add_argument("--action", type=float, default=None)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--flattop-turns", type=int, default=32)
    parser.add_argument("--perturb-quads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--apply-orbit-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--target-qx", type=float, default=0.28)
    parser.add_argument("--target-qy", type=float, default=0.31)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/compare_madng_xsuite"))
    return parser.parse_args()


def default_action(accelerator_name: str) -> float:
    return 4e-7 if accelerator_name == "sps" else 4e-8


def build_accelerator(
    args: argparse.Namespace,
    *,
    errors: dict[str, set[str]] | None = None,
):
    kinetic_energy = args.beam_energy
    if kinetic_energy is None:
        kinetic_energy = 450.0 if args.accelerator == "sps" else 6800.0

    if args.accelerator == "lhc":
        return LHC(
            beam=args.beam,
            sequence_file=args.sequence_file,
            kinetic_energy=kinetic_energy,
            errors=errors,
        )
    return SPS(
        sequence_file=args.sequence_file,
        kinetic_energy=kinetic_energy,
        errors=errors,
    )


def main() -> None:
    import logging
    # Set up logging to display info messages
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    action = args.action if args.action is not None else default_action(args.accelerator)

    base_accelerator = build_accelerator(args)
    base_interface = AbaMadInterface(accelerator=base_accelerator)

    track_path = args.output_dir / "xsuite_track.parquet"
    corrector_file = (
        args.output_dir / "corrector_strengths.tfs" if args.apply_orbit_correction else None
    )
    tune_knobs = args.output_dir / "tune_knobs.txt"

    env, magnet_strengths, matched_tunes, _corrector_table = generate_xsuite_env_with_errors(
        base_interface,
        dpp_value=args.dpp,
        corrector_file=corrector_file,
        perturb_quads=args.perturb_quads,
        perturb_bends=False,
        apply_orbit_correction=args.apply_orbit_correction,
        target_qx=args.target_qx,
        target_qy=args.target_qy,
    )
    save_knobs(matched_tunes, tune_knobs)

    _run_track_with_model(
        env=env,
        flattop_turns=args.flattop_turns,
        destination=track_path,
        dpp_value=args.dpp,
        action_list=[action],
        angle_list=[args.angle],
        line_name=base_accelerator.seq_name.lower(),
        start_marker=args.start_bpm,
        use_diagonal_kicks=True,
    )

    # Take the first turn of the xsuite track as reference and track it with MAD-NG to compare BPM readings turn-by-turn.
    xsuite_results = pd.read_parquet(track_path)
    xsuite_results = xsuite_results[xsuite_results["turn"] == 1].set_index("name")
    print("xsuite results:")
    print(xsuite_results)

    base_interface.observe_bpms()
    base_interface.set_magnet_strengths(magnet_strengths)
    base_interface.mad["tbl", "flw"] = base_interface.mad.track(
        sequence = "loaded_sequence",
        X0={
            "x": xsuite_results.loc[args.start_bpm, "x"],
            "y": xsuite_results.loc[args.start_bpm, "y"],
            "px": xsuite_results.loc[args.start_bpm, "px"],
            "py": xsuite_results.loc[args.start_bpm, "py"],
        },
        range=f"'{args.start_bpm}/{args.end_bpm}'",
        method=6,
    )
    ng_result = base_interface.mad.tbl.to_df().set_index("name")
    print("MAD-NG results:")
    print(ng_result)

    # remove everythin in xsuite results that is not in MAD-NG results
    xsuite_results = xsuite_results[xsuite_results.index.isin(ng_result.index)]

    # Plot the difference between xsuite and MAD-NG for the first 10 BPMs
    plt.figure(figsize=(10, 6))
    for plane in args.plane:
        plt.subplot(1, len(args.plane), args.plane.index(plane) + 1)
        diff = xsuite_results[plane] - ng_result[plane]
        print(diff)
        plt.plot(xsuite_results.index, diff, marker="o")
        plt.xlabel("BPM Name")
        plt.ylabel(f"Difference in {plane} [m]")
        plt.title(f"Difference in {plane}-plane BPM readings")
        plt.xticks(rotation=90)
        plt.grid()
    plt.tight_layout()
    plt.savefig(args.output_dir / "xsuite_madng_difference.png")
    print("Difference plot saved to:")
    print(args.output_dir / "xsuite_madng_difference.png")
    plt.show()


if __name__ == "__main__":
    main()
