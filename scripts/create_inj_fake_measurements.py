"""Create inj fake-measurement folders for five frequency points and archive them.

Workflow:
1. Loop over 0Hz, +/-150Hz, +/-300Hz.
2. Download tune and corrector knobs for each frequency using earliest inj measurement times.
3. Build MAD state from inj quad checkpoint + correctors + tune knobs.
4. Run Twiss with frequency-specific deltap and convert MAD-NG Twiss to MAD-X style Twiss.
5. Generate fake measurements into frequency-named folders.
6. Zip the output root and delete the unzipped folders.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from pymadng_utils.io.utils import read_knobs
from pymadng_utils.io.utils import save_knobs as write_knobs
from pymadng_utils.madx import convert_tfs_to_madx

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.measurements.online_knobs import save_online_knobs
from aba_optimiser.measurements.sequence import get_or_make_sequence
from aba_optimiser.measurements.squeeze.config import (
    MODEL_DIRS,
    PC,
    get_measurement_date,
    get_model_dir,
)

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)

FREQUENCIES = ["0Hz", "+150Hz", "-150Hz", "+300Hz", "-300Hz"]
TARGET_QX = 0.28
TARGET_QY = 0.31

# Earliest inj measurement timestamps from commented entries in optimise_squeeze_quads.py.
# Format is the raw time token used in measurement folder names: HH_MM_SS_mmm.
INJ_EARLIEST_TIMES = {
    1: {
        "0Hz": "16_49_56_490",
        "+150Hz": "17_10_45_523",
        "-150Hz": "18_02_51_331",
        "+300Hz": "17_33_14_338",
        "-300Hz": "18_16_36_372",
    },
    2: {
        "0Hz": "16_50_23_408",
        "+150Hz": "17_09_53_327",
        "-150Hz": "18_03_31_335",
        "+300Hz": "17_34_11_504",
        "-300Hz": "18_16_48_300",
    },
}

# Fixed inj deltap points aligned with the tune-vs-deltap scan conventions.
FREQ_TO_DELTAP = {
    "0Hz": 0.0,
    "+150Hz": 0.00107669,
    "-150Hz": -0.00107669,
    "+300Hz": 0.00215338,
    "-300Hz": -0.00215338,
}


def load_checkpoint_knobs(checkpoint_file: Path) -> dict[str, float]:
    LOGGER.info("Loading checkpoint knobs from %s", checkpoint_file)
    payload = json.loads(checkpoint_file.read_text())
    knob_map = payload.get("best_knobs") or payload.get("current_knobs") or {}
    out = {str(k): float(v) for k, v in knob_map.items()}
    LOGGER.info("Loaded %d checkpoint knobs", len(out))
    return out


def autodetect_quad_checkpoint(
    beam: int, squeeze_step: str, checkpoint_dir: Path | None
) -> Path:
    squeeze_step_id = squeeze_step.replace(".", "_")
    base_dir = (
        checkpoint_dir
        if checkpoint_dir is not None
        else PROJECT_ROOT / f"temp_analysis_squeeze_b{beam}_{squeeze_step_id}" / "checkpoints"
    )
    pattern = f"checkpoint_b{beam}_{squeeze_step_id}_arc*_quads.json"
    candidates = sorted(base_dir.glob(pattern))
    LOGGER.info("Searching quad checkpoints in %s with pattern %s", base_dir, pattern)
    if not candidates:
        raise FileNotFoundError(f"No quad checkpoints found in {base_dir} matching {pattern}")
    selected = max(candidates, key=lambda p: p.stat().st_mtime)
    LOGGER.info("Selected latest quad checkpoint: %s", selected)
    return selected


def autodetect_sequence_file(beam: int, squeeze_step: str) -> Path:
    try:
        model_dir = get_model_dir(beam, squeeze_step)
        sequence = get_or_make_sequence(beam, model_dir)
        LOGGER.info("Using sequence from model dir: %s", sequence)
        return sequence
    except (ValueError, FileNotFoundError):
        LOGGER.info("Model-dir sequence autodetect failed, trying local fallbacks")

    sequences_dir = PROJECT_ROOT / "sequences_from_models"
    expected_name = MODEL_DIRS.get(beam, {}).get(squeeze_step)
    if expected_name is not None:
        expected_seq = sequences_dir / f"{expected_name}.seq"
        if expected_seq.exists():
            LOGGER.info("Using expected cached sequence: %s", expected_seq)
            return expected_seq

    candidates = sorted(sequences_dir.glob("*.seq"), key=lambda p: p.stat().st_mtime)
    if candidates:
        LOGGER.info("Using latest cached sequence fallback: %s", candidates[-1])
        return candidates[-1]

    raise FileNotFoundError(
        "Could not auto-detect a sequence file. Provide --sequence explicitly."
    )


def autodetect_energy(beam: int, squeeze_step: str) -> float:
    """Auto-detect beam energy from temp-analysis metadata with fallback."""
    metadata_file = (
        PROJECT_ROOT
        / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
        / "metadata.json"
    )
    if not metadata_file.exists():
        LOGGER.info(
            "Energy metadata not found at %s, falling back to PC=%s GeV",
            metadata_file,
            PC,
        )
        return float(PC)

    payload = json.loads(metadata_file.read_text())
    energy = float(payload.get("energy", PC))
    LOGGER.info("Auto-detected beam energy from metadata: %.6f GeV", energy)
    return energy

def resolve_checkpoint_knobs(
    mad: GradientDescentMadInterface, checkpoint_knobs: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    knob_name_set = mad.knob_name_set
    direct = {k: v for k, v in checkpoint_knobs.items() if k in knob_name_set}
    unresolved_input = {k: v for k, v in checkpoint_knobs.items() if k not in knob_name_set}

    converted = mad.absolute_to_optimisation_knobs(unresolved_input) if unresolved_input else {}

    resolved = dict(direct)
    resolved.update(converted)
    unresolved = [k for k in unresolved_input if k not in converted]
    LOGGER.info(
        "Resolved checkpoint knobs: %d direct + %d converted, %d unresolved",
        len(direct),
        len(converted),
        len(unresolved),
    )
    if unresolved:
        preview = ", ".join(unresolved[:8])
        if len(unresolved) > 8:
            preview += ", ..."
        LOGGER.info("Unresolved checkpoint knob preview: %s", preview)
    return resolved, unresolved


def _parse_measurement_time(beam: int, squeeze_step: str, time_token: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    date = get_measurement_date(squeeze_step)
    hh_mm_ss = time_token.split("_")[:3]
    ts = f"{date} {'_'.join(hh_mm_ss)}"
    parsed = datetime.strptime(ts, "%Y-%m-%d %H_%M_%S").replace(tzinfo=ZoneInfo("UTC"))
    LOGGER.info(
        "Parsed measurement time for beam=%d step=%s token=%s -> %s",
        beam,
        squeeze_step,
        time_token,
        parsed.isoformat(),
    )
    return parsed


def download_knobs_for_frequency(
    *,
    beam: int,
    squeeze_step: str,
    freq: str,
    pc: float,
    freq_dir: Path,
    drop_ks_knobs: bool,
) -> tuple[Path, Path, str]:
    if beam not in INJ_EARLIEST_TIMES or freq not in INJ_EARLIEST_TIMES[beam]:
        raise ValueError(f"No earliest inj time configured for beam={beam}, frequency={freq}")

    time_token = INJ_EARLIEST_TIMES[beam][freq]
    meas_time = _parse_measurement_time(beam, squeeze_step, time_token)

    tune_file = freq_dir / f"tune_knobs_{squeeze_step}_{freq}.txt"
    corrector_file = freq_dir / f"corrector_strengths_{squeeze_step}_{freq}.txt"
    LOGGER.info(
        "Downloading knobs for %s at %s -> tune=%s, correctors=%s",
        freq,
        meas_time.isoformat(),
        tune_file,
        corrector_file,
    )
    save_online_knobs(
        meas_time,
        beam=beam,
        tune_knobs=tune_file,
        corrector_knobs=corrector_file,
        energy=pc,
    )
    if drop_ks_knobs:
        tune_knobs = read_knobs(tune_file)
        tune_clean = {
            k: v for k, v in tune_knobs.items() if not str(k).lower().startswith("ks")
        }
        write_knobs(tune_clean, tune_file)
        LOGGER.info(
            "Removed ks* keys from knob snapshots for %s: tune=%d",
            freq,
            len(tune_knobs) - len(tune_clean),
        )
    LOGGER.info("Saved online knobs for %s", freq)
    return tune_file, corrector_file, meas_time.isoformat()


def build_madx_twiss_for_frequency(
    *,
    beam: int,
    pc: float,
    sequence_file: Path,
    checkpoint_knobs: dict[str, float],
    tune_knobs: Path,
    corrector_knobs: Path,
    deltap: float,
    match_model_tunes: bool,
    model_twiss_onmom: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    LOGGER.info(
        "Building measurement Twiss (with knobs/checkpoint) and base model Twiss (clean) for beam=%d deltap=%+.8f",
        beam,
        deltap,
    )

    # 1) Measurement Twiss: apply checkpoint + tune/corrector knobs.
    LOGGER.info(
        "Building measurement Twiss for beam=%d deltap=%+.8f sequence=%s tune=%s correctors=%s",
        beam,
        deltap,
        sequence_file,
        tune_knobs,
        corrector_knobs,
    )
    accelerator_meas = LHC(
        beam=beam,
        sequence_file=sequence_file,
        kinetic_energy=pc,
        errors={"quad": {"k1"}, "bend": {"k0"}, "other_quad": {"k1"}},
        misalignments={"other_quad": {"dx", "dy"}, "quad": {"dy"}},
    )

    mad_meas = GradientDescentMadInterface(
        accelerator_meas,
        corrector_knobs=corrector_knobs,
        tune_knobs=tune_knobs,
    )

    resolved_knobs, unresolved_knobs = resolve_checkpoint_knobs(mad_meas, checkpoint_knobs)
    if not resolved_knobs:
        raise ValueError("No checkpoint knobs could be mapped to optimisation-space names.")
    mad_meas.update_knob_values(resolved_knobs)
    LOGGER.info("Applied %d resolved checkpoint knobs into measurement MAD", len(resolved_knobs))
    twiss_meas_madng = mad_meas.run_twiss(observe=1, deltap=deltap, coupling=True, chrom=True)
    print(twiss_meas_madng.head())
    print(twiss_meas_madng.columns)
    print(twiss_meas_madng.headers)
    twiss_meas_madx = convert_tfs_to_madx(twiss_meas_madng)
    LOGGER.info("Measurement Twiss generated and converted to MAD-X format")

    mad_meas.close()
    LOGGER.info("Closed measurement MAD interface for this frequency")

    # 2) Base model Twiss: clean state (no checkpoint, no tune/corrector knobs).
    LOGGER.info(
        "Building BASE model Twiss for beam=%d deltap=%+.8f sequence=%s (no knobs), matching tunes to qx=%.2f qy=%.2f",
        beam,
        deltap,
        sequence_file,
        TARGET_QX,
        TARGET_QY,
    )
    accelerator_model = LHC(
        beam=beam,
        sequence_file=sequence_file,
        kinetic_energy=pc,
    )

    mad_model = GradientDescentMadInterface(
        accelerator_model,
        corrector_knobs=None,
        tune_knobs=None,
    )

    if match_model_tunes:
        mad_model.match_tunes(TARGET_QX, TARGET_QY, deltap=deltap)
        LOGGER.info(
            "Matched model tunes to qx=%.2f qy=%.2f at deltap=%+.8f",
            TARGET_QX,
            TARGET_QY,
            deltap,
        )
    else:
        LOGGER.info(
            "Skipping model tune matching at deltap=%+.8f (--no-match-model-tunes set)",
            deltap,
        )

    if model_twiss_onmom:
        twiss_model_madng = mad_model.run_twiss(observe=1, coupling=True)
        LOGGER.info("Built model Twiss on-momentum (no deltap applied)")
    else:
        twiss_model_madng = mad_model.run_twiss(observe=1, deltap=deltap, coupling=True)
        LOGGER.info("Built model Twiss with deltap=%+.8f", deltap)
    twiss_model_madx = convert_tfs_to_madx(twiss_model_madng)
    LOGGER.info("Base model Twiss generated and converted to MAD-X format")

    mad_model.close()
    LOGGER.info("Closed base model MAD interface for this frequency")

    return (
        twiss_meas_madx,
        twiss_model_madx,
        len(resolved_knobs),
        len(unresolved_knobs),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Create and archive inj fake measurements")
    parser.add_argument("--beam", type=int, choices=[1, 2], default=1)
    parser.add_argument("--squeeze-step", type=str, default="inj")
    parser.add_argument("--sequence", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--energy",
        type=float,
        default=None,
        help="Optional beam energy override; auto-detected when omitted",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "fake_measurements_inj",
        help="Root directory containing one fake-measurement folder per frequency",
    )
    parser.add_argument(
        "--match-model-tunes",
        dest="match_model_tunes",
        action="store_true",
        default=True,
        help=(
            "Match base model tunes to target qx/qy before model Twiss generation "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-match-model-tunes",
        dest="match_model_tunes",
        action="store_false",
        help="Disable base model tune matching before model Twiss generation.",
    )
    parser.add_argument(
        "--drop-ks-knobs",
        action="store_true",
        help="Remove all ks* keys fsssssrom saved tune/corrector knob snapshots.",
    )
    parser.add_argument(
        "--model-twiss-onmom",
        dest="model_twiss_onmom",
        action="store_true",
        default=True,
        help="Build base model Twiss on-momentum (no deltap).",
    )
    parser.add_argument(
        "--model-twiss-with-deltap",
        dest="model_twiss_onmom",
        action="store_false",
        help="Build base model Twiss with each frequency deltap.",
    )
    args = parser.parse_args()
    LOGGER.info("Starting inj fake-measurement generation with args: %s", vars(args))

    sequence_file = args.sequence or autodetect_sequence_file(args.beam, args.squeeze_step)
    checkpoint_file = args.checkpoint or autodetect_quad_checkpoint(
        args.beam, args.squeeze_step, args.checkpoint_dir
    )
    pc = (
        float(args.energy)
        if args.energy is not None
        else autodetect_energy(args.beam, args.squeeze_step)
    )
    LOGGER.info("Resolved sequence file: %s", sequence_file)
    LOGGER.info("Resolved checkpoint file: %s", checkpoint_file)
    LOGGER.info("Using beam energy [GeV]: %.6f", pc)

    checkpoint_knobs = load_checkpoint_knobs(checkpoint_file)

    output_root = args.output_root
    if output_root.exists():
        LOGGER.info("Removing existing output root: %s", output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Created output root: %s", output_root)

    summary: list[dict[str, object]] = []

    for freq in FREQUENCIES:
        LOGGER.info("Processing frequency: %s", freq)
        if freq not in FREQ_TO_DELTAP:
            raise ValueError(f"Missing deltap mapping for frequency {freq}")
        deltap = FREQ_TO_DELTAP[freq]
        LOGGER.info("Using deltap for %s: %+.8f", freq, deltap)

        freq_dir = output_root / freq
        freq_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Frequency output dir: %s", freq_dir)
        tune_snapshot, corrector_snapshot, knob_time_iso = download_knobs_for_frequency(
            beam=args.beam,
            squeeze_step=args.squeeze_step,
            freq=freq,
            pc=pc,
            freq_dir=freq_dir,
            drop_ks_knobs=args.drop_ks_knobs,
        )

        twiss_madx, model_madx, n_resolved, n_unresolved = build_madx_twiss_for_frequency(
            beam=args.beam,
            pc=pc,
            sequence_file=sequence_file,
            checkpoint_knobs=checkpoint_knobs,
            tune_knobs=tune_snapshot,
            corrector_knobs=corrector_snapshot,
            deltap=deltap,
            match_model_tunes=args.match_model_tunes,
            model_twiss_onmom=args.model_twiss_onmom,
        )

        LOGGER.info("Writing Twiss TFS for %s", freq)
        tfs.write(freq_dir / "twiss_madx.tfs", twiss_madx, save_index="NAME")
        tfs.write(freq_dir / "model_madx.tfs", model_madx, save_index="NAME")
        LOGGER.info("Generating fake measurement for %s", freq)
        fake_measurement(twiss=twiss_madx, model=model_madx, outputdir=freq_dir)
        LOGGER.info("Completed fake measurement generation for %s", freq)

        summary.append(
            {
                "frequency": freq,
                "tune_knobs": str(tune_snapshot),
                "corrector_knobs": str(corrector_snapshot),
                "knob_measurement_time_utc": knob_time_iso,
                "deltap": deltap,
                "resolved_checkpoint_knobs": n_resolved,
                "unresolved_checkpoint_knobs": n_unresolved,
                "output_dir": str(freq_dir),
            }
        )
        print(f"Built fake measurement for {freq} (deltap={deltap:+.8f}) in {freq_dir}")

    metadata = {
        "beam": args.beam,
        "squeeze_step": args.squeeze_step,
        "sequence_file": str(sequence_file),
        "checkpoint_file": str(checkpoint_file),
        "pc_GeV": pc,
        "measurements": summary,
        "match_model_tunes": args.match_model_tunes,
        "model_twiss_onmom": args.model_twiss_onmom,
        "drop_ks_knobs": args.drop_ks_knobs,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Wrote metadata file: %s", output_root / "metadata.json")


    zip_name = (
        output_root.name + "_tunes_matched"
        if args.match_model_tunes
        else output_root.name + "_no_tune_match"
    )
    if args.drop_ks_knobs:
        zip_name += "_drop_ks"
    if args.model_twiss_onmom:
        zip_name += "_model_onmom"
    zip_base = output_root.parent / zip_name
    LOGGER.info("Creating zip archive from %s", output_root)
    zip_path = Path(
        shutil.make_archive(
            base_name=str(zip_base),
            format="zip",
            root_dir=output_root.parent,
            base_dir=output_root.name,
        )
    )
    LOGGER.info("Created archive: %s", zip_path)

    shutil.rmtree(output_root)
    LOGGER.info("Deleted unzipped folder tree: %s", output_root)
    print(f"Archive created: {zip_path}")
    print(f"Deleted unzipped folder: {output_root}")


if __name__ == "__main__":
    main()
