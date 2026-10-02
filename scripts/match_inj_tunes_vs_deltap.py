"""Quick-and-dirty inj tune matching over fixed frequency/deltap points.

Usage:
    python -m aba_optimiser.measurements.match_inj_tunes_vs_deltap \
        --beam 1 \
        --sequence /path/to/sequence.seq \
        --checkpoint /path/to/checkpoint_b1_inj_arc1_quads.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pymadng_utils.io.utils import read_knobs

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.mad import GenericMadInterface, GradientDescentMadInterface
from aba_optimiser.measurements.sequence import get_or_make_sequence
from aba_optimiser.measurements.squeeze.config import (
    MODEL_DIRS,
    PC,
    get_model_dir,
    get_results_dir,
)

if TYPE_CHECKING:
    import pandas as pd

TARGET_QX = 0.28
TARGET_QY = 0.31

DELTAPS = [
    0.0,
    -0.00035890,
    -0.00071779,
    -0.00107669,
    -0.00143559,
    -0.00179448,
    -0.00215338,
    -0.00251227,
    0.00035890,
    0.00071779,
    0.00107669,
    0.00143559,
    0.00179448,
    0.00215338,
    0.00251227,
]

FREQUENCIES = [
    "0Hz",
    "50Hz",
    "100Hz",
    "150Hz",
    "200Hz",
    "250Hz",
    "300Hz",
    "350Hz",
    "-50Hz",
    "-100Hz",
    "-150Hz",
    "-200Hz",
    "-250Hz",
    "-300Hz",
    "-350Hz",
]


def load_checkpoint_knobs(checkpoint_file: Path) -> dict[str, float]:
    payload = json.loads(checkpoint_file.read_text())
    knob_map = payload.get("best_knobs") or payload.get("current_knobs") or {}
    return {str(k): float(v) for k, v in knob_map.items()}


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
    if not candidates:
        raise FileNotFoundError(f"No quad checkpoints found in {base_dir} matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def autodetect_energy(beam: int, squeeze_step: str) -> float:
    metadata_file = (
        PROJECT_ROOT
        / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
        / "metadata.json"
    )
    if not metadata_file.exists():
        return float(PC)
    metadata = json.loads(metadata_file.read_text())
    return float(metadata.get("energy", PC))


def _autodetect_knob_file(
    results_dir: Path, squeeze_step: str, prefix: str
) -> Path | None:
    preferred = results_dir / f"{prefix}_{squeeze_step}_0Hz.txt"
    if preferred.exists():
        return preferred

    candidates = sorted(results_dir.glob(f"{prefix}_{squeeze_step}_*.txt"))
    if candidates:
        return candidates[0]
    return None


def autodetect_sequence_file(beam: int, squeeze_step: str) -> Path:
    """Auto-resolve a usable sequence file with local fallbacks.

    Preference order:
    1) Build/cache from model dir (if available in this environment)
    2) Expected local cached sequence by model-dir name
    3) Most recent local cached sequence in sequences_from_models/
    """
    try:
        model_dir = get_model_dir(beam, squeeze_step)
        return get_or_make_sequence(beam, model_dir)
    except (ValueError, FileNotFoundError):
        pass

    sequences_dir = PROJECT_ROOT / "sequences_from_models"
    expected_name = MODEL_DIRS.get(beam, {}).get(squeeze_step)
    if expected_name is not None:
        expected_seq = sequences_dir / f"{expected_name}.seq"
        if expected_seq.exists():
            return expected_seq

    candidates = sorted(sequences_dir.glob("*.seq"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        "Could not auto-detect a sequence file. Provide --sequence explicitly."
    )


def apply_tune_knobs_preserving_dq_definitions(
    mad: GradientDescentMadInterface,
    tune_knobs: Path | None,
) -> tuple[int, int]:
    """Apply tune-knob file while preserving dqx/dqy->kqtf/kqtd definitions.

    Skips only arc tune-family entries of the form kqt[fd].a[1-8][1-8]b[12],
    and applies all other knobs from the file.
    """
    if tune_knobs is None:
        return 0, 0

    knob_values = read_knobs(tune_knobs)
    skip_pattern = re.compile(r"^kqt[fd]\.a[1-8][1-8]b[12]$", flags=re.IGNORECASE)

    applied = 0
    skipped = 0
    for name, value in knob_values.items():
        if skip_pattern.fullmatch(str(name)) is not None:
            skipped += 1
            continue
        mad.mad.send(f"MADX['{name}'] = {float(value):.16e}")
        applied += 1

    mad.mad.send(f"{mad.py_name}:send('done')")
    mad._check_mad_response("done", f"Failed to set tune knobs from {tune_knobs}")
    return applied, skipped


def plot_pre_match_twiss_difference(
    baseline: GenericMadInterface,
    current: GradientDescentMadInterface,
    beam: int,
) -> None:
    """Plot the Twiss difference between a clean baseline and the current MAD state."""

    def _load_twiss(mad_iface: GenericMadInterface) -> tuple[pd.DataFrame, str, str, str, str]:
        twiss = mad_iface.run_twiss(observe=1).copy()
        if "name" in twiss.columns:
            twiss = twiss.set_index("name")
        if "s" not in twiss.columns:
            raise KeyError("Twiss output is missing the 's' column needed for plotting.")
        betx_col = "beta11" if "beta11" in twiss.columns else "betx"
        bety_col = "beta22" if "beta22" in twiss.columns else "bety"
        mux_col = "mu1" if "mu1" in twiss.columns else "mux"
        muy_col = "mu2" if "mu2" in twiss.columns else "muy"
        return twiss, betx_col, bety_col, mux_col, muy_col

    base_twiss, betx_col, bety_col, mux_col, muy_col = _load_twiss(baseline)
    current_twiss, _, _, _, _ = _load_twiss(current)

    common_index = base_twiss.index.intersection(current_twiss.index)
    if common_index.empty:
        raise ValueError("No common BPMs found between baseline and current Twiss outputs.")

    base_twiss = base_twiss.loc[common_index]
    current_twiss = current_twiss.loc[common_index]

    beta_x_rel = np.divide(
        current_twiss[betx_col] - base_twiss[betx_col],
        base_twiss[betx_col],
        out=np.zeros_like(current_twiss[betx_col], dtype=float),
        where=base_twiss[betx_col].to_numpy(dtype=float) != 0.0,
    )
    beta_y_rel = np.divide(
        current_twiss[bety_col] - base_twiss[bety_col],
        base_twiss[bety_col],
        out=np.zeros_like(current_twiss[bety_col], dtype=float),
        where=base_twiss[bety_col].to_numpy(dtype=float) != 0.0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(base_twiss["s"], beta_x_rel, label="Δbetx / betx")
    axes[0].plot(base_twiss["s"], beta_y_rel, label="Δbety / bety")
    axes[0].set_ylabel("Relative beta difference")
    axes[0].set_title(f"Twiss difference vs baseline (beam {beam})")
    axes[0].grid(visible=True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(base_twiss["s"], current_twiss[mux_col] - base_twiss[mux_col], label="Δmux")
    axes[1].plot(base_twiss["s"], current_twiss[muy_col] - base_twiss[muy_col], label="Δmuy")
    axes[1].set_ylabel("Delta phase")
    axes[1].set_xlabel("s [m]")
    axes[1].grid(visible=True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def resolve_checkpoint_knobs(
    mad: GradientDescentMadInterface, checkpoint_knobs: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    """Map checkpoint knobs onto the live optimisation-space knob set.

    Supports checkpoints that contain either optimisation-space knob names or
    absolute knob names by converting unknown entries via the MAD knob transform.
    """
    knob_name_set = mad.knob_name_set

    direct = {k: v for k, v in checkpoint_knobs.items() if k in knob_name_set}
    unresolved_input = {k: v for k, v in checkpoint_knobs.items() if k not in knob_name_set}

    converted = mad.absolute_to_optimisation_knobs(unresolved_input) if unresolved_input else {}

    resolved = dict(direct)
    resolved.update(converted)

    # Names that were neither direct optimisation knobs nor mappable absolute knobs.
    unresolved = [k for k in unresolved_input if k not in converted]
    return resolved, unresolved


def format_table(rows: list[tuple[str, float, float, float]]) -> str:
    headers = ("freq", "deltap", "qx_knob", "qy_knob", "qx/qy")
    table_rows = [
        (freq, f"{dpp:+.8f}", f"{qx:+.8e}", f"{qy:+.8e}", f"{TARGET_QX:.2f}/{TARGET_QY:.2f}")
        for freq, dpp, qx, qy in rows
    ]

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_line(values: tuple[str, str, str, str, str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    sep = "-+-".join("-" * w for w in widths)
    lines = [render_line(headers), sep]
    lines.extend(render_line(row) for row in table_rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Match inj tunes vs deltap from one quad checkpoint")
    parser.add_argument("--beam", type=int, choices=[1, 2], required=True)
    parser.add_argument("--squeeze-step", type=str, default="inj", help="Squeeze step")
    parser.add_argument("--sequence", type=Path, default=None, help="Optional sequence file override")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional quad checkpoint JSON override")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional checkpoint directory override")
    parser.add_argument("--energy", type=float, default=None, help="Optional beam energy override (GeV)")
    parser.add_argument(
        "--tune-knobs-file",
        type=Path,
        default=None,
        help="Optional tune knobs file override",
    )
    parser.add_argument(
        "--corrector-knobs-file",
        type=Path,
        default=None,
        help="Optional corrector knobs file override",
    )
    args = parser.parse_args()

    if len(DELTAPS) != len(FREQUENCIES):
        raise ValueError("DELTAPS and FREQUENCIES must have the same length")

    sequence_file = (
        args.sequence
        if args.sequence is not None
        else autodetect_sequence_file(args.beam, args.squeeze_step)
    )
    checkpoint_file = (
        args.checkpoint
        if args.checkpoint is not None
        else autodetect_quad_checkpoint(args.beam, args.squeeze_step, args.checkpoint_dir)
    )
    pc = args.energy if args.energy is not None else autodetect_energy(args.beam, args.squeeze_step)

    results_dir = get_results_dir(args.beam)
    tune_knobs = (
        args.tune_knobs
        if args.tune_knobs is not None
        else _autodetect_knob_file(results_dir, args.squeeze_step, "tune_knobs")
    )
    corrector_knobs = (
        args.corrector_knobs
        if args.corrector_knobs is not None
        else _autodetect_knob_file(results_dir, args.squeeze_step, "corrector_knobs")
    )

    checkpoint_knobs = load_checkpoint_knobs(checkpoint_file)

    accelerator = LHC(
        beam=args.beam,
        sequence_file=sequence_file,
        kinetic_energy=pc,
        errors={"quad": {"k1"}, "bend": {"k0"}, "other_quad": {"k1"}},
        misalignments={"other_quad": {"dx", "dy"}, "quad": {"dy"}},
    )
    baseline_mad = GenericMadInterface(
        accelerator,
        corrector_knobs=None,
        tune_knobs=None,
    )

    mad = GradientDescentMadInterface(
        accelerator,
        corrector_knobs=corrector_knobs,
    )
    applied_tune_knobs, skipped_tune_knobs = apply_tune_knobs_preserving_dq_definitions(
        mad, tune_knobs
    )
    resolved_knobs, unresolved_knobs = resolve_checkpoint_knobs(mad, checkpoint_knobs)
    if not resolved_knobs:
        raise ValueError(
            "No checkpoint knobs could be mapped to optimisation-space names for this sequence. "
            "Check that beam/sequence/checkpoint correspond to the same machine state."
        )
    mad.update_knob_values(resolved_knobs)

    print(
        "Resolved checkpoint knobs: "
        f"{len(resolved_knobs)} applied / {len(checkpoint_knobs)} total"
    )
    if unresolved_knobs:
        preview = ", ".join(unresolved_knobs[:8])
        if len(unresolved_knobs) > 8:
            preview += ", ..."
        print(f"Unresolved checkpoint knobs skipped ({len(unresolved_knobs)}): {preview}")

    print(f"Auto sequence: {sequence_file}")
    print(f"Auto quad checkpoint: {checkpoint_file}")
    print(f"Auto energy [GeV]: {pc}")
    print(f"Auto tune knobs file: {tune_knobs}")
    print(f"Auto corrector knobs file: {corrector_knobs}")
    print(
        "Manual tune knob apply: "
        f"{applied_tune_knobs} applied, {skipped_tune_knobs} skipped "
        "(kqtf/kqtd arc tune-family knobs)"
    )

    mad.match_tunes(TARGET_QX, TARGET_QY, deltap=0.0) # Match tunes before plotting
    plot_pre_match_twiss_difference(baseline_mad, mad, args.beam)

    qx_knob_name, qy_knob_name = accelerator.tune_variables
    rows: list[tuple[str, float, float, float]] = []

    for freq, dpp in zip(FREQUENCIES, DELTAPS, strict=True):
        matched = mad.match_tunes(TARGET_QX, TARGET_QY, deltap=dpp)
        rows.append((freq, dpp, float(matched[qx_knob_name]), float(matched[qy_knob_name])))

    print(f"Matched to tunes qx={TARGET_QX}, qy={TARGET_QY}")
    print(f"Using tune knobs: {qx_knob_name}, {qy_knob_name}\n")
    print(format_table(rows))


if __name__ == "__main__":
    main()
