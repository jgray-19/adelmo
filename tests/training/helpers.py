"""
Shared helper functions for controller integration tests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tfs
from xtrack_tools.env import initialise_env

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad import (
    AbaMadInterface,
    GenericMadInterface,
    GradientDescentMadInterface,
    merge_machine_states,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import xtrack as xt

TRACK_COLUMNS = (
    "turn",
    "bunch_number",
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

LOGGER = logging.getLogger(__name__)


def convert_rbends_to_true_rbends(mad: AbaMadInterface) -> None:
    """Convert all rbends in the sequence to true rbends for correct tracking."""
    mad.mad.send("""
for _, elm in loaded_sequence:iter() do
    if elm.kind == 'rbend' then
        elm.true_rbend = true
    end
end
    """)


def generate_model_with_errors(
    loaded_interface: AbaMadInterface,
    dpp_value: float,
    corrector_file: Path | None,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    rel_error: float | None = None,
    bpms: list[str] | None = None,
    correctors: list[str] | None = None,
) -> tuple[dict[str, float], dict, tfs.TfsDataFrame]:
    """
    Generate a MAD model with errors and return the xsuite environment.

    Args:
        sequence_file: Path to the sequence file
        dpp_value: Momentum deviation value
        corrector_file: Path to save corrector strengths
        perturb_quads: Whether to perturb quadrupoles
        perturb_bends: Whether to perturb bends

    Returns:
        Tuple of (magnet_strengths, matched_tunes, corrector_table)
    """
    accelerator_type = loaded_interface.accelerator.seq_name.lower()

    # Create MAD interface and load sequence
    interface = loaded_interface
    interface.mad["zero_twiss", "_"] = interface.mad.twiss(sequence="loaded_sequence")

    # Perform orbit correction for off-momentum beam
    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    magnet_strengths = {}
    magnet_type = ("q" if perturb_quads else "") + ("d" if perturb_bends else "")
    if magnet_type:
        LOGGER.info(f"Applying magnetic perturbations to {magnet_type}")
        magnet_strengths, _ = interface.apply_magnet_perturbations(
            rel_error=rel_error,  # None -> use default from ERROR_TABLE
            seed=42,
            magnet_type=magnet_type,
        )
        if accelerator_type == "lhcb1" and "q" in magnet_type:
            interface.set_madx_variables(**{"dQx.b1_op": -5.03401e-02, "dQy.b1_op": 9.70709e-02})

    # Convert all rbends into true rbends to ensure correct tracking
    # convert_rbends_to_true_rbends(loaded_interface)
    matched_tunes = {}
    if apply_orbit_correction:
        if corrector_file is None:
            raise ValueError("corrector_file must be provided when apply_orbit_correction=True")
        n_iterations = 15 if accelerator_type == "sps" else 1
        for i in range(n_iterations):  # Allow multiple iterations of correction if needed
            matched_tunes = interface.perform_orbit_correction(
                machine_deltap=dpp_value,
                target_qx=target_qx,
                target_qy=target_qy,
                corrector_file=corrector_file,
                bpms=bpms,
                correctors=correctors,
            )
            # interface.set_corrector_strengths(corrector_file)

        # Read corrector table
        corrector_table = tfs.read(corrector_file)
        possible_monitors = [f"{pre}monitor" for pre in ("h", "v", "")]
        corrector_table = corrector_table.loc[
            ~corrector_table.loc[:, "kind"].isin(possible_monitors)
        ]
    else:
        matched_tunes = interface.match_tunes(
            target_qx=target_qx,
            target_qy=target_qy,
            deltap=dpp_value,
        )
        corrector_table = tfs.TfsDataFrame(
            columns=["kind", "hkick", "hkick_old", "vkick", "vkick_old"]
        )
    magnet_strengths = interface.get_magnet_strengths(list(magnet_strengths.keys()))

    return magnet_strengths, matched_tunes, corrector_table


def generate_xsuite_env_with_errors(
    loaded_interface: AbaMadInterface,
    dpp_value: float,
    corrector_file: Path | None,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    rel_error: float | None = None,
    bpms: list[str] | None = None,
    correctors: list[str] | None = None,
) -> tuple[xt.Environment, dict[str, float], dict, tfs.TfsDataFrame]:
    """
    Generate a MAD model with errors and return the xsuite environment.

    Args:
        sequence_file: Path to the sequence file
        dpp_value: Momentum deviation value
        corrector_file: Path to save corrector strengths
        perturb_quads: Whether to perturb quadrupoles
        perturb_bends: Whether to perturb bends

    Returns:
        Tuple of (env, magnet_strengths, matched_tunes, corrector_table)
    """
    magnet_strengths, matched_tunes, corrector_table = generate_model_with_errors(
        loaded_interface,
        dpp_value,
        corrector_file,
        perturb_quads,
        perturb_bends,
        apply_orbit_correction,
        target_qx,
        target_qy,
        rel_error,
        bpms,
        correctors,
    )

    # Create xsuite environment with orbit correction applied
    accel = loaded_interface.accelerator
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        sequence_file=accel.sequence_file,
        seq_name=accel.seq_name,
        kinetic_energy=accel.kinetic_energy,
        strict_set=False,
    )
    return env, magnet_strengths, matched_tunes, corrector_table


def get_twiss_without_errors(
    sequence_file: Path,
    just_bpms: bool,
    beam: int = 1,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs: Path | None = None,
    corrector_file: Path | None = None,
) -> pd.DataFrame:
    """Get twiss data from a clean model without errors."""
    accelerator = LHC(
        beam=beam,
        bpm_pattern="BPM",
        sequence_file=sequence_file,
    )
    mad = GenericMadInterface(
        accelerator,
        machine_state=merge_machine_states(corrector_file, tune_knobs),
    )
    convert_rbends_to_true_rbends(mad)
    if estimated_magnets is not None:
        gradient = GradientDescentMadInterface(
            accelerator,
            machine_state=merge_machine_states(corrector_file, tune_knobs),
        )
        gradient.update_knob_values(estimated_magnets)
        mad = gradient
    return mad.run_twiss(observe=int(just_bpms))
