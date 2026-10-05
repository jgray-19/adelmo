"""
Shared utilities for controller integration tests.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xtrack as xt

pytest.importorskip("tmom_recon")
pytest.importorskip("xtrack_tools")
from pymadng_utils.io.utils import save_knobs
from xtrack_tools.coordinates import create_initial_conditions
from xtrack_tools.kicker import _insert_exciter_at, _knl_ksl
from xtrack_tools.monitors import (
    get_monitor_names_at_pattern,
    line_to_dataframes,
    process_tracking_data,
)
from xtrack_tools.tracking import run_tracking, run_tracking_without_ac_dipole

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.mad import merge_machine_states
from aba_optimiser.training.config.helpers import create_arc_measurement_config
from aba_optimiser.training.config.models import (
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.tracking_fitter import (
    ArcByArcFitter,
    TrackingFitter,
)
from aba_optimiser.training.workers.screening import OutlierScreener
from tests.training.helpers import TRACK_COLUMNS, generate_xsuite_env_with_errors

# Measurement noise assigned to the synthetic tracks' variance columns.
POSITION_STD_DEV = 1e-4
MOMENTUM_STD_DEV = 3e-6

KICKER_PATTERNS = (
    r"^mk",
    r"kick",
    r"^mcb",
    r"^mke",
    r"^mki",
    r"\.ksw",
    r"\.dhz",
    r"\.dvt",
)


def select_kicker_element(line) -> str | None:
    """Return the first element name that looks like a kicker/corrector."""
    for pattern in KICKER_PATTERNS:
        for name in line.element_names:
            if re.search(pattern, name, flags=re.IGNORECASE):
                return name
    return None


def strip_inline_flags(pattern: str) -> str:
    """Remove a leading inline case-insensitive flag for safe embedding."""
    if pattern.startswith("(?i)"):
        return pattern[4:]
    return pattern


def realign_kicker_turns(
    tracking_df: pd.DataFrame,
    *,
    kicker_name: str,
    logical_turns: int,
) -> pd.DataFrame:
    """Realign turns so each logical turn starts immediately after the kicker."""
    parts: list[pd.DataFrame] = []
    marker_name = kicker_name.upper()

    for _turn, turn_df in tracking_df.groupby("turn", sort=False):
        turn_names = turn_df["name"].str.upper().to_numpy()
        marker_rows = np.flatnonzero(turn_names == marker_name)
        if marker_rows.size == 0:
            parts.append(turn_df)
            continue

        marker_idx = int(marker_rows[0])
        if marker_idx > 0:
            before = turn_df.iloc[:marker_idx].copy()
            before["turn"] = before["turn"] - 1
            parts.append(before)
        parts.append(turn_df.iloc[marker_idx:].copy())

    realigned = pd.concat(parts, ignore_index=True)
    return realigned.loc[(realigned["turn"] >= 1) & (realigned["turn"] <= logical_turns)].copy()

if TYPE_CHECKING:
    from pathlib import Path

    import xtrack as xt

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


def _load_mad_twiss_for_tracking(
    interface_with_beam: AbaMadInterface,
    dpp_value: float,
) -> pd.DataFrame:
    """Load MAD-NG twiss data for off-momentum tracking initial conditions."""
    return interface_with_beam.run_twiss(observe=0, deltap=dpp_value).copy()


def _run_track_with_model(
    env: xt.Environment,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    action_list: list[float],
    angle_list: list[float],
    line_name: str,
    start_marker: str | None = None,
    bpm_pattern: str = "bpm.*[^k]",
    use_diagonal_kicks: bool = True,
    combine_particles_into_single_file: bool = False,
    return_dataframes: bool = False,
    twiss_data: pd.DataFrame | None = None,
) -> list[Path] | list[pd.DataFrame]:
    """Run tracking with the given model and initial conditions."""
    input_particles = len(action_list)
    if len(angle_list) != input_particles:
        raise ValueError("action_list and angle_list must have the same length")
    if twiss_data is None:
        raise ValueError("twiss_data must be provided; controller tests should use MAD-NG twiss")

    line: xt.Line = env[line_name]
    resolved_start_marker = start_marker
    if resolved_start_marker is None:
        monitor_names = get_monitor_names_at_pattern(line, bpm_pattern)
        if not monitor_names:
            raise ValueError(f"No BPMs matched pattern {bpm_pattern!r} in line {line_name!r}")
        # Start tracking at the first observed BPM so action-angle initialisation
        # uses a physical monitor rather than a synthetic drift element.
        resolved_start_marker = str(monitor_names[0])
    monitored_line = run_tracking_without_ac_dipole(
        line=line,
        tws=twiss_data,
        flattop_turns=flattop_turns,
        bpm_pattern=bpm_pattern,
        action_list=action_list,
        angle_list=angle_list,
        use_diagonal_kicks=use_diagonal_kicks,
        deltas=dpp_value,
        start_marker=resolved_start_marker,
    )

    true_dfs = line_to_dataframes(monitored_line)
    processed_dfs = []
    for true_df in true_dfs:
        df = true_df
        df["turn"] = df["turn"].astype(np.int32)
        df["bunch_number"] = 0
        df["var_x"] = df["var_y"] = POSITION_STD_DEV**2
        df["var_px"] = df["var_py"] = MOMENTUM_STD_DEV**2
        df = df.loc[:, TRACK_COLUMNS].copy()
        df["name"] = df["name"].astype(str)
        processed_dfs.append(df)

    if return_dataframes:
        return processed_dfs

    output_files = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    num_output_particles = len(processed_dfs)
    if combine_particles_into_single_file and num_output_particles > 1:
        offset_dfs = []
        for idx, df in enumerate(processed_dfs):
            particle_df = df.copy()
            particle_df["turn"] = particle_df["turn"] + idx * flattop_turns
            particle_df["bunch_number"] = idx
            offset_dfs.append(particle_df)
        combined_df = pd.concat(offset_dfs, ignore_index=True)
        combined_df.to_parquet(destination, index=False)
        output_files.append(destination)
        return output_files

    for idx, df in enumerate(processed_dfs):
        if num_output_particles == 1:
            output_path = destination
        else:
            output_path = (
                destination.parent / f"{destination.stem}_particle_{idx}{destination.suffix}"
            )
        df.to_parquet(output_path, index=False)
        output_files.append(output_path)
    return output_files


def _generate_nonoise_track(
    interface_with_beam: AbaMadInterface,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    start_marker: str | None = None,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
    num_particles: int = 1,
    phases: list[float] | None = None,
    bpm_pattern: str = "bpm.*[^k]",
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    use_diagonal_kicks: bool = True,
) -> tuple[Path | None, dict[str, float], Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    corrector_file: Path | None = None
    tune_knobs: Path | None = None
    if apply_orbit_correction:
        corrector_file = destination.parent / f"corrector_{destination.stem}.tfs"
    tune_knobs = destination.parent / f"tune_knobs_{destination.stem}.txt"

    env, magnet_strengths, matched_tunes, corrector_table = generate_xsuite_env_with_errors(
        interface_with_beam,
        dpp_value=dpp_value,
        corrector_file=corrector_file,
        perturb_quads=perturb_quads,
        perturb_bends=perturb_bends,
        apply_orbit_correction=apply_orbit_correction,
        target_qx=target_qx,
        target_qy=target_qy,
    )
    del corrector_table
    save_knobs(matched_tunes, tune_knobs)

    action = 4e-7 if interface_with_beam.accelerator.seq_name.lower() == "sps" else 4e-8
    angle = 0.0
    if num_particles == 1:
        action_list = [action]
        angle_list = [angle]
    else:
        if phases is not None:
            if len(phases) != num_particles:
                raise ValueError("Length of phases must equal num_particles")
            angle_list = phases
        else:
            angle_list = (np.linspace(0.0, 2 * np.pi, num=num_particles, endpoint=False)).tolist()
        action_list = [action] * num_particles

    # MAD-NG's off-momentum twiss is more robust here than xsuite's periodic
    # twiss solve, so pass it directly into xtrack_tools.
    mad_twiss = _load_mad_twiss_for_tracking(interface_with_beam, dpp_value)

    # run_madng_tracking(
    #     interface=interface_with_beam,
    #     flattop_turns=flattop_turns,
    #     action_list=action_list,
    #     angle_list=angle_list,
    #     use_diagonal_kicks=use_diagonal_kicks,
    #     start_marker=start_marker,
    #     destination=destination,
    # )

    _run_track_with_model(
        env=env,
        flattop_turns=flattop_turns,
        destination=destination,
        dpp_value=dpp_value,
        action_list=action_list,
        angle_list=angle_list,
        start_marker=start_marker,
        line_name=interface_with_beam.accelerator.seq_name.lower(),
        bpm_pattern=bpm_pattern,
        use_diagonal_kicks=use_diagonal_kicks,
        twiss_data=mad_twiss,
    )
    return corrector_file, magnet_strengths, tune_knobs


def _generate_kicker_track(
    interface_with_beam: AbaMadInterface,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    kick_strength: float = 2e-5,
    kick_plane: str = "diagonal",
    kick_turn: int = 0,
    bpm_pattern: str = r"(?i)bpm.*",
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    magnet_strengths: dict[str, float] | None = None,
    bpms: list[str] | None = None,
    correctors: list[str] | None = None,
) -> tuple[Path | None, dict[str, float], Path | None, str]:
    """Generate a parquet file containing kicker tracking data with x/px/y/py.

    ``magnet_strengths`` fixes the quadrupole truth instead of drawing a fresh
    random perturbation: pass ``None`` once to perturb and capture the truth,
    then pass that same dict back in for every other ``dpp_value`` against the
    same ``interface_with_beam`` so all measurements describe one machine
    viewed at different momenta, rather than compounding a new perturbation
    on top of the previous one on each call.
    """
    corrector_file: Path | None = None
    tune_knobs: Path | None = None
    if apply_orbit_correction:
        corrector_file = destination.parent / f"corrector_{destination.stem}.tfs"
    tune_knobs = destination.parent / f"tune_knobs_{destination.stem}.txt"

    env, generated_strengths, matched_tunes, _ = generate_xsuite_env_with_errors(
        interface_with_beam,
        dpp_value=dpp_value,
        corrector_file=corrector_file,
        perturb_quads=magnet_strengths is None,
        apply_orbit_correction=apply_orbit_correction,
        target_qx=target_qx,
        target_qy=target_qy,
        bpms=bpms,
        correctors=correctors,
    )
    magnet_strengths = magnet_strengths or generated_strengths
    save_knobs(matched_tunes, tune_knobs)

    seq_name = interface_with_beam.accelerator.seq_name.lower()
    line: xt.Line = env[seq_name]
    kicker_name = select_kicker_element(line)
    if kicker_name is None:
        pytest.skip("No kicker-like element found in sequence for test")

    mad_twiss = _load_mad_twiss_for_tracking(interface_with_beam, dpp_value)
    tws = line.twiss(method="4d")
    frev = float(1.0 / tws.t_rev0)
    s_kicker = float(line.get_s_position(kicker_name))
    knl, ksl = _knl_ksl(kick_strength, kick_plane)

    kicked_line = line.copy()
    _insert_exciter_at(
        kicked_line,
        name="single_turn_kicker",
        s=s_kicker,
        knl=knl,
        ksl=ksl,
        frev=frev,
        start_turn=kick_turn,
    )

    bpm_pattern_clean = strip_inline_flags(bpm_pattern)
    monitor_pattern = rf"(?i:{bpm_pattern_clean})|{re.escape(kicker_name)}"
    monitor_names = get_monitor_names_at_pattern(kicked_line, monitor_pattern)
    start_elem = kicked_line.element_names[0].upper()
    co_row = mad_twiss.loc[start_elem] if start_elem in mad_twiss.index else mad_twiss.iloc[0]
    particles: xt.Particles = kicked_line.build_particles(
        x=float(co_row["x"]),
        px=float(co_row["px"]),
        y=float(co_row["y"]),
        py=float(co_row["py"]),
        delta=dpp_value,
    )

    monitored_line = run_tracking(
        line=kicked_line,
        particles=particles,
        # Track one extra physical turn so we can recover the BPMs that lie
        # just before the kicker for the final logical post-kicker turn.
        nturns=flattop_turns + 1,
        monitor_names=monitor_names,
    )
    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=0,
        flattop_turns=flattop_turns + 1,
        add_variance_columns=True,
    )
    tracking_df["bunch_number"] = 0
    tracking_df = tracking_df.loc[:, TRACK_COLUMNS].copy()
    tracking_df["name"] = tracking_df["name"].astype(str)
    # realign re-groups each turn's rows to begin just after the kicker, giving the
    # cycled (kicker-relative) BPM order that MAD observes when it tracks the full
    # sequence cycled to the kicker. The worker compares observables positionally,
    # so the saved per-turn order must match that cycled order.
    tracking_df = realign_kicker_turns(
        tracking_df,
        kicker_name=kicker_name,
        logical_turns=flattop_turns,
    )
    if kicker_name.upper() not in set(tracking_df["name"]):
        raise ValueError(f"Kicker marker {kicker_name} missing from tracking output")
    tracking_df.to_parquet(destination, index=False)

    return corrector_file, magnet_strengths, tune_knobs, kicker_name.upper()


DPP_VALUE = 1.25e-4
FLATTOP_TURNS = 256
def _make_simulation_config_energy(optimise_momenta: bool = True) -> SimulationConfig:
    return SimulationConfig(
        num_workers=3,
        num_batches=2,
        optimise_momenta=optimise_momenta,
    )


def _build_energy_optimisation_case(
    *,
    tmp_path: Path,
    loaded_interface: AbaMadInterface,
    simulation_config: SimulationConfig,
    optimiser_config: OptimiserConfig,
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    magnet_range: str,
    mad_log_name: str,
    bpm_pattern: str = "bpm.*[^k]",
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    dpp_value: float = DPP_VALUE,
    flattop_turns: int = FLATTOP_TURNS,
) -> tuple[TrackingFitter, dict[str, float]]:
    """Build one energy optimisation controller and its true internal knob values."""
    off_dpp_path = tmp_path / "track_off_dpp.parquet"
    corrector_file, _, tune_knobs = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        off_dpp_path,
        dpp_value,
        bpm_pattern=bpm_pattern,
        apply_orbit_correction=apply_orbit_correction,
        target_qx=target_qx,
        target_qy=target_qy,
    )

    sequence_config = SequenceConfig(magnet_range=magnet_range)
    measurement_config = create_arc_measurement_config(off_dpp_path, machine_state=merge_machine_states(corrector_file, tune_knobs))

    accel = loaded_interface.accelerator.copy_with(optimise_energy=True)
    output_config = OutputConfig(
        mad_logfile=tmp_path / mad_log_name,
        write_tensorboard_logs=False,
    )
    ctrl = ArcByArcFitter(
        accel,
        optimiser_config,
        simulation_config,
        sequence_config,
        measurement_config,
        bpm_start_points,
        bpm_end_points,
        output_config=output_config,
    )
    true_knobs = {
        "pt": loaded_interface.accelerator.dp2pt(dpp_value)
    }
    return ctrl, true_knobs


def _run_energy_optimisation_case(
    *,
    tmp_path: Path,
    loaded_interface: AbaMadInterface,
    simulation_config: SimulationConfig,
    optimiser_config: OptimiserConfig,
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    magnet_range: str,
    mad_log_name: str,
    bpm_pattern: str = "bpm.*[^k]",
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    dpp_value: float = DPP_VALUE,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run one energy optimisation scenario and return estimate/uncertainty dictionaries."""
    ctrl, _true_knobs = _build_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
        magnet_range=magnet_range,
        mad_log_name=mad_log_name,
        bpm_pattern=bpm_pattern,
        apply_orbit_correction=apply_orbit_correction,
        target_qx=target_qx,
        target_qy=target_qy,
        dpp_value=dpp_value,
    )
    return ctrl.run()


def _make_optimiser_config_quad() -> OptimiserConfig:
    return OptimiserConfig(
        max_epochs=300,
        warmup_epochs=200,
        warmup_lr_start=1e-4,
        max_lr=1e-4,
        min_lr=1e-4,
        gradient_converged_value=5e-14,
    )


def _make_simulation_config_quad() -> SimulationConfig:
    return SimulationConfig(
        num_workers=8,
        num_batches=2,
        bpm_loss_outlier_sigma=4,
    )


def evaluate_controller_worker_loss(
    ctrl: TrackingFitter,
    knobs: dict[str, float],
    *,
    enable_validation: bool | None = None,
) -> float:
    """Return the total worker diagnostic loss summed across all workers at one knob setting."""
    return evaluate_controller_worker_losses(
        ctrl,
        [knobs],
        enable_validation=enable_validation,
    )[0]


def evaluate_controller_worker_losses(
    ctrl: TrackingFitter,
    knobs_list: list[dict[str, float]],
    *,
    enable_validation: bool | None = None,
) -> list[float]:
    """Return worker diagnostic losses for several knob settings using one worker startup."""
    if enable_validation is None:
        enable_validation = ctrl.tracking_plan.enable_validation
    if not hasattr(ctrl, "data_manager"):
        ctrl._init_data_manager()
    initial_worker_values = {
        **ctrl.config_manager.initial_model_values,
        **ctrl.initial_knobs,
    }
    ctrl.worker_manager.start_workers(
        ctrl.data_manager.tracks,
        ctrl.data_manager.turn_batches,
        ctrl.data_manager.validation_turn_batches,
        ctrl.data_manager.file_map,
        ctrl.config_manager.start_bpms,
        ctrl.config_manager.end_bpms,
        ctrl.simulation_config,
        ctrl.machine_deltaps,
        initial_worker_values,
        enable_validation=enable_validation,
    )
    try:
        losses = []
        for knobs in knobs_list:
            screener = OutlierScreener(ctrl.worker_manager.payload_builder)
            diags = screener.request_worker_diagnostics(ctrl.worker_manager._channels(), knobs)
            losses.append(sum(float(d["total_loss"]) for d in diags))  # type: ignore[arg-type]
    finally:
        ctrl.worker_manager.terminate_workers()
    return losses


def run_madng_tracking(
    interface: AbaMadInterface,
    flattop_turns: int,
    action_list: list[float],
    angle_list: list[float],
    use_diagonal_kicks: bool,
    start_marker: str | None,
    destination: Path,
):
    if isinstance(start_marker, str):
        interface.cycle_sequence(start_marker)
        interface.observe_elements([start_marker])

    tws = interface.run_twiss(observe=0)
    tws = tws[tws["s"] == 0]
    if isinstance(start_marker, str):
        interface.unobserve_elements([start_marker])
    else:
        start_marker = tws.index[0]
    if len(action_list) > 1:
        raise ValueError("Currently only supports single particle tracking for MAD-NG")
    coords = create_initial_conditions(
        action=action_list[0],
        angle=angle_list[0],
        twiss_data=tws,
        kick_plane="xy",
        starting_bpm=start_marker,
    )
    coords = {k: float(v) for k, v in coords.items()}
    interface.observe_bpms()
    interface.mad["trk", "flw"] = interface.mad.track(
        sequence="loaded_sequence", X0=coords, nturn=flattop_turns
    )
    df = pd.DataFrame(interface.mad.trk.to_df())
    # add variance columns
    df["var_x"] = (1e-4) ** 2
    df["var_y"] = (1e-4) ** 2
    df["var_px"] = (1e-6) ** 2
    df["var_py"] = (1e-6) ** 2
    df["bunch_number"] = 0
    df.to_parquet(destination, index=False)
