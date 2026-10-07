"""PSB quadrupole-focused integration tests for controller logic."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import OptimiserConfig
from aba_optimiser.mad import merge_machine_states
from aba_optimiser.training.config.helpers import create_arc_measurement_config
from aba_optimiser.training.config.models import (
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.tracking_fitter import ArcByArcFitter
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


PSB_TARGET_QX = 0.17
PSB_TARGET_QY = 0.225
PSB_TRACK_BPM_PATTERN = r"br3\.bpm.*"
PSB_BPM_START_POINTS = ["BR3.BPM1L3", "BR3.BPM5L3", "BR3.BPM9L3"]
PSB_BPM_END_POINTS = ["BR3.BPM13L3", "BR3.BPM15L3", "BR3.BPM16L3"]
pytestmark = pytest.mark.serial


def _build_psb_arc_quad_controller(
    *,
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
    flattop_turns: int = 64,
) -> tuple[ArcByArcFitter, dict[str, float]]:
    """Build a PSB ring-3 arc-by-arc quadrupole fitter and its true strengths."""
    off_magnet_path = tmp_path / "track_off_magnet_psb_val.parquet"
    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_psb_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        perturb_quads=True,
        bpm_pattern=PSB_TRACK_BPM_PATTERN,
        apply_orbit_correction=False,
        target_qx=PSB_TARGET_QX,
        target_qy=PSB_TARGET_QY,
    )
    simulation_config = dataclasses.replace(
        _make_simulation_config_quad(),
        num_workers=4,
        num_batches=1,
        bpm_loss_outlier_sigma=20,
        worker_loss_outlier_sigma=20,
    )
    ctrl = ArcByArcFitter(
        PSB(
            ring=3,
            kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors={"quad": {"k1"}},
        ),
        OptimiserConfig(
            max_epochs=300,
            warmup_epochs=40,
            warmup_lr_start=1e-6,
            max_lr=3e-4,
            min_lr=3e-4,
            gradient_converged_value=1e-13,
            optimiser_type="adam",
        ),
        simulation_config,
        SequenceConfig("$start/$end"),
        create_arc_measurement_config(
            off_magnet_path, machine_state=merge_machine_states(corrector_file, tune_knobs)
        ),
        bpm_start_points=PSB_BPM_START_POINTS,
        bpm_end_points=PSB_BPM_END_POINTS,
        output_config=OutputConfig(
            mad_logfile=tmp_path / "controller_quad_opt_psb_val.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths,
        debug=False,
    )
    return ctrl, magnet_strengths


def test_controller_quad_psb_validation_loss_is_real_out_of_sample(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
) -> None:
    """Arc-by-arc PSB fit yields a genuine held-out validation loss (not None).

    The arc-by-arc plan enables validation, so DataManager reserves a disjoint
    set of turns. compute_validation_loss evaluates the validation workers, which
    track ONLY those held-out turns, and must return a real number. Because the
    held-out turns are noise-free samples of the same machine, the out-of-sample
    loss must strongly prefer the true quadrupole strengths over the perturbed
    initial guess.
    """
    ctrl, true_values = _build_psb_arc_quad_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
    )
    assert ctrl.tracking_plan.enable_validation
    # Turns were genuinely held out, disjoint from every training batch.
    assert ctrl.data_manager.validation_turn_batches
    training_turns = {t for batch in ctrl.data_manager.turn_batches for t in batch}
    validation_turns = {
        t for batch in ctrl.data_manager.validation_turn_batches for t in batch
    }
    assert training_turns.isdisjoint(validation_turns)

    ctrl.worker_manager.start_workers(
        ctrl.data_manager.tracks,
        ctrl.data_manager.turn_batches,
        ctrl.data_manager.validation_turn_batches,
        ctrl.data_manager.file_map,
        ctrl.config_manager.start_bpms,
        ctrl.config_manager.end_bpms,
        ctrl.simulation_config,
        ctrl.machine_deltaps,
        ctrl.initial_knobs,
        enable_validation=True,
    )
    try:
        val_initial = ctrl.worker_manager.compute_validation_loss(ctrl.initial_knobs)
        val_true = ctrl.worker_manager.compute_validation_loss(true_values)
    finally:
        ctrl.worker_manager.terminate_workers()

    # A genuine out-of-sample number, never None when validation is enabled.
    assert val_initial is not None and math.isfinite(val_initial)
    assert val_true is not None and math.isfinite(val_true)
    # Held-out loss prefers the true strengths: a real overfitting signal.
    assert val_true < val_initial, (
        f"Held-out validation loss should prefer true strengths "
        f"(initial={val_initial:.3e}, true={val_true:.3e})"
    )


def test_controller_quad_opt_psb_ring3(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
    controller_test_mode: str,
) -> None:
    """Run a PSB ring-3 quadrupole optimisation scenario."""
    flattop_turns = 256
    off_magnet_path = tmp_path / "track_off_magnet_psb.parquet"

    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_psb_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        perturb_quads=True,
        bpm_pattern=PSB_TRACK_BPM_PATTERN,
        apply_orbit_correction=False,
        target_qx=PSB_TARGET_QX,
        target_qy=PSB_TARGET_QY,
    )

    base_sim = _make_simulation_config_quad()
    simulation_config = dataclasses.replace(
        base_sim,
        num_workers=4,
        num_batches=4,
        bpm_loss_outlier_sigma=20,
        worker_loss_outlier_sigma=20,
    )
    optimiser_config = OptimiserConfig(
        max_epochs=300,
        warmup_epochs=40,
        warmup_lr_start=1e-6,
        max_lr=1e-3,
        min_lr=1e-4,
        gradient_converged_value=1e-13,
        optimiser_type="adam",
        # Gradients are ~1e-10, far below Adam's default eps (1e-8), which would
        # otherwise set the step size and stall the fit.
        adam_eps=1e-16,
    )

    sequence_config = SequenceConfig("$start/$end")
    measurement_config = create_arc_measurement_config(off_magnet_path, machine_state=merge_machine_states(corrector_file, tune_knobs))
    accelerator = PSB(
        ring=3,
        kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
        sequence_file=seq_psb,
        errors={"quad": {"k1"}},
    )

    ctrl = ArcByArcFitter(
        accelerator,
        optimiser_config,
        simulation_config,
        sequence_config,
        measurement_config,
        bpm_start_points=PSB_BPM_START_POINTS,
        bpm_end_points=PSB_BPM_END_POINTS,
        output_config=OutputConfig(
            mad_logfile=tmp_path / "controller_quad_opt_psb.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths,
        debug=False,
    )
    if controller_test_mode == "loss_regression":
        initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.initial_knobs)
        true_loss = evaluate_controller_worker_loss(ctrl, magnet_strengths)
        assert true_loss < initial_loss * 1e-3
        return
    result = ctrl.run()
    estimate, unc = result.knobs, result.uncertainties

    psb_abs_tol = 1e-4
    # These knobs sit outside the recorded BPM response for the tracked turn, so their
    # kicks never reach an observation point. They remain valid knobs, but the optimiser
    # cannot recover them from this measurement layout.
    unobservable = {"BR.QFO11.dk1l", "BR.QDE16.dk1l", "BR.QFO162.dk1l"}
    assert set(estimate) == set(magnet_strengths)
    assert set(unc) == set(magnet_strengths)
    for magnet, true_value in magnet_strengths.items():
        if magnet in unobservable:
            continue
        est_value = estimate[magnet]
        abs_diff = abs(est_value - true_value)
        rel_diff = abs_diff / abs(true_value) if true_value != 0 else abs(est_value)
        assert abs_diff <= psb_abs_tol or rel_diff < 5e-3, (
            f"Relative difference for {magnet} is too high: {rel_diff:.2%} "
            f"(abs diff {abs_diff:.3e})"
        )
