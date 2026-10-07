"""Kicker-focused integration tests for controller logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from aba_optimiser.config import OptimiserConfig
from aba_optimiser.fitting.config import OutputConfig, SequenceConfig
from aba_optimiser.machine.accelerators import PSB
from aba_optimiser.machine.mad import merge_machine_states
from aba_optimiser.tracking.config.helpers import create_arc_measurement_config
from aba_optimiser.tracking.config.models import KickerConfig
from aba_optimiser.tracking.fitter import FitterOptions, KickerFitter
from tests.training.controller_test_utils import (
    _generate_kicker_track,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.machine.mad.aba_mad_interface import AbaMadInterface

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.serial


def _build_kicker_controller(
    *,
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
    flattop_turns: int = 10,
) -> tuple[KickerFitter, dict[str, float]]:
    magnet_range = "$start/$end"
    off_magnet_path = tmp_path / "track_kicker_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs, kicker_name = _generate_kicker_track(
        loaded_psb_interface,
        flattop_turns,
        off_magnet_path,
        dpp_value=0.0,
        bpm_pattern=r"(?i)br3\.bpm.*",
    )
    optimiser_config = OptimiserConfig(
        max_epochs=500,
        warmup_epochs=100,
        warmup_lr_start=1e-4,
        max_lr=1e-3,
        min_lr=1e-4,
        gradient_converged_value=5e-20,
    )

    ctrl = KickerFitter(
        PSB(
            ring=3,
            kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors={"quad": {"k1"}},
        ),
        optimiser_config,
        _make_simulation_config_quad(),
        SequenceConfig(magnet_range=magnet_range),
        create_arc_measurement_config(
            off_magnet_path, machine_state=merge_machine_states(corrector_file, tune_knobs)
        ),
        KickerConfig(
            kicker_name=kicker_name,
            turns_after_kicker=flattop_turns,
        ),
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "mad_logfile_kicker.log",
                write_tensorboard_logs=False,
            ),
            true_strengths=magnet_strengths.copy(),
        ),
    )
    return ctrl, magnet_strengths.copy()


def test_controller_kicker_has_no_held_out_validation(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
) -> None:
    """The kicker plan disables validation, so no turns are held out.

    KickerTrackingPlan sets enable_validation=False: DataManager reserves no
    validation turns and TrackingSession.validation_loss returns None (the caller then
    falls back to training loss). This asserts the real behaviour of a Kicker-style
    run, complementing the arc-by-arc case where validation returns a real number.
    """
    ctrl, _magnet_strengths = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4,
    )
    assert not ctrl.tracking_plan.enable_validation
    assert ctrl.data_manager.validation_turn_batches == []

    ctrl.start_session(ctrl.machine.initial_knobs)
    try:
        assert ctrl.session.validation_loss(ctrl.machine.initial_knobs) is None
    finally:
        ctrl.session.terminate()


def test_controller_quad_opt_with_kicker(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
    controller_test_mode: str,
) -> None:
    loss_regression = controller_test_mode == "loss_regression"
    ctrl, magnet_strengths = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4 if loss_regression else 10,
    )
    logger.info(
        "Starting kicker controller with logfile at %s",
        tmp_path / "mad_logfile_kicker.log",
    )

    initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.machine.initial_knobs)

    if loss_regression:
        true_loss = evaluate_controller_worker_loss(ctrl, magnet_strengths)
        # Round-off only: a run at the true strengths lands around 1e-18.
        assert true_loss < 1e-16, (
            f"True-strength kicker loss should be numerically tiny, got {true_loss:.3e}"
        )
        assert true_loss < initial_loss * 1e-3, (
            "Kicker loss should strongly prefer the true quadrupole strengths "
            f"(initial={initial_loss:.3e}, true={true_loss:.3e})"
        )
        return

    initial_sum_true_diff = sum(
        abs(ctrl.machine.initial_knobs[magnet] - magnet_strengths[magnet])
        for magnet in magnet_strengths
    )
    estimate = ctrl.run().knobs
    final_sum_true_diff = sum(
        abs(estimate[magnet] - magnet_strengths[magnet])
        for magnet in magnet_strengths
    )
    final_ctrl, _ = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
    )
    final_loss = evaluate_controller_worker_loss(final_ctrl, estimate)

    assert final_loss < initial_loss * 0.05, (
        "Kicker optimisation should reduce the worker loss substantially "
        f"(initial={initial_loss:.3e}, final={final_loss:.3e})"
    )
    assert final_sum_true_diff < initial_sum_true_diff, (
        "Kicker optimisation should move the estimate closer to the true strengths "
        f"(initial diff={initial_sum_true_diff:.3e}, final diff={final_sum_true_diff:.3e})"
    )
