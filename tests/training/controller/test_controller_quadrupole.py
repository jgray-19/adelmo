"""
Quadrupole-focused integration tests for controller logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.accelerators import LHC
from aba_optimiser.training.config.helpers import create_arc_measurement_config
from aba_optimiser.training.config.models import (
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.tracking_fitter import ArcByArcFitter
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_optimiser_config_quad,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.serial


def _build_lhc_quad_controller(
    *,
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    start_marker: str,
) -> tuple[ArcByArcFitter, dict[str, float]]:
    magnet_range = "BPM.13R1.B1/BPM.13L2.B1"
    bpm_start_points = [f"BPM.{i}R1.B1" for i in range(13, 14)]
    bpm_end_points = [f"BPM.{i}L2.B1" for i in range(13, 14)]

    flattop_turns = 100
    off_magnet_path = tmp_path / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        start_marker=start_marker,
        perturb_quads=True,
    )

    ctrl = ArcByArcFitter(
        LHC(
            beam=1,
            kinetic_energy=6800,
            sequence_file=seq_b1,
            errors={"quad": {"k1"}},
        ),
        _make_optimiser_config_quad(),
        _make_simulation_config_quad(),
        SequenceConfig(magnet_range=magnet_range),
        create_arc_measurement_config(
            off_magnet_path, corrector_knobs=corrector_file, tune_knobs=tune_knobs
        ),
        bpm_start_points,
        bpm_end_points,
        output_config=OutputConfig(
            mad_logfile=tmp_path / "mad_logfile.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths.copy(),
    )
    return ctrl, magnet_strengths.copy()


def _assert_estimate_matches_true(
    estimate: dict[str, float],
    true_values: dict[str, float],
    *,
    max_rel_diff: float,
    abs_tol: float = 2e-7,
) -> None:
    worst_magnet = ""
    worst_rel_diff = -np.inf
    for magnet, value in estimate.items():
        abs_diff = abs(value - true_values[magnet])
        rel_diff = (
            abs_diff / abs(true_values[magnet])
            if true_values[magnet] != 0
            else abs(value)
        )
        if rel_diff > worst_rel_diff:
            worst_magnet = magnet
            worst_rel_diff = rel_diff
        assert abs_diff <= abs_tol or rel_diff < max_rel_diff, (
            f"Relative difference for {magnet} is too high: {rel_diff:.2%} "
            f"(abs diff {abs_diff:.3e}; worst so far: {worst_magnet} at {worst_rel_diff:.2%})"
        )


@pytest.mark.parametrize("start_marker", ["MSIA.EXIT.B1", "E.CELL.12.B1"])
def test_controller_quad_opt_simple(
    tmp_path: Path,
    seq_b1: Path,
    start_marker: str,
    loaded_interface: AbaMadInterface,
    controller_test_mode: str,
) -> None:
    ctrl, true_values = _build_lhc_quad_controller(
        tmp_path=tmp_path,
        seq_b1=seq_b1,
        loaded_interface=loaded_interface,
        start_marker=start_marker,
    )
    logger.info("Starting controller with logfile at %s", tmp_path / "mad_logfile.log")
    if controller_test_mode == "loss_regression":
        initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.initial_knobs)
        true_loss = evaluate_controller_worker_loss(ctrl, true_values)
        assert true_loss < initial_loss * 1e-6
        return
    estimate, _unc = ctrl.run()
    _assert_estimate_matches_true(estimate, true_values, max_rel_diff=1e-5)


def test_controller_quad_opt_simple_without_early_stopping_reaches_truth(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    controller_test_mode: str,
) -> None:
    ctrl, true_values = _build_lhc_quad_controller(
        tmp_path=tmp_path,
        seq_b1=seq_b1,
        loaded_interface=loaded_interface,
        start_marker="MSIA.EXIT.B1",
    )
    if controller_test_mode == "loss_regression":
        initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.initial_knobs)
        true_loss = evaluate_controller_worker_loss(ctrl, true_values)
        assert true_loss < initial_loss * 1e-6
        return

    ctrl.optimisation_loop._should_stop_for_loss_change = (  # type: ignore[method-assign]
        lambda epoch, epoch_loss, prev_loss: False
    )
    ctrl.optimisation_loop.gradient_converged_value = -1.0

    estimate, _unc = ctrl.run()
    _assert_estimate_matches_true(estimate, true_values, max_rel_diff=1e-5)


