"""
Energy-focused integration tests for controller logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.config import OptimiserConfig
from tests.training.controller_test_utils import (
    DPP_VALUE,
    _build_energy_optimisation_case,
    _make_simulation_config_energy,
    _run_energy_optimisation_case,
    evaluate_controller_worker_losses,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


pytestmark = pytest.mark.serial


@pytest.mark.parametrize("optimise_momenta", [False, True], ids=["position_only", "with_momenta"])
def test_controller_energy_opt(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    optimise_momenta: bool,
    controller_test_mode: str,
) -> None:
    simulation_config = _make_simulation_config_energy(optimise_momenta)
    optimiser_config = OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=1,
        warmup_lr_start=1e-8,
        max_lr=2e-6,
        min_lr=2e-7,
        gradient_converged_value=5e-10,
    )

    if controller_test_mode == "loss_regression":
        ctrl, true_knobs = _build_energy_optimisation_case(
            tmp_path=tmp_path,
            loaded_interface=loaded_interface,
            simulation_config=simulation_config,
            optimiser_config=optimiser_config,
            bpm_start_points=["BPM.9R2.B1", "BPM.10R2.B1", "BPM.11R2.B1"],
            bpm_end_points=["BPM.9L3.B1", "BPM.10L3.B1", "BPM.11L3.B1"],
            magnet_range="BPM.9R2.B1/BPM.9L3.B1",
            mad_log_name="controller_energy_opt.log",
        )
        initial_loss, true_loss = evaluate_controller_worker_losses(
            ctrl, [ctrl.machine.initial_knobs, true_knobs]
        )
        assert true_loss < initial_loss * 1e-2
        return

    result = _run_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=["BPM.9R2.B1", "BPM.10R2.B1", "BPM.11R2.B1"],
        bpm_end_points=["BPM.9L3.B1", "BPM.10L3.B1", "BPM.11L3.B1"],
        magnet_range="BPM.9R2.B1/BPM.9L3.B1",
        mad_log_name="controller_energy_opt.log",
    )
    estimate, unc = result.knobs, result.uncertainties

    assert np.allclose(
        estimate.pop("pt"), loaded_interface.dp2pt(DPP_VALUE), rtol=2e-3, atol=1e-10
    )
    uncertainty = unc.pop("pt")
    assert 0 < uncertainty < 3e-6
    assert not estimate
    assert not unc

