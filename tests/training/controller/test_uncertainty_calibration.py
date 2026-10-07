"""Reported 1-sigma knob uncertainty must match the real scatter of the estimator.

For one knob near the truth the loss is a parabola in that knob, so three loss
evaluations give the least-squares estimate for one noise draw exactly, without
depending on how the workers normalise the loss. Repeating over noise seeds gives
the true estimator spread; the Hessian-based sigma must agree with it.

Both tests keep the forward and backward workers of the arc-by-arc plan and differ
only in the measured start coordinates:

- exact starts: the start BPMs keep their true x/y and are not observations; the
  remaining noise is still shared between workers observing the same readings;
- noisy starts: every reading is noisy, including the start coordinates, whose noise
  moves every downstream residual of the workers starting there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.fitting.config import OutputConfig, SequenceConfig
from aba_optimiser.fitting.uncertainty import sandwich_uncertainties
from aba_optimiser.machine.mad import merge_machine_states
from aba_optimiser.tracking.config.helpers import create_arc_measurement_config
from aba_optimiser.tracking.config.models import MeasurementConfig
from aba_optimiser.tracking.fitter import ArcByArcFitter, FitterOptions
from tests.training.controller_test_utils import (
    DPP_VALUE,
    FLATTOP_TURNS,
    POSITION_STD_DEV,
    _generate_nonoise_track,
    evaluate_controller_worker_losses,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.machine.mad.aba_mad_interface import AbaMadInterface

pytestmark = [pytest.mark.slow, pytest.mark.serial]

N_SEEDS = 20
STEP = 1e-5
BPM_START_POINTS = ["BPM.9R2.B1", "BPM.10R2.B1", "BPM.11R2.B1"]
BPM_END_POINTS = ["BPM.9L3.B1", "BPM.10L3.B1", "BPM.11L3.B1"]
# Forward workers start at the start points, backward workers at the end points.
START_BPMS = [*BPM_START_POINTS, *BPM_END_POINTS]


def _make_fitter(
    interface: AbaMadInterface, track: Path, corrector: Path | None, tune_knobs: Path | None, log: Path
) -> ArcByArcFitter:
    return ArcByArcFitter(
        interface.accelerator.copy_with(optimise_energy=True),
        OptimiserConfig(
            max_epochs=1,
            warmup_epochs=0,
            warmup_lr_start=1e-8,
            max_lr=2e-6,
            min_lr=2e-7,
            gradient_converged_value=5e-10,
        ),
        SimulationConfig(
            num_workers=3, num_batches=2, optimise_momenta=False, validation_fraction=0.0
        ),
        SequenceConfig(magnet_range="BPM.9R2.B1/BPM.9L3.B1"),
        create_arc_measurement_config(track, machine_state=merge_machine_states(corrector, tune_knobs)),
        BPM_START_POINTS,
        BPM_END_POINTS,
        options=FitterOptions(
            output_config=OutputConfig(mad_logfile=log, write_tensorboard_logs=False),
        ),
    )


def _reported_sigma(ctrl: ArcByArcFitter, true_knobs: dict[str, float]) -> float:
    ctrl.start_session({**ctrl.machine.initial_model_values, **true_knobs}, enable_validation=False)
    knob_names = list(ctrl.machine.knob_names)
    ctrl.session.set_training_knobs(true_knobs)
    normal, noise = ctrl.session.stop_and_collect_uncertainty(
        len(knob_names), propagate_uncertainty=True
    )
    return float(sandwich_uncertainties(normal, noise)[knob_names.index("pt")])


def _start_rows(track: pd.DataFrame) -> np.ndarray:
    rows = track["name"].str.upper().isin(START_BPMS).to_numpy()
    # A start BPM missing from the data would silently leave its start noisy.
    assert set(track.loc[rows, "name"].str.upper()) == set(START_BPMS)
    return rows


def _sigma_and_scatter(
    tmp_path: Path, interface: AbaMadInterface, *, noisy_starts: bool
) -> tuple[float, float]:
    """Return the reported sigma(pt) and the pt scatter over ``N_SEEDS`` noise draws."""
    clean_track = tmp_path / "track_clean.parquet"
    corrector, _, tune_knobs = _generate_nonoise_track(
        interface, FLATTOP_TURNS, clean_track, DPP_VALUE
    )
    clean = pd.read_parquet(clean_track)
    noisy_rows = np.ones(len(clean), dtype=bool)
    if not noisy_starts:
        # Exact start coordinates, never observed: their declared variance is infinite.
        start_rows = _start_rows(clean)
        clean.loc[start_rows, ["var_x", "var_y"]] = np.inf
        clean.to_parquet(clean_track, index=False)
        noisy_rows = ~start_rows

    pt_true = interface.accelerator.dp2pt(DPP_VALUE)
    true_knobs = {"pt": pt_true}
    # One fitter for everything: building it reloads the lattice (~18 s), while the
    # seeds only differ in the measurement file.
    ctrl = _make_fitter(interface, clean_track, corrector, tune_knobs, tmp_path / "mad.log")
    sigma = _reported_sigma(ctrl, true_knobs)

    rng = np.random.default_rng(20260914)
    estimates = []
    for seed in range(N_SEEDS):
        noisy = clean.copy()
        for column in ("x", "y"):
            noisy.loc[noisy_rows, column] += rng.normal(
                0.0, POSITION_STD_DEV, int(noisy_rows.sum())
            )
        noisy_track = tmp_path / f"track_seed{seed}.parquet"
        noisy.to_parquet(noisy_track, index=False)

        ctrl.measurement_config = MeasurementConfig({noisy_track: ctrl.measurement_config.details[0]})
        ctrl.data_manager = None
        low, mid, high = evaluate_controller_worker_losses(
            ctrl,
            [{"pt": pt_true - STEP}, true_knobs, {"pt": pt_true + STEP}],
            enable_validation=False,
        )
        estimates.append(pt_true - STEP * (high - low) / (2.0 * (high - 2.0 * mid + low)))

    return sigma, float(np.std(estimates, ddof=1))


def _assert_calibrated(sigma: float, scatter: float, label: str) -> None:
    ratio = scatter / sigma
    print(f"{label}: reported sigma={sigma:.3e}, seed scatter={scatter:.3e}, ratio={ratio:.2f}")
    assert 0.6 < ratio < 1.5


def test_sigma_matches_scatter_with_exact_start_coordinates(
    tmp_path: Path, loaded_interface: AbaMadInterface
) -> None:
    sigma, scatter = _sigma_and_scatter(tmp_path, loaded_interface, noisy_starts=False)
    _assert_calibrated(sigma, scatter, "exact starts")


def test_sigma_matches_scatter_with_noisy_start_coordinates(
    tmp_path: Path, loaded_interface: AbaMadInterface
) -> None:
    sigma, scatter = _sigma_and_scatter(tmp_path, loaded_interface, noisy_starts=True)
    _assert_calibrated(sigma, scatter, "noisy starts")
