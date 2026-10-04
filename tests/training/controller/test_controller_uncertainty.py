from __future__ import annotations

import dataclasses
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.training.config.models import OutputConfig
from aba_optimiser.training.tracking_fitter import (
    ArcByArcFitter,
    TrackingFitter,
)
from aba_optimiser.workers.common import (
    HESSIAN_MIN_EIGENVALUE,
    UncertaintyPart,
    WeightProcessor,
    merge_uncertainty_parts,
    noise_matrix,
    sandwich_uncertainties,
)


def test_finalise_results_uses_finite_non_negative_uncertainties_for_indefinite_hessian() -> None:
    ctrl = TrackingFitter.__new__(TrackingFitter)
    ctrl.output_config = OutputConfig(include_uncertainty=True)
    ctrl.final_knobs = {"kq1": 1.0, "kq2": 2.0}
    ctrl.filtered_true_strengths = {"kq1": 1.1, "kq2": 2.1}
    ctrl.accelerator = SimpleNamespace(optimise_energy=False)
    ctrl.config_manager = SimpleNamespace(
        knob_names=["kq1", "kq2"],
        mad_iface=SimpleNamespace(
            convert_uncertainties_to_absolute=lambda knob_names, uncertainties: np.asarray(
                uncertainties,
                dtype=np.float64,
            )
        ),
    )
    ctrl.output_knob_names = ["kq1", "kq2"]

    ctrl.final_knobs = {"kq1": 0.9, "kq2": 1.9}
    uncertainties = ctrl._finalise_results(
        (
            np.array([[4.0, 0.0], [0.0, -1e-12]], dtype=np.float64),
            np.diag([4.0, HESSIAN_MIN_EIGENVALUE]),
        ),
        writer=None,
    )

    assert np.all(np.isfinite(uncertainties))
    assert np.all(uncertainties >= 0.0)
    assert np.isclose(uncertainties[0], 0.5)
    assert np.isclose(uncertainties[1], 1.0 / np.sqrt(HESSIAN_MIN_EIGENVALUE))


def test_shared_reading_noise_is_summed_before_squaring() -> None:
    # Two workers observe reading 7; worker b also starts from reading 9.
    g_a, g_b, g_start = np.array([1.0, 2.0]), np.array([3.0, -1.0]), np.array([0.5, 0.5])
    part_a = UncertaintyPart(np.eye(2), np.array([7]), g_a[None, :], np.array([0.1]))
    part_b = UncertaintyPart(
        2.0 * np.eye(2), np.array([7, 9, 11]), np.stack([g_b, g_start, g_start]),
        np.array([0.1, 0.2, np.inf]),
    )

    merged = merge_uncertainty_parts([part_a, part_b], n_knobs=2)

    shared = g_a + g_b
    np.testing.assert_allclose(merged.normal, 3.0 * np.eye(2))
    # The unobserved reading 11 (infinite variance) carries no noise.
    np.testing.assert_allclose(
        noise_matrix(merged), 0.1 * np.outer(shared, shared) + 0.2 * np.outer(g_start, g_start)
    )


def test_incremental_merge_matches_one_shot_merge() -> None:
    # The manager folds workers into a running merge chunk by chunk.
    rng = np.random.default_rng(1)
    parts = [
        UncertaintyPart(
            np.eye(3) * (i + 1),
            rng.integers(0, 30, size=50),
            rng.normal(size=(50, 3)),
            np.full(50, 0.3),
        )
        for i in range(5)
    ]

    one_shot = merge_uncertainty_parts(parts, n_knobs=3)
    running = merge_uncertainty_parts(parts[:2], n_knobs=3)
    running = merge_uncertainty_parts([*parts[2:4], running], n_knobs=3)
    running = merge_uncertainty_parts([running, parts[4]], n_knobs=3)

    np.testing.assert_array_equal(running.reading_ids, one_shot.reading_ids)
    np.testing.assert_allclose(running.normal, one_shot.normal, rtol=1e-14)
    np.testing.assert_allclose(noise_matrix(running), noise_matrix(one_shot), rtol=1e-12)


def test_sandwich_reduces_to_inverse_normal_matrix_for_independent_readings() -> None:
    rng = np.random.default_rng(0)
    jacobian = rng.normal(size=(40, 3))
    variances = rng.uniform(0.5, 2.0, size=40)
    # Each reading is used once with w = 1/σ²: B = A, so Cov = A⁻¹.
    part = UncertaintyPart(
        jacobian.T @ (jacobian / variances[:, None]),
        np.arange(40),
        -jacobian / variances[:, None],
        variances,
    )

    merged = merge_uncertainty_parts([part], n_knobs=3)
    sigmas = sandwich_uncertainties(merged.normal, noise_matrix(merged))

    np.testing.assert_allclose(sigmas, np.sqrt(np.diag(np.linalg.inv(part.normal))))


def _collect_epoch_gradient(ctrl: TrackingFitter, knob_updates: dict[str, float]) -> np.ndarray:
    """Return the raw training gradient summed over all workers and batches.

    Workers divide their gradient by their own point count, which differs between
    ranges; multiplying it back makes the sum comparable with the Hessian parts.
    """
    gradient = np.zeros(len(ctrl.config_manager.knob_names), dtype=np.float64)
    points = {
        meta.worker_id: len(meta.bpm_names) * meta.n_run_turns
        for meta in ctrl.worker_manager.worker_metadata
    }
    channels = ctrl.worker_manager._channels()
    for batch in range(ctrl.simulation_config.num_batches):
        channels.send_all((knob_updates, batch))
        for result in channels.recv_all():
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError(f"Unexpected worker result payload: {result!r}")
            worker_id, grad, _loss = result
            gradient += points[worker_id] * np.asarray(grad, dtype=np.float64).reshape(-1)
    return gradient


def _compute_training_weight_normaliser(ctrl: TrackingFitter) -> float:
    """Rebuild the worker payload weights and return the global gradient normaliser.

    Production normalises weights over the training *and* validation payloads at once
    (see ``WorkerManager._build_payload_split``), so the normaliser is computed over the
    same combined set here.
    """
    build = partial(
        ctrl.worker_manager.create_worker_payloads,
        ctrl.data_manager.tracks,
        file_turn_map=ctrl.data_manager.file_map,
        start_bpms=ctrl.config_manager.start_bpms,
        end_bpms=ctrl.config_manager.end_bpms,
        simulation_config=ctrl.simulation_config,
        machine_deltaps=ctrl.machine_deltaps,
    )
    payloads = build(ctrl.data_manager.turn_batches)
    if ctrl.data_manager.validation_turn_batches:
        payloads = payloads + build(ctrl.data_manager.validation_turn_batches)
    if not payloads:
        raise AssertionError("Expected at least one worker payload")

    optimise_momenta = ctrl.simulation_config.optimise_momenta
    global_max = 0.0
    for data, config, _file_idx in payloads:
        planes = {"x": ("x"), "y": ("y")}.get(config.kick_plane, ("x", "y"))
        active = planes + tuple(f"p{plane}" for plane in planes) * optimise_momenta
        variances = {
            "x": data.position_variances[:, :, 0],
            "y": data.position_variances[:, :, 1],
            "px": data.momentum_variances[:, :, 0],
            "py": data.momentum_variances[:, :, 1],
        }
        for observable in active:
            weights = WeightProcessor.variance_to_weight(variances[observable])
            if weights.size:
                global_max = max(global_max, float(np.max(weights)))

    return global_max if global_max > 0.0 else 1.0


@pytest.mark.slow
@pytest.mark.serial
def test_controller_worker_hessian_matches_finite_difference_on_reduced_knob_subset(
    tmp_path,
    seq_b1,
    loaded_interface,
) -> None:
    from aba_optimiser.accelerators import LHC
    from aba_optimiser.training.config.helpers import create_arc_measurement_config
    from aba_optimiser.training.config.models import SequenceConfig
    from tests.training.controller_test_utils import (
        _generate_nonoise_track,
        _make_optimiser_config_quad,
        _make_simulation_config_quad,
    )

    magnet_range = "BPM.13R1.B1/BPM.13L2.B1"
    bpm_start_points = ["BPM.13R1.B1"]
    bpm_end_points = ["BPM.13L2.B1"]
    flattop_turns = 64
    start_marker = "MSIA.EXIT.B1"
    measurement_file = tmp_path / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        measurement_file,
        0.0,
        start_marker=start_marker,
        perturb_quads=True,
    )

    simulation_config = dataclasses.replace(
        _make_simulation_config_quad(),
        num_workers=4,
        num_batches=2,
    )
    ctrl = ArcByArcFitter(
        LHC(
            beam=1,
            kinetic_energy=6800,
            sequence_file=seq_b1,
            errors={"quad": {"k1"}},
        ),
        _make_optimiser_config_quad(),
        simulation_config,
        SequenceConfig(magnet_range=magnet_range),
        create_arc_measurement_config(
            measurement_file, corrector_knobs=corrector_file, tune_knobs=tune_knobs
        ),
        bpm_start_points,
        bpm_end_points,
        output_config=OutputConfig(
            mad_logfile=tmp_path / "controller_hessian.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths.copy(),
    )
    weight_normaliser = _compute_training_weight_normaliser(ctrl)

    terminated = False
    try:
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
        )
        base_knobs = ctrl.filtered_true_strengths.copy()
        base_vec = np.array(
            [base_knobs[name] for name in ctrl.config_manager.knob_names],
            dtype=np.float64,
        )
        n_knobs = len(base_vec)
        centre = n_knobs // 2
        subset = np.array([centre - 1, centre, centre + 1], dtype=int)

        fd_matrix = np.zeros((subset.size, subset.size), dtype=np.float64)
        for col, knob_idx in enumerate(subset):
            knob_name = ctrl.config_manager.knob_names[knob_idx]
            step = max(1e-6, 1e-2 * max(abs(base_vec[knob_idx]), 1e-4))

            plus_knobs = base_knobs.copy()
            minus_knobs = base_knobs.copy()
            plus_knobs[knob_name] += step
            minus_knobs[knob_name] -= step

            grad_plus = _collect_epoch_gradient(ctrl, plus_knobs)
            grad_minus = _collect_epoch_gradient(ctrl, minus_knobs)
            fd_matrix[:, col] = (grad_plus[subset] - grad_minus[subset]) / (2.0 * step)

        total_hessian, _ = ctrl.worker_manager.termination_and_hessian(
            n_knobs, estimate_hessian=True
        )
        terminated = True
    finally:
        if not terminated:
            ctrl.worker_manager.terminate_workers()

    assert total_hessian.shape == (n_knobs, n_knobs)
    assert np.all(np.isfinite(total_hessian))
    sym_hessian = 0.5 * (total_hessian + total_hessian.T)
    predicted = 2.0 * sym_hessian[np.ix_(subset, subset)] / weight_normaliser
    difference = fd_matrix - predicted
    reference_scale = max(np.linalg.norm(predicted), 1.0)

    assert np.linalg.norm(predicted) > 0.0
    assert np.linalg.norm(difference) / reference_scale < 0.25
    assert np.allclose(fd_matrix, predicted, rtol=0.25, atol=1e-2)


@pytest.mark.slow
@pytest.mark.serial
def test_controller_worker_hessian_matches_finite_difference_for_psb_100um_noise(
    tmp_path,
    seq_psb,
    loaded_psb_interface,
) -> None:
    from aba_optimiser.accelerators import PSB
    from aba_optimiser.config import OptimiserConfig
    from aba_optimiser.training.config.helpers import create_arc_measurement_config
    from aba_optimiser.training.config.models import SequenceConfig
    from tests.training.controller_test_utils import (
        _generate_nonoise_track,
        _make_simulation_config_quad,
    )

    flattop_turns = 256
    measurement_file = tmp_path / "track_off_magnet_psb.parquet"
    bpm_start_points = ["BR3.BPM1L3", "BR3.BPM5L3", "BR3.BPM9L3"]
    bpm_end_points = ["BR3.BPM13L3", "BR3.BPM15L3", "BR3.BPM16L3"]

    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_psb_interface,
        flattop_turns,
        measurement_file,
        0.0,
        perturb_quads=True,
        bpm_pattern=r"br3\.bpm.*",
        apply_orbit_correction=False,
        target_qx=0.17,
        target_qy=0.225,
    )

    measurement_df = pd.read_parquet(measurement_file)
    assert np.allclose(measurement_df["var_x"].dropna().to_numpy(), (1e-4) ** 2)
    assert np.allclose(measurement_df["var_y"].dropna().to_numpy(), (1e-4) ** 2)

    simulation_config = dataclasses.replace(
        _make_simulation_config_quad(),
        num_workers=4,
        num_batches=4,
        bpm_loss_outlier_sigma=20,
        worker_loss_outlier_sigma=20,
    )
    optimiser_config = OptimiserConfig(
        max_epochs=1,
        warmup_epochs=0,
        warmup_lr_start=1e-6,
        max_lr=3e-4,
        min_lr=3e-4,
        gradient_converged_value=5e-15,
        optimiser_type="adam",
    )
    ctrl = ArcByArcFitter(
        PSB(
            ring=3,
            kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors={"quad": {"k1"}},
        ),
        optimiser_config,
        simulation_config,
        SequenceConfig("$start/$end"),
        create_arc_measurement_config(
            measurement_file, corrector_knobs=corrector_file, tune_knobs=tune_knobs
        ),
        bpm_start_points,
        bpm_end_points,
        output_config=OutputConfig(
            mad_logfile=tmp_path / "controller_psb_hessian.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths.copy(),
    )
    weight_normaliser = _compute_training_weight_normaliser(ctrl)

    terminated = False
    try:
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
        )
        base_knobs = ctrl.filtered_true_strengths.copy()
        base_vec = np.array(
            [base_knobs[name] for name in ctrl.config_manager.knob_names],
            dtype=np.float64,
        )
        n_knobs = len(base_vec)
        centre = n_knobs // 2
        subset = np.array([centre - 1, centre, centre + 1], dtype=int)

        fd_matrix = np.zeros((subset.size, subset.size), dtype=np.float64)
        for col, knob_idx in enumerate(subset):
            knob_name = ctrl.config_manager.knob_names[knob_idx]
            step = max(1e-6, 1e-2 * max(abs(base_vec[knob_idx]), 1e-4))

            plus_knobs = base_knobs.copy()
            minus_knobs = base_knobs.copy()
            plus_knobs[knob_name] += step
            minus_knobs[knob_name] -= step

            grad_plus = _collect_epoch_gradient(ctrl, plus_knobs)
            grad_minus = _collect_epoch_gradient(ctrl, minus_knobs)
            fd_matrix[:, col] = (grad_plus[subset] - grad_minus[subset]) / (2.0 * step)

        total_hessian, _ = ctrl.worker_manager.termination_and_hessian(
            n_knobs, estimate_hessian=True
        )
        terminated = True
    finally:
        if not terminated:
            ctrl.worker_manager.terminate_workers()

    assert total_hessian.shape == (n_knobs, n_knobs)
    assert np.all(np.isfinite(total_hessian))
    sym_hessian = 0.5 * (total_hessian + total_hessian.T)
    eigenvalues = np.linalg.eigvalsh(sym_hessian)
    assert np.min(eigenvalues) >= -1e-9

    # No range crosses s = 0, so the quadrupoles outside BPM1L3..BPM16L3 (QFO11 before
    # the first BPM, QDE16 and QFO162 after the last) are never observed.
    row_norms = np.linalg.norm(sym_hessian, axis=1)
    zero_row_knobs = {
        ctrl.config_manager.knob_names[idx] for idx in np.where(row_norms == 0.0)[0]
    }
    assert zero_row_knobs == {
        "BR.QFO11.dk1l",
        "BR.QDE16.dk1l",
        "BR.QFO162.dk1l",
    }
    positive_modes = eigenvalues[eigenvalues > 1e-9]
    assert positive_modes.size == n_knobs - len(zero_row_knobs)

    predicted = 2.0 * sym_hessian[np.ix_(subset, subset)] / weight_normaliser
    difference = fd_matrix - predicted
    reference_scale = max(np.linalg.norm(predicted), 1.0)

    assert np.linalg.norm(predicted) > 0.0
    assert np.linalg.norm(difference) / reference_scale < 0.25
    assert np.allclose(fd_matrix, predicted, rtol=0.25, atol=1e-2)
