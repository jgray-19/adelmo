import numpy as np
import pytest

from aba_optimiser.optimisers.levenberg_marquardt import (
    LevenbergMarquardtConfig,
    LevenbergMarquardtOptimiser,
)


def test_lm_solves_quadratic_newton_step() -> None:
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(gradient_converged_value=0.0),
        initial_params=np.array([0.0]),
    )

    update = optim.update(
        params=np.array([0.0]),
        loss=4.0,
        grad=np.array([-4.0]),
        hessian=np.array([[2.0]]),
    )

    assert update.accepted
    assert not update.converged
    assert np.allclose(update.next_params, [2.0])
    assert np.allclose(optim.best.value, [0.0])


def test_lm_rejects_worse_loss_and_increases_damping() -> None:
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(lambda_floor=1e-4),
        initial_params=np.array([0.0]),
    )
    hessian = np.array([[1.0]])

    first = optim.update(np.array([0.0]), 1.0, np.array([1.0]), hessian)
    rejected = optim.update(np.array([10.0]), 2.0, np.array([1.0]), hessian)

    assert first.accepted
    assert not rejected.accepted
    assert rejected.reason == "rejected"
    assert rejected.damping == pytest.approx(1e-4)
    # The retry is measured from the best point, not from the rejected one.
    assert np.allclose(optim.best.value, [0.0])
    assert not np.allclose(rejected.next_params, optim.best.value)


def test_lm_retries_from_best_instead_of_re_evaluating_it() -> None:
    """A rejected step must come back as a *new* point to evaluate.

    Returning ``best_params`` unchanged deadlocks the solve: the caller
    re-evaluates a point already stored as best, gets exactly ``best_loss``,
    and is rejected again for every remaining iteration.
    """
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(lambda_floor=1.0, lambda_increase=10.0),
        initial_params=np.array([0.0]),
    )
    hessian = np.array([[1.0]])
    grad = np.array([2.0])

    optim.update(np.array([0.0]), 1.0, grad, hessian)
    first = optim.update(np.array([100.0]), 5.0, grad, hessian)
    second = optim.update(first.next_params, 4.0, grad, hessian)

    # Undamped the step would be -grad/H = -2. Each rejection damps it further.
    assert first.next_params[0] == pytest.approx(-2.0 / (1.0 + 1.0))
    assert second.next_params[0] == pytest.approx(-2.0 / (1.0 + 10.0))
    assert abs(second.next_params[0]) < abs(first.next_params[0])


def test_lm_recovers_from_an_overshooting_gauss_newton_step() -> None:
    """End-to-end: an overshoot must be backtracked, not fatal.

    ``f(x) = (x^2 - 1)^2`` with residual ``r = x^2 - 1``. From x=0.1 the
    Gauss-Newton step is ~+5.0, which lands far uphill and is rejected. Only a
    working retry path recovers the root at |x| = 1.
    """
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(max_iterations=200, lambda_floor=1e-6),
        initial_params=np.array([0.1]),
    )
    params = np.array([0.1])
    rejections = 0
    for _ in range(200):
        residual = params[0] ** 2 - 1.0
        jac = 2.0 * params[0]
        update = optim.update(
            params, residual**2, np.array([2.0 * jac * residual]), np.array([[2.0 * jac**2]])
        )
        rejections += not update.accepted
        params = update.next_params
        if update.converged:
            break

    assert rejections > 0, "the overshoot should have been rejected at least once"
    assert abs(optim.best.value[0]) == pytest.approx(1.0, abs=1e-6)


def test_lm_stops_when_damping_is_exhausted() -> None:
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(lambda_floor=1e-4, lambda_max=1e2),
        initial_params=np.array([0.0]),
    )
    hessian = np.array([[1.0]])
    optim.update(np.array([0.0]), 1.0, np.array([1.0]), hessian)

    for _ in range(20):
        update = optim.update(np.array([10.0]), 2.0, np.array([1.0]), hessian)
        if update.converged:
            break

    assert update.converged
    assert update.reason == "damping_exhausted"
    assert np.allclose(update.next_params, optim.best.value)


def test_lm_reports_no_progress_when_nothing_was_ever_accepted() -> None:
    """A first evaluation that fails leaves no curvature to retry from."""
    optim = LevenbergMarquardtOptimiser(initial_params=np.array([0.0]))

    update = optim.update(
        np.array([0.0]), float("nan"), np.array([np.nan]), np.array([[np.nan]]), failed=True
    )

    assert not update.accepted
    assert update.converged, "the caller must stop rather than spin on the same point"
    assert update.reason == "no_progress"


def test_lm_state_roundtrip() -> None:
    optim = LevenbergMarquardtOptimiser(initial_params=np.array([1.0, 2.0]))
    hessian = np.eye(2)
    optim.update(np.array([1.0, 2.0]), 3.0, np.array([0.5, -0.5]), hessian)

    restored = LevenbergMarquardtOptimiser()
    restored.load_state_dict(optim.state_dict())

    assert restored.damping == optim.damping
    assert restored.best.loss == optim.best.loss
    assert np.allclose(restored.best.value, optim.best.value)
    assert np.allclose(restored.best_hessian, optim.best_hessian)
    # Without the gradient at the best point a resumed run cannot backtrack.
    assert np.allclose(restored.best_grad, optim.best_grad)
