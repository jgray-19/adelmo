"""Levenberg-Marquardt optimiser for Gauss-Newton systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LevenbergMarquardtConfig:
    """Configuration for a Levenberg-Marquardt Gauss-Newton solve."""

    max_iterations: int = 50
    gradient_converged_value: float = 1e-10
    loss_relative_tolerance: float = 1e-12
    initial_lambda: float = 0.0
    lambda_floor: float = 1e-10
    lambda_decrease: float = 0.5
    lambda_increase: float = 10.0
    lambda_max: float = 1e10

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("LevenbergMarquardtConfig.max_iterations must be >= 1")
        if self.gradient_converged_value < 0.0:
            raise ValueError("LevenbergMarquardtConfig.gradient_converged_value must be >= 0")
        if self.loss_relative_tolerance < 0.0:
            raise ValueError("LevenbergMarquardtConfig.loss_relative_tolerance must be >= 0")
        if self.initial_lambda < 0.0 or self.lambda_floor < 0.0:
            raise ValueError("LevenbergMarquardtConfig damping values must be >= 0")
        if self.lambda_decrease <= 0.0 or self.lambda_increase <= 1.0:
            raise ValueError("LevenbergMarquardtConfig lambda multipliers are invalid")
        if self.lambda_max <= self.lambda_floor:
            raise ValueError("LevenbergMarquardtConfig.lambda_max must exceed lambda_floor")


@dataclass(frozen=True)
class LevenbergMarquardtUpdate:
    """Result of one LM accept/reject decision."""

    next_params: np.ndarray
    accepted: bool
    converged: bool
    loss: float
    grad_norm: float
    damping: float
    reason: str


class LevenbergMarquardtOptimiser:
    """Stateful Levenberg-Marquardt optimiser for dense Gauss-Newton systems."""

    def __init__(
        self,
        config: LevenbergMarquardtConfig | None = None,
        initial_params: np.ndarray | None = None,
    ):
        self.config = config or LevenbergMarquardtConfig()
        self.damping = self.config.initial_lambda
        self.best_loss = float("inf")
        self.best_params = (
            np.array(initial_params, dtype=float).copy()
            if initial_params is not None
            else np.array([], dtype=float)
        )
        self.best_hessian: np.ndarray | None = None
        # Gradient at the best point, kept so a rejected step can be retried from
        # there with the increased damping instead of re-evaluating it unchanged.
        self.best_grad: np.ndarray | None = None
        self.prev_loss: float | None = None
        self.t = 0

    def update(
        self,
        params: np.ndarray,
        loss: float,
        grad: np.ndarray,
        hessian: np.ndarray,
        failed: bool = False,
    ) -> LevenbergMarquardtUpdate:
        """Accept or reject the current point and return the next parameters."""
        self.t += 1
        params = np.asarray(params, dtype=float)
        grad = np.asarray(grad, dtype=float)
        hessian = np.asarray(hessian, dtype=float)

        if self.best_params.size == 0:
            self.best_params = params.copy()

        if failed or not np.isfinite(loss):
            self._increase_damping()
            return self._retry_from_best(loss, float("nan"), "failed")

        grad_norm = float(np.linalg.norm(grad))
        if loss >= self.best_loss:
            self._increase_damping()
            return self._retry_from_best(loss, grad_norm, "rejected")

        loss_converged = (
            self.prev_loss is not None
            and abs(self.prev_loss - loss)
            <= self.config.loss_relative_tolerance * max(1.0, abs(self.prev_loss))
        )
        grad_converged = grad_norm < self.config.gradient_converged_value

        self.best_loss = float(loss)
        self.best_params = params.copy()
        self.best_hessian = hessian.copy()
        self.best_grad = grad.copy()
        self._decrease_damping()
        self.prev_loss = float(loss)

        if grad_converged or loss_converged:
            reason = "gradient_converged" if grad_converged else "loss_converged"
            return LevenbergMarquardtUpdate(
                next_params=params.copy(),
                accepted=True,
                converged=True,
                loss=float(loss),
                grad_norm=grad_norm,
                damping=self.damping,
                reason=reason,
            )

        return LevenbergMarquardtUpdate(
            next_params=params + self.solve_step(hessian, grad),
            accepted=True,
            converged=False,
            loss=float(loss),
            grad_norm=grad_norm,
            damping=self.damping,
            reason="accepted",
        )

    def _retry_from_best(
        self, loss: float, grad_norm: float, reason: str
    ) -> LevenbergMarquardtUpdate:
        """Re-solve a shorter step from the best point with the raised damping.

        Returning ``best_params`` unchanged would be a dead end: the caller would
        re-evaluate a point already stored as best, get exactly ``best_loss``
        back, be rejected again, and repeat until ``max_iterations`` - with the
        damping raised on every pass but never reaching :meth:`solve_step`,
        because only the accepted branch computes a step. Retrying from the best
        point is the whole mechanism by which LM recovers from an overshoot: a
        larger lambda both shortens the step and rotates it toward ``-grad``.
        """
        if self.best_grad is None or self.best_hessian is None:
            # No step has been accepted, so there is no curvature to retry from.
            # Report convergence so the caller stops rather than re-evaluating a
            # point that cannot be evaluated.
            return LevenbergMarquardtUpdate(
                next_params=self.best_params.copy(),
                accepted=False,
                converged=True,
                loss=float(loss),
                grad_norm=grad_norm,
                damping=self.damping,
                reason="no_progress",
            )
        if self.damping > self.config.lambda_max:
            # The step has been shortened to nothing without finding an
            # improvement; the best point is as good as this solve gets.
            return LevenbergMarquardtUpdate(
                next_params=self.best_params.copy(),
                accepted=False,
                converged=True,
                loss=float(loss),
                grad_norm=grad_norm,
                damping=self.damping,
                reason="damping_exhausted",
            )
        return LevenbergMarquardtUpdate(
            next_params=self.best_params + self.solve_step(self.best_hessian, self.best_grad),
            accepted=False,
            converged=False,
            loss=float(loss),
            grad_norm=grad_norm,
            damping=self.damping,
            reason=reason,
        )

    def solve_step(self, hessian: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Solve ``(H + lambda diag(H)) step = -grad`` for the LM step."""
        damped = np.asarray(hessian, dtype=float)
        if self.damping > 0.0:
            diag = np.abs(np.diag(damped))
            floor = 1e-12 * max(1.0, float(np.max(diag)))
            damped = damped + np.diag(self.damping * np.maximum(diag, floor))
        try:
            return np.linalg.solve(damped, -grad)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(damped, -grad, rcond=None)[0]

    def state_to_dict(self) -> dict[str, Any]:
        """Return optimiser internal state as a serialisable dictionary."""
        return {
            "type": "levenberg_marquardt",
            "damping": float(self.damping),
            "best_loss": float(self.best_loss),
            "best_params": self.best_params.tolist(),
            "best_hessian": None
            if self.best_hessian is None
            else self.best_hessian.tolist(),
            "best_grad": None if self.best_grad is None else self.best_grad.tolist(),
            "prev_loss": self.prev_loss,
            "t": int(self.t),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimiser internal state from a dictionary."""
        if state.get("type") != "levenberg_marquardt":
            raise ValueError("State type does not match levenberg_marquardt")
        self.damping = float(state["damping"])
        self.best_loss = float(state["best_loss"])
        self.best_params = np.array(state["best_params"], dtype=float)
        best_hessian = state.get("best_hessian")
        self.best_hessian = (
            None if best_hessian is None else np.array(best_hessian, dtype=float)
        )
        best_grad = state.get("best_grad")
        self.best_grad = None if best_grad is None else np.array(best_grad, dtype=float)
        self.prev_loss = state.get("prev_loss")
        self.t = int(state["t"])

    def _increase_damping(self) -> None:
        self.damping = max(
            self.damping * self.config.lambda_increase,
            self.config.lambda_floor,
        )

    def _decrease_damping(self) -> None:
        if self.damping > self.config.lambda_floor:
            self.damping *= self.config.lambda_decrease
        else:
            self.damping = 0.0
