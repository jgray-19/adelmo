"""The Levenberg-Marquardt iteration shared by every closed-orbit / closed-twiss fitter."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from aba_optimiser.fitting.results import FitDiagnostics
from aba_optimiser.optimisers.levenberg_marquardt import LevenbergMarquardtOptimiser

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorboardX import SummaryWriter

    from aba_optimiser.optimisers.levenberg_marquardt import LevenbergMarquardtConfig

logger = logging.getLogger(__name__)


@dataclass
class LMPoint:
    """The objective at one point: loss, gradient and Gauss-Newton Hessian.

    ``failed`` marks a point where the closed orbit was lost. ``extra`` is whatever
    the evaluation wants handed back to ``on_accept``.
    """

    loss: float
    grad: np.ndarray
    hessian: np.ndarray
    failed: bool = False
    extra: Any = None


def run_levenberg_marquardt(
    params: np.ndarray,
    evaluate: Callable[[np.ndarray, int], LMPoint],
    config: LevenbergMarquardtConfig,
    *,
    trial_loss: Callable[[np.ndarray], float | None] | None = None,
    on_accept: Callable[[np.ndarray, LMPoint], None] | None = None,
    after_step: Callable[[np.ndarray, LevenbergMarquardtOptimiser], None] | None = None,
    writer: SummaryWriter | None = None,
) -> tuple[LevenbergMarquardtOptimiser, FitDiagnostics]:
    """Minimise with Levenberg-Marquardt steps ``(H + lam·diag H) delta = -g``.

    ``evaluate(params, iteration)`` returns the full :class:`LMPoint`. With
    ``trial_loss``, a trial point is first scored by its loss alone (``None`` when
    the orbit is lost) and only one that improves on the best loss is evaluated in
    full, so a rejected step costs a cheap loss evaluation. ``on_accept`` sees every
    accepted point; ``after_step`` runs once the next point is chosen.

    A rejected step is not a no-op: the optimiser retries from its best point with
    more damping, so the next point still needs evaluating. Only the terminal
    reasons (no curvature to retry from, or damping driven to the ceiling) end the
    solve early.

    Returns the optimiser, whose ``best_params`` are the solution, and how the fit ended.
    """
    optimiser = LevenbergMarquardtOptimiser(config, initial_params=params)
    n = len(params)
    zero_grad, zero_hess = np.zeros(n), np.zeros((n, n))
    run_start = time.time()
    last_update = None
    completed_iterations = 0
    accepted_evaluations = 0

    for iteration in range(config.max_iterations):
        completed_iterations = iteration + 1
        update = None
        point = None
        if trial_loss is not None and optimiser.best_hessian is not None:
            loss = trial_loss(params)
            if loss is None:
                update = optimiser.update(params, float("nan"), zero_grad, zero_hess, failed=True)
            elif not loss < optimiser.best.loss:
                update = optimiser.update(params, loss, zero_grad, zero_hess, failed=False)
        if update is None:
            point = evaluate(params, iteration)
            update = optimiser.update(params, point.loss, point.grad, point.hessian, point.failed)
        last_update = update
        if update.accepted:
            accepted_evaluations += 1
            if on_accept is not None:
                on_accept(params, point)
        params = update.next_params

        if update.converged and not update.accepted:
            logger.warning(
                "Levenberg-Marquardt stopped at iter %d (%s, lam=%.1e)",
                iteration,
                update.reason,
                update.damping,
            )
            break
        if after_step is not None:
            after_step(params, optimiser)
        if update.reason == "failed":
            logger.warning(
                "Iter %d: closed orbit lost; retrying from best (lam=%.1e)", iteration, update.damping
            )
            continue
        if not update.accepted:
            logger.info(
                "Iter %d: step rejected (loss %.6e >= best %.6e); retrying from best (lam=%.1e)",
                iteration,
                update.loss,
                optimiser.best.loss,
                update.damping,
            )
            continue

        _log_iteration(writer, iteration, update.loss, update.grad_norm, update.damping, run_start)
        if update.converged:
            logger.info("Levenberg-Marquardt converged (%s) at iter %d", update.reason, iteration)
            break

    diagnostics = FitDiagnostics(
        converged=bool(last_update is not None and last_update.converged),
        reason=None if last_update is None else last_update.reason,
        iterations=completed_iterations,
        best_loss=float(optimiser.best.loss),
        gradient_norm=None if last_update is None else float(last_update.grad_norm),
        damping=None if last_update is None else float(last_update.damping),
        accepted_evaluations=accepted_evaluations,
    )
    return optimiser, diagnostics


def _log_iteration(
    writer: SummaryWriter | None, iteration: int, loss: float, grad_norm: float, lam: float, run_start: float
) -> None:
    """Log one accepted iteration to the console and TensorBoard."""
    logger.info(
        "LM iter %d: loss=%.3e, |g|=%.3e, lam=%.1e, tt=%.1fs",
        iteration,
        loss,
        grad_norm,
        lam,
        time.time() - run_start,
    )
    if writer is not None:
        writer.add_scalar("loss", loss, iteration)
        writer.add_scalar("grad_norm", grad_norm, iteration)
        writer.add_scalar("lm_lambda", lam, iteration)
        writer.flush()
