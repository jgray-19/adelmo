"""The mini-batch gradient-descent loop of the tracking fit."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from adelmo.fitting.protocol import Evaluate
from adelmo.fitting.reduction import reduce_replies
from adelmo.fitting.results import FitDiagnostics
from adelmo.optimisers import adam as _adam  # noqa: F401
from adelmo.optimisers import lbfgs as _lbfgs  # noqa: F401
from adelmo.optimisers.base import BaseOptimiser, BestTracker
from adelmo.tracking.sgd.checkpointing import OptimisationCheckpointer
from adelmo.tracking.sgd.scheduler import LRScheduler

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorboardX import SummaryWriter

    from adelmo.config import OptimiserConfig
    from adelmo.fitting.protocol import WorkerChannels
    from adelmo.tracking.config.models import CheckpointConfig

LOGGER = logging.getLogger(__name__)
#: Starting value for a ``pt`` knob that starts at exactly zero.
PT_SEED = 1e-6


@dataclass
class EpochState:
    """What one epoch produced: the knobs after it and the numbers logged for it."""

    epoch: int
    knobs: dict[str, float]
    #: Training loss, averaged over the batches.
    loss: float
    grad_norm: float
    lr: float
    started: float
    #: Distance of ``knobs`` from the true strengths (see :meth:`SGDLoop._true_diff`).
    true_diff: float = 0.0
    validation_loss: float | None = None
    new_best: bool = False
    saved_checkpoint: bool = False
    #: Whatever the epoch-end hook returned, appended to the log line.
    note: str | None = None

    @property
    def selection_loss(self) -> float:
        """The loss best-selection and early stopping use: validation when available."""
        return self.validation_loss if self.validation_loss is not None else self.loss


class SGDLoop:
    """Runs epochs of mini-batch gradient descent over the training workers.

    Each epoch sends every batch to the workers, steps the optimiser after each
    one, then validates, picks the best knobs so far and checks the stopping
    rules. :attr:`best` holds the result; :attr:`diagnostics` how the run ended.
    """

    def __init__(
        self,
        knob_names: list[str],
        true_strengths: dict[str, float],
        config: OptimiserConfig,
        num_batches: int,
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.config = config
        self.num_batches = num_batches
        self.smoothed_grad_norm = 0.0
        self.smoothed_loss_change = 0.0
        #: Best knobs so far and their selection loss; empty until the first epoch.
        self.best: BestTracker[dict[str, float]] = BestTracker(value={})
        #: How the last :meth:`run` ended.
        self.diagnostics: FitDiagnostics | None = None
        self.optimiser = BaseOptimiser.from_config(config, len(knob_names))
        LOGGER.info("Using optimiser: %s", type(self.optimiser).__name__)
        self.scheduler = LRScheduler(
            warmup_epochs=config.warmup_epochs,
            decay_epochs=config.decay_epochs,
            start_lr=config.warmup_lr_start,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
        )

    def run(
        self,
        knobs: dict[str, float],
        channels: WorkerChannels,
        *,
        total_turns: int,
        writer: SummaryWriter | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        validation_loss_fn: Callable[[dict[str, float]], float | None] | None = None,
        loss_callback: Callable[[int, float, float | None, float, float], None] | None = None,
        epoch_end_hook: Callable[[dict[str, float], dict[str, float]], str | None] | None = None,
    ) -> dict[str, float]:
        """Optimise from ``knobs`` and return the best knobs found.

        ``total_turns`` normalises the summed gradients. ``validation_loss_fn``
        scores the knobs after each epoch on held-out data; ``epoch_end_hook``
        runs after best-selection and returns a note for the log line.
        """
        run_start = time.time()
        checkpointer = OptimisationCheckpointer(self, checkpoint_config)

        if knobs.get("pt") == 0.0:
            knobs = {**knobs, "pt": PT_SEED}

        prev_loss = None
        start_epoch = 0
        if checkpointer.restore:
            checkpoint_state = checkpointer.load(base_current_knobs=knobs)
            knobs = checkpoint_state["current_knobs"]
            prev_loss = checkpoint_state["prev_loss"]
            start_epoch = checkpoint_state["next_epoch"]
            LOGGER.info(
                "Restored optimisation checkpoint from %s at epoch %d",
                checkpointer.path,
                checkpoint_state["saved_epoch"],
            )

        last_completed_epoch = start_epoch - 1
        stop_reason = "max_epochs"
        epochs_run = start_epoch
        for epoch in range(start_epoch, self.config.max_epochs):
            epochs_run = epoch + 1
            state = self._run_epoch(epoch, knobs, channels, total_turns)
            knobs = state.knobs
            state.true_diff = self._true_diff(knobs)
            if validation_loss_fn is not None:
                state.validation_loss = validation_loss_fn(knobs)
            state.new_best = self._select_best(state)
            if epoch_end_hook is not None:
                state.note = epoch_end_hook(knobs, self.best.value)

            stop = self._should_stop(state, prev_loss)
            prev_loss = state.selection_loss
            last_completed_epoch = epoch
            if stop is None and checkpointer.should_save_periodic(epoch):
                checkpointer.save(epoch, knobs, prev_loss)
                state.saved_checkpoint = True

            self._log_epoch(writer, state, run_start)
            if loss_callback is not None:
                loss_callback(
                    epoch, state.loss, state.validation_loss, state.grad_norm, state.true_diff
                )
            if stop is not None:
                LOGGER.info(
                    "\nStopping early at epoch %d (%s): smoothed loss change %.3e, smoothed grad norm %.3e.",
                    epoch,
                    stop,
                    self.smoothed_loss_change,
                    self.smoothed_grad_norm,
                )
                stop_reason = stop
                break

        if checkpointer.should_save_final(last_completed_epoch):
            checkpointer.save(last_completed_epoch, knobs, prev_loss)
        self.diagnostics = FitDiagnostics(
            converged=stop_reason != "max_epochs",
            reason=stop_reason,
            iterations=epochs_run,
            best_loss=self.best.loss,
        )
        return self.best.value

    def _run_epoch(
        self,
        epoch: int,
        knobs: dict[str, float],
        channels: WorkerChannels,
        total_turns: int,
    ) -> EpochState:
        """Evaluate every batch, stepping the optimiser after each one.

        A batch in which any worker lost its particles stops the stepping, and the
        epoch's knobs revert to those it started from.
        """
        started = time.time()
        lr = self.scheduler(epoch)
        loss = 0.0
        grad = np.zeros(len(self.knob_names))
        start_knobs = knobs
        particle_lost = False
        for batch in range(self.num_batches):
            channels.send_all(Evaluate(knobs, batch))
            reduced = reduce_replies(channels.recv_all(), len(self.knob_names))
            loss += reduced.loss / max(1, reduced.n_valid)
            grad += reduced.grad
            particle_lost = particle_lost or reduced.particle_lost
            if not particle_lost:
                knobs = self._step(knobs, reduced.grad / total_turns, lr)

        if particle_lost:
            LOGGER.warning(
                "Epoch %d: particle loss detected — rejecting knob updates and restoring "
                "pre-epoch parameters",
                epoch,
            )
            knobs = start_knobs

        # Keep training loss on a single-worker scale by averaging over batches.
        loss /= max(1, self.num_batches)
        grad /= total_turns
        grad_norm = float(np.linalg.norm(grad[grad != 0.0]))
        self.smoothed_grad_norm = self._smooth(self.smoothed_grad_norm, grad_norm)
        return EpochState(epoch, knobs, loss, grad_norm, lr, started)

    def _step(self, knobs: dict[str, float], grad: np.ndarray, lr: float) -> dict[str, float]:
        """One optimiser step from ``knobs`` along ``grad``."""
        params = np.array([knobs[k] for k in self.knob_names])
        return dict(zip(self.knob_names, self.optimiser.step(params, grad, lr)))

    def _select_best(self, state: EpochState) -> bool:
        """Record ``state`` as the best epoch if its loss is the lowest so far."""
        if state.selection_loss < self.best.loss:
            self.best.record(state.selection_loss, state.knobs.copy())
            return True
        return False

    def _should_stop(self, state: EpochState, prev_loss: float | None) -> str | None:
        """The reason to stop after ``state``, or ``None`` to carry on.

        Always updates the smoothed relative loss change, so call it once per epoch.
        """
        if prev_loss is not None:
            loss = state.selection_loss
            change = abs(loss - prev_loss) / abs(prev_loss) if prev_loss != 0 else 0
            self.smoothed_loss_change = self._smooth(self.smoothed_loss_change, change)
            if (
                self.smoothed_loss_change < self.config.loss_change_tolerance
                and state.epoch
                > self.config.loss_change_min_epoch_fraction * self.config.max_epochs
            ):
                return "loss_converged"
        if self.smoothed_grad_norm < self.config.gradient_converged_value:
            return "gradient_converged"
        return None

    def _smooth(self, average: float, value: float) -> float:
        """Exponential moving average; the first value (``average`` exactly 0) starts it."""
        if average == 0.0:
            return value
        alpha = self.config.grad_norm_alpha
        return alpha * average + (1.0 - alpha) * value

    def _true_diff(self, knobs: dict[str, float]) -> float:
        """Sum of absolute differences from the true strengths (the sum of the knobs without them)."""
        if not self.true_strengths:
            return sum(knobs.values())
        return np.sum([abs(knobs[k] - self.true_strengths[k]) for k in self.knob_names])

    def _log_epoch(self, writer: SummaryWriter | None, state: EpochState, run_start: float) -> None:
        """Log the epoch to the console and TensorBoard."""
        if writer is not None:
            loss_scalars = {"train": state.loss}
            if state.validation_loss is not None:
                loss_scalars["validation"] = state.validation_loss
            writer.add_scalars("loss", loss_scalars, state.epoch)
            writer.add_scalar("grad_norm", state.grad_norm, state.epoch)
            writer.add_scalar("learning_rate", state.lr, state.epoch)
            writer.add_scalar("sum_true_diff", state.true_diff, state.epoch)
            writer.flush()

        now = time.time()
        parts = [f"Ep {state.epoch}: loss={state.loss:.3e}"]
        if state.validation_loss is not None:
            parts.append(f"val={state.validation_loss:.3e}")
        parts.append(f"g={state.grad_norm:.3e}")
        parts.append(f"td={state.true_diff:.3e}")
        if state.note:
            parts.append(state.note)
        parts.append(f"lr={state.lr:.2e}, et={now - state.started:.1f}s, tt={now - run_start:.1f}s")
        message = ", ".join(parts)
        if state.new_best:
            message += " [b]"
        if state.saved_checkpoint:
            message += " [s]"
        LOGGER.info("\r%s", message)
