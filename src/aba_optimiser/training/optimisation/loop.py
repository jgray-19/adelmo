"""Optimisation loop management for the fitter."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.optimisers import adam as _adam  # noqa: F401
from aba_optimiser.optimisers import lbfgs as _lbfgs  # noqa: F401
from aba_optimiser.optimisers.base import BaseOptimiser, BestTracker
from aba_optimiser.training.optimisation.checkpointing import OptimisationCheckpointer
from aba_optimiser.training.optimisation.scheduler import LRScheduler
from aba_optimiser.training.reduction import reduce_replies
from aba_optimiser.training.results import FitDiagnostics
from aba_optimiser.workers.protocol import Evaluate

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorboardX import SummaryWriter

    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.config.models import CheckpointConfig
    from aba_optimiser.workers.protocol import WorkerChannels

LOGGER = logging.getLogger(__name__)


class OptimisationLoop:
    """Manages the optimisation loop and statistics tracking."""

    def __init__(
        self,
        initial_strengths: np.ndarray,
        knob_names: list[str],
        true_strengths: dict[str, float],
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.use_true_strengths = len(true_strengths) > 0
        self.smoothed_grad_norm: float = 0.0
        self.smoothed_loss_change: float = 0.0
        self.grad_norm_alpha = optimiser_config.grad_norm_alpha

        #: Best knobs so far and their (validation, else training) loss; empty until the first epoch.
        self.best: BestTracker[dict[str, float]] = BestTracker(value={})
        #: How the last :meth:`run_optimisation` ended.
        self.diagnostics: FitDiagnostics | None = None
        self.loss_improvement_threshold = 1e-4  # Minimum relative improvement to accept new best

        self.max_epochs = optimiser_config.max_epochs
        self.gradient_converged_value = optimiser_config.gradient_converged_value
        self.optimiser: BaseOptimiser

        self.optimiser = BaseOptimiser.from_config(optimiser_config, len(knob_names))
        LOGGER.info(f"Using optimiser: {self.optimiser.__class__.__name__}")

        # Initialise scheduler
        self.scheduler = LRScheduler(
            warmup_epochs=optimiser_config.warmup_epochs,
            decay_epochs=optimiser_config.decay_epochs,
            start_lr=optimiser_config.warmup_lr_start,
            max_lr=optimiser_config.max_lr,
            min_lr=optimiser_config.min_lr,
        )
        self.num_batches = simulation_config.num_batches

    def _is_new_best(
        self,
        epoch_loss: float,
        prev_loss: float | None,
        sum_diff: float,
    ) -> bool:
        """Decide whether the current epoch should replace the best known state."""
        should_save_as_best = True
        if self.best.loss != float("inf") and prev_loss is not None:
            loss_improvement = (
                (self.best.loss - epoch_loss) / abs(prev_loss) if prev_loss != 0 else 0
            )
            if loss_improvement < self.loss_improvement_threshold:
                best_sum_diff = self._calculate_diff(self.best.value)
                if sum_diff > best_sum_diff:
                    should_save_as_best = False
                    LOGGER.debug(
                        f"Not saving as best: loss improvement {loss_improvement:.3e} < {self.loss_improvement_threshold:.3e} "
                        f"and rel_diff {sum_diff:.3e} > {best_sum_diff:.3e}."
                    )
        return should_save_as_best and epoch_loss < self.best.loss

    def _should_stop_for_loss_change(
        self,
        epoch: int,
        epoch_loss: float,
        prev_loss: float | None,
    ) -> bool:
        """Update smoothed loss-change metric and decide if loss-based early stop triggers."""
        if prev_loss is None:
            return False

        rel_loss_change = abs(epoch_loss - prev_loss) / abs(prev_loss) if prev_loss != 0 else 0
        if self.smoothed_loss_change == 0.0:  # Exact 0 case for first update
            self.smoothed_loss_change = rel_loss_change
        else:
            self.smoothed_loss_change = (
                self.grad_norm_alpha * self.smoothed_loss_change
                + (1.0 - self.grad_norm_alpha) * rel_loss_change
            )
        return self.smoothed_loss_change < 1e-6 and epoch > 0.2 * self.max_epochs

    def run_optimisation(
        self,
        current_knobs: dict[str, float],
        channels: WorkerChannels,
        writer: SummaryWriter | None,
        run_start: float,
        total_turns: int,
        checkpoint_config: CheckpointConfig | None = None,
        validation_loss_fn: Callable[[dict[str, float]], float | None] | None = None,
        loss_callback: Callable[[int, float, float | None, float, float], None]
        | None = None,
        epoch_end_hook: Callable[[dict[str, float], dict[str, float]], str | None]
        | None = None,
    ) -> dict[str, float]:
        """Run the main optimisation loop."""
        checkpointer = OptimisationCheckpointer(self, checkpoint_config)

        if "pt" in current_knobs and current_knobs["pt"] == 0.0:
            current_knobs["pt"] = 1e-6  # Initialise pt to non-zero

        prev_loss = None
        start_epoch = 0

        if checkpointer.restore:
            checkpoint_state = checkpointer.load(base_current_knobs=current_knobs)
            current_knobs = checkpoint_state["current_knobs"]
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
        for epoch in range(start_epoch, self.max_epochs):
            epochs_run = epoch + 1
            epoch_start = time.time()

            epoch_loss = 0.0
            epoch_grad = np.zeros(len(self.knob_names))
            lr = self.scheduler(epoch)
            pre_epoch_knobs = current_knobs
            epoch_had_particle_loss = False

            for batch in range(self.num_batches):
                channels.send_all(Evaluate(current_knobs, batch))

                batch_loss, batch_grad, had_particle_loss = self._collect_batch_results(channels)
                epoch_loss += batch_loss
                epoch_grad += batch_grad
                epoch_had_particle_loss = epoch_had_particle_loss or had_particle_loss

                # Update knobs after each batch (only when no particle loss this epoch)
                if not epoch_had_particle_loss:
                    current_knobs = self._update_knobs(
                        current_knobs,
                        batch_grad / total_turns,
                        lr,
                    )

            if epoch_had_particle_loss:
                LOGGER.warning(
                    "Epoch %d: particle loss detected — rejecting knob updates and restoring "
                    "pre-epoch parameters",
                    epoch,
                )
                current_knobs = pre_epoch_knobs

            # Keep training loss on a single-worker scale by averaging over batches.
            epoch_loss /= max(1, self.num_batches)
            epoch_grad /= total_turns

            grad_norm = float(np.linalg.norm(epoch_grad[epoch_grad != 0.0]))
            self._update_smoothed_grad_norm(grad_norm)

            # Calculate relative differences for rejection logic
            sum_true_diff = self._calculate_diff(current_knobs)

            validation_loss = (
                validation_loss_fn(current_knobs) if validation_loss_fn is not None else None
            )
            stop_loss = validation_loss if validation_loss is not None else epoch_loss

            new_best = False
            if self._is_new_best(stop_loss, prev_loss, sum_true_diff):
                self.best.record(stop_loss, current_knobs.copy())
                new_best = True

            hook_note = None
            if epoch_end_hook is not None:
                hook_note = epoch_end_hook(current_knobs, self.best.value)

            stop_for_loss_change = self._should_stop_for_loss_change(epoch, stop_loss, prev_loss)
            if not stop_for_loss_change:
                prev_loss = stop_loss
                last_completed_epoch = epoch

            stop_for_grad_norm = self.smoothed_grad_norm < self.gradient_converged_value
            saved_checkpoint = False
            if (
                not stop_for_loss_change
                and not stop_for_grad_norm
                and checkpointer.should_save_periodic(epoch)
            ):
                checkpointer.save(epoch, current_knobs, prev_loss)
                saved_checkpoint = True

            self._log_epoch_stats(
                writer,
                epoch,
                epoch_loss,
                grad_norm,
                lr,
                epoch_start,
                run_start,
                current_knobs,
                sum_true_diff,
                new_best,
                saved_checkpoint,
                validation_loss,
                hook_note,
            )
            if loss_callback is not None:
                loss_callback(epoch, epoch_loss, validation_loss, grad_norm, sum_true_diff)

            if stop_for_loss_change:
                LOGGER.info(f"\nLoss change below threshold. Stopping early at epoch {epoch}.")
                stop_reason = "loss_converged"
                break

            if stop_for_grad_norm:
                LOGGER.info(
                    f"\nGradient norm below threshold: {self.smoothed_grad_norm:.3e}. Stopping early at epoch {epoch}."
                )
                stop_reason = "gradient_converged"
                break
        if checkpointer.should_save_final(last_completed_epoch):
            checkpointer.save(last_completed_epoch, current_knobs, prev_loss)
        self.diagnostics = FitDiagnostics(
            converged=stop_reason != "max_epochs",
            reason=stop_reason,
            iterations=epochs_run,
            best_loss=self.best.loss,
        )

        return self.best.value

    def _collect_batch_results(
        self, channels: WorkerChannels
    ) -> tuple[float, np.ndarray, bool]:
        """Sum the workers' gradients for one batch and average their losses.

        Returns ``(loss, gradient, had_particle_loss)``; on particle loss the caller
        rejects the knob update for the enclosing epoch.
        """
        reduced = reduce_replies(channels.recv_all(), len(self.knob_names))
        return reduced.loss / len(channels.workers), reduced.grad, reduced.particle_lost

    def _update_knobs(
        self, current_knobs: dict[str, float], agg_grad: np.ndarray, lr: float
    ) -> dict[str, float]:
        """Update knob values using the optimiser."""
        param_vec = np.array([current_knobs[k] for k in self.knob_names])
        new_vec = self.optimiser.step(param_vec, agg_grad, lr)
        return dict(zip(self.knob_names, new_vec))

    def _update_smoothed_grad_norm(self, grad_norm: float) -> None:
        """Update the exponential moving average of the gradient norm."""
        if self.smoothed_grad_norm == 0.0:  # Exact 0 case for first update
            self.smoothed_grad_norm = grad_norm
        else:
            self.smoothed_grad_norm = (
                self.grad_norm_alpha * self.smoothed_grad_norm
                + (1.0 - self.grad_norm_alpha) * grad_norm
            )

    def _calculate_diff(self, current_knobs: dict[str, float]) -> float:
        """Calculate sum of absolute and relative differences from true strengths.

        Returns:
            Tuple of (sum_true_diff, sum_rel_diff)
        """
        if not self.use_true_strengths:
            return sum(current_knobs.values())

        true_diff = [abs(current_knobs[k] - self.true_strengths[k]) for k in self.knob_names]

        return np.sum(true_diff)

    def _log_epoch_stats(
        self,
        writer: SummaryWriter | None,
        epoch: int,
        loss: float,
        grad_norm: float,
        lr: float,
        epoch_start: float,
        run_start: float,
        current_knobs: dict[str, float],
        sum_true_diff: float = 0.0,
        new_best: bool = False,
        saved_checkpoint: bool = False,
        validation_loss: float | None = None,
        hook_note: str | None = None,
    ) -> None:
        """Log statistics for the current epoch.

        ``hook_note`` is whatever the epoch-end hook returned, appended to the
        line as-is. It is how a caller whose hook changes the run -- refreshing
        the workers' initial conditions, say -- gets that change onto the same
        line as the loss it moved, instead of into a second stream the reader has
        to interleave by hand.
        """
        # Log scalars to TensorBoard
        if writer is not None:
            loss_scalars = {"train": loss}
            if validation_loss is not None:
                loss_scalars["validation"] = validation_loss
            writer.add_scalars("loss", loss_scalars, epoch)

            scalars = {
                "grad_norm": grad_norm,
                "learning_rate": lr,
                "sum_true_diff": sum_true_diff,
            }
            for key, value in scalars.items():
                writer.add_scalar(key, value, epoch)
            writer.flush()

        # Calculate times
        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start

        # Build log message
        parts = [f"Ep {epoch}: loss={loss:.3e}"]
        if validation_loss is not None:
            parts.append(f"val={validation_loss:.3e}")
        parts.append(f"g={grad_norm:.3e}")
        parts.append(f"td={sum_true_diff:.3e}")
        if hook_note:
            parts.append(hook_note)
        parts.append(f"lr={lr:.2e}, et={epoch_time:.1f}s, tt={total_time:.1f}s")
        message = ", ".join(parts)

        if new_best:
            message += " [b]"
        if saved_checkpoint:
            message += " [s]"
        LOGGER.info(f"\r{message}")
