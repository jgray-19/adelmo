"""Adam optimiser implementation used by the training loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from adelmo.optimisers.base import BaseOptimiser

if TYPE_CHECKING:
    from adelmo.config import OptimiserConfig

LOGGER = logging.getLogger(__name__)


class AdamOptimiser(BaseOptimiser):
    """Stateful Adam optimiser for dense NumPy parameter vectors.

    Instantiate once for a given parameter shape and call :meth:`step` for
    each optimisation update.
    """

    OPTIMISER_NAME = "adam"
    KNOB_VECTORS = ("m", "v")

    @classmethod
    def config_kwargs(cls, config: OptimiserConfig, n_params: int) -> dict[str, Any]:
        eps = {} if config.adam_eps is None else {"eps": config.adam_eps}
        return {
            "shape": (n_params,),
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": config.adam_weight_decay,
            **eps,
        }

    def __init__(
        self,
        shape: tuple[int, ...],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialise the Adam optimiser state.

        Args:
            shape: Shape of the parameter vector to be optimised.
            beta1: Exponential decay rate for the first moment estimates.
            beta2: Exponential decay rate for the second moment estimates.
            eps: Small term to avoid division by zero.
            weight_decay: Coefficient for L2 weight decay (adds to gradient).
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        LOGGER.debug(
            f"Initialising Adam optimiser with shape={shape}, beta1={beta1}, beta2={beta2}, eps={eps}, weight_decay={weight_decay}"
        )

        # Initialise first and second moments and timestep
        self.m = np.zeros(shape, dtype=float)
        self.v = np.zeros(shape, dtype=float)
        self.t = 0

    def step(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Perform a single optimisation step.

        Args:
            params: Current parameters as a NumPy array.
            grads: Current gradients of loss w.r.t. params.
            lr: Learning rate for this step.

        Returns:
            Updated parameters after applying the Adam update.
        """
        self.t += 1

        LOGGER.debug(f"Adam step {self.t}: lr={lr}, weight_decay={self.weight_decay}")

        grads = self.with_weight_decay(grads, params, self.weight_decay)

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Parameter update
        update = lr * (m_hat / (np.sqrt(v_hat) + self.eps))

        new_params = params - update
        update_norm = np.linalg.norm(update)
        LOGGER.debug(f"Update norm: {update_norm:.6e}")

        return new_params

    def state_dict(self) -> dict[str, Any]:
        """Return optimiser internal state as a serialisable dictionary."""
        return {
            "type": self.OPTIMISER_NAME,
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "eps": float(self.eps),
            "weight_decay": float(self.weight_decay),
            "m": self.m.tolist(),
            "v": self.v.tolist(),
            "t": int(self.t),
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        """Restore optimiser internal state from a dictionary."""
        if state.get("type") != self.OPTIMISER_NAME:
            raise ValueError(
                f"State type {state.get('type')} does not match {self.OPTIMISER_NAME}"
            )

        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.m = np.array(state["m"], dtype=float)
        self.v = np.array(state["v"], dtype=float)
        self.t = int(state["t"])
