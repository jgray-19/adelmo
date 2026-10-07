"""Limited-memory BFGS optimiser with an adaptive step-length heuristic."""

from __future__ import annotations

from typing import Any

import numpy as np

from aba_optimiser.optimisers.base import BaseOptimiser


class LBFGSOptimiser(BaseOptimiser):
    OPTIMISER_NAME = "lbfgs"

    """
    Limited-memory BFGS with an adaptive, line-search-free step length.

    Adaptive LR:
      lr_eff = base_lr * eta_k
      eta_k  = EMA( clip( (s^T s) / (s^T y), eta_min, eta_max ) )

    Notes
    -----
    * Two-loop recursion computes d_k = -H_k g_k; the result is scaled by lr_eff.
    * Initial Hessian scaling gamma = (s^T y)/(y^T y) is retained for conditioning.
    * The BB1 multiplier reuses the (s, y) pair already built; no extra evaluations.
    """

    def __init__(
        self,
        history_size: int = 20,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        # --- adaptive LR knobs ---
        use_adaptive_lr: bool = True,
        bb_clip: tuple[float, float] = (1e-1, 1e1),  # clip for raw BB1
        ema_beta: float = 0.8,  # smoothing of BB1
        eta_init: float = 1.0,  # initial multiplier
        # --- stability knobs ---
        max_grad_norm: float | None = 1.0,   # gradient clipping before two-loop
        max_step_norm: float | None = 1.0,   # step-vector clipping after two-loop
        powell_damping: float = 0.2,         # minimum fraction of s^T s for y^T s
    ):
        self.history_size = int(history_size)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # State needed to build (s, y)
        self.prev_params: np.ndarray | None = None
        self.prev_grads: np.ndarray | None = None

        # L-BFGS memory
        self.S: list[np.ndarray] = []  # s_i = x_{i+1} - x_i
        self.Y: list[np.ndarray] = []  # y_i = g_{i+1} - g_i
        self.RHO: list[float] = []  # 1 / (y_i^T s_i)

        # time step
        self.t = 0

        # Adaptive LR state
        self.use_adaptive_lr = bool(use_adaptive_lr)
        self.bb_min, self.bb_max = bb_clip
        self.ema_beta = float(ema_beta)
        self.eta_ema = float(eta_init)

        # Stability
        self.max_grad_norm = max_grad_norm
        self.max_step_norm = max_step_norm
        self.powell_damping = float(powell_damping)

    def _push_pair(self, s: np.ndarray, y: np.ndarray) -> None:
        ss = float(np.dot(s, s))
        ys = float(np.dot(y, s))
        yy = float(np.dot(y, y))
        if ss <= 1e-20 or yy <= 1e-20:
            return
        # Powell's damping: if y^T s is too small relative to s^T s the curvature
        # pair would make H ill-conditioned; blend y towards the steepest-descent
        # direction so that y^T s >= powell_damping * s^T s.
        threshold = self.powell_damping * ss
        if ys < threshold:
            theta = (1.0 - self.powell_damping) * ss / (ss - ys)
            y = theta * y + (1.0 - theta) * s
            ys = float(np.dot(y, s))
        if ys <= 1e-20:
            return
        if len(self.S) == self.history_size:
            self.S.pop(0)
            self.Y.pop(0)
            self.RHO.pop(0)
        self.S.append(s)
        self.Y.append(y)
        self.RHO.append(1.0 / ys)

    def _two_loop(self, g: np.ndarray) -> np.ndarray:
        """Standard two-loop recursion to compute r ≈ H * g."""
        if not self.S:
            return g.copy()  # H ≈ I at start

        q = g.copy()
        alpha = [0.0] * len(self.S)

        # First loop: descending order
        for i in range(len(self.S) - 1, -1, -1):
            s_i, y_i, rho_i = self.S[i], self.Y[i], self.RHO[i]
            alpha[i] = rho_i * np.dot(s_i, q)
            q = q - alpha[i] * y_i

        # Initial H0 scaling using the most recent pair (gamma ≈ BB2)
        s_last, y_last = self.S[-1], self.Y[-1]
        yy = float(np.dot(y_last, y_last))
        ys = float(np.dot(y_last, s_last))
        gamma = ys / yy if yy > 0 else 1.0
        r = gamma * q

        # Second loop: ascending order
        for i in range(len(self.S)):
            s_i, y_i, rho_i = self.S[i], self.Y[i], self.RHO[i]
            beta = rho_i * np.dot(y_i, r)
            r = r + s_i * (alpha[i] - beta)
        return r

    def _bb1_multiplier(self, s: np.ndarray, y: np.ndarray) -> float:
        """
        BB1 spectral step (s^T s)/(s^T y); clipped + EMA-smoothed.
        Requires a fresh (s, y) pair.
        """
        denom = float(np.dot(s, y))
        if denom <= self.eps:
            return self.eta_ema  # retain previous multiplier if curvature is poor
        raw = float(np.dot(s, s)) / denom
        # clip to avoid outliers, then EMA
        raw = min(max(raw, self.bb_min), self.bb_max)
        self.eta_ema = self.ema_beta * self.eta_ema + (1.0 - self.ema_beta) * raw
        return self.eta_ema

    def step(self, params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
        """
        Perform one L-BFGS step with adaptive LR multiplier.

        Args
        ----
        params : np.ndarray
            Current parameter vector x_k.
        grads : np.ndarray
            Current gradient g_k = ∇f(x_k).
        lr : float
            Base learning rate (e.g. from your cosine scheduler).
        """
        self.t += 1

        # Weight decay (L2 regularisation)
        g = grads + (self.weight_decay * params if self.weight_decay != 0 else 0)

        # Gradient clipping — keeps noisy gradients from corrupting the (s,y) history
        if self.max_grad_norm is not None:
            g_norm = float(np.linalg.norm(g))
            if g_norm > self.max_grad_norm:
                g = g * (self.max_grad_norm / g_norm)

        # On k ≥ 1, build (s_{k-1}, y_{k-1}) from the PREVIOUS state
        eta_mult = self.eta_ema
        if self.prev_params is not None and self.prev_grads is not None:
            s = params - self.prev_params
            y = g - self.prev_grads
            self._push_pair(s, y)
            if self.use_adaptive_lr:
                eta_mult = self._bb1_multiplier(s, y)

        # Compute quasi-Newton direction: d = - H_k * g_k
        Hg = self._two_loop(g)  # noqa: N806
        d = -Hg

        # Step-norm clipping — prevents huge jumps when the Hessian estimate is poor
        if self.max_step_norm is not None:
            d_norm = float(np.linalg.norm(d))
            if d_norm > self.max_step_norm:
                d = d * (self.max_step_norm / d_norm)

        # Take step with adaptive multiplier (no line search)
        lr_eff = lr * eta_mult
        new_params = params + lr_eff * d

        # Save state for the next call
        self.prev_params = params.copy()
        self.prev_grads = g.copy()

        return new_params

    def state_to_dict(self) -> dict[str, Any]:
        """Return optimiser internal state as a serialisable dictionary."""
        return {
            "type": self.OPTIMISER_NAME,
            "history_size": int(self.history_size),
            "eps": float(self.eps),
            "weight_decay": float(self.weight_decay),
            "t": int(self.t),
            "S": [arr.tolist() for arr in self.S],
            "Y": [arr.tolist() for arr in self.Y],
            "RHO": [float(v) for v in self.RHO],
            "prev_params": None if self.prev_params is None else self.prev_params.tolist(),
            "prev_grads": None if self.prev_grads is None else self.prev_grads.tolist(),
            "use_adaptive_lr": bool(self.use_adaptive_lr),
            "bb_min": float(self.bb_min),
            "bb_max": float(self.bb_max),
            "ema_beta": float(self.ema_beta),
            "eta_ema": float(self.eta_ema),
            "max_grad_norm": self.max_grad_norm,
            "max_step_norm": self.max_step_norm,
            "powell_damping": float(self.powell_damping),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimiser internal state from a dictionary."""
        if state.get("type") != self.OPTIMISER_NAME:
            raise ValueError(
                f"State type {state.get('type')} does not match {self.OPTIMISER_NAME}"
            )

        self.history_size = int(state["history_size"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.t = int(state["t"])

        self.S = [np.array(arr, dtype=float) for arr in state["S"]]
        self.Y = [np.array(arr, dtype=float) for arr in state["Y"]]
        self.RHO = [float(v) for v in state["RHO"]]
        prev_params = state.get("prev_params")
        prev_grads = state.get("prev_grads")
        self.prev_params = None if prev_params is None else np.array(prev_params, dtype=float)
        self.prev_grads = None if prev_grads is None else np.array(prev_grads, dtype=float)

        self.use_adaptive_lr = bool(state["use_adaptive_lr"])
        self.bb_min = float(state["bb_min"])
        self.bb_max = float(state["bb_max"])
        self.ema_beta = float(state["ema_beta"])
        self.eta_ema = float(state["eta_ema"])
        self.max_grad_norm = state.get("max_grad_norm", 1.0)
        self.max_step_norm = state.get("max_step_norm", 1.0)
        self.powell_damping = float(state.get("powell_damping", 0.2))
