"""Measure map for BPM gain and corrector kick calibration in closed-orbit fits.

A measured orbit change is the model's change at the kick ``nominal + (1 + g_corrector) * delta`` (the corrector calibration acts on
the kick put into the lattice), scaled per BPM and plane by ``(1 + b)`` (a readout error). The corrector kick is a parameter of the
MAD-NG damap, so ``d(orbit)/d(kick)`` is exact and the orbit's nonlinearity in the kick is kept; the BPM gains are linear scalings
and their columns are analytic. BPM gain columns are one-hot (one column per BPM and plane), so the normal matrix is block-structured and the
BPM gains are eliminated by a Schur complement (:func:`schur_step`) instead of carrying a dense matrix.

Parameter blocks: ``u = (q, g)`` (magnet knobs then corrector gains, dense) and ``b`` (BPM gains, diagonal Hessian).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Gain parameter names: ``bpmgain.<x|y>.<BPM>`` and ``corrgain.<CORRECTOR>`` (upper-case, as MAD-NG names them).
PLANES = ("x", "y")


def bpm_gain_name(plane: str, bpm: str) -> str:
    return f"bpmgain.{plane}.{bpm}"


def corrector_gain_name(corrector: str) -> str:
    return f"corrgain.{corrector}"


def corrector_gain_knob_name(corrector: str) -> str:
    """Model knob (constant part 0) added to a corrector's ``1 + g``, so its Jacobian column is d(orbit)/d(g)."""
    return f"dg.{corrector}"


@dataclass(frozen=True)
class CalibrationSpec:
    """Global gain layout shared by every series: BPM names (both planes) and corrector names."""

    bpms: tuple[str, ...]
    correctors: tuple[str, ...]


class CalibrationBlocks:
    """Summed loss, gradient and Gauss-Newton Hessian blocks (``2 JᵀWJ`` convention, unnormalised)."""

    def __init__(self, n_q: int, n_g: int, n_b: int) -> None:
        self.n_q, self.n_g, self.n_b = n_q, n_g, n_b
        n_u = n_q + n_g
        self.loss = 0.0
        self.grad_u = np.zeros(n_u)
        self.hess_uu = np.zeros((n_u, n_u))
        self.hess_ub = np.zeros((n_u, n_b))
        self.hess_bb = np.zeros(n_b)
        self.grad_b = np.zeros(n_b)

    def add(self, other: CalibrationBlocks) -> None:
        self.loss += other.loss
        self.grad_u += other.grad_u
        self.hess_uu += other.hess_uu
        self.hess_ub += other.hess_ub
        self.hess_bb += other.hess_bb
        self.grad_b += other.grad_b

    def scale(self, factor: float) -> None:
        self.loss *= factor
        self.grad_u *= factor
        self.hess_uu *= factor
        self.hess_ub *= factor
        self.hess_bb *= factor
        self.grad_b *= factor

    ARRAYS = ("grad_u", "hess_uu", "hess_ub", "hess_bb", "grad_b")

    def to_payload(self) -> dict:
        """The scalars; the arrays travel as raw bytes (:meth:`send_arrays`): pickling them copies ~170 MB twice."""
        return {"n_q": self.n_q, "n_g": self.n_g, "n_b": self.n_b, "loss": self.loss}

    @classmethod
    def from_payload(cls, payload: dict) -> CalibrationBlocks:
        blocks = cls(payload["n_q"], payload["n_g"], payload["n_b"])
        blocks.loss = payload["loss"]
        return blocks

    def send_arrays(self, conn) -> None:
        for name in self.ARRAYS:
            conn.send_bytes(memoryview(getattr(self, name)).cast("B"))

    def recv_arrays(self, conn) -> None:
        for name in self.ARRAYS:
            conn.recv_bytes_into(memoryview(getattr(self, name)).cast("B"))


def add_series(
    blocks: CalibrationBlocks,
    model: np.ndarray,
    jacobian: np.ndarray,
    d_gain: np.ndarray,
    targets,
    weights,
    bpm_gain: np.ndarray,
    bpm_cols: np.ndarray,
    corrector_indices: list[int],
) -> None:
    """Add one series' contribution.

    ``model`` is the orbit change with every corrector kick scaled by ``1 + g`` (reference subtracted) and ``jacobian`` its
    quad derivatives; ``d_gain[..., j]`` is its derivative with respect to the gain of corrector ``corrector_indices[j]``.
    Only the BPM gains scale the model: ``(1 + b) * model`` against the target.
    """
    n_q = blocks.n_q
    g_rows = n_q + np.asarray(corrector_indices, dtype=int)
    for o in range(model.shape[0]):
        scale = 1.0 + bpm_gain[o]
        mp = scale * model[o]
        jac = jacobian[o]  # not scaled: the BPM gain scale is folded into the weights below, so no copy of the Jacobian
        d_g = scale[:, None] * d_gain[o]
        w = np.asarray(weights[o], dtype=float)
        w_scaled = w * scale**2
        r = np.where(w > 0.0, mp - np.asarray(targets[o], dtype=float), 0.0)
        wr = w * r
        d_b = model[o]
        cols = bpm_cols[o]

        blocks.loss += float(np.sum(w * r * r))
        blocks.grad_u[:n_q] += 2.0 * (scale * wr) @ jac
        two_w_jac = (2.0 * w_scaled)[:, None] * jac  # 2 (J^T w scale^2) transposed
        blocks.hess_uu[:n_q, :n_q] += two_w_jac.T @ jac
        cross = two_w_jac.T @ d_gain[o]
        del two_w_jac
        blocks.grad_u[g_rows] += 2.0 * wr @ d_g
        blocks.hess_uu[np.ix_(g_rows, g_rows)] += 2.0 * (d_g.T * w) @ d_g
        blocks.hess_uu[:n_q, g_rows] += cross
        blocks.hess_uu[g_rows, :n_q] += cross.T
        blocks.grad_b[cols] += 2.0 * wr * d_b
        blocks.hess_bb[cols] += 2.0 * w * d_b * d_b
        blocks.hess_ub[:n_q, cols] += jac.T * (2.0 * scale * w * d_b)
        blocks.hess_ub[np.ix_(g_rows, cols)] += 2.0 * (d_g * (w * d_b)[:, None]).T


def apply_prior(
    blocks: CalibrationBlocks,
    u: np.ndarray,
    b: np.ndarray,
    sigma_g: float,
    sigma_b: float,
    c: float,
    sigma_q: np.ndarray | None = None,
    *,
    loss_only: bool = False,
):
    """Gaussian priors ``c * g²/σ_g²``, ``c * b²/σ_b²`` and ``c * q²/σ_q²`` (``c`` converts χ² units to the normalised loss).

    ``sigma_q`` holds one absolute width per magnet knob (the knob's units); an entry of 0 leaves that knob free.
    ``loss_only`` adds the priors to ``blocks.loss`` alone (``blocks`` then needs only ``loss`` and ``n_q``): a loss-only trial
    evaluation has no gradient or Hessian blocks.
    """
    n_q = blocks.n_q
    if sigma_q is not None:
        inv_var = np.zeros(n_q)
        np.divide(1.0, sigma_q**2, out=inv_var, where=sigma_q > 0)
        q = u[:n_q]
        blocks.loss += c * float(q @ (inv_var * q))
        if not loss_only:
            blocks.grad_u[:n_q] += 2.0 * c * inv_var * q
            blocks.hess_uu[np.diag_indices(n_q)] += 2.0 * c * inv_var
    if sigma_g > 0:
        g = u[n_q:]
        blocks.loss += c * float(g @ g) / sigma_g**2
        if not loss_only:
            blocks.grad_u[n_q:] += 2.0 * c * g / sigma_g**2
            blocks.hess_uu[n_q:, n_q:] += np.eye(blocks.n_g) * 2.0 * c / sigma_g**2
    if sigma_b > 0:
        blocks.loss += c * float(b @ b) / sigma_b**2
        if not loss_only:
            blocks.grad_b += 2.0 * c * b / sigma_b**2
            blocks.hess_bb += 2.0 * c / sigma_b**2


def reduce_blocks(blocks: CalibrationBlocks) -> tuple[np.ndarray, np.ndarray]:
    """Schur complement over ``b``: reduced ``(gradient, Hessian)`` of ``u``."""
    inv = 1.0 / blocks.hess_bb
    ub = blocks.hess_ub
    scaled = ub * np.sqrt(inv)
    hess = blocks.hess_uu - scaled @ scaled.T
    grad = blocks.grad_u - ub @ (inv * blocks.grad_b)
    return grad, hess


def back_substitute(blocks: CalibrationBlocks, delta_u: np.ndarray) -> np.ndarray:
    """Step of ``b`` that is optimal for the step ``delta_u`` (conditional Gauss-Newton minimiser)."""
    return -(blocks.grad_b + blocks.hess_ub.T @ delta_u) / blocks.hess_bb
