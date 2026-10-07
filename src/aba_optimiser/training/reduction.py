"""Summing the workers' replies to one evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.workers.protocol import GradReply

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class Reduced:
    """The sum over the workers that returned a valid loss.

    ``hessian`` and ``normal`` are ``None`` when the workers send none (tracking).
    ``particle_lost`` is set when any worker lost its particles or closed orbit;
    those workers are left out of every sum and of ``n_valid``.
    """

    loss: float
    grad: np.ndarray
    hessian: np.ndarray | None
    normal: np.ndarray | None
    n_valid: int
    particle_lost: bool


def reduce_replies(replies: Sequence[object], n_knobs: int) -> Reduced:
    """Sum the :class:`~aba_optimiser.workers.protocol.GradReply` of every worker.

    A NaN loss marks a worker that lost its particles or closed orbit; an infinite
    one a worker error, which is raised.
    """
    if not replies:
        raise RuntimeError("No workers returned results")
    with_curvature = isinstance(replies[0], GradReply) and replies[0].hessian is not None
    loss = 0.0
    grad = np.zeros(n_knobs, dtype=float)
    hessian = np.zeros((n_knobs, n_knobs)) if with_curvature else None
    normal = np.zeros((n_knobs, n_knobs)) if with_curvature else None
    n_valid = 0
    particle_lost = False
    for reply in replies:
        if not isinstance(reply, GradReply):
            raise RuntimeError(f"Unexpected worker reply: {reply!r}")
        if reply.loss == float("inf"):
            raise RuntimeError("Worker error detected during optimisation")
        if math.isnan(float(reply.loss)):
            particle_lost = True
            continue
        grad += np.asarray(reply.grad, dtype=float).ravel()
        if with_curvature:
            hessian += np.asarray(reply.hessian, dtype=float)
            normal += np.asarray(reply.normal, dtype=float)
        loss += float(reply.loss)
        n_valid += 1
    return Reduced(loss, grad, hessian, normal, n_valid, particle_lost)
