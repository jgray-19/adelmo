"""The parent <-> worker protocol: typed messages and the parent-side channels that carry them.

A worker receives :class:`Start` once, then any number of :class:`Evaluate` and
:class:`Command` messages, then :class:`Stop`. Each :class:`Evaluate` is answered
with a :class:`GradReply`, each :class:`Command` with an :class:`Ack` or a
:class:`LossReply`. A failure is answered with an :class:`ErrorReply`, which the
receiving :class:`WorkerChannels` raises.
"""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterable
    from multiprocessing import Process
    from multiprocessing.connection import Connection

    import numpy as np


# ---------------------------------------------------------------------------
# Parent -> worker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Start:
    """The first message: the model values the worker builds its MAD-NG session with."""

    knobs: dict[str, float]


@dataclass(frozen=True)
class Evaluate:
    """Evaluate the loss and, unless ``loss_only``, its knob derivatives at ``knobs``.

    ``batch`` selects a tracking worker's particle batch. ``loss_only`` asks a
    closed-orbit worker for the loss alone, from plain closed-orbit solves without
    knob derivatives (about 6x cheaper); the LM fit uses it for trial points.
    """

    knobs: dict[str, float]
    batch: int = 0
    loss_only: bool = False


class CommandKind(Enum):
    """Control commands a tracking worker answers between evaluations."""

    #: Total and per-point loss at ``payload["knobs"]``; answered with a :class:`LossReply`.
    DIAGNOSE = auto()
    #: Held-out loss at ``payload["knobs"]``; answered with a :class:`LossReply`.
    VALIDATE = auto()
    #: Install ``payload["keep_bpm_mask"]`` and ``payload["disable_worker"]``.
    APPLY_MASK = auto()
    #: Load ``payload["knobs"]`` into the model without evaluating.
    SET_KNOBS = auto()
    #: Replace the start coordinates ``payload["x" | "px" | "y" | "py"]``.
    UPDATE_INIT_COORDS = auto()
    #: Whether to propagate uncertainty on :class:`Stop` (``payload["enabled"]``).
    SET_UNCERTAINTY_MODE = auto()


@dataclass(frozen=True)
class Command:
    """A control command and its arguments."""

    kind: CommandKind
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Stop:
    """End the worker. A tracking worker first answers with its uncertainty part."""


STOP = Stop()


# ---------------------------------------------------------------------------
# Worker -> parent
# ---------------------------------------------------------------------------


@dataclass
class GradReply:
    """One evaluation's result.

    ``loss`` is NaN when the worker lost its particles or its closed orbit, and the
    caller rejects the step. ``hessian`` is the normalised Gauss-Newton Hessian the
    LM step uses and ``normal`` the physical ``JᵀWJ``; tracking workers send
    neither. ``extra`` carries worker-specific results: a calibrated worker's block
    header, or the orbit a reference worker published.
    """

    worker_id: int
    loss: float
    grad: np.ndarray
    hessian: np.ndarray | None = None
    normal: np.ndarray | None = None
    extra: Any = None


@dataclass
class LossReply:
    """A loss without derivatives; ``loss`` is ``None`` for a worker disabled by screening."""

    worker_id: int
    loss: float | None
    per_point: np.ndarray | None = None


@dataclass(frozen=True)
class Ack:
    """A control command was applied."""

    worker_id: int


@dataclass(frozen=True)
class ErrorReply:
    """The worker failed during ``phase``."""

    worker_id: int
    phase: str
    error_type: str
    error: str
    traceback: str

    def raise_error(self, worker: Process | None = None) -> NoReturn:
        """Raise this failure in the parent."""
        tb_section = f"\n\nWorker traceback:\n{self.traceback}" if self.traceback else ""
        exitcode = None if worker is None else worker.exitcode
        raise RuntimeError(
            f"Worker {self.worker_id} failed during {self.phase} (exitcode={exitcode}): "
            f"{self.error_type}: {self.error}{tb_section}"
        )


# ---------------------------------------------------------------------------
# Parent-side channels
# ---------------------------------------------------------------------------


class WorkerChannels:
    """Reusable worker communication state for fast send/receive rounds."""

    __slots__ = ("parent_conns", "workers", "_count")

    def __init__(self, parent_conns: list[Connection], workers: list[Process]) -> None:
        if len(parent_conns) != len(workers):
            raise ValueError(
                f"Connection/worker count mismatch: {len(parent_conns)} != {len(workers)}"
            )
        self.parent_conns = tuple(parent_conns)
        self.workers = tuple(workers)
        self._count = len(self.parent_conns)

    def send_all(self, message: object) -> None:
        """Send one message to every worker."""
        payload = ForkingPickler.dumps(message)
        for conn in self.parent_conns:
            conn.send_bytes(payload)

    @staticmethod
    def _recv(conn: Connection, worker: Process) -> object:
        try:
            payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"Worker process {worker.pid} closed its pipe before sending a response "
                f"(exitcode={worker.exitcode})"
            ) from exc
        if isinstance(payload, ErrorReply):
            payload.raise_error(worker)
        return payload

    def send_to(self, index: int, message: object) -> None:
        """Send one message to the worker at ``index``."""
        self.parent_conns[index].send(message)

    def recv_all(self) -> list[Any]:
        """Receive one message from each worker, preserving connection order."""
        return self.recv_some(range(self._count))

    def recv_some(self, indices: Iterable[int]) -> list[Any]:
        """Receive one message from each listed worker, in the order of ``indices``."""
        indices = list(indices)
        if not indices:
            return []
        if len(indices) == 1:
            conn = self.parent_conns[indices[0]]
            wait([conn])
            return [self._recv(conn, self.workers[indices[0]])]

        position = {self.parent_conns[i]: k for k, i in enumerate(indices)}  # connection -> slot in results
        results: list[object] = [None] * len(indices)
        seen = bytearray(len(indices))
        remaining = len(indices)

        while remaining:
            for conn in wait(list(position)):
                k = position[conn]
                if seen[k]:
                    continue
                results[k] = self._recv(conn, self.workers[indices[k]])
                seen[k] = 1
                remaining -= 1

        return results

    def ack_all(self, command: Command) -> None:
        """Send ``command`` to every worker and require an :class:`Ack` from each."""
        self.send_all(command)
        for reply in self.recv_all():
            if not isinstance(reply, Ack):
                raise RuntimeError(f"Unexpected reply to {command.kind.name}: {reply!r}")


#: pymadng waits on each MAD process with ``select()``, which rejects fd numbers >= 1024.
FD_SETSIZE = 1024
#: Descriptors a forked worker leaves open in every later worker (pipe pair, both ends).
FDS_PER_WORKER = 4
#: Descriptors the main process and each MAD-NG process need before any worker exists.
FD_RESERVE = 64


def machine_worker_limit() -> tuple[int, str]:
    """Most MAD-NG workers this machine can run at once, and what limits it.

    CPU count is the working limit; the open-file bound only matters on a host with
    more cores than ``FD_SETSIZE`` allows workers.
    """
    cpus = len(os.sched_getaffinity(0))
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    fd_room = min(FD_SETSIZE, soft if soft != resource.RLIM_INFINITY else FD_SETSIZE)
    open_files = max(1, (fd_room - FD_RESERVE) // FDS_PER_WORKER)
    return (cpus, "cpus") if cpus <= open_files else (open_files, "open files")


def distribute(costs: list[int], n_groups: int) -> list[list[int]]:
    """Split item indices into ``n_groups`` of near-equal total cost (longest first)."""
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    totals = [0] * n_groups
    for index in sorted(range(len(costs)), key=lambda i: -costs[i]):
        target = totals.index(min(totals))
        groups[target].append(index)
        totals[target] += costs[index]
    return [sorted(group) for group in groups if group]
