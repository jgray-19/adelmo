"""Shared worker communication helpers for parent-side IPC."""

from __future__ import annotations

import os
import resource
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, NoReturn, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable
    from multiprocessing import Process
    from multiprocessing.connection import Connection

#: ``batch`` slot of a closed-orbit worker message: 0 = loss, gradient and Hessian (knob derivatives), ``LOSS_ONLY`` = the loss alone,
#: from a plain closed-orbit solve without the knob parameters (about 6x cheaper). Used for trial points of the LM fit.
LOSS_ONLY = 1


class WorkerErrorPayload(TypedDict):
    """Structured worker failure payload sent across the pipe."""

    worker_id: int
    status: str
    phase: str
    error_type: str
    error: str
    traceback: str

def raise_for_worker_error_payload(
    payload: object,
    worker: Process | None = None,
) -> NoReturn:
    """Raise a RuntimeError from a worker payload."""
    if isinstance(payload, dict) and payload.get("status") == "error":
        p: WorkerErrorPayload = payload  # type: ignore[assignment]
        worker_id = p.get("worker_id", "?")
        phase = p.get("phase", "unknown")
        error_type = p.get("error_type", "Exception")
        error = p.get("error", "unknown worker error")
        tb = p.get("traceback", "")
        tb_section = f"\n\nWorker traceback:\n{tb}" if tb else ""
        raise RuntimeError(
            f"Worker {worker_id} failed during {phase}: {error_type}: {error}{tb_section}"
        )

    exitcode = None if worker is None else worker.exitcode
    raise RuntimeError(f"Unexpected worker payload: {payload!r} (worker exitcode={exitcode})")


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
        if isinstance(payload, dict) and payload.get("status") == "error":
            raise_for_worker_error_payload(payload, worker)
        return payload

    def send_to(self, index: int, message: object) -> None:
        """Send one message to the worker at ``index``."""
        self.parent_conns[index].send(message)

    def recv_all(self) -> list[object]:
        """Receive one message from each worker, preserving connection order."""
        return self.recv_some(range(self._count))

    def recv_some(self, indices: Iterable[int]) -> list[object]:
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
