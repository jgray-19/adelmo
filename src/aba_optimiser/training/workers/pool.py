"""A pool of worker processes and the parent-side state kept for each one."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aba_optimiser.workers.protocol import STOP, WorkerChannels

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection

    from aba_optimiser.training.workers.setup import WorkerRuntimeMetadata

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkerPool:
    """Worker processes, their pipes and, for tracking workers, their metadata.

    ``metadata`` and ``particle_counts`` are parallel to ``workers`` when filled;
    pools whose workers need neither (closed-twiss) leave them empty.
    """

    conns: list[Connection] = field(default_factory=list)
    workers: list[mp.Process] = field(default_factory=list)
    metadata: list[WorkerRuntimeMetadata] = field(default_factory=list)
    particle_counts: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.workers)

    def spawn(self, worker_class: Callable[..., mp.Process], worker_id: int, *args: object) -> Connection:
        """Start ``worker_class(child_conn, worker_id, *args)`` and return the parent pipe end."""
        parent, child = mp.Pipe()
        worker = worker_class(child, worker_id, *args)
        worker.start()
        self.conns.append(parent)
        self.workers.append(worker)
        return parent

    @property
    def channels(self) -> WorkerChannels:
        """Channels over every worker in the pool."""
        if not self.workers:
            raise RuntimeError("Worker pool is empty")
        return WorkerChannels(self.conns, self.workers)

    def stop(self) -> None:
        """Send :data:`~aba_optimiser.workers.protocol.STOP`, join (forcing stragglers) and close the pipes."""
        for conn in self.conns:
            with contextlib.suppress(OSError, EOFError):
                conn.send(STOP)
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                LOGGER.warning("Worker %s did not terminate, forcing...", worker.name)
                worker.terminate()
                worker.join()
        for conn in self.conns:
            conn.close()

    def kill(self) -> None:
        """Terminate every worker without waiting for it to respond."""
        for worker in self.workers:
            worker.terminate()
            worker.join()
