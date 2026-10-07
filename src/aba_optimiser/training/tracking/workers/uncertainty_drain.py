"""Stopping the training workers and folding their uncertainty parts into ``(A, B)``."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np
import psutil

from aba_optimiser.workers.common import (
    KickPlane,
    UncertaintyPart,
    merge_uncertainty_parts,
    noise_matrix,
)
from aba_optimiser.workers.protocol import STOP, Command, CommandKind, WorkerChannels

if TYPE_CHECKING:
    from aba_optimiser.training.pool import WorkerPool

LOGGER = logging.getLogger(__name__)


def drain_uncertainty(
    pool: WorkerPool, n_knobs: int, propagate_uncertainty: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Stop ``pool`` in memory-bounded chunks and return the normal matrix ``A`` and noise ``B``.

    Feed both to :func:`~aba_optimiser.workers.common.sandwich_uncertainties`.
    Reading ids are file-scoped, so once every worker of a file has reported, that
    file's merged readings become its ``B`` contribution and are dropped. Workers are
    drained in file order, so besides the chunk being received at most one partly
    merged file is held.
    """
    normal = np.zeros((n_knobs, n_knobs), dtype=np.float64)
    noise = np.zeros((n_knobs, n_knobs), dtype=np.float64)
    files = [meta.file_idx for meta in pool.metadata]
    order = sorted(range(len(pool)), key=files.__getitem__)
    remaining = Counter(files)
    chunk_size = _concurrency(pool, n_knobs) if propagate_uncertainty else max(1, len(order))

    pending: dict[int, UncertaintyPart] = {}
    for start in range(0, len(order), chunk_size):
        chunk = order[start : start + chunk_size]
        parts_by_file: dict[int, list[UncertaintyPart]] = defaultdict(list)
        for idx, part in zip(chunk, _drain_parts(pool, chunk, propagate_uncertainty)):
            parts_by_file[files[idx]].append(part)
        for file_idx, parts in parts_by_file.items():
            remaining[file_idx] -= len(parts)
            if file_idx in pending:
                parts.append(pending.pop(file_idx))
            merged = merge_uncertainty_parts(parts, n_knobs)
            if remaining[file_idx]:
                pending[file_idx] = merged
            else:
                normal += merged.normal
                noise += noise_matrix(merged)
    return normal, noise


def _drain_parts(
    pool: WorkerPool, indices: list[int], propagate_uncertainty: bool
) -> list[UncertaintyPart]:
    """Stop the given workers and return their parts in order."""
    workers = [pool.workers[idx] for idx in indices]
    channels = WorkerChannels([pool.conns[idx] for idx in indices], workers)
    if not propagate_uncertainty:
        channels.ack_all(Command(CommandKind.SET_UNCERTAINTY_MODE, {"enabled": False}))
    channels.send_all(STOP)
    parts = channels.recv_all()
    for part in parts:
        if not isinstance(part, UncertaintyPart):
            raise RuntimeError(f"Unexpected uncertainty payload from worker: {part!r}")
    for worker in workers:
        worker.join()
    return parts


def _concurrency(pool: WorkerPool, n_knobs: int) -> int:
    """How many workers may send their parts at once within half the free memory.

    A part has at most one row per observed point for up to two observables per
    kicked plane, plus one start row per plane and particle. Each row holds
    ``n_knobs`` sensitivities, an id and a variance, 8 bytes each.
    """
    largest = 0
    for meta, particles in zip(pool.metadata, pool.particle_counts):
        planes = 2 if meta.kick_plane == KickPlane.XY else 1
        points = len(meta.bpm_names) * meta.n_run_turns
        rows = particles * planes * (2 * points + 1)
        largest = max(largest, rows * (n_knobs + 2) * 8)
    available = psutil.virtual_memory().available
    concurrency = max(1, min(len(pool), int(0.5 * available // max(largest, 1))))
    LOGGER.info(
        "Uncertainty stage: largest worker part <= %.1f MiB, %.1f MiB available, "
        "%d worker(s) at once",
        largest / 2**20,
        available / 2**20,
        concurrency,
    )
    return concurrency
