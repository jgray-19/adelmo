"""Stopping the training workers and folding their uncertainty parts into ``(A, B)``."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import psutil

from adelmo.fitting.protocol import STOP, Command, CommandKind, WorkerChannels
from adelmo.fitting.worker import KickPlane

if TYPE_CHECKING:
    from adelmo.fitting.pool import WorkerPool
    from adelmo.tracking.dispatch.setup import WorkerRuntimeMetadata

LOGGER = logging.getLogger(__name__)


@dataclass
class UncertaintyPart:
    """One worker's contribution to the propagated knob covariance.

    ``normal`` is ``Σ w J Jᵀ``. Each row ``r`` of ``sensitivities`` is the change in
    that worker's gradient per unit noise on reading ``reading_ids[r]`` (an
    observation, a start coordinate, or both), with ``variances[r]`` its declared
    variance.
    """

    normal: np.ndarray  # Shape: (n_knobs, n_knobs)
    reading_ids: np.ndarray  # Shape: (n_rows,)
    sensitivities: np.ndarray  # Shape: (n_rows, n_knobs)
    variances: np.ndarray  # Shape: (n_rows,)

    @classmethod
    def empty(cls, n_knobs: int) -> UncertaintyPart:
        """A worker that contributes nothing (disabled, or uncertainty not requested)."""
        return cls(
            normal=np.zeros((n_knobs, n_knobs)),
            reading_ids=np.zeros(0, dtype=np.int64),
            sensitivities=np.zeros((0, n_knobs)),
            variances=np.zeros(0),
        )


def merge_uncertainty_parts(parts: list[UncertaintyPart], n_knobs: int) -> UncertaintyPart:
    """Sum the normal matrices and merge rows that share a reading id.

    ``G_e`` adds every part's sensitivity to reading ``e`` (shared between workers, or
    used as both an observation and a start), so :func:`noise_matrix` counts its noise
    once. Merging a merged part with further parts gives the same result.
    """
    normal = np.zeros((n_knobs, n_knobs), dtype=np.float64)
    for part in parts:
        normal += part.normal
    ids = np.concatenate([part.reading_ids for part in parts]) if parts else np.zeros(0, np.int64)
    if ids.size == 0:
        return UncertaintyPart(normal, ids, np.zeros((0, n_knobs)), np.zeros(0))

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    sensitivities = np.concatenate([part.sensitivities for part in parts])[order]
    return UncertaintyPart(
        normal=normal,
        reading_ids=sorted_ids[starts],
        sensitivities=np.add.reduceat(sensitivities, starts, axis=0),
        variances=np.concatenate([part.variances for part in parts])[order][starts],
    )


def noise_matrix(part: UncertaintyPart) -> np.ndarray:
    """``B = Σ_e σ_e² G_e G_eᵀ`` for a merged part; readings without a finite variance add nothing."""
    variances = np.where(np.isfinite(part.variances), part.variances, 0.0)
    return (part.sensitivities * variances[:, None]).T @ part.sensitivities


def drain_uncertainty(
    pool: WorkerPool[WorkerRuntimeMetadata], n_knobs: int, propagate_uncertainty: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Stop ``pool`` in memory-bounded chunks and return the normal matrix ``A`` and noise ``B``.

    Feed both to :func:`~adelmo.fitting.uncertainty.sandwich_uncertainties`.
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
