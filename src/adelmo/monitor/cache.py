"""Computing optics from logged knobs in the background, and keeping the results."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import queue
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from adelmo.monitor.source import MonitorSpec, OpticsSource

LOGGER = logging.getLogger(__name__)

#: Priorities: lower runs first.
REFERENCES, SELECTED, LATEST, BACKFILL = 0, 1, 2, 10


def refinement_order(items: Sequence[Any]) -> list[Any]:
    """``items`` ordered first, last, then the midpoints of ever finer halves.

    Computing a history in this order gives an evenly spread picture early,
    which fills in as more is done.
    """
    n = len(items)
    if n <= 2:
        return list(items)
    order, seen = [0, n - 1], {0, n - 1}
    halves = deque([(0, n - 1)])
    while halves:
        lo, hi = halves.popleft()
        if hi - lo < 2:
            continue
        mid = (lo + hi) // 2
        if mid not in seen:
            order.append(mid)
            seen.add(mid)
        halves.extend([(lo, mid), (mid, hi)])
    return [items[i] for i in order]


def _digest(*parts: bytes) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part)
    return h.hexdigest()[:20]


@dataclass(order=True)
class _Task:
    priority: int
    order: int
    model_key: str = field(compare=False)
    metadata: Mapping[str, Any] = field(compare=False)
    #: ``None`` for the references, else the knobs to compute optics for.
    knobs: Mapping[str, float] | None = field(compare=False, default=None)
    key: str = field(compare=False, default="")


class OpticsCache:
    """Computes references, optics and summaries with the spec's source, on worker threads.

    Results are written below ``directory`` (one folder per model, one file per
    distinct set of knob values) and so survive restarts of the monitor. Reads
    never block: anything not computed yet is queued and ``None`` returned.
    Nothing is kept in memory after it is read unless ``memory`` asks for that
    many tables to be (the monitor does not: it rereads from disk to draw).
    """

    def __init__(self, spec: MonitorSpec, directory: Path, *, workers: int = 1, memory: int = 0):
        if spec.make_source is None:
            raise ValueError("OpticsCache needs a MonitorSpec with make_source")
        self.spec = spec
        self.directory = Path(directory)
        self._queue: queue.PriorityQueue[_Task] = queue.PriorityQueue()
        self._order = itertools.count()
        self._queued: dict[str, int] = {}  # key -> best priority queued
        self._lock = threading.Lock()
        self._tables: OrderedDict[Path, Any] = OrderedDict()
        self._memory = memory
        self._local = threading.local()
        #: The last error per key, shown in the monitor instead of retrying forever.
        self.errors: dict[str, str] = {}
        self.busy = 0
        for i in range(workers):
            threading.Thread(target=self._work, name=f"optics-{i}", daemon=True).start()

    # -- keys and paths ----------------------------------------------------------------

    def model_key(self, metadata: Mapping[str, Any]) -> str:
        """Identifies the model: the stage metadata and the spec version."""
        text = json.dumps(metadata, sort_keys=True, default=str) + self.spec.version
        return _digest(text.encode())

    @staticmethod
    def knobs_key(knobs: pd.Series) -> str:
        """Identifies a set of knob values, whatever step or stage they come from."""
        return _digest(
            "\0".join(knobs.index).encode(), np.ascontiguousarray(knobs.to_numpy(float)).tobytes()
        )

    def _path(self, model_key: str, name: str) -> Path:
        return self.directory / model_key / name

    # -- reads -------------------------------------------------------------------------

    def has_references(self, metadata: Mapping[str, Any]) -> bool:
        """Whether the references are on disk; queues them if not."""
        model_key = self.model_key(metadata)
        if self._path(model_key, "references.pkl").exists():
            return True
        self._submit(_Task(REFERENCES, next(self._order), model_key, metadata, key=model_key))
        return False

    def references(self, metadata: Mapping[str, Any]) -> dict[str, pd.DataFrame] | None:
        if not self.has_references(metadata):
            return None
        return self._load(self._path(self.model_key(metadata), "references.pkl"))

    def request(self, metadata: Mapping[str, Any], knobs: pd.Series, priority: int) -> bool:
        """Queue the optics for ``knobs`` at ``priority`` unless done; True when done."""
        model_key = self.model_key(metadata)
        key = self.knobs_key(knobs)
        if self._path(model_key, f"{key}.pkl").exists():
            return True
        self._submit(_Task(priority, next(self._order), model_key, metadata, knobs.to_dict(), key))
        return False

    def optics(self, metadata: Mapping[str, Any], knobs: pd.Series, priority: int) -> pd.DataFrame | None:
        """The optics for ``knobs``, or ``None`` after queueing them at ``priority``."""
        if not self.request(metadata, knobs, priority):
            return None
        return self._load(self._path(self.model_key(metadata), f"{self.knobs_key(knobs)}.pkl"))

    def has_summary(self, metadata: Mapping[str, Any], knobs: pd.Series) -> bool:
        """Whether the summary for ``knobs`` is on disk; never queues anything."""
        return self._path(self.model_key(metadata), f"{self.knobs_key(knobs)}.json").exists()

    def summary(self, metadata: Mapping[str, Any], knobs: pd.Series) -> dict[str, float] | None:
        """The summary for ``knobs`` if computed; never queues anything."""
        path = self._path(self.model_key(metadata), f"{self.knobs_key(knobs)}.json")
        return json.loads(path.read_text()) if path.exists() else None

    def error(self, metadata: Mapping[str, Any], knobs: pd.Series | None = None) -> str | None:
        key = self.model_key(metadata) if knobs is None else self.knobs_key(knobs)
        return self.errors.get(key)

    @property
    def pending(self) -> int:
        return len(self._queued) + self.busy

    def _load(self, path: Path) -> Any:
        if self._memory <= 0:
            return pd.read_pickle(path)
        with self._lock:
            if path in self._tables:
                self._tables.move_to_end(path)
                return self._tables[path]
        value = pd.read_pickle(path)
        with self._lock:
            self._tables[path] = value
            while len(self._tables) > self._memory:
                self._tables.popitem(last=False)
        return value

    # -- the work ----------------------------------------------------------------------

    def _submit(self, task: _Task) -> None:
        with self._lock:
            if task.key in self.errors:
                return
            queued = self._queued.get(task.key)
            if queued is not None and queued <= task.priority:
                return
            self._queued[task.key] = task.priority
        self._queue.put(task)

    def _source(self, task: _Task) -> OpticsSource:
        """This thread's source for the task's model, built on first use."""
        sources = self._local.__dict__.setdefault("sources", {})
        if task.model_key not in sources:
            LOGGER.info("Building optics source for model %s", task.model_key)
            sources[task.model_key] = self.spec.make_source(task.metadata)
        return sources[task.model_key]

    def _work(self) -> None:
        while True:
            task = self._queue.get()
            with self._lock:
                if self._queued.get(task.key) != task.priority:
                    continue  # superseded by a higher-priority copy, or already done
                del self._queued[task.key]
                self.busy += 1
            try:
                self._run(task)
            except Exception as exc:
                LOGGER.exception("Optics computation failed for %s", task.key)
                self.errors[task.key] = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self.busy -= 1

    def _run(self, task: _Task) -> None:
        folder = self.directory / task.model_key
        folder.mkdir(parents=True, exist_ok=True)
        source = self._source(task)
        if task.knobs is None:
            _write_pickle(folder / "references.pkl", source.references())
            (folder / "metadata.json").write_text(json.dumps(task.metadata, indent=2, default=str))
            return
        optics = source.optics(task.knobs)
        summary = source.summary(optics) if hasattr(source, "summary") else {}
        tmp = folder / f"{task.key}.json.tmp"
        tmp.write_text(json.dumps(summary, default=float))
        tmp.replace(folder / f"{task.key}.json")  # atomic: summary() may be reading concurrently
        _write_pickle(folder / f"{task.key}.pkl", optics)


def _write_pickle(path: Path, value: Any) -> None:
    """Pickle ``value`` atomically, so a reader never loads half a file."""
    tmp = path.with_suffix(".tmp")
    pd.to_pickle(value, tmp)
    tmp.replace(path)
