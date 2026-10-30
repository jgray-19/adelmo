"""Writing a run log from inside a fit."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

STAGE_FILE = "stage.json"
HISTORY_FILE = "history.jsonl"
KNOBS_FILE = "knobs.f64"
RESULT_FILE = "result.json"

#: The step under which :meth:`RunLog.begin` records the starting knobs.
START_STEP = -1

#: Seconds between closing and reopening the append-only files. A flush only reaches
#: the local cache on AFS; the server (and so other hosts) sees the data on close.
REOPEN_SECONDS = 10.0


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically, so a reader never sees half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_jsonable))
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    """Make numpy scalars and paths JSON-serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class RunLog:
    """Records one fit in ``directory``; see :mod:`adelmo.runlog` for the files.

    Call :meth:`begin` with the starting knobs, then :meth:`log` and
    :meth:`record_knobs` once per step, :meth:`finish` with how the fit ended and
    :meth:`close` at the end. Every write is flushed straight away, so a monitor
    in another process sees each step as soon as it is logged.
    """

    def __init__(
        self,
        directory: Path,
        *,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        resume: bool = False,
    ):
        self.directory = directory
        self.resume = resume
        self.name = name
        self.metadata = dict(metadata or {})
        self.knob_names: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)
        self._history = (directory / HISTORY_FILE).open("a")
        self._knobs = (directory / KNOBS_FILE).open("ab")
        self._last_reopen = time.monotonic()

    def begin(self, knobs: Mapping[str, float], *, resume_step: int | None = None) -> None:
        """Fix the knob order to that of ``knobs`` and record them as the start.

        On a log opened with ``resume=True`` that already has a stage file, the
        existing run is continued instead: rows from ``resume_step`` on (written after
        the checkpoint being restored) are dropped, and nothing else is rewritten.
        """
        self.knob_names = list(knobs)
        stage_path = self.directory / STAGE_FILE
        if self.resume and stage_path.exists():
            recorded = json.loads(stage_path.read_text())["knob_names"]
            if recorded != self.knob_names:
                msg = f"Cannot resume {self.directory}: its knobs differ from this fit's"
                raise ValueError(msg)
            if resume_step is not None:
                self._truncate_from(resume_step)
            return
        write_json(
            self.directory / STAGE_FILE,
            {
                "name": self.name,
                "started": time.time(),
                "knob_names": self.knob_names,
                "metadata": self.metadata,
            },
        )
        self.record_knobs(START_STEP, knobs)

    def log(self, step: int, **scalars: float | bool | None) -> None:
        """Append one line of scalars for ``step``."""
        line = {"step": int(step), "time": time.time(), **scalars}
        self._history.write(json.dumps(line, default=_jsonable) + "\n")
        self._history.flush()
        self._reopen_if_due()

    def record_knobs(self, step: int, knobs: Mapping[str, float]) -> None:
        """Append the knob values after ``step``, in the order fixed by :meth:`begin`."""
        row = np.array([step, *(knobs[name] for name in self.knob_names)], dtype=np.float64)
        self._knobs.write(row.tobytes())
        self._knobs.flush()
        self._reopen_if_due()

    def _truncate_from(self, step: int) -> None:
        """Drop the history lines and knob rows of ``step`` and later."""
        self._history.close()
        self._knobs.close()
        history = self.directory / HISTORY_FILE
        kept = []
        for line in history.read_text().splitlines():
            try:
                if json.loads(line)["step"] < step:
                    kept.append(line)
            except json.JSONDecodeError:  # a line cut short by the crash
                break
        history.write_text("".join(f"{x}\n" for x in kept))
        knobs = self.directory / KNOBS_FILE
        width = len(self.knob_names) + 1
        flat = np.fromfile(knobs, dtype=np.float64)
        rows = flat[: len(flat) // width * width].reshape(-1, width)
        rows[rows[:, 0] < step].tofile(knobs)
        self._history = history.open("a")
        self._knobs = knobs.open("ab")
        self._last_reopen = time.monotonic()

    def _reopen_if_due(self) -> None:
        """Close and reopen both files every :data:`REOPEN_SECONDS`, to publish them on AFS."""
        if time.monotonic() - self._last_reopen < REOPEN_SECONDS:
            return
        self._history.close()
        self._knobs.close()
        self._history = (self.directory / HISTORY_FILE).open("a")
        self._knobs = (self.directory / KNOBS_FILE).open("ab")
        self._last_reopen = time.monotonic()

    def finish(self, **summary: Any) -> None:
        """Record how the fit ended (stop reason, best loss, ...)."""
        write_json(self.directory / RESULT_FILE, {"finished": time.time(), **summary})

    def close(self) -> None:
        self._history.close()
        self._knobs.close()
