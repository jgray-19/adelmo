"""Reading run logs, including those of fits that are still running."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adelmo.runlog.writer import HISTORY_FILE, KNOBS_FILE, RESULT_FILE, STAGE_FILE


def find_stages(root: Path | Sequence[Path]) -> list[StageLog]:
    """Every stage logged anywhere below ``root`` (or ``root`` itself), oldest first.

    ``root`` may be several directories, e.g. the run logs of runs that sit side by side.
    """
    roots = [Path(root)] if isinstance(root, str | Path) else [Path(r) for r in root]
    stages = [StageLog(path.parent) for r in roots for path in r.rglob(STAGE_FILE)]
    return sorted(stages, key=lambda stage: stage.started)


@dataclass(frozen=True)
class StageLog:
    """Read access to one stage directory written by :class:`~adelmo.runlog.RunLog`.

    Nothing is cached: every call reads the files again, so a stage that is
    still running shows its newest steps. Use :meth:`version` to tell whether
    anything changed since the last read.
    """

    directory: Path

    @property
    def info(self) -> dict[str, Any]:
        """The contents of ``stage.json``."""
        return json.loads((self.directory / STAGE_FILE).read_text())

    @property
    def name(self) -> str:
        return self.info["name"]

    @property
    def started(self) -> float:
        return float(self.info["started"])

    @property
    def knob_names(self) -> list[str]:
        return list(self.info["knob_names"])

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.info.get("metadata", {}))

    @property
    def result(self) -> dict[str, Any] | None:
        """How the fit ended, or ``None`` while it is still running (or was killed)."""
        path = self.directory / RESULT_FILE
        return json.loads(path.read_text()) if path.exists() else None

    @property
    def label(self) -> str:
        """``name`` plus the run's own ``label`` (metadata) and the directory, to tell repeated stages apart."""
        run = self.metadata.get("label")
        return f"{self.name} [{run}] ({self.directory.name})" if run else f"{self.name} ({self.directory.name})"

    def version(self) -> tuple[int, int]:
        """Sizes of the growing files: changes whenever a step is logged."""
        return tuple(  # type: ignore[return-value]
            (self.directory / name).stat().st_size if (self.directory / name).exists() else 0
            for name in (HISTORY_FILE, KNOBS_FILE)
        )

    def history(self) -> pd.DataFrame:
        """The logged scalars, one row per step, indexed by step."""
        path = self.directory / HISTORY_FILE
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # the last line is still being written
        if not rows:
            return pd.DataFrame(index=pd.Index([], name="step"))
        return pd.DataFrame(rows).set_index("step")

    def knob_history(self) -> pd.DataFrame:
        """Knob values, one row per recorded step (the start is step -1), one column per knob."""
        names = self.knob_names
        raw = np.fromfile(self.directory / KNOBS_FILE, dtype=np.float64)
        width = len(names) + 1
        rows = raw[: len(raw) // width * width].reshape(-1, width)  # drop a half-written row
        frame = pd.DataFrame(rows[:, 1:], columns=names, index=rows[:, 0].astype(int))
        frame.index.name = "step"
        return frame[~frame.index.duplicated(keep="last")]

    def steps(self) -> list[int]:
        """Every step with recorded knobs, in order."""
        return self.knob_history().index.tolist()

    def knobs_at(self, step: int) -> dict[str, float]:
        """The knobs after ``step``."""
        return self.knob_history().loc[step].to_dict()

    def best_step(self) -> int | None:
        """The last step logged as a new best, if the fit logs ``new_best``."""
        history = self.history()
        if "new_best" not in history:
            return None
        best = history.index[history["new_best"].fillna(value=False).astype(bool)]
        return int(best[-1]) if len(best) else None
