"""Run logs: what a fit records while it runs, for live monitoring and later replay.

Each fit (a *stage*) writes one directory::

    <stage dir>/
        stage.json      name, start time, knob names and free-form metadata, written once
        history.jsonl   one JSON object per step: ``step``, ``time`` and the logged scalars
        knobs.f64       raw float64 rows ``[step, knob_0, knob_1, ...]``, one per step;
                        the row with step -1 holds the starting knobs
        result.json     how the fit ended, written when it finishes

Only the knobs are stored, not the optics they produce: anything derived from
them (a twiss, a tune, a chromaticity) is recomputed from the knobs when needed,
so a later plot is never limited by what was saved. ``metadata`` is where the
caller puts what is needed to rebuild its model (sequence, energy, knob files);
adelmo stores it without reading it.

Every file is append-only or written once, so a reader can follow a live fit
from another process without locking.
"""

from adelmo.runlog.reader import StageLog, find_stages
from adelmo.runlog.writer import RunLog

__all__ = ["RunLog", "StageLog", "find_stages"]
