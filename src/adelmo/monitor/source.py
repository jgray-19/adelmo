"""What the machine side supplies to the monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd
    from bokeh.model import Model

    from adelmo.monitor.view import MonitorView


class OpticsSource(Protocol):
    """Turns knobs into optics for one model.

    The monitor builds one per distinct stage metadata (and per worker thread) and
    calls it from a background thread, so it may hold a live MAD process. Every
    method returns plain data; the monitor caches it on disk, so the same knobs
    are never computed twice.
    """

    def references(self) -> dict[str, pd.DataFrame]:
        """Tables that do not depend on the knobs: the measurement, design, ..."""
        ...

    def optics(self, knobs: Mapping[str, float]) -> pd.DataFrame:
        """The optics of the model with ``knobs`` applied (e.g. a twiss table)."""
        ...

    def summary(self, optics: pd.DataFrame) -> dict[str, float]:
        """A few numbers describing ``optics`` (e.g. RMS beta-beating), plotted against step."""
        ...


@dataclass
class MonitorSpec:
    """How to monitor one kind of fit: its optics source and its panels.

    ``make_source`` builds an :class:`OpticsSource` from a stage's metadata (the
    ``OutputConfig.run_metadata`` the fit was given). ``panels`` are ``(title,
    function)`` pairs shown before the generic ones. Bump ``version`` whenever
    the source would give different optics for the same knobs, so old cached
    results are not reused.
    """

    make_source: Callable[[Mapping[str, Any]], OpticsSource] | None = None
    panels: Sequence[tuple[str, Callable[[MonitorView], Model]]] = field(default_factory=list)
    version: str = "1"
