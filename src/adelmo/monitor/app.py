"""The monitor web app: pick a stage and a step, see every panel for it.

Nothing the panels draw is kept in memory: a draw reads the run log and the
optics cache from disk, builds the figures, hands them to the browser and lets
go of the data. The page changes only when the user asks (stage, step, tab or
the Redraw button), so a long run does not make the browser rebuild figures
every few seconds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn
from bokeh.models import DataRange1d

from adelmo.monitor.cache import BACKFILL, LATEST, SELECTED, OpticsCache, refinement_order
from adelmo.monitor.panels import GENERIC_PANELS, OPTICS_PANELS, enable_middle_click_fit, message
from adelmo.monitor.source import MonitorSpec
from adelmo.monitor.view import MonitorView
from adelmo.runlog import StageLog, find_stages

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from bokeh.model import Model

LOGGER = logging.getLogger(__name__)

#: Panels that show :attr:`MonitorView.summaries`, and so go stale as those grow.
SUMMARY_PANELS = {panel for _, panel in OPTICS_PANELS}
#: The step dropdown's value for "whichever step is the newest".
LATEST_STEP = "latest"


class Monitor:
    """One browser session's view of the run logs below ``root``.

    Every ``refresh_seconds`` it rescans for stages, queues the optics it needs
    (the selected step first, then the latest, then the whole history
    coarse-to-fine) and says in the sidebar whether the visible tab is out of
    date. It redraws only on a user action; the Redraw button draws the visible
    tab afresh if (and only if) there is new data for it.
    """

    def __init__(self, root: Path | Sequence[Path], spec: MonitorSpec, cache: OpticsCache | None, refresh_seconds: float):
        self.root = Path(root) if isinstance(root, str | Path) else [Path(r) for r in root]
        self.spec = spec
        self.cache = cache
        self.refresh_seconds = refresh_seconds

        self.stage_select = pn.widgets.Select(label="Stage", options={})
        self.step_select = pn.widgets.Select(label="Step", options={"Latest": LATEST_STEP}, value=LATEST_STEP)
        self.best_button = pn.widgets.Button(label="Go to best step")
        self.redraw_button = pn.widgets.Button(label="Redraw")
        self.backfill = pn.widgets.Checkbox(label="Compute optics for the whole history", value=True)
        self.status = pn.pane.Markdown()

        panels: list[tuple[str, Callable[[MonitorView], Model]]] = list(spec.panels)
        if cache is not None:
            panels += OPTICS_PANELS
        panels += GENERIC_PANELS
        self.panels = panels
        # One container per tab; a draw swaps its content for a brand new pane, so
        # nothing of the previous figures stays reachable from the page.
        self.holders = [pn.Column(sizing_mode="stretch_width") for _ in panels]
        self.tabs = pn.Tabs(*((title, holder) for (title, _), holder in zip(panels, self.holders)))

        self._steps: tuple[int, ...] = ()
        #: Tab index -> fingerprint of what it shows (a few small values, never data).
        self._rendered: dict[int, Any] = {}

        self.stage_select.param.watch(self._on_stage, "value")
        self.step_select.param.watch(lambda _: self._draw_active(), "value")
        self.best_button.on_click(self._go_to_best)
        self.redraw_button.on_click(lambda _: self._on_redraw())
        self.tabs.param.watch(lambda _: self._draw_active(), "active")
        self.backfill.param.watch(lambda _: self.refresh(), "value")

    # -- layout ------------------------------------------------------------------------

    def layout(self) -> pn.template.BaseTemplate:
        template = pn.template.FastListTemplate(
            title="adelmo monitor",
            sidebar=[
                pn.pane.Markdown(f"**Run logs:** `{self.root}`"),
                self.stage_select,
                self.step_select,
                self.best_button,
                self.redraw_button,
                self.backfill,
                self.status,
            ],
            main=[self.tabs],
        )
        pn.state.onload(self._start)
        pn.state.add_periodic_callback(self.refresh, period=int(self.refresh_seconds * 1000))
        return template

    # -- events ------------------------------------------------------------------------

    def _start(self) -> None:
        self.refresh()
        self._draw_active()

    def _on_stage(self, _event) -> None:
        self._rendered.clear()
        self.step_select.value = LATEST_STEP
        self.refresh()
        self._draw_active()

    def _on_redraw(self) -> None:
        self.refresh()
        self._draw_active(announce=True)

    def _go_to_best(self, _event) -> None:
        stage = self._stage()
        best = stage.best_step() if stage is not None else None
        if best is not None and best in self._steps:
            self.step_select.value = best

    # -- reading the run log -----------------------------------------------------------

    def _stage(self) -> StageLog | None:
        return StageLog(Path(self.stage_select.value)) if self.stage_select.value else None

    def _step(self, knobs: pd.DataFrame) -> int:
        """The selected step: the dropdown's, or the newest recorded."""
        value = self.step_select.value
        if value == LATEST_STEP or value not in knobs.index:
            return int(knobs.index[-1])
        return int(value)

    def _set_step_options(self, knobs: pd.DataFrame) -> None:
        steps = tuple(int(s) for s in knobs.index)
        if steps == self._steps:
            return
        self._steps = steps
        options = {"Latest": LATEST_STEP} | {f"Step {s}": s for s in reversed(steps)}
        self.step_select.options = options
        if self.step_select.value != LATEST_STEP and self.step_select.value not in steps:
            self.step_select.value = LATEST_STEP

    def _queue_optics(self, stage: StageLog, knobs: pd.DataFrame, step: int) -> tuple[bool, bool, int]:
        """Queue what is missing; return whether the references and the optics at ``step`` are
        on disk, and how many steps have a summary."""
        cache = self.cache
        if cache is None or not stage.metadata:
            return False, False, 0
        metadata = stage.metadata
        references = cache.has_references(metadata)
        optics = cache.request(metadata, knobs.loc[step], SELECTED)
        cache.request(metadata, knobs.iloc[-1], LATEST)
        steps = list(knobs.index)
        summaries = 0
        missing = []
        for s in steps:
            if cache.has_summary(metadata, knobs.loc[s]):
                summaries += 1
            else:
                missing.append(s)
        if self.backfill.value:
            missing = set(missing)
            for rank, s in enumerate(refinement_order(steps)):
                if s in missing:
                    cache.request(metadata, knobs.loc[s], BACKFILL + rank)
        return references, optics, summaries

    def _read_summaries(self, stage: StageLog, knobs: pd.DataFrame) -> pd.DataFrame:
        cache = self.cache
        rows = {}
        if cache is not None and stage.metadata:
            for s in knobs.index:
                summary = cache.summary(stage.metadata, knobs.loc[s])
                if summary is not None:
                    rows[int(s)] = summary
        summaries = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        summaries.index.name = "step"
        return summaries

    # -- the page ----------------------------------------------------------------------

    @staticmethod
    def _fingerprint(panel, stage: StageLog, step: int, references: bool, optics: bool, summaries: int) -> tuple:
        """What a tab shows, as a few comparable values; summaries count only for the panels that plot them."""
        return (stage.directory, stage.version(), step, references, optics, summaries if panel in SUMMARY_PANELS else None)

    def refresh(self) -> None:
        """Rescan stages, queue optics, update the sidebar. Never touches the plots."""
        stages = find_stages(self.root)
        options = {stage.label: str(stage.directory) for stage in reversed(stages)}
        if options != self.stage_select.options:
            self.stage_select.options = options
        if not options:
            self.status.object = f"No run logs below `{self.root}` yet."
            return
        if self.stage_select.value not in options.values():
            self.stage_select.value = next(iter(options.values()))  # newest; re-enters via _on_stage
            return

        stage = self._stage()
        knobs = stage.knob_history()
        if knobs.empty:
            self.status.object = f"No knob history recorded yet in `{stage.directory}`."
            return
        self._set_step_options(knobs)
        step = self._step(knobs)
        references, optics, summaries = self._queue_optics(stage, knobs, step)

        index = self.tabs.active
        fresh = self._rendered.get(index) == self._fingerprint(
            self.panels[index][1], stage, step, references, optics, summaries
        )
        self.redraw_button.button_type = "default" if fresh else "warning"
        self._update_status(stage, knobs, step, references, optics, summaries, fresh)

    def _draw_active(self, *, announce: bool = False) -> None:
        """Draw the visible tab from disk if it is out of date; hidden tabs are drawn when first shown."""
        stage = self._stage()
        if stage is None:
            return
        knobs = stage.knob_history()
        if knobs.empty:
            return
        index = self.tabs.active
        title, panel = self.panels[index]
        step = self._step(knobs)
        references, optics, summaries = self._queue_optics(stage, knobs, step)
        state = self._fingerprint(panel, stage, step, references, optics, summaries)
        if self._rendered.get(index) == state:
            if announce:
                self.status.object += "\n\nNo new data for this tab."
            return
        self._rendered[index] = state

        metadata = stage.metadata
        view = MonitorView(
            stage=stage,
            step=step,
            history=stage.history(),
            knobs=knobs,
            references=self.cache.references(metadata) if references else None,
            optics=self.cache.optics(metadata, knobs.loc[step], SELECTED) if optics else None,
            summaries=self._read_summaries(stage, knobs),
            s_range=DataRange1d(),  # fresh: a range kept across draws keeps every old figure alive
        )
        try:
            body = panel(view)
        except Exception as exc:
            LOGGER.exception("Panel %s failed", title)
            body = message(f"Panel failed: {type(exc).__name__}: {exc}")
        del view
        enable_middle_click_fit(body)
        self.holders[index][:] = [pn.pane.Bokeh(body, sizing_mode="stretch_width")]
        self.redraw_button.button_type = "default"
        self._update_status(stage, knobs, step, references, optics, summaries, True)

    def _update_status(self, stage, knobs, step, references, optics, summaries, fresh) -> None:
        result = stage.result
        lines = [
            f"**{stage.name}** — {'finished: ' + str(result.get('reason')) if result else 'running'}",
            f"Step {step} of {int(knobs.index[-1])} ({len(knobs)} recorded)",
        ]
        if self.cache is not None and stage.metadata:
            lines.append(f"Optics: {summaries}/{len(knobs)} steps, {self.cache.pending} queued")
            error = self.cache.error(stage.metadata) or self.cache.error(stage.metadata, knobs.loc[step])
            if error:
                lines.append(f"<span style='color:#c0392b'>Error: {error}</span>")
            elif not references or not optics:
                lines.append("Computing the selected step ...")
        elif self.cache is not None:
            lines.append("This stage has no metadata, so no optics.")
        lines.append("Up to date." if fresh else "**New data: press Redraw.**")
        self.status.object = "\n\n".join(lines)


def serve(
    root: Path | Sequence[Path],
    spec: MonitorSpec | None = None,
    *,
    port: int = 5006,
    address: str = "localhost",
    refresh_seconds: float = 5.0,
    workers: int = 1,
    cache_dir: Path | None = None,
    show: bool = False,
) -> None:
    """Serve the monitor for the run logs below ``root`` until interrupted."""
    spec = spec or MonitorSpec()
    cache = None
    if spec.make_source is not None:
        cache = OpticsCache(spec, cache_dir or Path(root if isinstance(root, str | Path) else root[0]) / ".optics_cache", workers=workers)
    pn.extension(sizing_mode="stretch_width")
    LOGGER.info("Serving the monitor of %s on http://%s:%d", root, address, port)
    pn.serve(
        lambda: Monitor(root, spec, cache, refresh_seconds).layout(),
        port=port,
        address=address,
        show=show,
        title="adelmo monitor",
    )
