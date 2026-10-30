"""Panels that work for any fit: they need only the run log, not the optics.

A panel is a function ``(MonitorView) -> bokeh model``. The helpers here
(:func:`step_figure`, :func:`message`, :data:`PALETTE`) are for machine-specific
panels too.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from bokeh.events import MouseEnter, MouseLeave
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, LogScale, Plot, Span
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

if TYPE_CHECKING:
    from bokeh.model import Model

    from adelmo.monitor.view import MonitorView

PALETTE = Category10_10
#: History columns that are bookkeeping, not quantities worth plotting.
NOT_PLOTTED = {"time", "new_best"}
#: How many knobs the trajectory plot follows.
N_TRAJECTORIES = 12


def message(text: str) -> Div:
    """A panel body that only says something (waiting, missing data, an error)."""
    return Div(text=f"<p style='font-size:1.1em'>{text}</p>")


def step_figure(
    view: MonitorView, title: str, *, y_axis_type: str = "linear", height: int = 260, **kwargs
) -> figure:
    """A figure against step with the selected step marked."""
    fig = figure(
        title=title,
        x_axis_label="step",
        y_axis_type=y_axis_type,
        sizing_mode="stretch_width",
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        **kwargs,
    )
    fig.add_layout(Span(location=view.step, dimension="height", line_dash="dashed", line_alpha=0.5))
    return fig


#: The figure under the pointer, and the one window-wide listener that fits it on a middle click.
_MIDDLE_FIT_ENTER = """
const state = (window.__adelmoMiddleFit ??= {});
state.fig = new WeakRef(fig);
state.log = log;
if (state.installed) return;
state.installed = true;
const fit = (f, log) => {
  const [xr, yr] = [f.x_range, f.y_range];
  if (typeof xr.start !== "number" || typeof yr.start !== "number") return;
  const x0 = Math.min(xr.start, xr.end), x1 = Math.max(xr.start, xr.end);
  let lo = Infinity, hi = -Infinity;
  for (const r of f.renderers) {
    const g = r.glyph, source = r.data_source;
    if (!r.visible || !g || !source) continue;
    const xf = g.x?.field, yf = (g.y ?? g.top)?.field;
    const X = xf && source.data[xf], Y = yf && source.data[yf];
    if (!X || !Y) continue;
    for (let i = 0; i < Y.length; i++) {
      const [x, y] = [X[i], Y[i]];
      if (!(x >= x0 && x <= x1) || !Number.isFinite(y) || (log && y <= 0)) continue;
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
  }
  if (!(lo <= hi)) return;
  if (log) {
    const [a, b] = [Math.log10(lo), Math.log10(hi)];
    const pad = (b - a || 1) * 0.05;
    [yr.start, yr.end] = [10 ** (a - pad), 10 ** (b + pad)];
  } else {
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.05;
    [yr.start, yr.end] = [lo - pad, hi + pad];
  }
};
window.addEventListener("mousedown", (e) => {
  const f = e.button === 1 && state.fig?.deref();
  if (!f) return;
  e.preventDefault();  // no browser autoscroll
  fit(f, state.log);
}, true);
"""
_MIDDLE_FIT_LEAVE = "if (window.__adelmoMiddleFit) window.__adelmoMiddleFit.fig = null;"


def enable_middle_click_fit(model: Model) -> None:
    """Make a middle click on any figure in ``model`` refit its y axis to the data in view."""
    for fig in model.select({"type": Plot}):
        log = isinstance(fig.y_scale, LogScale)
        fig.js_on_event(MouseEnter, CustomJS(args={"fig": fig, "log": log}, code=_MIDDLE_FIT_ENTER))
        fig.js_on_event(MouseLeave, CustomJS(code=_MIDDLE_FIT_LEAVE))


def _positive(values) -> bool:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return values.size > 0 and bool(np.all(values > 0))


def scalars_panel(view: MonitorView) -> Model:
    """Every logged scalar against step, one plot each; positive ones on a log scale."""
    history = view.history
    if history.empty:
        return message("No steps logged yet.")
    columns = [
        c
        for c in history.columns
        if c not in NOT_PLOTTED
        and np.issubdtype(history[c].dtype, np.number)
        and history[c].notna().any()
    ]
    figures = []
    for name in columns:
        values = history[name]
        fig = step_figure(view, name, y_axis_type="log" if _positive(values) else "linear")
        fig.line(history.index, values, line_width=1.5)
        fig.scatter(history.index, values, size=3)
        figures.append(fig)
    return gridplot(figures, ncols=2, sizing_mode="stretch_width")


def loss_panel(view: MonitorView) -> Model:
    """Training (and validation) loss against step, best step marked."""
    history = view.history
    if history.empty or "loss" not in history:
        return message("No loss logged yet.")
    fig = step_figure(
        view, "Loss", y_axis_type="log" if _positive(history["loss"]) else "linear", height=420
    )
    for i, name in enumerate(
        c for c in ("loss", "validation_loss") if c in history and history[c].notna().any()
    ):
        fig.line(history.index, history[name], legend_label=name, color=PALETTE[i], line_width=2)
    best = view.stage.best_step()
    if best is not None and best in history.index:
        fig.scatter(
            [best],
            [history.loc[best, "loss"]],
            size=10,
            marker="star",
            color="gold",
            legend_label="best",
        )
    fig.legend.location = "top_right"
    return fig


def knobs_panel(view: MonitorView) -> Model:
    """Change of every knob from the start at the selected step, and the largest movers over time."""
    knobs = view.knobs
    if view.step not in knobs.index:
        return message(f"No knobs recorded for step {view.step}.")
    start = knobs.iloc[0]
    change = knobs.loc[view.step] - start

    bar = figure(
        title=f"Knob change from start at step {view.step}",
        x_axis_label="knob index",
        sizing_mode="stretch_width",
        height=320,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    source = ColumnDataSource(
        {"index": np.arange(len(change)), "change": change.to_numpy(), "name": change.index}
    )
    bar.scatter("index", "change", source=source, size=4)
    bar.add_tools(HoverTool(tooltips=[("knob", "@name"), ("change", "@change{0.000e}")]))

    movers = change.abs().nlargest(N_TRAJECTORIES).index
    lines = step_figure(view, f"The {len(movers)} knobs that moved most", height=360)
    for i, name in enumerate(movers):
        lines.line(
            knobs.index,
            knobs[name] - start[name],
            legend_label=name,
            color=PALETTE[i % len(PALETTE)],
        )
    lines.legend.label_text_font_size = "8pt"
    lines.legend.click_policy = "hide"
    return column(bar, lines, sizing_mode="stretch_width")


#: Summary names of the two planes: ``Δβx`` / ``Δβy``, ``Δx orbit`` / ``Δy orbit``, ``ΔQ1`` / ``ΔQ2``.
_PLANE_NAMES = [
    re.compile(r"^(?P<pre>.*[βμ])(?P<plane>[xy])(?P<post>.*)$"),
    re.compile(r"^(?P<pre>.*Δ)(?P<plane>[xy])(?P<post> orbit.*)$"),
    re.compile(r"^(?P<pre>.*Q)(?P<plane>[12])(?P<post>)$"),
]


def _plane_pairs(names) -> list[tuple[str | None, str | None]]:
    """``(x name, y name)`` of every summary row; names that differ only in the plane share a row.

    A name with no plane goes in the x column, with ``None`` for y.
    """
    pairs: dict[tuple[str, str], list[str | None]] = {}
    for name in names:
        for pattern in _PLANE_NAMES:
            match = pattern.match(name)
            if match:
                key = (match["pre"], match["post"])
                pairs.setdefault(key, [None, None])[0 if match["plane"] in "x1" else 1] = name
                break
        else:
            pairs[(name, "")] = [name, None]
    return [(x_name, y_name) for x_name, y_name in pairs.values()]


def summaries_panel(view: MonitorView) -> Model:
    """The optics source's summary against step: one row per quantity, x (or Q1) left, y (or Q2) right."""
    summaries = view.summaries
    if summaries.empty:
        return message("No optics summaries computed yet.")
    rows = []
    for x_name, y_name in _plane_pairs(summaries.columns):
        row = []
        for name in (x_name, y_name):
            if name is None:
                row.append(None)
                continue
            values = summaries[name].dropna()
            fig = step_figure(view, name)
            fig.line(values.index, values.to_numpy(), line_width=1.5)
            fig.scatter(values.index, values.to_numpy(), size=4)
            row.append(fig)
        rows.append(row)
    return gridplot(rows, sizing_mode="stretch_width")


#: The panels every monitor shows, after the machine-specific ones.
GENERIC_PANELS = [
    ("Loss", loss_panel),
    ("Scalars", scalars_panel),
    ("Knobs", knobs_panel),
]
#: Shown only when an optics source is configured.
OPTICS_PANELS = [("Optics summary", summaries_panel)]
