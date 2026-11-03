"""The monitor computes optics from a run log in the background and draws every panel."""

from __future__ import annotations

import time

import pandas as pd
import pytest

pytest.importorskip("panel")

from adelmo.monitor import MonitorSpec
from adelmo.monitor.app import Monitor
from adelmo.monitor.cache import OpticsCache, refinement_order
from adelmo.runlog import RunLog


class FakeSource:
    """Optics are the knobs themselves; counts how often it is asked."""

    calls = 0

    def __init__(self, metadata):
        self.offset = metadata["offset"]

    def references(self):
        return {"design": pd.DataFrame({"s": [0.0, 1.0], "betx": [1.0, 1.0]})}

    def optics(self, knobs):
        FakeSource.calls += 1
        return pd.DataFrame({"s": [0.0, 1.0], "betx": [1.0 + knobs["a"], 1.0 + knobs["b"]]})

    def summary(self, optics):
        return {"max_betx": float(optics["betx"].max()) + self.offset}


def _wait(predicate, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def _write_run(directory, steps=5):
    run_log = RunLog(directory, name="fit", metadata={"offset": 100.0})
    run_log.begin({"a": 0.0, "b": 0.0})
    for step in range(steps):
        run_log.log(step, loss=1.0 / (step + 1), new_best=True)
        run_log.record_knobs(step, {"a": 0.1 * step, "b": -0.1 * step})
    run_log.close()


def test_refinement_order_starts_with_the_ends_then_halves() -> None:
    assert refinement_order(list(range(9))) == [0, 8, 4, 2, 6, 1, 3, 5, 7]
    assert refinement_order([5]) == [5]


def test_cache_computes_once_and_survives_a_new_cache(tmp_path) -> None:
    spec = MonitorSpec(make_source=FakeSource)
    cache = OpticsCache(spec, tmp_path / "cache")
    knobs = pd.Series({"a": 0.5, "b": 0.25})
    metadata = {"offset": 1.0}

    assert cache.optics(metadata, knobs, 0) is None
    _wait(lambda: cache.optics(metadata, knobs, 0) is not None)
    assert cache.optics(metadata, knobs, 0)["betx"].tolist() == [1.5, 1.25]
    assert cache.summary(metadata, knobs) == {"max_betx": 2.5}

    calls = FakeSource.calls
    again = OpticsCache(spec, tmp_path / "cache")
    assert again.optics(metadata, knobs, 0) is not None
    assert FakeSource.calls == calls


def test_a_failing_source_is_reported_not_retried(tmp_path) -> None:
    def broken(_metadata):
        raise RuntimeError("no MAD here")

    cache = OpticsCache(MonitorSpec(make_source=broken), tmp_path)
    assert cache.references({"x": 1}) is None
    _wait(lambda: cache.error({"x": 1}) is not None)
    assert "no MAD here" in cache.error({"x": 1})


def test_monitor_follows_the_latest_step_and_draws_the_active_panel(tmp_path, caplog) -> None:
    _write_run(tmp_path / "runs" / "stage")
    seen = []

    def optics_panel(view):
        seen.append((view.step, view.optics is not None))
        from adelmo.monitor.panels import message

        return message("optics")

    spec = MonitorSpec(make_source=FakeSource, panels=[("Optics", optics_panel)])
    cache = OpticsCache(spec, tmp_path / "cache")
    monitor = Monitor(tmp_path / "runs", spec, cache, refresh_seconds=1)

    monitor.refresh()
    assert monitor.step_slider.value == 4
    _wait(lambda: (monitor.refresh(), len(monitor._summaries) == 6)[1])

    assert seen[-1] == (4, True)
    assert sorted(monitor._summaries) == [-1, 0, 1, 2, 3, 4]
    assert monitor.panes[0].object is not None
    assert all(pane.object is None for pane in monitor.panes[1:]), "hidden tabs must not be drawn"
    for index, pane in enumerate(monitor.panes[1:], start=1):
        monitor.tabs.active = index  # drawn on first show
        assert pane.object is not None
    monitor.tabs.active = 0
    assert not [r for r in caplog.records if r.name == "adelmo.monitor.app"], "a panel failed"

    monitor.step_slider.value = 2
    assert monitor.follow.value is False
    assert seen[-1] == (2, True)
