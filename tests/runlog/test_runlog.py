"""The run log round-trips through its reader, including while it is being written."""

from __future__ import annotations

import numpy as np
import pytest

from adelmo.runlog import RunLog, StageLog, find_stages
from adelmo.runlog.writer import KNOBS_FILE


def _write_stage(directory, name="fit", steps=3) -> RunLog:
    run_log = RunLog(directory, name=name, metadata={"sequence": "lhcb1.seq", "energy": 450.0})
    run_log.begin({"a": 0.0, "b": 1.0})
    for step in range(steps):
        run_log.log(step, loss=10.0 - step, new_best=step != 1)
        run_log.record_knobs(step, {"b": 1.0 + step, "a": 0.1 * step})
    return run_log


def test_stage_round_trips(tmp_path) -> None:
    run_log = _write_stage(tmp_path / "stage")
    run_log.finish(reason="max_epochs")
    run_log.close()

    stage = StageLog(tmp_path / "stage")
    assert stage.name == "fit"
    assert stage.knob_names == ["a", "b"]
    assert stage.metadata == {"sequence": "lhcb1.seq", "energy": 450.0}
    assert stage.steps() == [-1, 0, 1, 2]
    assert stage.knobs_at(-1) == {"a": 0.0, "b": 1.0}
    assert stage.knobs_at(2) == pytest.approx({"a": 0.2, "b": 3.0})
    assert stage.history()["loss"].tolist() == [10.0, 9.0, 8.0]
    assert stage.best_step() == 2
    assert stage.result["reason"] == "max_epochs"


def test_a_running_stage_reads_without_its_result_or_a_half_written_row(tmp_path) -> None:
    run_log = _write_stage(tmp_path / "stage", steps=2)
    with (tmp_path / "stage" / KNOBS_FILE).open("ab") as f:
        f.write(np.array([2.0], dtype=np.float64).tobytes())  # a row cut short
    with (tmp_path / "stage" / "history.jsonl").open("a") as f:
        f.write('{"step": 2, "lo')  # a line cut short

    stage = StageLog(tmp_path / "stage")
    assert stage.result is None
    assert stage.steps() == [-1, 0, 1]
    assert len(stage.history()) == 2
    run_log.close()


def test_version_changes_when_a_step_is_logged(tmp_path) -> None:
    run_log = _write_stage(tmp_path / "stage", steps=1)
    stage = StageLog(tmp_path / "stage")
    before = stage.version()
    run_log.log(1, loss=1.0)
    assert stage.version() != before
    run_log.close()


def test_find_stages_orders_by_start_time(tmp_path) -> None:
    _write_stage(tmp_path / "run" / "x" / "bends", name="bends").close()
    _write_stage(tmp_path / "run" / "a" / "quads", name="quads").close()

    assert [stage.name for stage in find_stages(tmp_path / "run")] == ["bends", "quads"]


def test_files_are_reopened_while_logging(tmp_path, monkeypatch) -> None:
    """Closing and reopening the files (so AFS publishes them) loses no rows."""
    monkeypatch.setattr("adelmo.runlog.writer.REOPEN_SECONDS", 0.0)
    run_log = _write_stage(tmp_path / "stage", steps=4)
    first_handle = run_log._history
    run_log.log(4, loss=1.0)
    assert run_log._history is not first_handle
    run_log.record_knobs(4, {"a": 0.4, "b": 5.0})
    run_log.close()

    stage = StageLog(tmp_path / "stage")
    assert stage.steps() == [-1, 0, 1, 2, 3, 4]
    assert stage.history()["loss"].tolist() == [10.0, 9.0, 8.0, 7.0, 1.0]


def test_resume_appends_to_the_interrupted_run_and_drops_rows_after_the_checkpoint(tmp_path) -> None:
    run_log = _write_stage(tmp_path / "stage", steps=4)
    run_log.close()  # no finish(): the fit died

    resumed = RunLog(tmp_path / "stage", name="fit", resume=True)
    resumed.begin({"a": 0.0, "b": 1.0}, resume_step=2)
    for step in (2, 3, 4):
        resumed.log(step, loss=1.0 - step, new_best=True)
        resumed.record_knobs(step, {"b": 5.0, "a": 0.5})
    resumed.finish(reason="max_epochs")
    resumed.close()

    stage = StageLog(tmp_path / "stage")
    assert stage.steps() == [-1, 0, 1, 2, 3, 4]
    assert stage.history()["loss"].tolist() == [10.0, 9.0, 1.0 - 2, 1.0 - 3, 1.0 - 4]
    assert stage.knobs_at(1) == pytest.approx({"a": 0.1, "b": 2.0})
    assert stage.knobs_at(2) == {"a": 0.5, "b": 5.0}


def test_resume_refuses_a_run_with_other_knobs(tmp_path) -> None:
    _write_stage(tmp_path / "stage").close()
    resumed = RunLog(tmp_path / "stage", name="fit", resume=True)
    with pytest.raises(ValueError, match="knobs differ"):
        resumed.begin({"a": 0.0, "c": 1.0}, resume_step=1)
    resumed.close()
