from __future__ import annotations

import numpy as np

from aba_optimiser.mad.scripts import dump_debug_script
from aba_optimiser.workers.tracking import TrackingWorker, active_observables


def _build_worker(*, include_momentum: bool = True) -> TrackingWorker:
    worker = object.__new__(TrackingWorker)
    worker.observables = active_observables("xy", include_momentum)
    worker.comparisons = {
        "x": [np.array([[1.0, 5.0]])],
        "y": [np.array([[2.0, 6.0]])],
        "px": [np.array([[3.0, 7.0]])],
        "py": [np.array([[4.0, 8.0]])],
    }
    worker.weights = {
        "x": [np.array([[1.0, 10.0]])],
        "y": [np.array([[2.0, 20.0]])],
        "px": [np.array([[3.0, 30.0]])],
        "py": [np.array([[4.0, 40.0]])],
    }
    worker.keep_bpm_mask = np.array([True, False])
    return worker


def test_tracking_worker_accumulates_all_configured_observables() -> None:
    worker = _build_worker()
    results = {
        "x": np.array([[2.0, 100.0]]),
        "y": np.array([[4.0, 100.0]]),
        "px": np.array([[6.0, 100.0]]),
        "py": np.array([[8.0, 100.0]]),
        "dx_dk": np.array([[[1.0, 99.0]]]),
        "dy_dk": np.array([[[1.0, 99.0]]]),
        "dpx_dk": np.array([[[1.0, 99.0]]]),
        "dpy_dk": np.array([[[1.0, 99.0]]]),
    }

    grad, loss = worker._compute_loss_and_gradients(results, batch=0)

    assert np.allclose(grad, [60.0])
    assert loss == 100.0


def test_position_only_tracking_worker_ignores_momentum_observables() -> None:
    worker = _build_worker(include_momentum=False)
    results = {
        "x": np.array([[2.0, 100.0]]),
        "y": np.array([[4.0, 100.0]]),
        "dx_dk": np.array([[[1.0, 99.0]]]),
        "dy_dk": np.array([[[1.0, 99.0]]]),
    }

    grad, loss = worker._compute_loss_and_gradients(results, batch=0)

    assert np.allclose(grad, [10.0])
    assert loss == 9.0


def test_tracking_worker_supports_single_plane_observable_sets() -> None:
    worker = _build_worker()
    worker.observables = ("x", "px")
    results = {
        "x": np.array([[2.0, 100.0]]),
        "px": np.array([[6.0, 100.0]]),
        "dx_dk": np.array([[[1.0, 99.0]]]),
        "dpx_dk": np.array([[[1.0, 99.0]]]),
    }

    grad, loss = worker._compute_loss_and_gradients(results, batch=0)

    assert np.allclose(grad, [20.0])
    assert loss == 28.0


def test_dump_debug_script_writes_generated_text(tmp_path) -> None:
    logfile = tmp_path / "worker.log"
    output = dump_debug_script(
        "run_track",
        "python:send('ok')\n",
        debug=True,
        mad_logfile=logfile,
        worker_id=3,
    )

    assert output == tmp_path / "generated_mad_scripts" / "run_track_worker_3.mad"
    assert output.read_text() == "python:send('ok')\n"
