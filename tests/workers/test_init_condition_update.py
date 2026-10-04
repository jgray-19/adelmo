"""Tests for the initial-condition update path.

Covers:
- TrackingWorker._send_init_condition_update updates _init_coords_np in Python
- TrackingWorker._handle_control_command dispatches 'update_init_coords'
- WorkerManager.send_init_condition_updates validates shape and sends per-worker slices
- TrackingFitter._make_epoch_end_hook dispatches the callback's coordinates
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from aba_optimiser.workers.tracking import TrackingWorker

# ---------------------------------------------------------------------------
# Stubs (no mocking library)
# ---------------------------------------------------------------------------


class FakeMAD:
    """Records every object passed to .send() so tests can inspect sent values."""

    def __init__(self) -> None:
        self.sent: list = []

    def send(self, obj: object) -> FakeMAD:
        self.sent.append(obj)
        return self  # support chaining


class FakeConn:
    """Minimal pipe-like object that stores the last sent message."""

    def __init__(self) -> None:
        self.last_sent = None

    def send(self, obj) -> None:
        self.last_sent = obj


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_worker_with_init_coords(n_particles: int = 6, num_batches: int = 2) -> TrackingWorker:
    """Return a TrackingWorker with _prepare_batches already called (no subprocess)."""
    worker = object.__new__(TrackingWorker)
    worker.worker_id = 0
    worker.observables = ("x", "px")
    worker.simulation_config = SimpleNamespace(num_batches=num_batches)
    worker.config = SimpleNamespace(kick_plane="x")

    init_coords = np.zeros((n_particles, 6), dtype=np.float64)
    init_coords[:, 0] = np.arange(n_particles, dtype=float)        # x
    init_coords[:, 1] = np.arange(n_particles, dtype=float) * 0.1  # px
    init_coords[:, 3] = np.arange(n_particles, dtype=float) * 0.01 # py
    init_pts = np.ones(n_particles, dtype=np.float64) * 1e-3

    worker.comparison_arrays = {
        "x": np.zeros((n_particles, 2)),
        "px": np.zeros((n_particles, 2)),
    }
    worker.weight_arrays = {
        "x": np.ones((n_particles, 2)),
        "px": np.ones((n_particles, 2)),
    }

    worker._prepare_batches(init_coords, init_pts, num_batches)
    return worker


# ---------------------------------------------------------------------------
# _send_init_condition_update updates _init_coords_np in Python
# ---------------------------------------------------------------------------

def test_send_init_condition_update_patches_all_four_transverse_coords_in_python() -> None:
    n = 6
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=2)
    expected_pt = worker._init_coords_np[:, 5].copy()

    new = {
        "x": np.linspace(3.0, 4.0, n),
        "px": np.linspace(1.0, 2.0, n),
        "y": np.linspace(-3.0, -4.0, n),
        "py": np.linspace(-1.0, -2.0, n),
    }

    worker._send_init_condition_update(FakeMAD(), *new.values())  # type: ignore[arg-type]

    for column, name in enumerate(("x", "px", "y", "py")):
        assert np.allclose(worker._init_coords_np[:, column], new[name])
    # The longitudinal columns (t, pt) must be untouched: the update is
    # transverse, and pt carries each file's energy offset.
    assert np.allclose(worker._init_coords_np[:, 5], expected_pt)


def test_send_init_condition_update_sends_column_matrices_to_mad() -> None:
    n = 4
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=2)
    sent_values = [np.ones(n) * scale for scale in (3.0, 1.0, -3.0, -1.0)]

    mad = FakeMAD()
    worker._send_init_condition_update(mad, *sent_values)  # type: ignore[arg-type]

    # The Lua script string is sent first, then x, px, y, py in that order --
    # the order the script's four python:recv() calls read them in.
    assert len(mad.sent) >= 5
    for sent, expected in zip(mad.sent[-4:], sent_values):
        assert isinstance(sent, np.ndarray) and sent.shape == (n, 1)
        assert np.allclose(sent[:, 0], expected)


# ---------------------------------------------------------------------------
# _handle_control_command dispatches update_init_coords
# ---------------------------------------------------------------------------

def test_handle_control_command_update_init_coords_updates_arrays_and_acks() -> None:
    n = 4
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=2)
    conn = FakeConn()
    worker.conn = conn

    new = {
        "x": np.linspace(0.3, 0.6, n),
        "px": np.linspace(0.1, 0.4, n),
        "y": np.linspace(-0.3, -0.6, n),
        "py": np.linspace(-0.1, -0.4, n),
    }
    command = {"cmd": "update_init_coords", **new}

    worker._handle_control_command(FakeMAD(), command)  # type: ignore[arg-type]

    for column, name in enumerate(("x", "px", "y", "py")):
        assert np.allclose(worker._init_coords_np[:, column], new[name])
    assert conn.last_sent == {"worker_id": 0, "status": "ok"}


# ---------------------------------------------------------------------------
# WorkerManager.send_init_condition_updates — real pipes + threads
# ---------------------------------------------------------------------------

def _recv_and_ack(child_conn, received_store: list, idx: int) -> None:
    """Thread target: receive one message from the child end and send ack back."""
    msg = child_conn.recv()
    received_store[idx] = msg
    child_conn.send({"worker_id": idx, "status": "ok"})


def _make_real_channels(counts: list[int]):
    """Build a WorkerManager stub backed by real mp.Pipe() connections."""
    from aba_optimiser.training.workers.manager import WorkerManager
    from aba_optimiser.workers.protocol import WorkerChannels

    parent_conns, child_conns = zip(*[mp.Pipe() for _ in counts])

    channels = object.__new__(WorkerChannels)
    channels.parent_conns = tuple(parent_conns)
    # workers just need a .exitcode attribute for error-handling
    channels.workers = tuple(SimpleNamespace(pid=i, exitcode=None) for i in range(len(counts)))
    channels._count = len(counts)

    wm = object.__new__(WorkerManager)
    wm._worker_particle_counts = list(counts)
    wm._validation_worker_particle_counts = []
    wm.channels = channels
    wm.validation_channels = None
    wm._channels = lambda: channels

    return wm, list(child_conns)


def _make_real_channels_with_validation(
    training_counts: list[int], validation_counts: list[int]
):
    """Build a WorkerManager stub with both training and validation pipe connections."""
    from aba_optimiser.training.workers.manager import WorkerManager
    from aba_optimiser.workers.protocol import WorkerChannels

    def _build_channels(counts: list[int], id_offset: int):
        parent_conns, child_conns = zip(*[mp.Pipe() for _ in counts])
        ch = object.__new__(WorkerChannels)
        ch.parent_conns = tuple(parent_conns)
        ch.workers = tuple(
            SimpleNamespace(pid=id_offset + i, exitcode=None) for i in range(len(counts))
        )
        ch._count = len(counts)
        return ch, list(child_conns)

    trn_channels, trn_children = _build_channels(training_counts, id_offset=0)
    val_channels, val_children = _build_channels(validation_counts, id_offset=len(training_counts))

    wm = object.__new__(WorkerManager)
    wm._worker_particle_counts = list(training_counts)
    wm._validation_worker_particle_counts = list(validation_counts)
    wm.channels = trn_channels
    wm.validation_channels = val_channels
    wm._channels = lambda: trn_channels
    wm._validation_channels = lambda: val_channels

    return wm, trn_children, val_children


def test_send_init_condition_updates_slices_correctly() -> None:
    counts = [3, 2, 4]
    wm, child_conns = _make_real_channels(counts)

    total = sum(counts)
    new_coords = np.column_stack([
        np.arange(total, dtype=float) + 0.5,
        np.arange(total, dtype=float),
        np.arange(total, dtype=float) + 0.25,
        -np.arange(total, dtype=float),
    ])

    received: list = [None] * len(counts)
    threads = [
        threading.Thread(target=_recv_and_ack, args=(child_conns[i], received, i))
        for i in range(len(counts))
    ]
    for t in threads:
        t.start()

    wm.send_init_condition_updates(new_coords)

    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "Worker thread did not finish in time"

    offset = 0
    for i, n in enumerate(counts):
        msg = received[i]
        assert isinstance(msg, dict), f"Worker {i} received unexpected value: {msg!r}"
        assert msg["cmd"] == "update_init_coords"
        for column, name in enumerate(("x", "px", "y", "py")):
            assert isinstance(msg[name], np.ndarray)
            assert msg[name].shape == (n, 1)
            assert np.allclose(msg[name][:, 0], new_coords[offset : offset + n, column])
        offset += n


def test_send_init_condition_updates_rejects_wrong_shape() -> None:
    counts = [3, 2]
    wm, _ = _make_real_channels(counts)

    with pytest.raises(ValueError, match="shape"):
        wm.send_init_condition_updates(np.zeros((4, 2)))  # wrong total (should be 5)


def test_send_init_condition_updates_also_updates_validation_workers() -> None:
    trn_counts = [3, 2]
    val_counts = [4, 1]
    wm, trn_children, val_children = _make_real_channels_with_validation(trn_counts, val_counts)

    total = sum(trn_counts) + sum(val_counts)
    new_coords = np.column_stack([
        np.arange(total, dtype=float) + 0.5,
        np.arange(total, dtype=float),
        np.arange(total, dtype=float) + 0.25,
        -np.arange(total, dtype=float),
    ])

    all_children = trn_children + val_children
    all_counts = trn_counts + val_counts
    received: list = [None] * len(all_children)
    threads = [
        threading.Thread(target=_recv_and_ack, args=(all_children[i], received, i))
        for i in range(len(all_children))
    ]
    for t in threads:
        t.start()

    wm.send_init_condition_updates(new_coords)

    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "Worker thread did not finish in time"

    offset = 0
    for i, n in enumerate(all_counts):
        msg = received[i]
        assert isinstance(msg, dict), f"Worker {i} received unexpected value: {msg!r}"
        assert msg["cmd"] == "update_init_coords"
        for column, name in enumerate(("x", "px", "y", "py")):
            assert msg[name].shape == (n, 1)
            assert np.allclose(msg[name][:, 0], new_coords[offset : offset + n, column])
        offset += n


# ---------------------------------------------------------------------------
# TrackingFitter._make_epoch_end_hook — no subprocess needed
# ---------------------------------------------------------------------------


def test_make_epoch_end_hook_calls_callback_and_dispatches() -> None:
    from aba_optimiser.training.tracking_fitter import TrackingFitter

    dispatched: list[np.ndarray] = []

    class FakeWorkerManager:
        def send_init_condition_updates(self, arr: np.ndarray) -> None:
            dispatched.append(arr)

    ctrl = object.__new__(TrackingFitter)
    ctrl.worker_manager = FakeWorkerManager()
    ctrl.config_manager = SimpleNamespace(initial_model_values={})

    new_coords = np.zeros((5, 2))
    ctrl.initial_conditions_callback = lambda knobs, best: new_coords

    hook = ctrl._make_epoch_end_hook()
    assert hook is not None

    hook({"k1": 1.0}, {"k1": 1.0})
    assert len(dispatched) == 1
    assert dispatched[0] is new_coords


def test_make_epoch_end_hook_skips_dispatch_when_callback_returns_none() -> None:
    from aba_optimiser.training.tracking_fitter import TrackingFitter

    dispatched: list = []

    class FakeWorkerManager:
        def send_init_condition_updates(self, arr: np.ndarray) -> None:
            dispatched.append(arr)

    ctrl = object.__new__(TrackingFitter)
    ctrl.worker_manager = FakeWorkerManager()
    ctrl.config_manager = SimpleNamespace(initial_model_values={})
    ctrl.initial_conditions_callback = lambda knobs, best: None

    hook = ctrl._make_epoch_end_hook()
    assert hook is not None

    hook({"k1": 1.0}, {"k1": 1.0})
    assert len(dispatched) == 0


def test_make_epoch_end_hook_includes_non_optimised_strengths() -> None:
    """The callback must see fixed strengths, not just this stage's knobs.

    The optimisation loop rebuilds ``current_knobs`` from the knob names alone, so
    strengths supplied via ``initial_knob_strengths`` but not optimised only survive
    in ``config_manager.initial_model_values``. A callback that rebuilds a model from
    the knobs it is handed would otherwise fall back to the bare model defaults.
    """
    from aba_optimiser.training.tracking_fitter import TrackingFitter

    seen: list[dict[str, float]] = []

    class FakeWorkerManager:
        def send_init_condition_updates(self, arr: np.ndarray) -> None:
            pass

    ctrl = object.__new__(TrackingFitter)
    ctrl.worker_manager = FakeWorkerManager()
    ctrl.config_manager = SimpleNamespace(
        initial_model_values={"kfixed": 3.0, "kopt": 0.0, "pt": 1e-3}
    )

    def callback(current: dict[str, float], best: dict[str, float]) -> None:
        seen.append(current)
        seen.append(best)
        return

    ctrl.initial_conditions_callback = callback
    hook = ctrl._make_epoch_end_hook()

    hook({"kopt": 1.0}, {"kopt": 2.0})
    current, best = seen
    assert current == {"kfixed": 3.0, "kopt": 1.0, "pt": 1e-3}
    assert best == {"kfixed": 3.0, "kopt": 2.0, "pt": 1e-3}


def test_make_epoch_end_hook_keeps_empty_best_knobs_empty() -> None:
    """An empty ``best_knobs`` must stay empty so callbacks can skip early epochs."""
    from aba_optimiser.training.tracking_fitter import TrackingFitter

    seen: list[dict[str, float]] = []

    class FakeWorkerManager:
        def send_init_condition_updates(self, arr: np.ndarray) -> None:
            pass

    ctrl = object.__new__(TrackingFitter)
    ctrl.worker_manager = FakeWorkerManager()
    ctrl.config_manager = SimpleNamespace(initial_model_values={"kfixed": 3.0})
    ctrl.initial_conditions_callback = lambda current, best: seen.append(best)

    ctrl._make_epoch_end_hook()({"kopt": 1.0}, {})
    assert seen == [{}]
