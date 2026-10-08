"""Tests for the initial-condition update path.

Covers:
- TrackingWorker._send_init_condition_update updates _init_coords_np in Python
- TrackingWorker.handle_command dispatches UPDATE_INIT_COORDS
- TrackingSession.send_init_condition_updates validates shape and sends per-worker slices
- initial_conditions_hook dispatches the callback's coordinates
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from adelmo.config import SimulationConfig
from adelmo.fitting.protocol import Ack, Command, CommandKind
from adelmo.fitting.worker import WorkerConfig
from adelmo.tracking.worker import PrecomputedTrackingWeights, TrackingData, TrackingWorker

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


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_worker_with_init_coords(n_particles: int = 6, num_batches: int = 2) -> TrackingWorker:
    """Return a prepared TrackingWorker (its process is never started)."""
    init_coords = np.zeros((n_particles, 6), dtype=np.float64)
    init_coords[:, 0] = np.arange(n_particles, dtype=float)        # x
    init_coords[:, 1] = np.arange(n_particles, dtype=float) * 0.1  # px
    init_coords[:, 3] = np.arange(n_particles, dtype=float) * 0.01 # py
    init_coords[:, 5] = 1e-3                                       # pt
    shape = (n_particles, 2)
    data = TrackingData(
        position_comparisons=np.zeros((*shape, 2)),
        momentum_comparisons=np.zeros((*shape, 2)),
        position_variances=np.ones((*shape, 2)),
        momentum_variances=np.ones((*shape, 2)),
        init_coords=init_coords,
        init_pts=np.ones(n_particles) * 1e-3,
        reading_ids=np.zeros(shape, dtype=np.int64),
        init_reading_ids=np.zeros(n_particles, dtype=np.int64),
        init_variances=np.ones((n_particles, 2)),
        precomputed_weights=PrecomputedTrackingWeights(
            x=np.ones(shape), y=np.ones(shape), px=np.ones(shape), py=np.ones(shape), scale=1.0
        ),
    )
    config = WorkerConfig(
        accelerator=None,  # type: ignore[arg-type]  # only the MAD session needs it
        tracking_start_bpm="BPM.A",
        tracking_end_bpm="BPM.B",
        magnet_range="$start/$end",
        kick_plane="x",
    )
    return TrackingWorker(
        None,  # type: ignore[arg-type]
        0,
        data,
        config,
        SimulationConfig(num_workers=1, num_batches=num_batches),
    )


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
# handle_command dispatches UPDATE_INIT_COORDS
# ---------------------------------------------------------------------------

def test_handle_command_update_init_coords_updates_arrays_and_acks() -> None:
    n = 4
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=2)

    new = {
        "x": np.linspace(0.3, 0.6, n),
        "px": np.linspace(0.1, 0.4, n),
        "y": np.linspace(-0.3, -0.6, n),
        "py": np.linspace(-0.1, -0.4, n),
    }
    command = Command(CommandKind.UPDATE_INIT_COORDS, new)

    reply = worker.handle_command(FakeMAD(), command)  # type: ignore[arg-type]

    for column, name in enumerate(("x", "px", "y", "py")):
        assert np.allclose(worker._init_coords_np[:, column], new[name])
    assert reply == Ack(0)


# ---------------------------------------------------------------------------
# TrackingSession.send_init_condition_updates — real pipes + threads
# ---------------------------------------------------------------------------

def _recv_and_ack(child_conn, received_store: list, idx: int) -> None:
    """Thread target: receive one message from the child end and send ack back."""
    msg = child_conn.recv()
    received_store[idx] = msg
    child_conn.send(Ack(idx))


def _make_pool(counts: list[int], id_offset: int = 0):
    """A WorkerPool backed by real mp.Pipe() connections; returns it and the child ends."""
    from adelmo.fitting.pool import WorkerPool

    parent_conns, child_conns = zip(*[mp.Pipe() for _ in counts])
    pool = WorkerPool(
        conns=list(parent_conns),
        # workers just need pid/exitcode attributes for error-handling
        workers=[SimpleNamespace(pid=id_offset + i, exitcode=None) for i in range(len(counts))],
        particle_counts=list(counts),
    )
    return pool, list(child_conns)


def _make_real_channels(counts: list[int]):
    """Build a session with real training pipe connections."""
    return _make_real_channels_with_validation(counts, [])[:2]


def _make_real_channels_with_validation(
    training_counts: list[int], validation_counts: list[int]
):
    """Build a session with both training and validation pipe connections.

    Only its pools are set: pushing coordinates touches nothing else.
    """
    from adelmo.fitting.pool import WorkerPool
    from adelmo.tracking.session import TrackingSession

    wm = object.__new__(TrackingSession)
    wm.training, trn_children = _make_pool(training_counts)
    wm.validation, val_children = (
        _make_pool(validation_counts, id_offset=len(training_counts))
        if validation_counts
        else (WorkerPool(), [])
    )
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
        assert isinstance(msg, Command), f"Worker {i} received unexpected value: {msg!r}"
        assert msg.kind is CommandKind.UPDATE_INIT_COORDS
        msg = msg.payload
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
        assert isinstance(msg, Command), f"Worker {i} received unexpected value: {msg!r}"
        assert msg.kind is CommandKind.UPDATE_INIT_COORDS
        msg = msg.payload
        for column, name in enumerate(("x", "px", "y", "py")):
            assert msg[name].shape == (n, 1)
            assert np.allclose(msg[name][:, 0], new_coords[offset : offset + n, column])
        offset += n


# ---------------------------------------------------------------------------
# initial_conditions_hook -- no subprocess needed
# ---------------------------------------------------------------------------


def test_initial_conditions_hook_pushes_the_callbacks_coordinates() -> None:
    from adelmo.tracking.fitter import initial_conditions_hook

    pushed: list[np.ndarray] = []
    new_coords = np.zeros((5, 4))
    hook = initial_conditions_hook(lambda knobs, best: new_coords, {}, pushed.append)

    note = hook({"k1": 1.0}, {"k1": 1.0})

    assert len(pushed) == 1
    assert pushed[0] is new_coords
    assert note == "dic=0.00e+00, dic0=0.00e+00"


def test_initial_conditions_hook_pushes_nothing_when_the_callback_returns_none() -> None:
    from adelmo.tracking.fitter import initial_conditions_hook

    pushed: list[np.ndarray] = []
    hook = initial_conditions_hook(lambda knobs, best: None, {}, pushed.append)

    assert hook({"k1": 1.0}, {"k1": 1.0}) is None
    assert pushed == []


def test_initial_conditions_hook_reports_step_and_drift() -> None:
    from adelmo.tracking.fitter import initial_conditions_hook

    coords = iter([np.zeros((2, 4)), np.ones((2, 4)), np.full((2, 4), 3.0)])
    hook = initial_conditions_hook(lambda knobs, best: next(coords), {}, lambda _: None)

    hook({}, {})
    assert hook({}, {}) == "dic=1.00e+00, dic0=1.00e+00"
    assert hook({}, {}) == "dic=2.00e+00, dic0=3.00e+00"


def test_initial_conditions_hook_includes_non_optimised_strengths() -> None:
    """The callback must see fixed strengths, not just this stage's knobs.

    The optimisation loop rebuilds ``current_knobs`` from the knob names alone, so
    strengths supplied via ``initial_knob_strengths`` but not optimised only survive
    in ``MachineSetup.initial_model_values``. A callback that rebuilds a model from
    the knobs it is handed would otherwise fall back to the bare model defaults.
    """
    from adelmo.tracking.fitter import initial_conditions_hook

    seen: list[dict[str, float]] = []

    def callback(current: dict[str, float], best: dict[str, float]) -> None:
        seen.extend((current, best))

    fixed = {"kfixed": 3.0, "kopt": 0.0, "pt": 1e-3}
    initial_conditions_hook(callback, fixed, lambda _: None)({"kopt": 1.0}, {"kopt": 2.0})

    current, best = seen
    assert current == {"kfixed": 3.0, "kopt": 1.0, "pt": 1e-3}
    assert best == {"kfixed": 3.0, "kopt": 2.0, "pt": 1e-3}


def test_initial_conditions_hook_keeps_empty_best_knobs_empty() -> None:
    """An empty ``best_knobs`` must stay empty so callbacks can skip early epochs."""
    from adelmo.tracking.fitter import initial_conditions_hook

    seen: list[dict[str, float]] = []
    hook = initial_conditions_hook(
        lambda current, best: seen.append(best), {"kfixed": 3.0}, lambda _: None
    )

    hook({"kopt": 1.0}, {})
    assert seen == [{}]
