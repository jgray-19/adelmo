from __future__ import annotations

import json
import multiprocessing as mp
import threading

import numpy as np
import pytest

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.optimisers.adam import AdamOptimiser
from aba_optimiser.training.config.models import CheckpointConfig
from aba_optimiser.training.optimisation.checkpointing import OptimisationCheckpointer
from aba_optimiser.training.optimisation.loop import OptimisationLoop
from aba_optimiser.workers.protocol import Evaluate, GradReply, Start, WorkerChannels


def _make_loop(knob_names: list[str]) -> OptimisationLoop:
    optimiser_config = OptimiserConfig(
        max_epochs=2,
        warmup_epochs=1,
        warmup_lr_start=1e-3,
        max_lr=1e-3,
        min_lr=1e-3,
        gradient_converged_value=1e-6,
        optimiser_type="adam",
    )
    simulation_config = SimulationConfig(
        num_workers=1,
        num_batches=1,
    )
    initial_strengths = np.zeros(len(knob_names), dtype=float)
    return OptimisationLoop(
        initial_strengths=initial_strengths,
        knob_names=knob_names,
        true_strengths={},
        optimiser_config=optimiser_config,
        simulation_config=simulation_config,
    )


def _make_checkpointer(loop: OptimisationLoop, checkpoint_path) -> OptimisationCheckpointer:
    return OptimisationCheckpointer(loop, CheckpointConfig(checkpoint_path=checkpoint_path))


def test_load_checkpoint_allows_current_knob_superset(tmp_path) -> None:
    loop = _make_loop(["k1", "k2", "k3"])

    checkpoint_payload = {
        "saved_epoch": 3,
        "next_epoch": 4,
        "knob_names": ["k1", "k2"],
        "current_knobs": {"k1": 1.5, "k2": -2.0},
        "best_knobs": {"k1": 1.0, "k2": -1.0},
        "best_loss": 0.25,
        "prev_loss": 0.3,
        "smoothed_grad_norm": 1e-3,
        "smoothed_loss_change": 2e-3,
        "max_clipping_ratio": 1.2,
        "optimiser_state": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "m": [1.0, 2.0],
            "v": [3.0, 4.0],
            "t": 7,
        },
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    base_current = {"k1": 10.0, "k2": 20.0, "k3": 30.0}
    checkpoint_state = _make_checkpointer(loop, checkpoint_path).load(base_current_knobs=base_current)

    assert checkpoint_state["saved_epoch"] == 3
    assert checkpoint_state["next_epoch"] == 4
    assert checkpoint_state["current_knobs"] == {"k1": 1.5, "k2": -2.0, "k3": 30.0}
    assert checkpoint_state["prev_loss"] == 0.3

    assert loop.best.value == {"k1": 1.0, "k2": -1.0, "k3": 30.0}
    assert loop.best.loss == 0.25

    # Optimiser state should be remapped and padded for the extra knob.
    assert isinstance(loop.optimiser, AdamOptimiser)
    assert loop.optimiser.t == 7
    assert np.allclose(loop.optimiser.m, [1.0, 2.0, 0.0])
    assert np.allclose(loop.optimiser.v, [3.0, 4.0, 0.0])


def test_load_checkpoint_rejects_missing_current_checkpoint_knobs(tmp_path) -> None:
    loop = _make_loop(["k1"])  # current setup is missing k2 from checkpoint

    checkpoint_payload = {
        "knob_names": ["k1", "k2"],
        "current_knobs": {"k1": 1.0, "k2": 2.0},
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    with pytest.raises(ValueError, match="missing checkpoint knobs"):
        _make_checkpointer(loop, checkpoint_path).load()


def test_load_checkpoint_rejects_non_finite_knob_values(tmp_path) -> None:
    loop = _make_loop(["k1", "k2"])

    checkpoint_payload = {
        "knob_names": ["k1", "k2"],
        "current_knobs": {"k1": float("nan"), "k2": 2.0},
        "best_knobs": {"k1": 1.0, "k2": 2.0},
    }
    checkpoint_path = tmp_path / "checkpoint_nan.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    with pytest.raises(ValueError, match="non-finite knob values"):
        _make_checkpointer(loop, checkpoint_path).load()


def test_load_checkpoint_remaps_and_pads_in_current_knob_order(tmp_path) -> None:
    # Current optimisation order differs from checkpoint order and adds one extra knob.
    loop = _make_loop(["k3", "k1", "k2", "k4"])

    checkpoint_payload = {
        "saved_epoch": 5,
        "next_epoch": 6,
        "knob_names": ["k1", "k2", "k3"],
        "current_knobs": {"k1": 10.0, "k2": 20.0, "k3": 30.0},
        "best_knobs": {"k1": 1.0, "k2": 2.0, "k3": 3.0},
        "optimiser_state": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "m": [100.0, 200.0, 300.0],
            "v": [1.0, 2.0, 3.0],
            "t": 11,
        },
    }
    checkpoint_path = tmp_path / "checkpoint_order.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    base_current = {"k3": -3.0, "k1": -1.0, "k2": -2.0, "k4": 99.0}
    checkpoint_state = _make_checkpointer(loop, checkpoint_path).load(base_current_knobs=base_current)

    # Values should be in current order: [k3, k1, k2, k4].
    assert checkpoint_state["current_knobs"] == {
        "k3": 30.0,
        "k1": 10.0,
        "k2": 20.0,
        "k4": 99.0,
    }
    assert loop.best.value == {
        "k3": 3.0,
        "k1": 1.0,
        "k2": 2.0,
        "k4": 99.0,
    }

    # Optimiser vectors are remapped by knob name then padded for k4.
    # checkpoint m/v order was [k1, k2, k3] = [100, 200, 300] / [1, 2, 3]
    # current order is [k3, k1, k2, k4] -> [300, 100, 200, 0] / [3, 1, 2, 0]
    assert isinstance(loop.optimiser, AdamOptimiser)
    assert np.allclose(loop.optimiser.m, [300.0, 100.0, 200.0, 0.0])
    assert np.allclose(loop.optimiser.v, [3.0, 1.0, 2.0, 0.0])
    assert loop.optimiser.t == 11


def _run_fake_worker(conn, n_epochs: int, n_batches: int, grad: np.ndarray, loss: float) -> None:
    """Thread target: act as a gradient-descent worker for n_epochs * n_batches rounds."""
    worker_id = 0
    conn.recv()  # Start
    for _ in range(n_epochs * n_batches):
        msg = conn.recv()
        if not isinstance(msg, Evaluate):
            break
        conn.send(GradReply(worker_id, loss, grad.copy()))


def _make_real_channels(n_knobs: int, n_epochs: int, n_batches: int) -> WorkerChannels:
    """Return real WorkerChannels backed by a thread acting as a single worker."""
    parent, child = mp.Pipe()
    worker_thread = threading.Thread(
        target=_run_fake_worker,
        args=(child, n_epochs, n_batches, np.zeros(n_knobs), 0.0),
        daemon=True,
    )
    worker_thread.start()

    # Send the startup handshake that the worker expects before the loop begins.
    parent.send(Start({f"k{i}": 0.0 for i in range(n_knobs)}))

    from types import SimpleNamespace

    proc = SimpleNamespace(pid=0, exitcode=None)

    channels = object.__new__(WorkerChannels)
    channels.parent_conns = (parent,)
    channels.workers = (proc,)
    channels._count = 1
    return channels


def _make_real_channels_nonzero_grad(n_knobs: int, n_epochs: int, n_batches: int) -> WorkerChannels:
    """Real channels where the fake worker returns a non-zero gradient."""
    from types import SimpleNamespace

    parent, child = mp.Pipe()
    threading.Thread(
        target=_run_fake_worker,
        args=(child, n_epochs, n_batches, np.ones(n_knobs), 1.0),
        daemon=True,
    ).start()
    parent.send(Start({f"k{i}": 0.0 for i in range(n_knobs)}))

    proc = SimpleNamespace(pid=0, exitcode=None)

    channels = object.__new__(WorkerChannels)
    channels.parent_conns = (parent,)
    channels.workers = (proc,)
    channels._count = 1
    return channels


def test_epoch_end_hook_called_once_per_epoch() -> None:
    """epoch_end_hook must be invoked exactly once after each completed epoch."""
    n_epochs, n_batches = 2, 1
    loop = _make_loop(["k1"])
    # Pin the loop to exactly n_epochs and disable gradient-norm early stopping.
    loop.max_epochs = n_epochs
    loop.gradient_converged_value = -1.0

    hook_calls: list[dict[str, float]] = []

    def hook(knobs: dict[str, float], _best: dict[str, float]) -> None:
        hook_calls.append(knobs.copy())

    loop.run_optimisation(
        current_knobs={"k1": 0.0},
        channels=_make_real_channels(1, n_epochs, n_batches),
        writer=None,
        run_start=0.0,
        total_turns=1,
        epoch_end_hook=hook,
    )

    assert len(hook_calls) == n_epochs
    assert all("k1" in call for call in hook_calls)


def test_epoch_end_hook_receives_updated_knobs() -> None:
    """The hook must receive knob values *after* the gradient update for that epoch."""
    n_epochs, n_batches = 2, 1
    loop = _make_loop(["k1"])
    loop.max_epochs = n_epochs
    loop.gradient_converged_value = -1.0

    seen_knobs: list[float] = []

    def hook(knobs: dict[str, float], _best: dict[str, float]) -> None:
        seen_knobs.append(knobs["k1"])

    loop.run_optimisation(
        current_knobs={"k1": 0.0},
        channels=_make_real_channels_nonzero_grad(1, n_epochs, n_batches),
        writer=None,
        run_start=0.0,
        total_turns=1,
        epoch_end_hook=hook,
    )

    assert seen_knobs[0] != 0.0


def test_epoch_end_hook_note_is_appended_to_the_epoch_log_line(caplog) -> None:
    """What the hook returns lands on that epoch's own log line.

    A hook that changes the run -- refreshing the workers' initial conditions --
    otherwise reports into a second stream the reader has to interleave with the
    losses by hand, which is exactly the comparison being made.
    """
    import logging

    n_epochs, n_batches = 2, 1
    loop = _make_loop(["k1"])
    loop.max_epochs = n_epochs
    loop.gradient_converged_value = -1.0

    notes = iter(["dic=1.00e-09", "dic=2.00e-09"])

    def hook(_knobs: dict[str, float], _best: dict[str, float]) -> str:
        return next(notes)

    with caplog.at_level(logging.INFO, logger="aba_optimiser.training.optimisation.loop"):
        loop.run_optimisation(
            current_knobs={"k1": 0.0},
            channels=_make_real_channels(1, n_epochs, n_batches),
            writer=None,
            run_start=0.0,
            total_turns=1,
            epoch_end_hook=hook,
        )

    epoch_lines = [
        record.getMessage() for record in caplog.records if "Ep " in record.getMessage()
    ]
    assert len(epoch_lines) == n_epochs
    assert "dic=1.00e-09" in epoch_lines[0]
    assert "dic=2.00e-09" in epoch_lines[1]
    # Placed between the existing fields, not tacked past the [b]/[s] markers.
    assert epoch_lines[0].index("dic=") < epoch_lines[0].index("lr=")


def test_epoch_line_omits_the_note_when_the_hook_returns_none(caplog) -> None:
    """A hook that did nothing this epoch must not leave an empty field behind."""
    import logging

    loop = _make_loop(["k1"])
    loop.max_epochs = 1
    loop.gradient_converged_value = -1.0

    with caplog.at_level(logging.INFO, logger="aba_optimiser.training.optimisation.loop"):
        loop.run_optimisation(
            current_knobs={"k1": 0.0},
            channels=_make_real_channels(1, 1, 1),
            writer=None,
            run_start=0.0,
            total_turns=1,
            epoch_end_hook=lambda _knobs, _best: None,
        )

    epoch_lines = [
        record.getMessage() for record in caplog.records if "Ep " in record.getMessage()
    ]
    assert epoch_lines
    assert ", ," not in epoch_lines[0]
    assert "dic=" not in epoch_lines[0]


def test_epoch_end_hook_none_does_not_raise() -> None:
    """Passing epoch_end_hook=None (the default) must not raise."""
    n_epochs, n_batches = 1, 1
    loop = _make_loop(["k1"])
    loop.max_epochs = n_epochs
    loop.gradient_converged_value = -1.0

    loop.run_optimisation(
        current_knobs={"k1": 0.0},
        channels=_make_real_channels(1, n_epochs, n_batches),
        writer=None,
        run_start=0.0,
        total_turns=1,
        epoch_end_hook=None,
    )
