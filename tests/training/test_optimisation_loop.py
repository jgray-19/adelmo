from __future__ import annotations

import json
import multiprocessing as mp
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from adelmo.config import OptimiserConfig
from adelmo.fitting.protocol import Evaluate, GradReply, Start, WorkerChannels
from adelmo.optimisers.adam import AdamOptimiser
from adelmo.tracking.config.models import CheckpointConfig
from adelmo.tracking.sgd.checkpointing import OptimisationCheckpointer
from adelmo.tracking.sgd.loop import SGDLoop


def _make_loop(
    knob_names: list[str],
    *,
    max_epochs: int = 2,
    gradient_converged_value: float = 1e-6,
    true_strengths: dict[str, float] | None = None,
) -> SGDLoop:
    optimiser_config = OptimiserConfig(
        max_epochs=max_epochs,
        warmup_epochs=1,
        warmup_lr_start=1e-3,
        max_lr=1e-3,
        min_lr=1e-3,
        gradient_converged_value=gradient_converged_value,
        optimiser_type="adam",
    )
    return SGDLoop(
        knob_names, true_strengths=true_strengths or {}, config=optimiser_config, num_batches=1
    )


def _make_checkpointer(loop: SGDLoop, checkpoint_path) -> OptimisationCheckpointer:
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
    checkpoint_state = _make_checkpointer(loop, checkpoint_path).load(
        base_current_knobs=base_current
    )

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
    checkpoint_state = _make_checkpointer(loop, checkpoint_path).load(
        base_current_knobs=base_current
    )

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


def _run_fake_worker(conn, rounds: int, grad: np.ndarray, loss: float | list[float]) -> None:
    """Thread target: answer ``rounds`` evaluations with a fixed gradient.

    ``loss`` is the loss of every round, or a list with one loss per round.
    """
    conn.recv()  # Start
    for i in range(rounds):
        msg = conn.recv()
        if not isinstance(msg, Evaluate):
            break
        conn.send(GradReply(0, loss[i] if isinstance(loss, list) else loss, grad.copy()))


def _make_channels(
    n_knobs: int,
    rounds: int,
    *,
    losses: tuple[float | list[float], ...] = (0.0,),
    grad: float = 0.0,
) -> WorkerChannels:
    """Real WorkerChannels backed by one thread per entry of ``losses``."""
    parents = []
    for loss in losses:
        parent, child = mp.Pipe()
        threading.Thread(
            target=_run_fake_worker,
            args=(child, rounds, np.full(n_knobs, grad), loss),
            daemon=True,
        ).start()
        parent.send(Start({f"k{i}": 0.0 for i in range(n_knobs)}))
        parents.append(parent)
    return WorkerChannels(parents, [SimpleNamespace(pid=0, exitcode=None) for _ in parents])


def test_epoch_end_hook_called_once_per_epoch() -> None:
    """epoch_end_hook must be invoked exactly once after each completed epoch."""
    n_epochs, n_batches = 2, 1
    # Pin the loop to exactly n_epochs and disable gradient-norm early stopping.
    loop = _make_loop(["k1"], max_epochs=n_epochs, gradient_converged_value=-1.0)

    hook_calls: list[dict[str, float]] = []

    def hook(knobs: dict[str, float], _best: dict[str, float]) -> None:
        hook_calls.append(knobs.copy())

    loop.run(
        {"k1": 0.0},
        _make_channels(1, n_epochs * n_batches),
        total_turns=1,
        epoch_end_hook=hook,
    )

    assert len(hook_calls) == n_epochs
    assert all("k1" in call for call in hook_calls)


def test_epoch_end_hook_receives_updated_knobs() -> None:
    """The hook must receive knob values *after* the gradient update for that epoch."""
    n_epochs, n_batches = 2, 1
    loop = _make_loop(["k1"], max_epochs=n_epochs, gradient_converged_value=-1.0)

    seen_knobs: list[float] = []

    def hook(knobs: dict[str, float], _best: dict[str, float]) -> None:
        seen_knobs.append(knobs["k1"])

    loop.run(
        {"k1": 0.0},
        _make_channels(1, n_epochs * n_batches, losses=(1.0,), grad=1.0),
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
    loop = _make_loop(["k1"], max_epochs=n_epochs, gradient_converged_value=-1.0)

    notes = iter(["dic=1.00e-09", "dic=2.00e-09"])

    def hook(_knobs: dict[str, float], _best: dict[str, float]) -> str:
        return next(notes)

    with caplog.at_level(logging.INFO, logger="adelmo.tracking.sgd.loop"):
        loop.run(
            {"k1": 0.0},
            _make_channels(1, n_epochs * n_batches),
            total_turns=1,
            epoch_end_hook=hook,
        )

    epoch_lines = [record.getMessage() for record in caplog.records if "Ep " in record.getMessage()]
    assert len(epoch_lines) == n_epochs
    assert "dic=1.00e-09" in epoch_lines[0]
    assert "dic=2.00e-09" in epoch_lines[1]
    # Placed between the existing fields, not tacked past the [b]/[s] markers.
    assert epoch_lines[0].index("dic=") < epoch_lines[0].index("lr=")


def test_epoch_line_omits_the_note_when_the_hook_returns_none(caplog) -> None:
    """A hook that did nothing this epoch must not leave an empty field behind."""
    import logging

    loop = _make_loop(["k1"], max_epochs=1, gradient_converged_value=-1.0)

    with caplog.at_level(logging.INFO, logger="adelmo.tracking.sgd.loop"):
        loop.run(
            {"k1": 0.0},
            _make_channels(1, 1),
            total_turns=1,
            epoch_end_hook=lambda _knobs, _best: None,
        )

    epoch_lines = [record.getMessage() for record in caplog.records if "Ep " in record.getMessage()]
    assert epoch_lines
    assert ", ," not in epoch_lines[0]
    assert "dic=" not in epoch_lines[0]


def test_epoch_end_hook_none_does_not_raise() -> None:
    """Passing epoch_end_hook=None (the default) must not raise."""
    n_epochs, n_batches = 1, 1
    loop = _make_loop(["k1"], max_epochs=n_epochs, gradient_converged_value=-1.0)

    loop.run(
        {"k1": 0.0},
        _make_channels(1, n_epochs * n_batches),
        total_turns=1,
        epoch_end_hook=None,
    )


def test_epoch_loss_averages_only_the_workers_with_a_valid_loss() -> None:
    """A worker that lost its particles (NaN loss) must not dilute the epoch loss."""
    loop = _make_loop(["k0"], max_epochs=1, gradient_converged_value=-1.0)
    losses: list[float] = []

    loop.run(
        {"k0": 0.0},
        _make_channels(1, 1, losses=(2.0, 4.0, float("nan"))),
        total_turns=1,
        loss_callback=lambda _epoch, loss, *_: losses.append(loss),
    )

    assert losses == [3.0]


def test_best_epoch_is_the_lowest_loss_even_when_further_from_the_truth() -> None:
    """Best-selection uses the loss alone; the true strengths are only diagnostics."""
    loop = _make_loop(
        ["k0"], max_epochs=2, gradient_converged_value=-1.0, true_strengths={"k0": 0.0}
    )
    epoch_knobs: list[dict[str, float]] = []

    best = loop.run(
        {"k0": 0.0},
        # A tiny loss improvement while the gradient pushes k0 away from its truth.
        _make_channels(1, 2, losses=([1.0, 1.0 - 1e-6],), grad=1.0),
        total_turns=1,
        epoch_end_hook=lambda knobs, _best: epoch_knobs.append(knobs.copy()),
    )

    assert abs(epoch_knobs[1]["k0"]) > abs(epoch_knobs[0]["k0"])
    assert best == epoch_knobs[1]


def test_run_leaves_the_callers_knobs_untouched() -> None:
    """Seeding a zero ``pt`` must not write into the dict the caller passed."""
    loop = _make_loop(["k0", "pt"], max_epochs=1, gradient_converged_value=-1.0)
    knobs = {"k0": 0.0, "pt": 0.0}

    loop.run(knobs, _make_channels(2, 1), total_turns=1)

    assert knobs == {"k0": 0.0, "pt": 0.0}


def test_final_checkpoint_records_the_epoch_that_triggered_the_loss_stop(tmp_path) -> None:
    """The epoch that stops on a converged loss still counts as completed."""
    loop = _make_loop(["k0"], max_epochs=5, gradient_converged_value=-1.0)
    checkpoint_path = tmp_path / "checkpoint.json"

    loop.run(
        {"k0": 0.0},
        _make_channels(1, 5, losses=(1.0,), grad=1.0),
        total_turns=1,
        checkpoint_config=CheckpointConfig(
            checkpoint_path=checkpoint_path, checkpoint_every_n_epochs=10
        ),
    )

    assert loop.diagnostics.reason == "loss_converged"
    assert loop.diagnostics.iterations == 3
    assert json.loads(checkpoint_path.read_text())["saved_epoch"] == 2
