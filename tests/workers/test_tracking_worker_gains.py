"""BPM gains in the tracking worker: readings, loss gradient and Jacobian."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from adelmo.config import SimulationConfig
from adelmo.fitting.worker import WorkerConfig
from adelmo.machine.accelerators import PSB
from adelmo.fitting.bpm_gains import bpm_gain_names, point_gain_index
from adelmo.tracking.worker import PrecomputedTrackingWeights, TrackingData, TrackingWorker

BPMS = ("B1", "B2", "B3")
# Two turns of B1, B2 and a marker; B3 has a gain but is not observed here.
POINTS = ["B1", "B2", "MARKER"] * 2
N_PARTICLES, N_KNOBS = 4, 3


def _worker(tmp_path, optimise_momenta: bool) -> TrackingWorker:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    rng = np.random.default_rng(0)
    shape = (N_PARTICLES, len(POINTS))
    data = TrackingData(
        position_comparisons=rng.normal(size=(*shape, 2)),
        momentum_comparisons=rng.normal(size=(*shape, 2)),
        position_variances=np.ones((*shape, 2)),
        momentum_variances=np.ones((*shape, 2)),
        init_coords=np.ones((N_PARTICLES, 6)),
        init_pts=np.zeros(N_PARTICLES),
        reading_ids=np.zeros(shape, dtype=np.int64),
        init_reading_ids=np.zeros(N_PARTICLES, dtype=np.int64),
        init_variances=np.ones((N_PARTICLES, 2)),
        precomputed_weights=PrecomputedTrackingWeights(
            **{o: rng.uniform(0.5, 2.0, shape) for o in ("x", "y", "px", "py")}, scale=1.0
        ),
        gain_bpms=BPMS,
        point_gain_index=point_gain_index(POINTS, BPMS),
    )
    config = WorkerConfig(
        accelerator=PSB(ring=3, sequence_file=seq_file),
        tracking_start_bpm="BR3.BPM1L3",
        tracking_end_bpm="BR3.BPM3L3",
        magnet_range="$start/$end",
    )
    sim = SimulationConfig(num_workers=1, num_batches=1, optimise_momenta=optimise_momenta)
    return TrackingWorker(mp.Pipe()[1], 0, data, config, sim)


class _LinearModel:
    """A stand-in for MAD: every observable is ``m = A k + c``, exactly linear in the knobs."""

    def __init__(self, observables):
        rng = np.random.default_rng(1)
        self.a = {o: rng.normal(size=(N_PARTICLES, N_KNOBS, len(POINTS))) for o in observables}
        self.c = {o: rng.normal(size=(N_PARTICLES, len(POINTS))) for o in observables}

    def results(self, knobs: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for o in self.a:
            out[o] = np.einsum("pkn,k->pn", self.a[o], knobs) + self.c[o]
            out[f"d{o}_dk"] = self.a[o].copy()
        return out


def _loss_and_grad(worker, model, params, *, gain_columns=True):
    knobs, gains = params[:N_KNOBS], params[N_KNOBS:]
    worker._update_gains(dict(zip(bpm_gain_names(BPMS), gains)))
    results = worker._apply_gains(model.results(knobs), gain_columns=gain_columns)
    grad, loss = worker._compute_loss_and_gradients(results, 0)
    return loss, grad, results


def _finite_differences(worker, model, params, step=1e-6):
    fd = np.empty(params.size)
    for i in range(params.size):
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        fd[i] = (_loss_and_grad(worker, model, up)[0] - _loss_and_grad(worker, model, down)[0]) / (2 * step)
    return fd


def test_gain_derivatives_match_finite_differences(tmp_path) -> None:
    for optimise_momenta in (False, True):
        worker = _worker(tmp_path, optimise_momenta)
        model = _LinearModel(worker.observables)
        rng = np.random.default_rng(2)
        params = np.concatenate([rng.normal(size=N_KNOBS), rng.normal(0, 0.05, 2 * len(BPMS))])
        fd = _finite_differences(worker, model, params)

        # The uncertainty Jacobian carries the gain columns after the knobs.
        _, full, _ = _loss_and_grad(worker, model, params)
        np.testing.assert_allclose(full, fd, rtol=1e-6, atol=1e-8)

        # The optimiser's reply: the knob gradient alone, and the gain sums.
        _, knob_grad, results = _loss_and_grad(worker, model, params, gain_columns=False)
        np.testing.assert_allclose(knob_grad, fd[:N_KNOBS], rtol=1e-6, atol=1e-8)
        numerator, curvature = worker._gain_sums(results, 0)
        np.testing.assert_allclose(2 * (params[N_KNOBS:] * curvature - numerator), fd[N_KNOBS:], rtol=1e-6, atol=1e-8)
        # B3 is never observed: its sums are zero.
        assert curvature[2] == 0.0 and curvature[5] == 0.0


def test_gain_sums_give_the_least_squares_gains(tmp_path) -> None:
    worker = _worker(tmp_path, optimise_momenta=False)
    model = _LinearModel(worker.observables)
    params = np.zeros(N_KNOBS + 2 * len(BPMS))
    _, _, results = _loss_and_grad(worker, model, params, gain_columns=False)
    numerator, curvature = worker._gain_sums(results, 0)
    best = np.divide(numerator, curvature, out=np.zeros_like(numerator), where=curvature > 0)
    params[N_KNOBS:] = best
    np.testing.assert_allclose(_finite_differences(worker, model, params)[N_KNOBS:], 0.0, atol=1e-8)


def test_gains_scale_positions_only(tmp_path) -> None:
    worker = _worker(tmp_path, optimise_momenta=True)
    model = _LinearModel(worker.observables)
    knobs = np.ones(N_KNOBS)
    plain = model.results(knobs)
    worker._update_gains({"bpmgain.x.B2": 0.1, "bpmgain.y.B1": -0.2})
    gained = worker._apply_gains(model.results(knobs))
    np.testing.assert_allclose(gained["x"][:, [1, 4]], 1.1 * plain["x"][:, [1, 4]])
    np.testing.assert_allclose(gained["x"][:, [0, 2, 3, 5]], plain["x"][:, [0, 2, 3, 5]])
    np.testing.assert_allclose(gained["y"][:, [0, 3]], 0.8 * plain["y"][:, [0, 3]])
    np.testing.assert_allclose(gained["px"], plain["px"])
    np.testing.assert_allclose(gained["dpx_dk"][:, N_KNOBS:], 0.0)
