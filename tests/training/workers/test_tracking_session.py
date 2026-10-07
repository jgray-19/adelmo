from __future__ import annotations

import multiprocessing
import threading
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import SimulationConfig
from aba_optimiser.fitting.pool import WorkerPool
from aba_optimiser.fitting.protocol import STOP, Ack, CommandKind, LossReply
from aba_optimiser.fitting.worker import KickPlane, WorkerConfig
from aba_optimiser.machine.accelerators import PSB
from aba_optimiser.tracking.config.tracking import TrackingPlan
from aba_optimiser.tracking.data_manager import FileTracks
from aba_optimiser.tracking.dispatch.payloads import WorkerPayloadBuilder
from aba_optimiser.tracking.dispatch.screening import OutlierScreener
from aba_optimiser.tracking.dispatch.setup import WorkerRuntimeMetadata, WorkerSetupHelper
from aba_optimiser.tracking.session import TrackingSession
from aba_optimiser.tracking.uncertainty import UncertaintyPart, drain_uncertainty
from aba_optimiser.tracking.worker import PrecomputedTrackingWeights, TrackingData, TrackingWorker

if TYPE_CHECKING:
    from pathlib import Path


class _FakeConn:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.sent: list[dict[str, object]] = []
        self._responses = responses

    def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    def recv(self) -> dict[str, object]:
        return self._responses.pop(0)


class _FakeWorker:
    def __init__(self) -> None:
        self.join_calls = 0
        self.terminate_calls = 0
        self.exitcode = 0
        self.pid = 1234

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminate_calls += 1


class _ConnThatMustNotPoll:
    def send(self, payload: object) -> None:
        self.sent = payload

    def poll(self, timeout: float | None = None) -> bool:
        raise AssertionError(f"Validation cleanup should not poll for payloads, got {timeout}")

    def recv(self) -> object:
        raise AssertionError("Validation cleanup should not receive a final payload")

    def close(self) -> None:
        pass


def _start_pipe_worker(child_conn: multiprocessing.connection.Connection, responses: list[object]) -> threading.Thread:
    """Start a daemon thread that acts as a fake worker over a real pipe.

    The thread receives one binary message per response (via recv_bytes), then sends
    the corresponding response back via send(). This matches the WorkerChannels
    protocol where the parent sends via send_bytes and receives via recv.
    """
    def _run() -> None:
        for response in responses:
            child_conn.recv_bytes()
            child_conn.send(response)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _start_recording_worker(sink: list[object]) -> multiprocessing.connection.Connection:
    """Return a parent pipe end whose fake worker records one message and acks it."""
    parent, child = multiprocessing.Pipe()

    def _run() -> None:
        sink.append(child.recv())
        child.send(Ack(0))

    threading.Thread(target=_run, daemon=True).start()
    return parent


BPMS = ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3", "BR3.BPM4L3"]


def _make_psb(tmp_path: Path) -> PSB:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    return PSB(ring=3, sequence_file=seq_file)


def _make_setup_helper(
    tmp_path: Path,
    *,
    all_bpms: list[str] | None = None,
    interface_options_per_file: list[dict] | None = None,
) -> WorkerSetupHelper:
    bpms = all_bpms or BPMS
    return WorkerSetupHelper(
        accelerator=_make_psb(tmp_path),
        all_bpms=bpms,
        fixed_start=bpms[0],
        fixed_end=bpms[-1],
        use_fixed_bpm=True,
        bad_bpms=None,
        file_kick_planes=dict.fromkeys(range(4), "xy"),
        magnet_range="$start/$end",
        interface_options_per_file=interface_options_per_file
        or [{"machine_state": tmp_path / "correctors_state.txt"}],
        debug=False,
        mad_logfile=None,
        python_logfile=None,
        tracking_plan=TrackingPlan(),
    )


def _screener(tmp_path: Path) -> OutlierScreener:
    return OutlierScreener(WorkerPayloadBuilder(_make_psb(tmp_path)))


def _make_tracks(all_bpms: list[str], turns: list[int]) -> FileTracks:
    grid = np.array(
        [[10.0 * (t + 1) + (b + 1) for b in range(len(all_bpms))] for t in range(len(turns))]
    )
    values = {
        "x": grid,
        "px": grid * 0.01,
        "y": grid * 0.1,
        "py": grid * 0.001,
        "var_x": np.ones_like(grid),
        "var_px": np.ones_like(grid),
        "var_y": np.ones_like(grid),
        "var_py": np.ones_like(grid),
    }
    return FileTracks(
        turns=np.array(turns), markers=list(all_bpms), values=values, kick_plane="xy"
    )


def test_session_payloads_take_per_file_artifacts_from_the_file_turn_map(tmp_path: Path) -> None:
    session = TrackingSession(
        _make_setup_helper(
            tmp_path,
            all_bpms=BPMS[:3],
            interface_options_per_file=[
                {"machine_state": tmp_path / "corr0_state.txt"},
                {"machine_state": tmp_path / "corr1_state.txt"},
            ],
        ),
        SimulationConfig(num_workers=2, num_batches=1, optimise_momenta=False),
        turn_batches=[[2], [202]],
        validation_turn_batches=[],
        file_turn_map={2: 0, 202: 1},
        start_bpms=[BPMS[0]],
        end_bpms=[BPMS[1]],
        machine_deltaps=[0.0, 1e-3],
    )

    payloads = session.create_worker_payloads(
        tracks={
            0: _make_tracks(BPMS[:3], [1, 2, 3]),
            1: _make_tracks(BPMS[:3], [201, 202, 203]),
        },
        turn_batches=[[2], [202]],
    )

    assert [file_idx for _, _, file_idx in payloads] == [0, 1, 0, 1]
    assert [config.interface_options for _, config, _ in payloads] == [
        {"machine_state": tmp_path / "corr0_state.txt"},
        {"machine_state": tmp_path / "corr1_state.txt"},
        {"machine_state": tmp_path / "corr0_state.txt"},
        {"machine_state": tmp_path / "corr1_state.txt"},
    ]

    init_pts = [float(data.init_pts[0]) for data, _, _ in payloads]
    assert init_pts[0] == init_pts[2]
    assert init_pts[1] == init_pts[3]
    assert init_pts[0] != init_pts[1]


def test_build_bpm_masks_from_diagnostics_aggregates_multi_turn_losses(tmp_path: Path) -> None:
    metadata = [
        WorkerRuntimeMetadata(
            worker_id=0,
            file_idx=0,
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        )
    ]

    masks = _screener(tmp_path).build_bpm_masks_from_diagnostics(
        diagnostics=[LossReply(0, 102.0, np.array([1.0, 50.0, 1.0, 50.0]))],
        worker_metadata=metadata,
        bpm_sigma_threshold=0.5,
    )

    assert len(masks) == 1
    assert masks[0].tolist() == [True, False]


def test_apply_screening_actions_expands_masks_across_turns(tmp_path: Path) -> None:
    pool = WorkerPool()
    received: list[list[object]] = [[], []]
    pool.conns = [_start_recording_worker(sink) for sink in received]
    pool.workers = [_FakeWorker(), _FakeWorker()]  # type: ignore[list-item]
    pool.metadata = [
        WorkerRuntimeMetadata(
            worker_id=0,
            file_idx=0,
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        ),
        WorkerRuntimeMetadata(
            worker_id=1,
            file_idx=1,
            start_bpm="BPV.13308",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=1,
            bpm_names=["BPV.13308", "BPV.20108"],
        ),
    ]

    _screener(tmp_path).apply_screening_actions(
        pool,
        bpm_masks=[np.array([True, False]), np.array([False, True])],
        worker_disabled=[False, True],
    )

    for sink, mask, disabled in zip(
        received, ([True, False, True, False], [False, True]), (False, True), strict=True
    ):
        (command,) = sink
        assert command.kind is CommandKind.APPLY_MASK
        assert command.payload["keep_bpm_mask"].tolist() == mask
        assert command.payload["disable_worker"] is disabled

def test_summarise_screening_losses_logs_pre_and_projected_loss(tmp_path: Path, caplog) -> None:
    metadata = [
        WorkerRuntimeMetadata(
            worker_id=0,
            file_idx=0,
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        )
    ]

    with caplog.at_level("INFO"):
        _screener(tmp_path).summarise_screening_losses(
            diagnostics=[LossReply(0, 20.0, np.array([1.0, 9.0, 1.0, 9.0]))],
            bpm_masks=[np.array([True, False])],
            worker_disabled=[False],
            worker_metadata=metadata,
        )

    assert "Pre-screening loss summary" in caplog.text
    assert "Projected post-screening loss summary" in caplog.text
    assert "total=2.000000e+01" in caplog.text
    assert "total=2.000000e+00" in caplog.text


def test_validation_worker_keeps_all_held_out_turns_with_nondividing_batch_count(
    tmp_path: Path,
) -> None:
    accelerator = _make_psb(tmp_path)
    n_tracks = 10
    n_points = 3
    shape = (n_tracks, n_points)
    data = TrackingData(
        position_comparisons=np.zeros((n_tracks, n_points, 2), dtype=np.float64),
        momentum_comparisons=np.zeros((n_tracks, n_points, 2), dtype=np.float64),
        position_variances=np.ones((n_tracks, n_points, 2), dtype=np.float64),
        momentum_variances=np.ones((n_tracks, n_points, 2), dtype=np.float64),
        init_coords=np.zeros((n_tracks, 6), dtype=np.float64),
        init_pts=np.zeros((n_tracks,), dtype=np.float64),
        reading_ids=np.zeros(shape, dtype=np.int64),
        init_reading_ids=np.zeros((n_tracks,), dtype=np.int64),
        init_variances=np.ones((n_tracks, 2), dtype=np.float64),
        precomputed_weights=PrecomputedTrackingWeights(
            x=np.ones(shape, dtype=np.float64),
            y=np.ones(shape, dtype=np.float64),
            px=np.ones(shape, dtype=np.float64),
            py=np.ones(shape, dtype=np.float64),
            scale=1.0,
        ),
    )
    config = WorkerConfig(
        accelerator=accelerator,
        tracking_start_bpm="BPH.13008",
        tracking_end_bpm="BPH.13408",
        magnet_range="$start/$end",
        interface_options={},
        sdir=1,
        kick_plane="xy",
    )

    worker = TrackingWorker(
        conn=_FakeConn([]),  # type: ignore[arg-type]
        worker_id=0,
        data=data,
        config=config,
        simulation_config=SimulationConfig(num_workers=1, num_batches=8),
        validation=True,
    )

    assert sum(len(batch) for batch in worker.init_coords) == n_tracks
    assert worker.num_batches == 5
    assert [len(batch) for batch in worker.init_coords] == [2, 2, 2, 2, 2]


def _uncertainty_pool(file_indices: list[int], responses: list[list[object]]) -> WorkerPool:
    """A pool whose workers are pipe threads, one per entry of ``file_indices``."""
    pool = WorkerPool()
    for worker_responses in responses:
        parent, child = multiprocessing.Pipe()
        _start_pipe_worker(child, worker_responses)
        pool.conns.append(parent)
    pool.workers = [_FakeWorker() for _ in file_indices]  # type: ignore[misc]
    pool.metadata = [
        WorkerRuntimeMetadata(
            worker_id=idx,
            file_idx=file_idx,
            start_bpm=BPMS[0],
            end_bpm=BPMS[-1],
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=1,
            bpm_names=BPMS,
        )
        for idx, file_idx in enumerate(file_indices)
    ]
    pool.particle_counts = [1 for _ in file_indices]
    return pool


def test_drain_uncertainty_merges_shared_readings_per_file() -> None:
    # Workers 0 and 2 share reading 7 of file 0; worker 1 (file 1) is drained between them.
    g_a, g_b, g_c = np.array([1.0, 2.0]), np.array([3.0, -1.0]), np.array([0.5, 0.5])
    parts = [
        UncertaintyPart(np.eye(2), np.array([7]), g_a[None, :], np.array([0.1])),
        UncertaintyPart(2.0 * np.eye(2), np.array([8]), g_c[None, :], np.array([0.2])),
        UncertaintyPart(3.0 * np.eye(2), np.array([7]), g_b[None, :], np.array([0.1])),
    ]
    pool = _uncertainty_pool([0, 1, 0], [[part] for part in parts])

    normal, noise = drain_uncertainty(pool, 2, propagate_uncertainty=True)

    shared = g_a + g_b
    np.testing.assert_allclose(normal, 6.0 * np.eye(2))
    np.testing.assert_allclose(noise, 0.1 * np.outer(shared, shared) + 0.2 * np.outer(g_c, g_c))
    assert [worker.join_calls for worker in pool.workers] == [1, 1, 1]


def test_terminate_kills_training_and_validation_workers(tmp_path: Path) -> None:
    session = TrackingSession(
        _make_setup_helper(tmp_path),
        SimulationConfig(num_workers=1, num_batches=1),
        turn_batches=[],
        validation_turn_batches=[],
        file_turn_map={},
        start_bpms=[],
        end_bpms=[],
        machine_deltaps=[],
    )
    training = [_FakeWorker(), _FakeWorker()]
    validation = [_FakeWorker()]
    session.training = WorkerPool(workers=training)  # type: ignore[arg-type]
    session.validation = WorkerPool(workers=validation)  # type: ignore[arg-type]

    session.terminate()

    for worker in (*training, *validation):
        assert worker.terminate_calls == 1
        assert worker.join_calls == 1


def test_pool_stop_does_not_wait_for_final_payload() -> None:
    worker = _FakeWorker()
    conn = _ConnThatMustNotPoll()
    pool = WorkerPool(conns=[conn], workers=[worker])  # type: ignore[list-item]

    pool.stop()

    assert conn.sent == STOP
    assert worker.join_calls == 1


def test_drain_uncertainty_disables_the_hessian_before_shutdown() -> None:
    # Two messages: ack for set_uncertainty_mode, then the empty part on termination
    pool = _uncertainty_pool([0], [[Ack(0), UncertaintyPart.empty(2)]])

    total, _ = drain_uncertainty(pool, 2, propagate_uncertainty=False)

    np.testing.assert_allclose(total, np.zeros((2, 2), dtype=np.float64))
    assert pool.workers[0].join_calls == 1
