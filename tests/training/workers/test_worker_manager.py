from __future__ import annotations

import multiprocessing
import threading
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.config.tracking import TrackingPlan
from aba_optimiser.training.data_manager import FileTracks
from aba_optimiser.training.workers.manager import WorkerManager
from aba_optimiser.training.workers.pool import WorkerPool
from aba_optimiser.training.workers.screening import OutlierScreener
from aba_optimiser.training.workers.setup import WorkerRuntimeMetadata, WorkerSetupHelper
from aba_optimiser.workers import TrackingData, WorkerConfig
from aba_optimiser.workers.common import KickPlane, PrecomputedTrackingWeights, UncertaintyPart
from aba_optimiser.workers.tracking_validation import ValidationTrackingWorker

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
        child.send({"status": "ok"})

    threading.Thread(target=_run, daemon=True).start()
    return parent


BPMS = ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3", "BR3.BPM4L3"]


def _make_psb(tmp_path: Path) -> PSB:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    return PSB(ring=3, sequence_file=seq_file)


def _make_manager(
    tmp_path: Path,
    *,
    all_bpms: list[str] | None = None,
    interface_options_per_file: list[dict] | None = None,
) -> WorkerManager:
    bpms = all_bpms or BPMS
    return WorkerManager(
        WorkerSetupHelper(
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
    )


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


def test_create_worker_payloads_assigns_per_file_artifacts_from_file_turn_map(tmp_path: Path) -> None:
    manager = _make_manager(
        tmp_path,
        all_bpms=BPMS[:3],
        interface_options_per_file=[
            {"machine_state": tmp_path / "corr0_state.txt"},
            {"machine_state": tmp_path / "corr1_state.txt"},
        ],
    )

    manager.file_turn_map = {2: 0, 202: 1}
    manager.start_bpms = [BPMS[0]]
    manager.end_bpms = [BPMS[1]]
    manager.simulation_config = SimulationConfig(
        num_workers=2, num_batches=1, optimise_momenta=False
    )
    manager.machine_deltaps = [0.0, 1e-3]

    payloads = manager.create_worker_payloads(
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
    manager = _make_manager(tmp_path)
    manager.training.metadata = [
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

    masks = OutlierScreener(manager.payload_builder).build_bpm_masks_from_diagnostics(
        diagnostics=[
            {
                "worker_id": 0,
                "loss_per_bpm": [1.0, 50.0, 1.0, 50.0],
            }
        ],
        worker_metadata=manager.training.metadata,
        bpm_sigma_threshold=0.5,
    )

    assert len(masks) == 1
    assert masks[0].tolist() == [True, False]


def test_apply_screening_actions_expands_masks_across_turns(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    received: list[list[object]] = [[], []]
    manager.training.conns = [_start_recording_worker(sink) for sink in received]
    manager.training.workers = [_FakeWorker(), _FakeWorker()]  # type: ignore[list-item]
    manager.training.metadata = [
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

    OutlierScreener(manager.payload_builder).apply_screening_actions(
        manager.training,
        bpm_masks=[np.array([True, False]), np.array([False, True])],
        worker_disabled=[False, True],
    )

    assert received[0] == [
        {
            "cmd": "apply_mask",
            "keep_bpm_mask": [True, False, True, False],
            "disable_worker": False,
        }
    ]
    assert received[1] == [
        {
            "cmd": "apply_mask",
            "keep_bpm_mask": [False, True],
            "disable_worker": True,
        }
    ]

def test_summarise_screening_losses_logs_pre_and_projected_loss(tmp_path: Path, caplog) -> None:
    manager = _make_manager(tmp_path)
    manager.training.metadata = [
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
        OutlierScreener(manager.payload_builder).summarise_screening_losses(
            diagnostics=[{"worker_id": 0, "loss_per_bpm": [1.0, 9.0, 1.0, 9.0]}],
            bpm_masks=[np.array([True, False])],
            worker_disabled=[False],
            worker_metadata=manager.training.metadata,
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

    worker = ValidationTrackingWorker(
        conn=_FakeConn([]),  # type: ignore[arg-type]
        worker_id=0,
        payloads=[(data, config, 0)],
        simulation_config=SimulationConfig(num_workers=1, num_batches=8),
    )

    assert worker.track_count == n_tracks
    assert worker.num_batches == 5
    assert [len(batch) for batch in worker.init_coords] == [2, 2, 2, 2, 2]


def _uncertainty_manager(
    tmp_path: Path, file_indices: list[int], responses: list[list[object]]
) -> WorkerManager:
    """A manager whose training workers are pipe threads, one per entry of ``file_indices``."""
    manager = _make_manager(tmp_path)
    for worker_responses in responses:
        parent, child = multiprocessing.Pipe()
        _start_pipe_worker(child, worker_responses)
        manager.training.conns.append(parent)
    manager.training.workers = [_FakeWorker() for _ in file_indices]  # type: ignore[misc]
    manager.training.metadata = [
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
    manager.training.particle_counts = [1 for _ in file_indices]
    return manager


def test_termination_and_hessian_merges_shared_readings_per_file(tmp_path: Path) -> None:
    # Workers 0 and 2 share reading 7 of file 0; worker 1 (file 1) is drained between them.
    g_a, g_b, g_c = np.array([1.0, 2.0]), np.array([3.0, -1.0]), np.array([0.5, 0.5])
    parts = [
        UncertaintyPart(np.eye(2), np.array([7]), g_a[None, :], np.array([0.1])),
        UncertaintyPart(2.0 * np.eye(2), np.array([8]), g_c[None, :], np.array([0.2])),
        UncertaintyPart(3.0 * np.eye(2), np.array([7]), g_b[None, :], np.array([0.1])),
    ]
    manager = _uncertainty_manager(tmp_path, [0, 1, 0], [[part] for part in parts])

    normal, noise = manager.termination_and_hessian(2)

    shared = g_a + g_b
    np.testing.assert_allclose(normal, 6.0 * np.eye(2))
    np.testing.assert_allclose(noise, 0.1 * np.outer(shared, shared) + 0.2 * np.outer(g_c, g_c))
    assert [worker.join_calls for worker in manager.training.workers] == [1, 1, 1]


def test_terminate_workers_kills_training_and_validation_workers(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    training = [_FakeWorker(), _FakeWorker()]
    validation = [_FakeWorker()]
    manager.training = WorkerPool(workers=training)  # type: ignore[arg-type]
    manager.validation = WorkerPool(workers=validation)  # type: ignore[arg-type]

    manager.terminate_workers()

    for worker in (*training, *validation):
        assert worker.terminate_calls == 1
        assert worker.join_calls == 1


def test_pool_stop_does_not_wait_for_final_payload() -> None:
    worker = _FakeWorker()
    conn = _ConnThatMustNotPoll()
    pool = WorkerPool(conns=[conn], workers=[worker])  # type: ignore[list-item]

    pool.stop()

    assert conn.sent == (None, None)
    assert worker.join_calls == 1


def test_termination_and_hessian_disables_hessian_before_shutdown(tmp_path: Path) -> None:
    # Two messages: ack for set_hessian_mode, then the empty part on termination
    manager = _uncertainty_manager(
        tmp_path, [0], [[{"worker_id": 0, "status": "ok"}, UncertaintyPart.empty(2)]]
    )

    total, _ = manager.termination_and_hessian(2, estimate_hessian=False)

    np.testing.assert_allclose(total, np.zeros((2, 2), dtype=np.float64))
    assert manager.training.workers[0].join_calls == 1
