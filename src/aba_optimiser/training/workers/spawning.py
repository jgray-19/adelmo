"""Tracking-worker process spawning.

:class:`WorkerSpawner` turns prepared training/validation payloads into running
worker processes. It owns the worker-class registry and the per-payload startup
sequence (pipe creation, process start, initial-knob handshake, runtime metadata),
returning the spawned processes and metadata as plain records so the
:class:`~aba_optimiser.training.workers.manager.WorkerManager` stays out of the
process-creation business.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aba_optimiser.training.workers.validation import payload_track_count
from aba_optimiser.workers import (
    PositionOnlyValidationTrackingWorker,
    TrackingWorker,
    ValidationTrackingWorker,
)
from aba_optimiser.workers.tracking_position_only import PositionOnlyTrackingWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.workers.setup import WorkerRuntimeMetadata, WorkerSetupHelper

# Maps (optimise_momenta, validation) -> worker class.
# Add new worker types here without touching WorkerManager.
_WORKER_CLASS_REGISTRY: dict[tuple[bool, bool], type] = {
    (True, False): TrackingWorker,
    (False, False): PositionOnlyTrackingWorker,
    (True, True): ValidationTrackingWorker,
    (False, True): PositionOnlyValidationTrackingWorker,
}

LOGGER = logging.getLogger(__name__)


@dataclass
class SpawnedTrainingWorkers:
    """Processes and metadata produced when spawning training workers."""

    parent_conns: list[Connection] = field(default_factory=list)
    workers: list[mp.Process] = field(default_factory=list)
    worker_metadata: list[WorkerRuntimeMetadata] = field(default_factory=list)
    particle_counts: list[int] = field(default_factory=list)


@dataclass
class SpawnedValidationWorkers:
    """Processes and metadata produced when spawning validation workers."""

    parent_conns: list[Connection] = field(default_factory=list)
    workers: list[mp.Process] = field(default_factory=list)
    metadata: list[WorkerRuntimeMetadata] = field(default_factory=list)
    loss_weights: list[float] = field(default_factory=list)
    particle_counts: list[int] = field(default_factory=list)


class WorkerSpawner:
    """Spawn tracking and validation worker processes from prepared payloads."""

    def __init__(self, setup_helper: WorkerSetupHelper) -> None:
        self.setup_helper = setup_helper

    @staticmethod
    def select_worker_class(
        kick_plane: str,
        optimise_momenta: bool,
        *,
        validation: bool = False,
    ) -> type:
        """Select the worker implementation for a payload."""
        if kick_plane not in {"xy", "x", "y"}:
            raise ValueError(f"Unsupported kick plane {kick_plane!r}")
        worker_class = _WORKER_CLASS_REGISTRY.get((optimise_momenta, validation))
        if worker_class is None:
            raise ValueError(
                f"No worker class registered for optimise_momenta={optimise_momenta}, validation={validation}"
            )
        return worker_class

    def spawn_training(
        self,
        training_payloads: list,
        simulation_config: SimulationConfig,
        n_run_turns: int,
        initial_knobs: dict[str, float],
    ) -> SpawnedTrainingWorkers:
        """Spawn one process per training payload and record its runtime metadata."""
        spawned = SpawnedTrainingWorkers()
        for worker_id, (data, config, file_idx) in enumerate(training_payloads):
            parent, child = mp.Pipe()
            worker_class = self.select_worker_class(
                config.kick_plane,
                simulation_config.optimise_momenta,
                validation=False,
            )
            worker = worker_class(
                child,
                worker_id,
                data,
                config,
                simulation_config,
            )
            worker.start()
            spawned.parent_conns.append(parent)
            spawned.workers.append(worker)
            spawned.particle_counts.append(len(data.init_coords))
            parent.send((initial_knobs, -1))
            # The config's bad BPMs already exclude every BPM blind to its plane.
            bpm_names = self.setup_helper.get_range_bpm_names(
                config.tracking_start_bpm,
                config.tracking_end_bpm,
                config.sdir,
                config.bad_bpms,
            )
            spawned.worker_metadata.append(
                self.setup_helper.make_runtime_metadata(
                    worker_id=worker_id,
                    file_idx=file_idx,
                    config=config,
                    bpm_names=bpm_names,
                    n_run_turns=n_run_turns,
                )
            )
            LOGGER.debug(
                "Trn worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d",
                worker_id,
                file_idx,
                config.tracking_start_bpm,
                config.tracking_end_bpm,
                config.sdir,
                config.kick_plane,
                len(bpm_names),
            )
        return spawned

    def spawn_validation(
        self,
        validation_payloads: list,
        training_worker_count: int,
        simulation_config: SimulationConfig,
        n_run_turns: int,
        initial_knobs: dict[str, float],
    ) -> SpawnedValidationWorkers:
        """Spawn one validation process per validation payload and record its metadata."""
        spawned = SpawnedValidationWorkers()
        covered_ranges: set[tuple[int, str, str]] = set()
        for val_offset, validation_payload in enumerate(validation_payloads):
            val_worker_id = training_worker_count + val_offset
            val_parent, val_child = mp.Pipe()
            val_data, val_config, val_file_idx = validation_payload
            validation_class = self.select_worker_class(
                val_config.kick_plane,
                simulation_config.optimise_momenta,
                validation=True,
            )
            val_worker = validation_class(
                val_child,
                val_worker_id,
                [validation_payload],
                simulation_config,
            )
            val_worker.start()
            val_parent.send((initial_knobs, -1))
            spawned.parent_conns.append(val_parent)
            spawned.workers.append(val_worker)

            val_bpm_names = self.setup_helper.get_range_bpm_names(
                val_config.tracking_start_bpm,
                val_config.tracking_end_bpm,
                val_config.sdir,
                val_config.bad_bpms,
            )
            spawned.metadata.append(
                self.setup_helper.make_runtime_metadata(
                    worker_id=val_worker_id,
                    file_idx=val_file_idx,
                    config=val_config,
                    bpm_names=val_bpm_names,
                    n_run_turns=n_run_turns,
                )
            )
            val_tracks = payload_track_count(validation_payload)
            spawned.loss_weights.append(float(val_tracks))
            spawned.particle_counts.append(len(val_data.init_coords))
            covered_ranges.add(
                (
                    val_file_idx,
                    val_config.tracking_start_bpm,
                    val_config.tracking_end_bpm,
                )
            )
            LOGGER.debug(
                "Val worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d, tracks=%d",
                val_worker_id,
                val_file_idx,
                val_config.tracking_start_bpm,
                val_config.tracking_end_bpm,
                val_config.sdir,
                val_config.kick_plane,
                len(val_bpm_names),
                val_tracks,
            )

        LOGGER.info(
            "Validation setup: payloads=%d, covered_ranges=%d, tracks=%d",
            len(validation_payloads),
            len(covered_ranges),
            int(sum(spawned.loss_weights)),
        )
        return spawned
