"""Worker orchestration for tracking optimisation.

`WorkerManager` builds payloads, spawns the training and held-out validation
:class:`~aba_optimiser.training.workers.pool.WorkerPool` s, and coordinates them at
runtime: screening, init-condition updates, validation loss and the uncertainty
drain. Worker-range selection lives in :mod:`setup`, payload arrays in :mod:`payloads`.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np
import psutil

from aba_optimiser.training.workers.payloads import WorkerPayloadBuilder
from aba_optimiser.training.workers.pool import WorkerPool
from aba_optimiser.training.workers.screening import OutlierScreener
from aba_optimiser.workers.common import (
    KickPlane,
    UncertaintyPart,
    merge_uncertainty_parts,
    noise_matrix,
)
from aba_optimiser.workers.protocol import (
    STOP,
    Command,
    CommandKind,
    LossReply,
    Start,
    WorkerChannels,
)
from aba_optimiser.workers.tracking import TrackingWorker

if TYPE_CHECKING:
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.config.tracking import WorkerRangeSpec
    from aba_optimiser.training.data_manager import FileTracks
    from aba_optimiser.training.workers.payloads import WorkerPayload
    from aba_optimiser.training.workers.setup import WorkerObservationPlan, WorkerSetupHelper


LOGGER = logging.getLogger(__name__)

def _summarise_file_usage(
    payloads: list[WorkerPayload],
    num_files: int,
    files_covered: frozenset[int],
) -> None:
    """Log measurement-file usage and validate that at least one worker exists."""
    file_usage = Counter(file_idx for _, _, file_idx in payloads)
    LOGGER.info(
        "Created %d workers using files: %s",
        len(payloads),
        ", ".join(f"file_{idx}={count} workers" for idx, count in sorted(file_usage.items())),
    )
    if len(files_covered) < num_files:
        LOGGER.warning(
            "Only %d/%d measurement files are being used by workers! "
            "This may lead to poor optimisation if different files have different deltap values.",
            len(files_covered),
            num_files,
        )
    if not file_usage:
        raise ValueError(
            "No worker payloads were created; check your input data and batch configuration"
        )


class WorkerManager:
    """Create worker payloads, launch processes, and manage runtime coordination."""

    def __init__(self, setup_helper: WorkerSetupHelper) -> None:
        self.setup_helper = setup_helper
        self.payload_builder = WorkerPayloadBuilder(setup_helper.accelerator)
        self.training = WorkerPool()
        self.validation = WorkerPool()
        self.turn_batches: list[list[int]] = []
        self.validation_turn_batches: list[list[int]] = []
        self.file_turn_map: dict[int, int] = {}
        self.start_bpms: list[str] = []
        self.end_bpms: list[str] = []
        self.simulation_config: SimulationConfig | None = None
        self.machine_deltaps: list[float] = []

    def create_worker_payloads(
        self, tracks: dict[int, FileTracks], turn_batches: list[list[int]]
    ) -> list[WorkerPayload]:
        """Create one payload per (range, worker plane, turn batch); each batch belongs to one file."""
        payloads: list[WorkerPayload] = []
        plans: dict[tuple[WorkerRangeSpec, int], list[WorkerObservationPlan]] = {}
        range_specs = self.setup_helper.build_range_specs(self.start_bpms, self.end_bpms)

        LOGGER.info("Creating %d range specs x %d batches", len(range_specs), len(turn_batches))

        for range_spec in range_specs:
            for batch_idx, turn_batch in enumerate(turn_batches):
                if not turn_batch:
                    raise ValueError(
                        f"Empty batch {batch_idx} for {range_spec.start_bpm}/{range_spec.end_bpm}"
                    )
                file_idx = self.file_turn_map[turn_batch[0]]
                key = (range_spec, file_idx)
                if key not in plans:
                    plans[key] = self.setup_helper.build_observation_plans(
                        range_spec, file_idx, available_bpms=set(tracks[file_idx].marker_col)
                    )
                for plan in plans[key]:
                    data = self.payload_builder.make_tracking_data(
                        turn_batch=turn_batch,
                        file_turn_map=self.file_turn_map,
                        plan=plan,
                        machine_deltaps=self.machine_deltaps,
                        tracks=tracks,
                        n_run_turns=self.simulation_config.n_run_turns,
                    )
                    payloads.append((data, self.setup_helper.make_worker_config(plan), file_idx))

        _summarise_file_usage(
            payloads,
            len(self.setup_helper.interface_options_per_file),
            frozenset(self.file_turn_map[turn] for batch in turn_batches for turn in batch),
        )
        return payloads

    def _build_payloads(
        self, tracks: dict[int, FileTracks], with_validation: bool
    ) -> tuple[list[WorkerPayload], list[WorkerPayload]]:
        """Build weighted training and held-out validation payloads from current track data.

        The two sets come from disjoint turns (``DataManager`` removed the held-out
        turns from ``turn_batches``). Weights are normalised across the *combined*
        set so training and validation losses live on the same scale.
        """
        training = self.create_worker_payloads(tracks, self.turn_batches)
        validation = (
            self.create_worker_payloads(tracks, self.validation_turn_batches)
            if with_validation and self.validation_turn_batches
            else []
        )
        self.payload_builder.attach_global_weights(
            training + validation, optimise_momenta=self.simulation_config.optimise_momenta
        )
        return training, validation

    def start_workers(
        self,
        tracks: dict[int, FileTracks],
        turn_batches: list[list[int]],
        validation_turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
        machine_deltaps: list[float],
        initial_knobs: dict[str, float],
        enable_validation: bool = True,
    ) -> None:
        """Start training workers plus held-out validation workers."""
        self.turn_batches = turn_batches
        self.validation_turn_batches = validation_turn_batches if enable_validation else []
        self.file_turn_map = file_turn_map
        self.start_bpms = start_bpms
        self.end_bpms = end_bpms
        self.simulation_config = simulation_config
        self.machine_deltaps = machine_deltaps

        training_payloads, validation_payloads = self._build_payloads(tracks, with_validation=True)
        LOGGER.info(
            "Starting %d trn worker(s) + %d held-out val worker(s)",
            len(training_payloads),
            len(validation_payloads),
        )
        self.training = self._spawn(training_payloads, 0, initial_knobs, validation=False)
        self.validation = self._spawn(
            validation_payloads, len(training_payloads), initial_knobs, validation=True
        )
        if validation_payloads:
            covered = {(m.file_idx, m.start_bpm, m.end_bpm) for m in self.validation.metadata}
            LOGGER.info(
                "Validation setup: payloads=%d, covered_ranges=%d, tracks=%d",
                len(validation_payloads),
                len(covered),
                sum(self.validation.particle_counts),
            )

    def _spawn(
        self,
        payloads: list[WorkerPayload],
        first_id: int,
        initial_knobs: dict[str, float],
        *,
        validation: bool,
    ) -> WorkerPool:
        """Start one worker per payload, send it the initial knobs and record its metadata."""
        pool = WorkerPool()
        n_run_turns = self.simulation_config.n_run_turns
        for worker_id, (data, config, file_idx) in enumerate(payloads, start=first_id):
            pool.spawn(
                TrackingWorker, worker_id, data, config, self.simulation_config, validation
            ).send(Start(initial_knobs))
            # The config's bad BPMs already exclude every BPM blind to its plane.
            bpm_names = self.setup_helper.get_range_bpm_names(
                config.tracking_start_bpm, config.tracking_end_bpm, config.sdir, config.bad_bpms
            )
            pool.metadata.append(
                self.setup_helper.make_runtime_metadata(
                    worker_id=worker_id,
                    file_idx=file_idx,
                    config=config,
                    bpm_names=bpm_names,
                    n_run_turns=n_run_turns,
                )
            )
            pool.particle_counts.append(len(data.init_coords))
            LOGGER.debug(
                "%s worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d, tracks=%d",
                "Val" if validation else "Trn",
                worker_id,
                file_idx,
                config.tracking_start_bpm,
                config.tracking_end_bpm,
                config.sdir,
                config.kick_plane,
                len(bpm_names),
                len(data.init_coords),
            )
        return pool

    def screen_initial_outliers(
        self,
        initial_knobs: dict[str, float],
        bpm_sigma_threshold: float = 2.0,
        worker_sigma_threshold: float = 2.0,
    ) -> None:
        """Screen and mask outliers before optimisation starts."""
        screener = OutlierScreener(self.payload_builder)
        result = screener.screen(
            self.training,
            initial_knobs=initial_knobs,
            bpm_sigma_threshold=bpm_sigma_threshold,
            worker_sigma_threshold=worker_sigma_threshold,
        )
        # The same decisions must reach the validation workers: they partition
        # the same measurement files, so leaving them unscreened makes the
        # held-out loss score precisely the data the fit was told to ignore.
        if result is not None and self.validation:
            masks, disabled = screener.build_validation_screening(
                self.training.metadata,
                result.bpm_masks,
                result.worker_disabled,
                self.validation.metadata,
            )
            screener.apply_screening_actions(self.validation, masks, disabled)

    def send_init_condition_updates(self, new_coords: np.ndarray) -> None:
        """Push updated initial ``x, px, y, py`` to every training and validation worker.

        ``new_coords`` must be a float64 array of shape ``(n_total_particles, 4)``
        whose columns are ``x, px, y, py`` and whose rows are ordered: training
        workers first (in creation order), then validation workers (in creation
        order), and within each worker in particle order.

        Positions travel with the momenta because the launch point can sit on a
        closed orbit that the fitted magnets themselves shape; see
        ``TrackingWorker._send_init_condition_update``. A caller with nothing new
        to say about position passes the current x/y straight back.

        Workers handle the update before processing the next gradient batch, so
        this method is safe to call between epochs (from the epoch_end_hook).
        """
        pools = [pool for pool in (self.training, self.validation) if pool]
        expected = sum(sum(pool.particle_counts) for pool in pools)
        if new_coords.shape != (expected, 4):
            raise ValueError(
                f"new_coords must have shape ({expected}, 4) of x, px, y, py; "
                f"got {new_coords.shape}"
            )

        offset = 0
        for pool in pools:
            for conn, n in zip(pool.conns, pool.particle_counts):
                chunk = new_coords[offset : offset + n]
                conn.send(
                    Command(
                        CommandKind.UPDATE_INIT_COORDS,
                        {
                            name: chunk[:, [column]]
                            for column, name in enumerate(("x", "px", "y", "py"))
                        },
                    )
                )
                offset += n
            pool.channels.recv_all()

    def build_update_coords(self, tracks: dict[int, FileTracks]) -> np.ndarray:
        """Build the combined ``x, px, y, py`` array for training and validation workers.

        Returns a float64 array of shape ``(n_total_particles, 4)`` suitable for
        passing directly to :meth:`send_init_condition_updates`: training worker
        rows first, followed by validation worker rows.
        """
        training, validation = self._build_payloads(tracks, with_validation=bool(self.validation))
        return np.concatenate(
            [data.init_coords[:, :4] for data, _config, _file_idx in training + validation]
        ).astype(np.float64)

    def compute_validation_loss(self, current_knobs: dict[str, float]) -> float | None:
        """Evaluate the held-out validation workers at the current knobs.

        The validation workers track turns that were removed from training, so this
        is a genuine out-of-sample loss. Returns ``None`` when no validation workers
        exist (validation disabled or too little data), in which case the caller
        falls back to training loss.
        """
        if not self.validation:
            return None

        channels = self.validation.channels
        channels.send_all(Command(CommandKind.VALIDATE, {"knobs": current_knobs}))
        results = channels.recv_all()
        losses: list[float] = []
        for result in results:
            if not isinstance(result, LossReply):
                raise RuntimeError(f"Unexpected validation reply from worker: {result!r}")
            # A worker screened out before the loop holds no usable data.
            if result.loss is not None:
                losses.append(float(result.loss))

        if not losses:
            if results:
                # All validation workers were screened out, so early stopping falls
                # back to the training loss; warn to distinguish this from
                # validation never having been configured.
                LOGGER.warning(
                    "All %d validation workers were disabled by screening; no held-out "
                    "loss is available and early stopping falls back to the training loss.",
                    len(results),
                )
            return None

        # Each validation worker already reports a per-turn, per-BPM-point loss, so
        # combine them with an unweighted mean over workers -- the same reduction the
        # training loop uses (loop.py: total_loss / n_workers). This keeps the
        # validation number on the same scale as the reported training loss.
        return float(np.mean(np.asarray(losses, dtype=np.float64)))

    def terminate_workers(self) -> None:
        """Kill all workers immediately, for aborting after an error or interrupt.

        Unlike the clean shutdown in ``stop_and_collect_uncertainty``, this does not
        drain payloads or join gracefully. Workers may be stuck in a failed
        simulation and not respond to a termination sentinel, so SIGTERM is sent and
        the processes are reaped.
        """
        LOGGER.info("Terminating workers...")
        self.training.kill()
        self.validation.kill()

    def set_training_knobs(self, knobs: dict[str, float]) -> None:
        """Load ``knobs`` into every training worker, e.g. the best knobs before the Hessian."""
        self.training.channels.ack_all(Command(CommandKind.SET_KNOBS, {"knobs": knobs}))

    def stop_and_collect_uncertainty(
        self,
        n_knobs: int,
        propagate_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Terminate training workers, collect ``(A, B)``, then stop validation.

        Feed both matrices to :func:`sandwich_uncertainties`.
        """
        LOGGER.info("Terminating workers...")
        normal, noise = self._collect_uncertainty(n_knobs, propagate_uncertainty)
        self.validation.stop()
        return normal, noise

    def _collect_uncertainty(
        self, n_knobs: int, propagate_uncertainty: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stop training workers in memory-bounded chunks and fold their parts into ``(A, B)``.

        Reading ids are file-scoped, so once every worker of a file has reported, that
        file's merged readings become its ``B`` contribution and are dropped. Workers are
        drained in file order, so besides the chunk being received at most one partly
        merged file is held.
        """
        normal = np.zeros((n_knobs, n_knobs), dtype=np.float64)
        noise = np.zeros((n_knobs, n_knobs), dtype=np.float64)
        files = [meta.file_idx for meta in self.training.metadata]
        order = sorted(range(len(self.training)), key=files.__getitem__)
        remaining = Counter(files)
        chunk_size = (
            self._uncertainty_concurrency(n_knobs) if propagate_uncertainty else max(1, len(order))
        )

        pending: dict[int, UncertaintyPart] = {}
        for start in range(0, len(order), chunk_size):
            chunk = order[start : start + chunk_size]
            parts_by_file: dict[int, list[UncertaintyPart]] = defaultdict(list)
            for idx, part in zip(chunk, self._drain_uncertainty_parts(chunk, propagate_uncertainty)):
                parts_by_file[files[idx]].append(part)
            for file_idx, parts in parts_by_file.items():
                remaining[file_idx] -= len(parts)
                if file_idx in pending:
                    parts.append(pending.pop(file_idx))
                merged = merge_uncertainty_parts(parts, n_knobs)
                if remaining[file_idx]:
                    pending[file_idx] = merged
                else:
                    normal += merged.normal
                    noise += noise_matrix(merged)
        return normal, noise

    def _drain_uncertainty_parts(
        self, indices: list[int], propagate_uncertainty: bool
    ) -> list[UncertaintyPart]:
        """Send the termination sentinel to the given workers and return their parts in order."""
        workers = [self.training.workers[idx] for idx in indices]
        channels = WorkerChannels([self.training.conns[idx] for idx in indices], workers)
        if not propagate_uncertainty:
            channels.ack_all(Command(CommandKind.SET_UNCERTAINTY_MODE, {"enabled": False}))
        channels.send_all(STOP)
        parts = channels.recv_all()
        for part in parts:
            if not isinstance(part, UncertaintyPart):
                raise RuntimeError(f"Unexpected uncertainty payload from worker: {part!r}")
        for worker in workers:
            worker.join()
        return parts

    def _uncertainty_concurrency(self, n_knobs: int) -> int:
        """Return how many workers may send uncertainty parts at once within half the free memory.

        A part has at most one row per observed point for up to two observables per
        kicked plane, plus one start row per plane and particle. Each row holds
        ``n_knobs`` sensitivities, an id and a variance, 8 bytes each.
        """
        largest = 0
        for meta, particles in zip(self.training.metadata, self.training.particle_counts):
            planes = 2 if meta.kick_plane == KickPlane.XY else 1
            points = len(meta.bpm_names) * meta.n_run_turns
            rows = particles * planes * (2 * points + 1)
            largest = max(largest, rows * (n_knobs + 2) * 8)
        available = psutil.virtual_memory().available
        concurrency = max(1, min(len(self.training), int(0.5 * available // max(largest, 1))))
        LOGGER.info(
            "Uncertainty stage: largest worker part <= %.1f MiB, %.1f MiB available, "
            "%d worker(s) at once",
            largest / 2**20,
            available / 2**20,
            concurrency,
        )
        return concurrency
