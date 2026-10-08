"""The running workers of one tracking fit.

A :class:`TrackingSession` builds the worker payloads, spawns the training and
held-out validation :class:`~adelmo.fitting.pool.WorkerPool` s, and talks
to them until they are stopped: outlier screening, initial-condition updates, the
validation loss and the uncertainty drain. Worker ranges come from
:mod:`~adelmo.tracking.dispatch.setup`, payload arrays from
:mod:`~adelmo.tracking.dispatch.payloads`.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from adelmo.fitting.pool import WorkerPool
from adelmo.fitting.protocol import Command, CommandKind, LossReply, Start
from adelmo.tracking.dispatch.payloads import WorkerPayloadBuilder
from adelmo.tracking.dispatch.screening import OutlierScreener
from adelmo.tracking.uncertainty import drain_uncertainty
from adelmo.tracking.worker import TrackingWorker

if TYPE_CHECKING:
    from adelmo.config import SimulationConfig
    from adelmo.tracking.config.tracking import WorkerRangeSpec
    from adelmo.tracking.data_manager import FileTracks
    from adelmo.tracking.dispatch.payloads import WorkerPayload
    from adelmo.tracking.dispatch.setup import (
        WorkerObservationPlan,
        WorkerRuntimeMetadata,
        WorkerSetupHelper,
    )

LOGGER = logging.getLogger(__name__)


class TrackingSession:
    """The training and validation workers of one tracking fit, from start to shutdown.

    ``turn_batches`` and ``validation_turn_batches`` are the training and held-out
    turns, one list per worker batch; ``file_turn_map`` maps each turn to its
    measurement file. Pass an empty ``validation_turn_batches`` to run without
    validation workers.
    """

    def __init__(
        self,
        setup_helper: WorkerSetupHelper,
        simulation_config: SimulationConfig,
        *,
        turn_batches: list[list[int]],
        validation_turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        machine_deltaps: list[float],
    ) -> None:
        self.setup_helper = setup_helper
        self.simulation_config = simulation_config
        self.turn_batches = turn_batches
        self.validation_turn_batches = validation_turn_batches
        self.file_turn_map = file_turn_map
        self.start_bpms = start_bpms
        self.end_bpms = end_bpms
        self.machine_deltaps = machine_deltaps
        self.payload_builder = WorkerPayloadBuilder(setup_helper.accelerator)
        self.training: WorkerPool[WorkerRuntimeMetadata] = WorkerPool()
        self.validation: WorkerPool[WorkerRuntimeMetadata] = WorkerPool()

    def start(self, tracks: dict[int, FileTracks], initial_knobs: dict[str, float]) -> None:
        """Spawn the training and validation workers, each starting from ``initial_knobs``."""
        training_payloads, validation_payloads = self.build_payloads(tracks)
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

    def build_payloads(
        self, tracks: dict[int, FileTracks]
    ) -> tuple[list[WorkerPayload], list[WorkerPayload]]:
        """Weighted training and held-out validation payloads from ``tracks``.

        The two sets come from disjoint turns. Weights are normalised across the
        *combined* set so training and validation losses live on the same scale.
        """
        training = self.create_worker_payloads(tracks, self.turn_batches)
        validation = (
            self.create_worker_payloads(tracks, self.validation_turn_batches)
            if self.validation_turn_batches
            else []
        )
        self.payload_builder.attach_global_weights(
            training + validation, optimise_momenta=self.simulation_config.optimise_momenta
        )
        return training, validation

    def create_worker_payloads(
        self, tracks: dict[int, FileTracks], turn_batches: list[list[int]]
    ) -> list[WorkerPayload]:
        """One payload per (range, worker plane, turn batch); each batch belongs to one file."""
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

        self._log_file_usage(
            payloads, frozenset(self.file_turn_map[turn] for batch in turn_batches for turn in batch)
        )
        return payloads

    def _log_file_usage(self, payloads: list[WorkerPayload], files_covered: frozenset[int]) -> None:
        """Log measurement-file usage and check that at least one worker exists."""
        num_files = len(self.setup_helper.interface_options_per_file)
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

    def _spawn(
        self,
        payloads: list[WorkerPayload],
        first_id: int,
        initial_knobs: dict[str, float],
        *,
        validation: bool,
    ) -> WorkerPool[WorkerRuntimeMetadata]:
        """Start one worker per payload, send it the initial knobs and record its metadata."""
        pool: WorkerPool[WorkerRuntimeMetadata] = WorkerPool()
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

    def screen_outliers(
        self,
        initial_knobs: dict[str, float],
        bpm_sigma_threshold: float = 2.0,
        worker_sigma_threshold: float = 2.0,
    ) -> None:
        """Mask outlier BPMs and workers before optimisation starts."""
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
        this method is safe to call between epochs (from the epoch-end hook).
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
        """The combined ``x, px, y, py`` start coordinates of the running workers, from ``tracks``.

        Returns a float64 array of shape ``(n_total_particles, 4)`` suitable for
        :meth:`send_init_condition_updates`: training worker rows first, followed
        by validation worker rows.
        """
        training, validation = self.build_payloads(tracks)
        return np.concatenate(
            [data.init_coords[:, :4] for data, _config, _file_idx in training + validation]
        ).astype(np.float64)

    def validation_loss(self, knobs: dict[str, float]) -> float | None:
        """The held-out loss at ``knobs``, or ``None`` without validation workers.

        The validation workers track turns that were removed from training, so this
        is a genuine out-of-sample loss; with ``None`` the caller falls back to the
        training loss.
        """
        if not self.validation:
            return None

        channels = self.validation.channels
        channels.send_all(Command(CommandKind.VALIDATE, {"knobs": knobs}))
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
                LOGGER.warning(
                    "All %d validation workers were disabled by screening; no held-out "
                    "loss is available and early stopping falls back to the training loss.",
                    len(results),
                )
            return None

        # Each validation worker already reports a per-turn, per-BPM-point loss, so
        # combine them with an unweighted mean over workers -- the same reduction the
        # training loop uses. This keeps the validation number on the training scale.
        return float(np.mean(np.asarray(losses, dtype=np.float64)))

    def set_training_knobs(self, knobs: dict[str, float]) -> None:
        """Load ``knobs`` into every training worker, e.g. the best knobs before the uncertainty."""
        self.training.channels.ack_all(Command(CommandKind.SET_KNOBS, {"knobs": knobs}))

    def stop_and_collect_uncertainty(
        self, n_knobs: int, propagate_uncertainty: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stop every worker and return the normal matrix ``A`` and noise ``B`` of the training workers."""
        LOGGER.info("Terminating workers...")
        normal, noise = drain_uncertainty(self.training, n_knobs, propagate_uncertainty)
        self.validation.stop()
        return normal, noise

    def terminate(self) -> None:
        """Kill every worker without waiting, for aborting after an error or interrupt.

        Workers may be stuck in a failed simulation and not respond to a stop
        message, so SIGTERM is sent and the processes are reaped.
        """
        LOGGER.info("Terminating workers...")
        self.training.kill()
        self.validation.kill()
