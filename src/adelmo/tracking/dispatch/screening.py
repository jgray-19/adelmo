"""Pre-optimisation outlier screening for tracking workers.

:class:`OutlierScreener` is the cohesive algorithm that the
:class:`~adelmo.tracking.session.TrackingSession` runs once before
optimisation begins: it asks each worker for per-BPM diagnostics, flags BPMs and
whole workers whose loss is an outlier (positive-side z-score above a sigma
threshold), and pushes the resulting keep-masks back to the workers. It operates
purely on the :class:`~adelmo.fitting.pool.WorkerPool` passed in, so it holds
no worker-process state of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from adelmo.fitting.protocol import Ack, Command, CommandKind, LossReply

if TYPE_CHECKING:
    from adelmo.fitting.pool import WorkerPool
    from adelmo.fitting.protocol import WorkerChannels
    from adelmo.tracking.dispatch.payloads import WorkerPayloadBuilder
    from adelmo.tracking.dispatch.setup import WorkerRuntimeMetadata

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreeningResult:
    """The screening decisions taken for one set of workers.

    Returned so the caller can replay the same decisions on other worker sets --
    the validation workers above all.
    """

    bpm_masks: list[np.ndarray]
    worker_disabled: list[bool]


class OutlierScreener:
    """Detect and mask high-loss BPMs and workers before optimisation starts."""

    def __init__(self, payload_builder: WorkerPayloadBuilder) -> None:
        self.payload_builder = payload_builder

    def screen(
        self,
        pool: WorkerPool,
        *,
        initial_knobs: dict[str, float],
        bpm_sigma_threshold: float = 2.0,
        worker_sigma_threshold: float = 2.0,
    ) -> ScreeningResult | None:
        """Screen and mask outliers before optimisation starts."""
        if not pool:
            LOGGER.warning("No workers available for pre-optimisation outlier screening")
            return None

        LOGGER.info(
            "Running pre-optimisation outlier screening (BPM=%.1fσ, worker=%.1fσ)",
            bpm_sigma_threshold,
            worker_sigma_threshold,
        )

        worker_metadata = pool.metadata
        diagnostics = self.request_worker_diagnostics(pool.channels, initial_knobs)
        worker_losses = [float(diag.loss) for diag in diagnostics]

        worker_disabled = self.classify_worker_outliers(
            np.asarray(worker_losses, dtype=np.float64),
            worker_metadata,
            worker_sigma_threshold,
        )
        bpm_masks = self.build_bpm_masks_from_diagnostics(
            diagnostics, worker_metadata, bpm_sigma_threshold
        )
        self.summarise_screening_losses(diagnostics, bpm_masks, worker_disabled, worker_metadata)
        self.apply_screening_actions(pool, bpm_masks, worker_disabled)

        LOGGER.warning(
            "Pre-optimisation screening complete: masked %d BPM entries across workers, disabled %d/%d workers",
            sum(int((~mask).sum()) for mask in bpm_masks),
            sum(worker_disabled),
            len(pool),
        )
        return ScreeningResult(bpm_masks=bpm_masks, worker_disabled=worker_disabled)

    @staticmethod
    def compute_positive_z_scores(values: np.ndarray) -> np.ndarray:
        """Compute positive-side z-scores; values below mean map to 0."""
        v = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(v)
        z = np.zeros_like(v, dtype=np.float64)
        if finite.sum() < 2:
            return z

        mean = float(np.mean(v[finite]))
        std = float(np.std(v[finite]))
        if std <= 0.0:
            return z

        z_vals = (v - mean) / std
        z[finite] = np.maximum(z_vals[finite], 0.0)
        return z

    def request_worker_diagnostics(
        self,
        channels: WorkerChannels,
        initial_knobs: dict[str, float],
    ) -> list[LossReply]:
        """Request every worker's total and per-point loss at ``initial_knobs``."""
        channels.send_all(Command(CommandKind.DIAGNOSE, {"knobs": initial_knobs}))
        diagnostics = channels.recv_all()
        for result in diagnostics:
            if not isinstance(result, LossReply):
                raise RuntimeError(f"Unexpected diagnostics reply from worker: {result!r}")
        return diagnostics

    def build_bpm_masks_from_diagnostics(
        self,
        diagnostics: list[LossReply],
        worker_metadata: list[WorkerRuntimeMetadata],
        bpm_sigma_threshold: float,
    ) -> list[np.ndarray]:
        """Build keep-masks from per-BPM losses."""
        bpm_masks: list[np.ndarray] = []

        for meta, diag in zip(worker_metadata, diagnostics, strict=True):
            worker_id = diag.worker_id
            loss_per_point = np.asarray(diag.per_point, dtype=np.float64)
            loss_per_bpm = self.payload_builder.diagnostic_loss_per_bpm(
                loss_per_point=loss_per_point,
                bpm_names=meta.bpm_names,
                n_run_turns=meta.n_run_turns,
                worker_id=worker_id,
            )
            bpm_z = self.compute_positive_z_scores(loss_per_bpm)
            keep_mask = np.ones(len(meta.bpm_names), dtype=bool)
            outlier_indices = np.where(bpm_z > bpm_sigma_threshold)[0]

            for bpm_idx in outlier_indices:
                keep_mask[bpm_idx] = False
                LOGGER.warning(
                    "Worker %d: loss at BPM %s is %.2f standard deviations away from the mean, ignoring for optimisation.",
                    worker_id,
                    meta.bpm_names[bpm_idx],
                    bpm_z[bpm_idx],
                )

            bpm_masks.append(keep_mask)

        return bpm_masks

    def build_validation_screening(
        self,
        training_metadata: list[WorkerRuntimeMetadata],
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
        validation_metadata: list[WorkerRuntimeMetadata],
    ) -> tuple[list[np.ndarray], list[bool]]:
        """Carry the training screening decisions over to the validation workers.

        Screening removes BPMs and whole workers from the *fit*. Validation
        workers are a separate partition of the same measurement files, so
        unless the same decisions reach them the held-out loss keeps scoring
        exactly the data the optimiser was told to ignore. Those residuals can
        only grow as the fit moves onto the data it kept, which shows up as a
        validation loss rising monotonically from the first epoch while the
        training loss falls -- a screening artefact, not overfitting.

        Decisions are matched on (file, range, direction, plane) when the
        validation worker's file also carries training workers, and on
        (range, direction, plane) otherwise: a pooled training worker reports
        only its primary file, so a validation worker for one of the other
        pooled files has no exact counterpart.
        """
        dropped_by_file: dict[tuple, set[str]] = {}
        dropped_by_range: dict[tuple, set[str]] = {}
        disabled_by_file: dict[tuple, bool] = {}
        disabled_by_range: dict[tuple, bool] = {}

        for meta, keep_mask, disable in zip(
            training_metadata, bpm_masks, worker_disabled, strict=True
        ):
            range_key = (meta.start_bpm, meta.end_bpm, int(meta.sdir), str(meta.kick_plane))
            file_key = (meta.file_idx, *range_key)
            dropped_by_file.setdefault(file_key, set())
            dropped_by_range.setdefault(range_key, set())
            # Only a worker that stays in the fit gets to mask BPMs. A disabled
            # worker is by construction a high-loss one, so it carries BPM
            # outliers of its own; letting those through would blank them in the
            # validation counterpart of a *surviving* sibling worker, which is
            # the same "score data the fit did not use" defect, inverted.
            if not disable:
                dropped = {
                    name for name, keep in zip(meta.bpm_names, keep_mask, strict=True) if not keep
                }
                dropped_by_file[file_key].update(dropped)
                dropped_by_range[range_key].update(dropped)
            # A key counts as disabled only when every training worker sharing
            # it was disabled; one surviving worker still constrains the fit.
            disabled_by_file[file_key] = disabled_by_file.get(file_key, True) and disable
            disabled_by_range[range_key] = disabled_by_range.get(range_key, True) and disable

        masks: list[np.ndarray] = []
        disabled: list[bool] = []
        fallback_keys = 0
        for meta in validation_metadata:
            range_key = (meta.start_bpm, meta.end_bpm, int(meta.sdir), str(meta.kick_plane))
            file_key = (meta.file_idx, *range_key)
            if file_key in dropped_by_file:
                dropped = dropped_by_file[file_key]
                is_disabled = disabled_by_file[file_key]
            else:
                # No training worker reported this file at this range. That is the
                # normal case under file pooling, because ``get_primary_file_idx``
                # reports only the lowest file index of a pooled batch, so every
                # other pooled file falls through to here. The range key unions
                # drops across files, which is coarser than a per-file decision:
                # a BPM that was an outlier in one file is masked for all of them,
                # and because the disable flag is an AND across files it will
                # almost never survive. Counted and logged so this is visible.
                fallback_keys += 1
                dropped = dropped_by_range.get(range_key, set())
                is_disabled = disabled_by_range.get(range_key, False)
            masks.append(
                np.array([name not in dropped for name in meta.bpm_names], dtype=bool)
            )
            disabled.append(bool(is_disabled))

        if fallback_keys:
            LOGGER.info(
                "Screening propagation fell back to the range key for %d/%d validation "
                "workers (no training worker reported their file at that range; expected "
                "under file pooling). Their masks are the union over every file sharing "
                "the range, and their disable flag the intersection.",
                fallback_keys,
                len(validation_metadata),
            )
        n_masked = sum(int((~mask).sum()) for mask in masks)
        if n_masked or any(disabled):
            LOGGER.info(
                "Propagated screening to validation: masked %d BPM entries, "
                "disabled %d/%d validation workers.",
                n_masked,
                sum(disabled),
                len(validation_metadata),
            )
        return masks, disabled

    def classify_worker_outliers(
        self,
        worker_losses: np.ndarray,
        worker_metadata: list[WorkerRuntimeMetadata],
        worker_sigma_threshold: float,
    ) -> list[bool]:
        """Identify high-loss worker outliers from adjusted worker losses."""
        worker_z = self.compute_positive_z_scores(worker_losses)
        worker_disabled: list[bool] = []
        n_disabled = 0

        for idx, meta in enumerate(worker_metadata):
            z_score = float(worker_z[idx])
            disable = z_score > worker_sigma_threshold
            worker_disabled.append(disable)
            if disable:
                n_disabled += 1
                LOGGER.warning(
                    "Worker %d with starting BPM %s is %.2f standard deviations away from the mean, ignoring.",
                    meta.worker_id,
                    meta.start_bpm,
                    z_score,
                )

        if n_disabled == 0:
            max_z = float(np.max(worker_z)) if worker_z.size else 0.0
            LOGGER.warning(
                "Worker outlier screening: no workers exceeded threshold %.2fσ (max z-score %.2f).",
                worker_sigma_threshold,
                max_z,
            )

        return worker_disabled

    def summarise_screening_losses(
        self,
        diagnostics: list[LossReply],
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
        worker_metadata: list[WorkerRuntimeMetadata],
    ) -> None:
        """Log loss before masking and projected loss after masking/disabling."""
        raw_worker_losses: list[float] = []
        projected_worker_losses: list[float] = []

        for idx, (diag, mask, disable, meta) in enumerate(
            zip(diagnostics, bpm_masks, worker_disabled, worker_metadata, strict=True)
        ):
            loss_per_point = np.asarray(diag.per_point, dtype=np.float64)
            expanded_mask = self.payload_builder.expand_bpm_mask(mask, meta.n_run_turns)
            if loss_per_point.size != expanded_mask.size:
                raise RuntimeError(
                    f"Worker diagnostics at index {idx} has incompatible mask/point lengths "
                    f"({expanded_mask.size} mask points vs {loss_per_point.size} losses)"
                )

            raw_loss = float(np.nansum(loss_per_point))
            kept_loss = float(np.nansum(loss_per_point[expanded_mask])) if not disable else 0.0
            raw_worker_losses.append(raw_loss)
            projected_worker_losses.append(kept_loss)

        raw_total = float(np.sum(raw_worker_losses))
        projected_total = float(np.sum(projected_worker_losses))
        n_workers = max(1, len(raw_worker_losses))
        raw_mean = raw_total / n_workers
        projected_mean = projected_total / n_workers
        reduction = 100.0 * (1.0 - projected_total / raw_total) if raw_total > 0.0 else 0.0

        LOGGER.info(
            "Pre-screening loss summary: total=%.6e, mean/worker=%.6e",
            raw_total,
            raw_mean,
        )
        LOGGER.info(
            "Projected post-screening loss summary: total=%.6e, mean/worker=%.6e (reduction=%.2f%%)",
            projected_total,
            projected_mean,
            reduction,
        )

    def apply_screening_actions(
        self,
        pool: WorkerPool,
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
    ) -> None:
        """Send mask/disable settings to workers and verify acknowledgements."""
        for conn, keep_mask, disable, meta in zip(
            pool.conns, bpm_masks, worker_disabled, pool.metadata, strict=True
        ):
            expanded_mask = self.payload_builder.expand_bpm_mask(keep_mask, meta.n_run_turns)
            conn.send(
                Command(
                    CommandKind.APPLY_MASK,
                    {"keep_bpm_mask": expanded_mask, "disable_worker": disable},
                )
            )

        for ack in pool.channels.recv_all():
            if not isinstance(ack, Ack):
                raise RuntimeError(f"Failed to apply worker mask settings: {ack!r}")
