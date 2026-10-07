"""Validation-only tracking workers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.mad.scripts import (
    PYTHON_IN_MAD,
    build_validation_init_script,
    build_validation_script,
)
from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.common import (
    TrackingData,
    WorkerConfig,
    split_array_to_batches,
)
from aba_optimiser.workers.tracking import OBSERVABLE_SPECS, TrackingWorker
from aba_optimiser.workers.tracking_position_only import PositionOnlyConfigMixin

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)


def _largest_divisor_at_most(value: int, limit: int) -> int:
    """Return the largest positive divisor of ``value`` not exceeding ``limit``."""
    if value <= 0:
        return 0
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class ValidationTrackingWorker(TrackingWorker):
    """Validation worker for one payload."""

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        payloads: list[tuple[TrackingData, WorkerConfig, int]],
        simulation_config: SimulationConfig,
    ) -> None:
        if len(payloads) != 1:
            raise ValueError(
                f"ValidationTrackingWorker requires exactly one payload, got {len(payloads)}"
            )

        self.file_idx = payloads[0][2]
        AbstractWorker.__init__(self, conn, worker_id, payloads[0][0], payloads[0][1], simulation_config)

    @staticmethod
    def _resolve_observables_for_config(
        config: WorkerConfig,
        include_momentum: bool,
    ) -> tuple[str, ...]:
        kick_plane = config.kick_plane
        if kick_plane == "xy":
            return ("x", "y", "px", "py") if include_momentum else ("x", "y")
        if kick_plane == "x":
            return ("x", "px") if include_momentum else ("x",)
        if kick_plane == "y":
            return ("y", "py") if include_momentum else ("y",)
        raise ValueError(f"Unsupported kick plane {kick_plane!r}")

    def prepare_data(self, data: TrackingData) -> None:
        """Prepare one validation payload."""
        observables = self._resolve_observables_for_config(self.config, self.include_momentum)
        num_batches = _largest_divisor_at_most(
            len(data.init_coords),
            self.simulation_config.num_batches,
        )
        if num_batches <= 0:
            raise ValueError(f"Worker {self.worker_id}: No initial coordinates available")

        # The MAD track script iterates a single fixed ``batch_size`` over every
        # sub-batch. Pick a divisor batch count so validation keeps all held-out
        # turns while preserving equal-size MAD batches.
        n_init = len(data.init_coords)
        init_coords = data.init_coords
        if np.isnan(init_coords).any():
            raise ValueError(f"Worker {self.worker_id}: NaNs found in initial coordinates")
        # Keep the flat python copy in sync with MAD so per-epoch init-coord
        # updates (update_init_coords) can mirror the new px/py here too.
        self._init_coords_np = np.ascontiguousarray(init_coords, dtype=np.float64)
        if data.precomputed_weights is None:
            raise ValueError("Precomputed weights must be provided for ValidationTrackingWorker")

        comparison_arrays: dict[str, np.ndarray] = {}
        weight_arrays: dict[str, np.ndarray] = {}
        for observable in observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            source = getattr(data, source_attr)[:n_init]
            comparison_arrays[observable] = source[:, :, plane_idx]
            weight_arrays[observable] = getattr(data.precomputed_weights, observable)[:n_init]

        self.observables = observables
        self.comparisons = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in comparison_arrays.items()
        }
        self.weights = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in weight_arrays.items()
        }
        self.init_coords = [
            batch.tolist() for batch in split_array_to_batches(init_coords, num_batches)
        ]
        self.init_pts = [
            batch.tolist()
            for batch in split_array_to_batches(data.init_pts[:n_init], num_batches)
        ]
        self.batch_size = len(self.init_coords[0])
        self.num_batches = num_batches
        self.track_count = int(n_init)
        self.normalisation_points = self.comparisons[observables[0]][0].shape[1]
        self.keep_bpm_mask = np.ones(self.normalisation_points, dtype=bool)
        self.run_track_init_text = build_validation_init_script(observables)
        self.run_track_script = build_validation_script(observables)

    def _setup_da_maps(self, mad: MAD) -> None:
        """Create only the coordinate DA state needed for numeric tracking."""
        mad.send("da_x0_base = damap{nv=6, np=0, mo=1, po=1}")

    def setup_mad_interface(self, init_knobs: dict[str, float]) -> tuple[MAD, int]:
        """Set up a non-gradient MAD interface."""
        LOGGER.debug("Worker %s: Setting up validation MAD interface", self.worker_id)
        LOGGER.debug("Worker %s: Using BPM range %s", self.worker_id, self.bpm_range)

        worker_logfile = self._resolve_per_worker_logfile(self.config.mad_logfile)
        mad_iface = GradientDescentMadInterface(
            accelerator=self.config.accelerator,
            magnet_range=self.config.magnet_range,
            bpm_range=self.bpm_range,
            **self.config.interface_options,
            initial_model_values=init_knobs,
            bad_bpms=self.config.bad_bpms,
            debug=self.config.debug,
            mad_logfile=worker_logfile,
            py_name=PYTHON_IN_MAD,
            tracking_anchor_mode=self.config.tracking_anchor_mode,
            tracking_anchor_markers=self.config.tracking_anchor_sources,
            observed_tracking_anchor_markers=self.config.observed_tracking_anchor_markers,
        )

        # Same cycling and BPM sizing as the training workers
        # (AbstractWorker.setup_mad_interface); without it the tracking range is
        # taken on an uncycled sequence and validation scores a different path.
        cycle_target = self._cycle_target()
        if cycle_target is not None:
            mad_iface.cycle_to_start(cycle_target)

        self.knob_name_set = set(mad_iface.knob_names)
        self.fixed_pt = (
            float(init_knobs.get("pt", 0.0)) if "pt" not in self.knob_name_set else 0.0
        )

        mad = mad_iface.mad
        nbpms = len(mad_iface.all_bpms) if self.tracking_range is None else mad_iface.nbpms
        mad["nbpms"] = nbpms
        mad["sdir"] = self.config.sdir
        mad.load("MAD", "damap", "matrix", "vector")

        self.setup_mad_sequence(mad)
        self._setup_da_maps(mad)
        return mad, nbpms

    def _initialise_mad_computation(self, mad: MAD) -> None:
        """Initialise MAD-NG environment for validation computation."""
        mad.send(self.run_track_init_text)

    def _receive_tracking_results(self, mad: MAD) -> dict[str, np.ndarray]:
        """Receive only observable arrays from MAD-NG."""
        results: dict[str, np.ndarray] = {}
        for observable in self.observables:
            results[observable] = np.asarray(mad.recv()).squeeze(-1)
        return results

    def _run_tracking_batch(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> dict[str, np.ndarray]:
        """Run MAD-NG tracking for one validation batch."""
        machine_pt = knob_updates.get("pt", getattr(self, "fixed_pt", 0.0))

        update_commands = [
            f"loaded_sequence['{name}'] = {val:.15e}"
            for name, val in knob_updates.items()
            if name != "pt" and name in self.knob_name_set
        ]
        if update_commands:
            mad.send("\n".join(update_commands))

        mad.send(f"batch = {batch + 1}")
        mad.send(f"""
for i = 1, batch_size do
    da_x0_c[batch][i].pt:set0({machine_pt:.15e} + init_pts[batch][i])
end
""")
        mad.send(self.run_track_script)
        return self._receive_tracking_results(mad)

    def compute_gradients_and_loss(self, mad: MAD, knob_updates: dict[str, float], batch: int):
        """Validation workers do not compute gradients."""
        del mad, knob_updates, batch
        raise NotImplementedError("ValidationTrackingWorker does not compute gradients")

    def compute_validation_loss(self, mad: MAD, knob_updates: dict[str, float]) -> float:
        """Return a per-turn, per-BPM-point validation loss.

        Uses the same normalisation as the training worker's reported loss
        (divide by BPM points and by the number of turns in the batch) so the
        held-out validation loss is directly comparable to the training loss,
        independent of how many turns each worker holds.
        """
        total_loss = 0.0
        for batch in range(self.num_batches):
            results = self._run_tracking_batch(mad, knob_updates, batch)
            batch_loss, _ = self._compute_loss_and_bpm_contributions(results, batch)
            n_turns = max(1, len(self.init_coords[batch]))
            total_loss += batch_loss / (self.normalisation_points * n_turns)
        return total_loss / max(1, self.num_batches)

    def _send_init_condition_update(
        self,
        mad: MAD,
        new_x: np.ndarray,
        new_px: np.ndarray,
        new_y: np.ndarray,
        new_py: np.ndarray,
    ) -> None:
        """Keep update arrays aligned with this worker's prepared particles.

        Validation normally keeps every held-out turn. The slice also keeps older
        or externally rebuilt payload chunks compatible if their update arrays are
        longer than this worker's prepared coordinate table.
        """
        n_init = len(self._init_coords_np)
        super()._send_init_condition_update(
            mad,
            new_x.ravel()[:n_init],
            new_px.ravel()[:n_init],
            new_y.ravel()[:n_init],
            new_py.ravel()[:n_init],
        )

    def _replace_validation_payloads(
        self,
        payloads: list[tuple[TrackingData, WorkerConfig, int]],
    ) -> None:
        """Replace the single validation payload while preserving the MAD session."""
        if len(payloads) != 1:
            raise ValueError(
                f"Worker {self.worker_id}: expected exactly one validation payload, got {len(payloads)}"
            )

        mad = getattr(self, "mad", None)
        nbpms = getattr(self, "nbpms", None)
        # ``prepare_data`` rebuilds ``keep_bpm_mask`` as all-ones, so a screening
        # mask applied earlier would be silently dropped here, leaving the worker
        # scoring BPMs the fit was told to ignore. Carry it across when the new
        # payload has the same BPM count; a different count means the mask no
        # longer refers to the same points and must not be reused.
        previous_mask = getattr(self, "keep_bpm_mask", None)
        data, config, file_idx = payloads[0]
        self.config = config
        self.file_idx = file_idx
        self.prepare_data(data)
        if previous_mask is not None and previous_mask.size == self.keep_bpm_mask.size:
            self._apply_runtime_mask(previous_mask)
        self.mad = mad
        self.nbpms = nbpms
        if self.mad is not None:
            self.send_initial_conditions(self.mad)

    @staticmethod
    def _parse_knobs(command: dict[str, object], worker_id: int) -> dict[str, float]:
        raw_knobs = command.get("knobs", {})
        if not isinstance(raw_knobs, dict):
            raise ValueError(f"Worker {worker_id}: validation command missing knob dictionary")

        parsed: dict[str, float] = {}
        for knob_name, knob_value in raw_knobs.items():
            if not isinstance(knob_name, str):
                raise ValueError(f"Worker {worker_id}: knob name {knob_name!r} is not a string")
            if not isinstance(knob_value, int | float | np.floating):
                raise ValueError(
                    f"Worker {worker_id}: knob {knob_name!r} has non-numeric value {knob_value!r}"
                )
            parsed[knob_name] = float(knob_value)
        return parsed

    def run(self) -> None:
        """Main validation-worker run loop."""
        try:
            self.configure_python_worker_logging()
            self.configure_worker_threads()
            knob_values, _batch = self.conn.recv()
            if knob_values is None:
                return

            self.mad, self.nbpms = self.setup_mad_interface(knob_values)
            self.send_initial_conditions(self.mad)
            self._initialise_mad_computation(self.mad)
            self.run_preflight_check(self.mad, self.nbpms)
            LOGGER.debug(
                "Worker %s: Ready for validation file=%d range=%s/%s sdir=%d kick_plane=%s tracks=%d bpms=%d",
                self.worker_id,
                self.file_idx,
                self.config.tracking_start_bpm,
                self.config.tracking_end_bpm,
                self.config.sdir,
                self.config.kick_plane,
                self.track_count,
                self.nbpms,
            )

            while True:
                message = self.conn.recv()
                if isinstance(message, tuple):
                    knob_values, batch = message
                    if knob_values is None or batch is None:
                        LOGGER.debug("Worker %s: Received termination signal", self.worker_id)
                        break
                    raise ValueError(
                        f"Worker {self.worker_id}: unexpected batch command for validation worker"
                    )

                if not isinstance(message, dict):
                    raise ValueError(
                        f"Worker {self.worker_id}: unexpected validation payload {type(message)}"
                    )

                cmd = message.get("cmd")
                if cmd == "replace_validation_payloads":
                    raw_payloads = message.get("payloads")
                    if not isinstance(raw_payloads, list):
                        raise ValueError(
                            f"Worker {self.worker_id}: replace_validation_payloads missing payload list"
                        )
                    self._replace_validation_payloads(raw_payloads)
                    self.conn.send({"worker_id": self.worker_id, "status": "ok"})
                    continue

                if cmd == "update_init_coords":
                    new_x = np.asarray(message["x"], dtype=np.float64)
                    new_px = np.asarray(message["px"], dtype=np.float64)
                    new_y = np.asarray(message["y"], dtype=np.float64)
                    new_py = np.asarray(message["py"], dtype=np.float64)
                    self._send_init_condition_update(self.mad, new_x, new_px, new_y, new_py)
                    self.conn.send({"worker_id": self.worker_id, "status": "ok"})
                    continue

                if cmd == "apply_mask":
                    # Pre-loop screening decisions are taken on the training
                    # workers, but validation partitions the same measurement
                    # files. Leaving these workers unscreened would score the
                    # held-out loss on precisely the data the fit was told to
                    # ignore, so the mask has to reach them too.
                    keep_bpm_mask = np.asarray(message.get("keep_bpm_mask", []), dtype=bool)
                    if keep_bpm_mask.size:
                        self._apply_runtime_mask(keep_bpm_mask)
                    self.worker_disabled = bool(message.get("disable_worker", False))
                    self.conn.send({"worker_id": self.worker_id, "status": "ok"})
                    continue

                if cmd != "validate":
                    raise ValueError(f"Worker {self.worker_id}: unknown command {cmd}")

                parsed_knobs = self._parse_knobs(message, self.worker_id)
                # A screened-out worker contributes no loss at all rather than a
                # zero: averaging a zero in would dilute the validation loss by
                # the fraction of workers that were disabled.
                loss = (
                    None
                    if self.worker_disabled
                    else self.compute_validation_loss(self.mad, parsed_knobs)
                )
                self.conn.send(
                    {
                        "worker_id": self.worker_id,
                        "loss": loss,
                        "disabled": bool(self.worker_disabled),
                        "payloads": 1,
                        "tracks": self.track_count,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            self.send_error_payload(exc, phase="validation")
        finally:
            LOGGER.debug("Worker %s: Terminating", self.worker_id)
            if getattr(self, "mad", None) is not None:
                self.mad.send("shush()")
                self.mad = None


class PositionOnlyValidationTrackingWorker(PositionOnlyConfigMixin, ValidationTrackingWorker):
    """Validation worker that compares only x/y position observables."""
