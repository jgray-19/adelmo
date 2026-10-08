"""Particle tracking worker for multi-turn beam dynamics simulations.

A :class:`TrackingWorker` tracks its particles through its BPM range and returns
the weighted least-squares loss and its knob gradient. Constructed with
``validation=True`` it instead scores held-out turns: no knob parameters, no
gradients, only the loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from adelmo.fitting.protocol import Ack, CommandKind, GradReply, LossReply
from adelmo.fitting.worker import AbstractWorker
from adelmo.machine.mad.scripts import (
    build_tracking_init_script,
    build_tracking_preflight_script,
    build_tracking_script,
    build_validation_init_script,
    build_validation_script,
    dump_debug_script,
)
from adelmo.tracking.uncertainty import UncertaintyPart

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

    from adelmo.config import SimulationConfig
    from adelmo.fitting.protocol import Command, Evaluate
    from adelmo.fitting.worker import WorkerConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class PrecomputedTrackingWeights:
    """Per-observable, globally normalised weights for the loss and gradient.

    ``scale`` undoes the normalisation (``weight · scale = 1/σ²``), so the
    uncertainty propagation can report a physical normal matrix.
    """

    x: np.ndarray
    y: np.ndarray
    px: np.ndarray
    py: np.ndarray
    scale: float


@dataclass
class TrackingData:
    """Reference data for a tracking-loss evaluation.

    Position and momentum comparison arrays use shape
    ``(n_particles, n_data_points, 2)``, with the last axis storing the two
    transverse components for each observable family.

    Reading ids identify one measured grid cell (file, turn, marker) across all
    workers, so the uncertainty propagation can add up every use of the same noisy
    reading -- as an observation or as a start coordinate.
    """

    position_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    position_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    init_coords: np.ndarray  # Shape: (n_particles, 6)
    init_pts: np.ndarray  # Shape: (n_particles,)
    reading_ids: np.ndarray  # Shape: (n_particles, n_data_points)
    init_reading_ids: np.ndarray  # Shape: (n_particles,)
    init_variances: np.ndarray  # Shape: (n_particles, 2), var of the start x, y
    precomputed_weights: PrecomputedTrackingWeights | None


class ParticleLostError(Exception):
    """Raised when MAD-NG returns non-finite values indicating particle loss."""


OBSERVABLE_SPECS: dict[str, tuple[str, int]] = {
    "x": ("position_comparisons", 0),
    "y": ("position_comparisons", 1),
    "px": ("momentum_comparisons", 0),
    "py": ("momentum_comparisons", 1),
}

#: Last digit of a reading id: which number in a (file, turn, marker) grid cell was
#: read. The start coordinates x0, y0 are the x and y readings of their cell.
READING_CODES: dict[str, int] = {"x": 0, "y": 1, "px": 2, "py": 3}
START_CODES = np.array([READING_CODES["x"], READING_CODES["y"]])


def active_observables(kick_plane: str, include_momentum: bool) -> tuple[str, ...]:
    """Observables a worker of this kick plane compares."""
    if kick_plane == "xy":
        return ("x", "y", "px", "py") if include_momentum else ("x", "y")
    if kick_plane == "x":
        return ("x", "px") if include_momentum else ("x",)
    if kick_plane == "y":
        return ("y", "py") if include_momentum else ("y",)
    raise ValueError(f"Unsupported kick plane {kick_plane!r}")


def _largest_divisor_at_most(value: int, limit: int) -> int:
    """Return the largest positive divisor of ``value`` not exceeding ``limit``."""
    if value <= 0:
        return 0
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class TrackingWorker(AbstractWorker[TrackingData]):
    """Worker for particle tracking simulations.

    Tracks every particle of a batch through the worker's range, compares the
    observed positions (and, with ``SimulationConfig.optimise_momenta``, momenta)
    with the measurement, and returns the loss and its knob gradient from the
    differential-algebra map.
    """

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: TrackingData,
        config: WorkerConfig,
        simulation_config: SimulationConfig,
        validation: bool = False,
    ) -> None:
        self.validation = validation
        super().__init__(conn, worker_id, data, config, simulation_config)

    def prepare_data(self, data: TrackingData) -> None:
        """Split the measurement and start coordinates into batches and build the MAD scripts."""
        self.observables = active_observables(
            self.config.kick_plane, self.simulation_config.optimise_momenta
        )
        n_init = len(data.init_coords)
        if self.validation:
            # The MAD track script iterates a single fixed ``batch_size`` over every
            # batch, so a divisor batch count keeps every held-out turn.
            num_batches = _largest_divisor_at_most(n_init, self.simulation_config.num_batches)
        else:
            num_batches = min(self.simulation_config.num_batches, n_init)
        if num_batches <= 0:
            raise ValueError(f"Worker {self.worker_id}: No initial coordinates available")
        LOGGER.debug(
            f"Worker {self.worker_id}: Processing {n_init} particles in {num_batches} batches"
        )

        init_coords = data.init_coords
        if np.isnan(init_coords).any():
            raise ValueError(f"Worker {self.worker_id}: NaNs found in initial coordinates")
        if data.precomputed_weights is None:
            raise ValueError("Precomputed weights must be provided for TrackingWorker")

        self.comparisons: dict[str, list[np.ndarray]] = {}
        self.weights: dict[str, list[np.ndarray]] = {}
        for observable in self.observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            values = getattr(data, source_attr)[:n_init][:, :, plane_idx]
            weights = getattr(data.precomputed_weights, observable)[:n_init]
            self.comparisons[observable] = np.array_split(values, num_batches)
            self.weights[observable] = np.array_split(weights, num_batches)
        self.weight_scale = data.precomputed_weights.scale

        # Flat copy kept in sync with MAD by per-epoch start-coordinate updates.
        self._init_coords_np = np.ascontiguousarray(init_coords, dtype=np.float64)
        self.init_coords = [batch.tolist() for batch in np.array_split(init_coords, num_batches)]
        self.init_pts = [batch.tolist() for batch in np.array_split(data.init_pts, num_batches)]
        self.batch_size = len(self.init_coords[0])
        self.num_batches = num_batches

        self.worker_disabled = False
        self.propagate_uncertainty_on_exit = True
        self.normalisation_points = self.comparisons[self.observables[0]][0].shape[1]
        self.keep_bpm_mask = np.ones(self.normalisation_points, dtype=bool)

        if self.validation:
            self.run_track_init_text = build_validation_init_script(self.observables)
            self.run_track_script = build_validation_script(self.observables)
            return

        self._prepare_uncertainty_batches(data, n_init, num_batches)
        self.run_track_init_text = build_tracking_init_script(
            self.observables,
            start_on_first_turn=self.config.initial_condition_marker is not None,
        )
        self.run_track_script = build_tracking_script(self.observables)
        self.uncertainty_track_script = build_tracking_script(
            self.observables, include_start_derivatives=True
        )
        for name, text in (
            ("run_track_init", self.run_track_init_text),
            ("run_track", self.run_track_script),
            ("run_track_uncertainty", self.uncertainty_track_script),
        ):
            dump_debug_script(
                name,
                text,
                debug=self.config.debug,
                mad_logfile=self.config.mad_logfile,
                worker_id=self.worker_id,
            )

    def _prepare_uncertainty_batches(self, data: TrackingData, n_init: int, num_batches: int) -> None:
        """Split reading ids and variances like the comparisons, for the uncertainty part."""
        self.variances = {}
        for observable in self.observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            variances = getattr(data, source_attr.replace("comparisons", "variances"))
            self.variances[observable] = np.array_split(
                variances[:n_init, :, plane_idx], num_batches
            )
        self.reading_ids = np.array_split(data.reading_ids[:n_init], num_batches)
        self.init_reading_ids = np.array_split(data.init_reading_ids[:n_init], num_batches)
        self.init_variances = np.array_split(data.init_variances[:n_init], num_batches)

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Configure MAD-NG sequence for tracking."""
        mad["batch_size"] = self.batch_size
        mad["num_batches"] = self.num_batches
        mad["optimise_energy"] = self.config.accelerator.optimise_energy
        mad["tracking_range"] = self.tracking_range
        mad["n_run_turns"] = self.simulation_config.n_run_turns

    def _setup_da_maps(self, mad: MAD) -> None:
        """Install the knobs as TPSA parameters; a validation worker tracks numbers only."""
        if self.validation:
            mad.send("da_x0_base = damap{nv=6, np=0, mo=1, po=1}")
            return

        # ``pt`` is applied per particle, never as a sequence knob.
        knob_names = list(mad["knob_names"])
        if "pt" in knob_names:
            knob_names.remove("pt")
            mad["knob_names"] = knob_names

        self.create_base_damap(mad, knob_order=1)
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    loaded_sequence[param] = loaded_sequence[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

    def on_start(self, knobs: dict[str, float]) -> MAD:
        """Set up MAD, load the particles and check the observation geometry."""
        # Replies carry one gradient entry per start value.
        self.n_reply_knobs = len(knobs)
        mad, nbpms = self.setup_mad_interface(knobs)
        self.send_initial_conditions(mad)
        mad.send(self.run_track_init_text)
        self.run_preflight_check(mad, nbpms)
        LOGGER.debug(f"Worker {self.worker_id}: Ready for computation with {nbpms} BPMs")
        return mad

    def send_initial_conditions(self, mad: MAD) -> None:
        """Create one DA map per particle in each batch, at its start coordinates."""
        mad.send("""
init_coords = python:recv()
init_pts = python:recv()
""")
        mad.send(self.init_coords).send(self.init_pts)

        mad.send("""
da_x0_c = table.new(num_batches, 0)
for i=1,num_batches do
    da_x0_c[i] = table.new(batch_size, 0)
    for j=1,batch_size do
        da_x0_c[i][j] = da_x0_base:copy()
        da_x0_c[i][j]:set0(init_coords[i][j])
    end
end
""")

    def run_preflight_check(self, mad: MAD, nbpms: int) -> None:
        """Validate the observation geometry once with a single-particle dry run.

        Tracks one particle through the configured range/turns and confirms MAD
        observes exactly ``nbpms * n_run_turns`` points — the size the result
        vectors are allocated with and filled by ``seti`` every real run. Doing this
        up front turns a mis-scoped observe/range (e.g. an off-plane BPM left
        observed outside the range) into a clear worker error here instead of an
        opaque MAD ``index out of bounds`` deep in the optimisation loop, and lets
        the hot tracking loop run without re-validating.
        """
        mad.send(build_tracking_preflight_script())
        report = mad.recv()
        if report.get("lost"):
            # A lost preflight particle is handled the same way at runtime (the
            # epoch is rejected); startup does not fail; the count check is
            # skipped because a truncated track gives a misleading observation count.
            LOGGER.warning(
                "Worker %d: preflight particle lost; skipping observation-count check "
                "(range=%s sdir=%d)",
                self.worker_id,
                self.tracking_range,
                self.config.sdir,
            )
            return
        observed = int(report["observed"])
        expected = int(report["expected"])
        if observed != expected:
            raise ValueError(
                f"Worker {self.worker_id}: preflight observed {observed} BPM points per run "
                f"but the result vectors hold {expected} (nbpms={nbpms} x n_run_turns). "
                f"An observe/range mismatch would overflow tracking; check the observe "
                f"pattern and bad_bpms for plane {self.config.kick_plane!r} "
                f"(range={self.tracking_range!r}, sdir={self.config.sdir})."
            )
        LOGGER.debug(
            "Worker %d: preflight OK (%d observed == %d allocated)",
            self.worker_id,
            observed,
            expected,
        )

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _set_particle_pts(self, mad: MAD, pt: float, first_batch: int, last_batch: int) -> None:
        """Set every particle's ``pt`` in batches ``first_batch..last_batch`` (1-based) to the machine ``pt`` plus its own offset."""
        mad.send(f"""
for b = {first_batch}, {last_batch} do
    for i = 1, batch_size do
        da_x0_c[b][i].pt:set0({pt:.15e} + init_pts[b][i])
    end
end
""")

    def _run_tracking_batch(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> dict[str, np.ndarray]:
        """Run MAD-NG tracking for a single batch and return all outputs."""
        machine_pt = knob_updates.get("pt", self.fixed_pt)
        self.send_knobs(mad, knob_updates, plain=self.validation)
        mad.send(f"batch = {batch + 1}")
        self._set_particle_pts(mad, machine_pt, batch + 1, batch + 1)
        mad.send(self.run_track_script)
        return self._receive_tracking_results(mad)

    def _receive_tracking_results(
        self, mad: MAD, *, include_start_derivatives: bool = False
    ) -> dict[str, np.ndarray]:
        """Receive tracking results from MAD-NG.

        A validation worker receives the observables alone. A training worker also
        receives the loss report, one knob-derivative array per observable and, with
        ``include_start_derivatives``, ``d{o}_dc0`` of shape ``(n_particles, 2,
        n_points)``, the derivatives by the start x and y.
        """
        if self.validation:
            return {
                observable: np.asarray(mad.recv()).squeeze(-1) for observable in self.observables
            }
        loss_info: dict = mad.recv()
        n_lost: int = loss_info["n_lost"]
        n_total: int = loss_info["n_total"]
        results: dict[str, np.ndarray] = {}
        for observable in self.observables:
            results[observable] = np.asarray(mad.recv()).squeeze(-1)
        for observable in self.observables:
            results[self._gradient_key(observable)] = np.stack(mad.recv(), axis=0)
        if include_start_derivatives:
            for observable in self.observables:
                results[f"d{observable}_dc0"] = np.stack(mad.recv(), axis=0)
        if n_lost > 0:
            pct = 100.0 * n_lost / n_total
            raise ParticleLostError(
                f"Worker {self.worker_id}: {n_lost}/{n_total} particles lost ({pct:.1f}%) during tracking"
            )
        return results

    @staticmethod
    def _gradient_key(observable: str) -> str:
        """Return the gradient result key for an observable."""
        return f"d{observable}_dk"

    def _masked_weights(self, batch: int) -> dict[str, np.ndarray]:
        """Return batch weights with runtime BPM masking applied."""
        bpm_mask = self.keep_bpm_mask.reshape(1, -1)
        return {
            observable: self.weights[observable][batch] * bpm_mask
            for observable in self.observables
        }

    def _residuals(self, results: dict[str, np.ndarray], batch: int) -> dict[str, np.ndarray]:
        """Return residual arrays for the current batch."""
        return {
            observable: results[observable] - self.comparisons[observable][batch]
            for observable in self.observables
        }

    def _compute_loss_and_bpm_contributions(
        self, results: dict[str, np.ndarray], batch: int
    ) -> tuple[float, np.ndarray]:
        """Compute total loss and per-BPM contributions for one batch."""
        weights = self._masked_weights(batch)
        residuals = self._residuals(results, batch)
        loss_bpm = np.zeros(self.keep_bpm_mask.size, dtype=np.float64)
        for observable in self.observables:
            loss_bpm += np.sum(weights[observable] * residuals[observable] ** 2, axis=0)
        return float(np.sum(loss_bpm)), loss_bpm

    def _compute_loss_and_gradients(
        self, results: dict[str, np.ndarray], batch: int
    ) -> tuple[np.ndarray, float]:
        """Weighted least-squares loss and its knob gradient for one batch."""
        weights = self._masked_weights(batch)
        residuals = self._residuals(results, batch)
        gradient_shape = results[self._gradient_key(self.observables[0])].shape[1]
        grad = np.zeros(gradient_shape, dtype=np.float64)
        for observable in self.observables:
            grad += np.einsum(
                "pkm,pm->k",
                results[self._gradient_key(observable)],
                weights[observable] * residuals[observable],
            )
        loss, _ = self._compute_loss_and_bpm_contributions(results, batch)
        return 2.0 * grad, loss

    def evaluate(self, mad: MAD, message: Evaluate) -> GradReply:
        """Loss and gradient of one batch.

        The loss is per particle and per BPM point, so it is comparable across
        workers (and to the held-out validation loss) however many particles a
        worker holds; the gradient is per point and the loop divides it by the
        total particle count.
        """
        if self.validation:
            raise ValueError(f"Worker {self.worker_id}: validation workers do not evaluate gradients")
        zeros = np.zeros(self.n_reply_knobs)
        if self.worker_disabled:
            return GradReply(self.worker_id, 0.0, zeros)
        batch = int(message.batch)
        try:
            results = self._run_tracking_batch(mad, message.knobs, batch)
        except ParticleLostError as exc:
            LOGGER.warning(
                "Worker %d: %s — sending zero contribution; epoch update will be rejected",
                self.worker_id,
                exc,
            )
            # NaN loss signals the aggregator to reject this epoch's knob updates.
            return GradReply(self.worker_id, float("nan"), zeros)
        grad, loss = self._compute_loss_and_gradients(results, batch)
        n_particles = max(1, len(self.init_coords[batch]))
        # ``normalisation_points`` is read live: ``APPLY_MASK`` shrinks it to the
        # kept BPM count.
        return GradReply(
            self.worker_id,
            loss / (self.normalisation_points * n_particles),
            grad / self.normalisation_points,
        )

    def compute_diagnostics(
        self, mad: MAD, knob_updates: dict[str, float]
    ) -> tuple[float, np.ndarray]:
        """Compute total and per-BPM losses across all batches at current knobs."""
        total_loss = 0.0
        loss_per_bpm = np.zeros_like(self.keep_bpm_mask, dtype=np.float64)

        for batch in range(self.num_batches):
            results = self._run_tracking_batch(mad, knob_updates, batch)
            batch_loss, batch_loss_bpm = self._compute_loss_and_bpm_contributions(results, batch)
            total_loss += batch_loss
            loss_per_bpm += batch_loss_bpm

        return total_loss, loss_per_bpm

    def compute_validation_loss(self, mad: MAD, knob_updates: dict[str, float]) -> float:
        """Held-out loss per particle and BPM point, normalised like the training loss."""
        total_loss = 0.0
        for batch in range(self.num_batches):
            results = self._run_tracking_batch(mad, knob_updates, batch)
            batch_loss, _ = self._compute_loss_and_bpm_contributions(results, batch)
            n_particles = max(1, len(self.init_coords[batch]))
            total_loss += batch_loss / (self.normalisation_points * n_particles)
        return total_loss / max(1, self.num_batches)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def handle_command(self, mad: MAD, command: Command) -> Ack | LossReply:
        payload = command.payload
        match command.kind:
            case CommandKind.DIAGNOSE:
                total_loss, loss_per_bpm = self.compute_diagnostics(mad, payload["knobs"])
                return LossReply(
                    self.worker_id,
                    total_loss / self.normalisation_points,
                    loss_per_bpm / self.normalisation_points,
                )
            case CommandKind.VALIDATE:
                # A screened-out worker contributes no loss rather than a zero, which
                # would dilute the mean by the fraction of disabled workers.
                loss = None if self.worker_disabled else self.compute_validation_loss(mad, payload["knobs"])
                return LossReply(self.worker_id, loss)
            case CommandKind.APPLY_MASK:
                keep_bpm_mask = np.asarray(payload["keep_bpm_mask"], dtype=bool)
                if keep_bpm_mask.size:
                    self._apply_runtime_mask(keep_bpm_mask)
                self.worker_disabled = bool(payload["disable_worker"])
            case CommandKind.SET_KNOBS:
                knobs = payload["knobs"]
                self.send_knobs(mad, knobs, plain=self.validation)
                self._set_particle_pts(mad, knobs.get("pt", self.fixed_pt), 1, self.num_batches)
            case CommandKind.UPDATE_INIT_COORDS:
                self._send_init_condition_update(
                    mad, *(np.asarray(payload[key], dtype=np.float64) for key in ("x", "px", "y", "py"))
                )
            case CommandKind.SET_UNCERTAINTY_MODE:
                self.propagate_uncertainty_on_exit = bool(payload["enabled"])
            case _:
                return super().handle_command(mad, command)
        return Ack(self.worker_id)

    def _apply_runtime_mask(self, keep_bpm_mask: np.ndarray) -> None:
        """Apply BPM keep-mask for subsequent optimisation and uncertainty steps."""
        if keep_bpm_mask.ndim != 1:
            raise ValueError("keep_bpm_mask must be a 1D array")
        if keep_bpm_mask.size != self.keep_bpm_mask.size:
            raise ValueError(
                f"Mask size mismatch for worker {self.worker_id}: "
                f"expected {self.keep_bpm_mask.size}, got {keep_bpm_mask.size}"
            )
        self.keep_bpm_mask = keep_bpm_mask.astype(bool, copy=True)
        self.normalisation_points = int(np.count_nonzero(self.keep_bpm_mask))

    def _send_init_condition_update(
        self,
        mad: MAD,
        new_x: np.ndarray,
        new_px: np.ndarray,
        new_y: np.ndarray,
        new_py: np.ndarray,
    ) -> None:
        """Push updated x/px/y/py into the MAD-NG DAMAP objects for all particles.

        Only the constant parts of the four transverse TPSA variables are
        touched; the longitudinal coordinates (t, pt) and all DA coefficients are
        left unchanged. The arrays are sent as binary column matrices, which is
        the fastest serialisation path in pymadng.

        The positions are updated alongside the momenta because a launch point
        that sits on a closed orbit moves when the lattice does: freeing the
        magnets that shape that orbit while holding the particle's starting x/y
        fixed would launch it off the orbit the very knobs being fitted define.
        A caller that only re-derives momenta passes the current positions back
        unchanged, which costs one extra column each way and keeps one code path.
        """
        # pymadng requires 2-D arrays for the binary matrix protocol, so the
        # coordinates arrive as N x 1 column matrices (the fastest serialisation
        # path). Indexing a MAD matrix with a single index is linear (row-major),
        # so new_px[particle] is the scalar value for that particle directly.
        mad.send("""
new_x  = python:recv()  -- N x 1 column matrix of updated x values
new_px = python:recv()  -- N x 1 column matrix of updated px values
new_y  = python:recv()  -- N x 1 column matrix of updated y values
new_py = python:recv()  -- N x 1 column matrix of updated py values

local particle = 0
for batch=1,num_batches do
    for j=1,#da_x0_c[batch] do
        particle = particle + 1
        da_x0_c[batch][j].x:set0(new_x[particle])
        da_x0_c[batch][j].px:set0(new_px[particle])
        da_x0_c[batch][j].y:set0(new_y[particle])
        da_x0_c[batch][j].py:set0(new_py[particle])
    end
end
""")
        for values in (new_x, new_px, new_y, new_py):
            mad.send(values.reshape(-1, 1))

        # Mirror in Python so _init_coords_np stays consistent
        self._init_coords_np[:, 0] = new_x.ravel()
        self._init_coords_np[:, 1] = new_px.ravel()
        self._init_coords_np[:, 2] = new_y.ravel()
        self._init_coords_np[:, 3] = new_py.ravel()

    # ------------------------------------------------------------------
    # Uncertainty propagation
    # ------------------------------------------------------------------

    def on_stop(self, mad: MAD, failed: bool) -> None:
        """A training worker answers :class:`Stop` with its uncertainty part.

        A failed worker has already sent its :class:`ErrorReply` and sends nothing more.
        """
        if self.validation or failed:
            return
        if not self.worker_disabled and self.propagate_uncertainty_on_exit:
            LOGGER.debug(f"Worker {self.worker_id}: Propagating uncertainty")
            try:
                self.send_reply(self._compute_uncertainty_part(mad, self.n_reply_knobs))
            except Exception as exc:  # noqa: BLE001
                self.send_error_payload(exc, phase="uncertainty")
        else:
            self.send_reply(UncertaintyPart.empty(self.n_reply_knobs))

    def _compute_uncertainty_part(self, mad: MAD, n_knobs: int) -> UncertaintyPart:
        """Track every batch at the loaded knobs and return this worker's uncertainty part.

        With ``r = m − y`` and loss weights ``w = 1/σ²`` (masked):
        ``A = Σ w J Jᵀ``; an observed reading moves the gradient by ``−w J``; a start
        coordinate by ``Σ_n w J ∂m/∂c0``. :func:`merge_uncertainty_parts` adds these
        over every worker using the same reading. Only rows that can carry noise are
        built: observations with a non-zero weight and starts with a finite variance.
        """
        normal = np.zeros((n_knobs, n_knobs))
        ids, sensitivities, variances = [], [], []
        mad.send("save_start_derivatives = true")
        for batch in range(self.num_batches):
            mad.send(f"batch = {batch + 1}")
            mad.send(self.uncertainty_track_script)
            results = self._receive_tracking_results(mad, include_start_derivatives=True)
            weights = self._masked_weights(batch)
            start = np.zeros((len(self.init_reading_ids[batch]), 2, n_knobs))
            for observable in self.observables:
                jacobian = results[self._gradient_key(observable)]  # (particles, knobs, points)
                weight = weights[observable] * self.weight_scale  # (particles, points)
                weighted = jacobian * weight[:, None, :]
                normal += np.einsum("pkn,pjn->kj", weighted, jacobian)
                noisy = weight != 0.0
                ids.append(self.reading_ids[batch][noisy] * 4 + READING_CODES[observable])
                sensitivities.append(-weighted.transpose(0, 2, 1)[noisy])
                variances.append(self.variances[observable][batch][noisy])
                start += np.einsum("pkn,pcn->pck", weighted, results[f"d{observable}_dc0"])
            noisy_start = np.isfinite(self.init_variances[batch])  # (particles, 2)
            ids.append((self.init_reading_ids[batch][:, None] * 4 + START_CODES)[noisy_start])
            sensitivities.append(start[noisy_start])
            variances.append(self.init_variances[batch][noisy_start])
        mad.send("save_start_derivatives = false")

        return UncertaintyPart(
            normal=normal,
            reading_ids=np.concatenate(ids),
            sensitivities=np.concatenate(sensitivities),
            variances=np.concatenate(variances),
        )
