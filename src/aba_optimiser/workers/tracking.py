"""Particle tracking worker for multi-turn beam dynamics simulations.

This module implements the TrackingWorker class which performs particle
tracking simulations and computes gradients for optimisation. It handles
both position and momentum observables with full symmetry between x/y planes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.scripts import (
    build_tracking_init_script,
    build_tracking_preflight_script,
    build_tracking_script,
    dump_debug_script,
)
from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.common import (
    PrecomputedTrackingWeights,
    TrackingData,
    UncertaintyPart,
    split_array_to_batches,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers.common import WorkerConfig

LOGGER = logging.getLogger(__name__)


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


class TrackingWorker(AbstractWorker[TrackingData]):
    """Worker for particle tracking simulations.

    This worker performs particle tracking through accelerator lattices,
    computing positions and momenta at each BPM. It calculates gradients
    of the loss function with respect to optimisation knobs using
    differential algebra techniques.

    The implementation treats x/y and position/momentum symmetrically,
    ensuring consistent handling of all phase space dimensions.
    """

    #: Set by the ``apply_mask`` control command. Declared at class level because
    #: subclasses override ``prepare_data`` without calling super (see
    #: ``ValidationTrackingWorker``), and the command loop reads this on every
    #: message -- an unset attribute would crash any run with screening disabled.
    worker_disabled: bool = False

    observables: tuple[str, ...] = ("x", "y", "px", "py")
    include_momentum = True

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: TrackingData,
        config: WorkerConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        """Initialize the tracking worker.

        Args:
            conn: Pipe connection for communicating with main process
            worker_id: Unique identifier for this worker
            data: TrackingData container with reference measurements
            config: Configuration parameters
            simulation_config: Simulation configuration settings
        """
        super().__init__(conn, worker_id, data, config, simulation_config)

    def prepare_data(self, data: TrackingData) -> None:
        """Process and prepare tracking data for computation.

        Extracts the active observables, loads precomputed weights, splits
        data into batches, and prepares initial conditions.

        Args:
            data: TrackingData container with reference measurements
        """
        self.observables = self._resolve_observables()
        num_batches = min(self.simulation_config.num_batches, len(data.init_coords))
        if num_batches <= 0:
            raise ValueError(f"Worker {self.worker_id}: No initial coordinates available")

        n_init = len(data.init_coords)
        init_coords = data.init_coords

        LOGGER.debug(
            f"Worker {self.worker_id}: Processing {n_init} particles in {num_batches} batches"
        )

        # Validate initial conditions
        if np.isnan(init_coords).any():
            raise ValueError(f"Worker {self.worker_id}: NaNs found in initial coordinates")

        self.comparison_arrays = self._extract_observable_arrays(data, n_init)
        if data.precomputed_weights is None:
            raise ValueError("Precomputed weights must be provided for TrackingWorker")
        self.weight_arrays = self._load_precomputed_weights(data.precomputed_weights, n_init)
        self.weight_scale = data.precomputed_weights.scale
        self._prepare_batches(init_coords, data.init_pts, num_batches)
        self._prepare_uncertainty_batches(data, n_init, num_batches)

        self.worker_disabled = False
        self.compute_hessian_on_exit = True
        self.normalisation_points = self.comparisons[self.observables[0]][0].shape[1]
        self.keep_bpm_mask = np.ones(self.normalisation_points, dtype=bool)

        self.run_track_init_text = build_tracking_init_script(
            self.observables,
            start_on_first_turn=self.config.initial_condition_marker is not None,
        )
        self.run_track_script = build_tracking_script(self.observables)
        self.uncertainty_track_script = build_tracking_script(
            self.observables, include_start_derivatives=True
        )
        self._dump_debug_scripts()

    def _dump_debug_scripts(self) -> None:
        """Write generated MAD scripts to disk when debugging is enabled."""
        dump_debug_script(
            "run_track_init",
            self.run_track_init_text,
            debug=self.config.debug,
            mad_logfile=self.config.mad_logfile,
            worker_id=self.worker_id,
        )
        dump_debug_script(
            "run_track",
            self.run_track_script,
            debug=self.config.debug,
            mad_logfile=self.config.mad_logfile,
            worker_id=self.worker_id,
        )
        dump_debug_script(
            "run_track_uncertainty",
            self.uncertainty_track_script,
            debug=self.config.debug,
            mad_logfile=self.config.mad_logfile,
            worker_id=self.worker_id,
        )

    def _resolve_observables(self) -> tuple[str, ...]:
        """Return the observables active for this worker configuration."""
        kick_plane = self.config.kick_plane
        if kick_plane == "xy":
            return ("x", "y", "px", "py") if self.include_momentum else ("x", "y")
        if kick_plane == "x":
            return ("x", "px") if self.include_momentum else ("x",)
        if kick_plane == "y":
            return ("y", "py") if self.include_momentum else ("y",)
        raise ValueError(f"Unsupported kick plane {kick_plane!r}")

    def _extract_observable_arrays(self, data: TrackingData, n_init: int) -> dict[str, np.ndarray]:
        """Return comparison arrays for the observables used by this worker."""
        arrays: dict[str, np.ndarray] = {}
        for observable in self.observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            source = getattr(data, source_attr)[:n_init]
            arrays[observable] = source[:, :, plane_idx]
        return arrays

    def _load_precomputed_weights(
        self,
        weights: PrecomputedTrackingWeights,
        n_init: int,
    ) -> dict[str, np.ndarray]:
        """Return per-particle weights for the active observables."""
        return {
            observable: getattr(weights, observable)[:n_init] for observable in self.observables
        }

    def _prepare_uncertainty_batches(self, data: TrackingData, n_init: int, num_batches: int) -> None:
        """Split reading ids and variances like the comparisons, for the uncertainty part."""
        self.variances = {}
        for observable in self.observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            variances = getattr(data, source_attr.replace("comparisons", "variances"))
            self.variances[observable] = split_array_to_batches(
                variances[:n_init, :, plane_idx], num_batches
            )
        self.reading_ids = split_array_to_batches(data.reading_ids[:n_init], num_batches)
        self.init_reading_ids = split_array_to_batches(data.init_reading_ids[:n_init], num_batches)
        self.init_variances = split_array_to_batches(data.init_variances[:n_init], num_batches)

    def _prepare_batches(
        self, init_coords: np.ndarray, init_pts: np.ndarray, num_batches: int
    ) -> None:
        """Split data and initial conditions into batches.

        Args:
            init_coords: Initial particle coordinates
            init_pts: Initial transverse momentum values
            num_batches: Number of batches to create
        """
        # Keep flat numpy copies for fast update transfers
        self._init_coords_np = np.ascontiguousarray(init_coords, dtype=np.float64)
        self._init_pts_np = np.ascontiguousarray(init_pts, dtype=np.float64)

        # Split initial conditions
        init_coords_batches = split_array_to_batches(init_coords, num_batches)
        init_pts_batches = split_array_to_batches(init_pts, num_batches)

        # Convert to nested lists for MAD-NG
        self.init_coords = [batch.tolist() for batch in init_coords_batches]
        self.init_pts = [batch.tolist() for batch in init_pts_batches]
        self.batch_size = len(self.init_coords[0])
        self.num_batches = num_batches

        self.comparisons = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in self.comparison_arrays.items()
        }
        self.weights = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in self.weight_arrays.items()
        }

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Configure MAD-NG sequence for tracking.

        Args:
            mad: MAD-NG interface object
        """
        mad["batch_size"] = self.batch_size
        mad["num_batches"] = self.num_batches
        mad["optimise_energy"] = self.config.accelerator.optimise_energy
        mad["tracking_range"] = self.tracking_range
        mad["n_run_turns"] = self.simulation_config.n_run_turns

    def _setup_da_maps(self, mad: MAD) -> None:
        """Setup differential algebra maps for tracking.

        Creates base DAMAP and adds knob parameters for differentiation.

        Args:
            mad: MAD-NG interface object
        """
        # Remove "pt" from knob names if present (handled separately)
        knob_names = list(mad["knob_names"])
        if "pt" in knob_names:
            knob_names.remove("pt")
            mad["knob_names"] = knob_names

        # Create base DAMAP
        self.create_base_damap(mad, knob_order=1)

        # Add knobs as TPSA variables
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    loaded_sequence[param] = loaded_sequence[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

    def send_initial_conditions(self, mad: MAD) -> None:
        """Send initial particle coordinates to MAD-NG.

        Creates DAMAP objects for each particle in each batch.

        Args:
            mad: MAD-NG interface object
        """
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

    def _initialise_mad_computation(self, mad: MAD) -> None:
        """Initialise MAD-NG environment for tracking computations.

        Args:
            mad: MAD-NG interface object
        """
        mad.send(self.run_track_init_text)

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

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute gradients and loss for a batch of particle tracking.

        Performs tracking simulation, receives position and momentum data
        along with their derivatives, and computes loss and gradients using
        weighted least-squares formulation.

        Args:
            mad: MAD-NG interface object
            knob_updates: Dictionary of knob names to values
            batch: Batch index to process

        Returns:
            Tuple of (gradient array, loss value)
        """
        results = self._run_tracking_batch(mad, knob_updates, batch)

        # Compute loss and gradients
        return self._compute_loss_and_gradients(results, batch)

    def _receive_tracking_results(
        self, mad: MAD, *, include_start_derivatives: bool = False
    ) -> dict[str, np.ndarray]:
        """Receive tracking results from MAD-NG.

        Args:
            mad: MAD-NG interface object
            include_start_derivatives: Also receive ``d{o}_dc0``, shape
                ``(n_particles, 2, n_points)``, the derivatives by the start x and y.

        Returns:
            Dictionary with one result array and one derivative array per active
            observable.
        """
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

    def _send_knobs(self, mad: MAD, knobs: dict[str, float]) -> None:
        """Set this worker's sequence knobs; ``pt`` is applied per particle instead."""
        update_commands = [
            f"loaded_sequence['{name}']:set0({val:.15e})"
            for name, val in knobs.items()
            if name != "pt" and name in self.knob_name_set
        ]
        if update_commands:
            mad.send("\n".join(update_commands))

    def _run_tracking_batch(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> dict[str, np.ndarray]:
        """Run MAD-NG tracking for a single batch and return all outputs."""
        machine_pt = knob_updates.get("pt", getattr(self, "fixed_pt", 0.0))
        self._send_knobs(mad, knob_updates)

        mad.send(f"batch = {batch + 1}")
        mad.send(f"""
for i = 1, batch_size do
    da_x0_c[batch][i].pt:set0({machine_pt:.15e} + init_pts[batch][i])
end
""")
        mad.send(self.run_track_script)
        return self._receive_tracking_results(mad)

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

    def _apply_runtime_mask(self, keep_bpm_mask: np.ndarray) -> None:
        """Apply BPM keep-mask for subsequent optimisation and Hessian steps."""
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

    def _handle_control_command(self, mad: MAD, command: dict[str, object]) -> None:
        """Handle control-plane commands from parent process."""
        cmd = command.get("cmd")
        if cmd == "diagnostics":
            raw_knobs = command.get("knobs", {})
            if not isinstance(raw_knobs, dict):
                raise ValueError(
                    f"Worker {self.worker_id}: diagnostics command missing knob dictionary"
                )
            diagnostic_knobs: dict[str, float] = {}
            for knob_name, knob_value in raw_knobs.items():
                if not isinstance(knob_name, str):
                    raise ValueError(
                        f"Worker {self.worker_id}: knob name {knob_name!r} is not a string"
                    )
                if not isinstance(knob_value, int | float | np.floating):
                    raise ValueError(
                        f"Worker {self.worker_id}: knob {knob_name!r} has non-numeric value {knob_value!r}"
                    )
                diagnostic_knobs[knob_name] = float(knob_value)
            total_loss, loss_per_bpm = self.compute_diagnostics(mad, diagnostic_knobs)
            self.conn.send(
                {
                    "worker_id": self.worker_id,
                    "total_loss": total_loss / self.normalisation_points,
                    "loss_per_bpm": (loss_per_bpm / self.normalisation_points).tolist(),
                }
            )
            return

        if cmd == "apply_mask":
            keep_bpm_mask = np.asarray(command.get("keep_bpm_mask", []), dtype=bool)
            disable_worker = bool(command.get("disable_worker", False))
            if keep_bpm_mask.size:
                self._apply_runtime_mask(keep_bpm_mask)
            self.worker_disabled = disable_worker
            self.conn.send({"worker_id": self.worker_id, "status": "ok"})
            return

        if cmd == "set_hessian_mode":
            self.compute_hessian_on_exit = bool(command.get("enabled", True))
            self.conn.send({"worker_id": self.worker_id, "status": "ok"})
            return

        if cmd == "set_knobs":
            knobs = command["knobs"]
            self._send_knobs(mad, knobs)
            mad.send(f"""
for b = 1, num_batches do
    for i = 1, batch_size do
        da_x0_c[b][i].pt:set0({knobs.get("pt", getattr(self, "fixed_pt", 0.0)):.15e} + init_pts[b][i])
    end
end
""")
            self.conn.send({"worker_id": self.worker_id, "status": "ok"})
            return

        if cmd == "update_init_coords":
            self._send_init_condition_update(
                mad,
                *(
                    np.asarray(command[key], dtype=np.float64)
                    for key in ("x", "px", "y", "py")
                ),
            )
            self.conn.send({"worker_id": self.worker_id, "status": "ok"})
            return

        raise ValueError(f"Worker {self.worker_id}: Unknown command {cmd}")

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

    def _compute_loss_and_gradients(
        self, results: dict[str, np.ndarray], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute weighted loss and gradients from tracking results.

        Uses symmetric treatment of all phase space dimensions.

        Args:
            results: Dictionary of tracking results and derivatives
            batch: Batch index

        Returns:
            Tuple of (gradient array, loss value)
        """
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

    def run(self) -> None:
        """Main worker run loop with Hessian calculation.

        Extends the base run method to compute approximate Hessian
        after the main optimisation loop completes.
        """
        mad: MAD | None = None
        n_knobs = 0
        computation_success = True

        try:
            self.configure_python_worker_logging()
            self.configure_worker_threads()
            knob_values, batch = self.conn.recv()
            if knob_values is None:
                return
            n_knobs = len(knob_values)

            mad, nbpms = self.setup_mad_interface(knob_values)
            self.send_initial_conditions(mad)
            self._initialise_mad_computation(mad)
            self.run_preflight_check(mad, nbpms)

            LOGGER.debug(f"Worker {self.worker_id}: Ready for computation with {nbpms} BPMs")

            message: tuple[dict[str, float] | None, int | None] | dict[str, object] = (
                self.conn.recv()
            )

            while True:
                while isinstance(message, dict):
                    self._handle_control_command(mad, message)
                    message = self.conn.recv()

                knob_values, batch = message
                if knob_values is None or batch is None:
                    LOGGER.debug(f"Worker {self.worker_id}: Received termination signal")
                    break
                try:
                    if self.worker_disabled:
                        self.conn.send((self.worker_id, np.zeros(n_knobs), 0.0))
                    else:
                        grad, loss = self.compute_gradients_and_loss(mad, knob_values, int(batch))
                        # Report a per-turn, per-BPM-point loss so it is comparable
                        # across workers (and to the held-out validation loss)
                        # regardless of how many turns a worker holds. The gradient
                        # keeps its own normalisation (per-point here, per-turn via
                        # total_turns in the loop) and is intentionally unchanged.
                        n_turns = max(1, len(self.init_coords[int(batch)]))
                        self.conn.send(
                            (
                                self.worker_id,
                                # Read live: ``apply_mask`` shrinks
                                # ``normalisation_points`` to the kept BPM count, and
                                # a snapshot taken before the command loop would keep
                                # dividing the masked loss by the unmasked count --
                                # putting training and validation on different scales.
                                grad / self.normalisation_points,
                                loss / (self.normalisation_points * n_turns),
                            )
                        )
                except ParticleLostError as exc:
                    LOGGER.warning(
                        "Worker %d: %s — sending zero contribution; epoch update will be rejected",
                        self.worker_id,
                        exc,
                    )
                    # NaN loss signals the aggregator to reject this epoch's knob updates.
                    self.conn.send((self.worker_id, np.zeros(n_knobs), float("nan")))
                except Exception as exc:  # noqa: BLE001
                    self.send_error_payload(exc, phase="computation")
                    computation_success = False
                    break

                message = self.conn.recv()

            if computation_success and not self.worker_disabled and self.compute_hessian_on_exit:
                LOGGER.debug(f"Worker {self.worker_id}: Propagating uncertainty")
                try:
                    self.conn.send(self._compute_uncertainty_part(mad, n_knobs))
                except Exception as exc:  # noqa: BLE001
                    self.send_error_payload(exc, phase="hessian")
                    computation_success = False
            else:
                self.conn.send(UncertaintyPart.empty(n_knobs))
        except Exception as exc:  # noqa: BLE001
            self.send_error_payload(exc, phase="startup")
        finally:
            LOGGER.debug(f"Worker {self.worker_id}: Terminating")
            if mad is not None:
                mad.send("shush()")
                del mad

    @staticmethod
    def get_n_data_points(nbpms: int, n_turns: int = 1) -> int:
        """Get number of data points for tracking.

        Args:
            nbpms: Number of BPMs in the range
            n_turns: Number of tracking turns (default 1 for arc-by-arc)

        Returns:
            Total number of data points (nbpms * n_turns)
        """
        return nbpms * n_turns
