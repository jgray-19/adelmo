# src/adelmo/config.py
"""
Configuration constants for the knob optimisation pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# OPTIMISATION SETTINGS
# =============================================================================

logger = logging.getLogger(__name__)


@dataclass
class OptimiserConfig:
    """Configuration for gradient descent optimiser algorithms.

    Controls learning rates, convergence criteria, and optimiser type
    for the gradient descent training process.
    """

    max_epochs: int
    warmup_epochs: int
    warmup_lr_start: float
    max_lr: float
    min_lr: float
    gradient_converged_value: float
    optimiser_type: str = field(default="adam")  # Options: "adam", "lbfgs"

    # Smoothing factor of the moving averages of the gradient norm and the
    # relative loss change that the stopping rules use.
    grad_norm_alpha: float = field(default=0.2)
    # Stop once the smoothed relative loss change falls below this, after at least
    # loss_change_min_epoch_fraction * max_epochs epochs. 0 disables the rule.
    loss_change_tolerance: float = field(default=1e-6)
    loss_change_min_epoch_fraction: float = field(default=0.2)

    # L-BFGS-specific parameters (ignored for adam)
    lbfgs_history_size: int = field(default=10)
    lbfgs_max_grad_norm: float | None = field(default=1.0)
    lbfgs_max_step_norm: float | None = field(default=1.0)
    lbfgs_powell_damping: float = field(default=0.2)

    # Adam-specific parameters (ignored for lbfgs)
    adam_weight_decay: float = field(default=0.0)
    # None keeps each optimiser's own default. Lower it when the gradients are far
    # below the default, or eps rather than the gradient scale sets the step size.
    adam_eps: float | None = field(default=None)

    # Computed fields
    decay_epochs: int = field(init=False)

    def __post_init__(self):
        self.decay_epochs = self.max_epochs - self.warmup_epochs

    def log_state(self) -> None:
        """Log the current optimiser config settings."""
        logger.info("OptimiserConfig: %s", self)


@dataclass
class SimulationConfig:
    """Configuration for simulation and worker process settings.

    Controls parallel worker distribution, tracking data, and which
    physical parameters (energy, quadrupoles, bends) to optimise.
    """

    num_workers: int
    num_batches: int

    # Fraction of the post-validation-split training data to use and
    # distribute among the workers. 1.0 (default) uses every available training
    # turn; smaller values keep that fraction of each file's turns (per-file
    # stratified sampling). Must be in (0, 1].
    data_fraction: float = field(default=1.0)

    # Fraction of the available turns held out per file as a disjoint validation
    # set used only to measure generalisation. These turns are removed from
    # training, so the validation loss is out-of-sample. `data_fraction` is applied to the remaining training
    # turns. Must be in [0, 1); 0.0 disables held-out validation.
    validation_fraction: float = field(default=0.1)

    # Whether to include momenta (px, py) in loss function
    # When False, only positions (x, y) are used for optimisation
    optimise_momenta: bool = field(default=True)

    # Number of turns tracked from each start turn. Arc-by-arc and AC-dipole
    # marker modes force this to 1; kicker mode sets it to the kicker track length.
    n_run_turns: int = field(default=1)

    # How start x end BPM points expand into ranges. True (default) pairs every
    # start with one fixed end and every end with one fixed start; False takes the
    # full cartesian product, giving many more (but more correlated) ranges.
    use_fixed_bpm: bool = field(default=True)

    # Logging level for worker processes (separate from main process)
    worker_logging_level: int = field(default=logging.WARNING)

    # BLAS/LAPACK threads per worker process. Every worker does dense linear algebra
    # (e.g. the Gauss-Newton Hessian) while the other workers do the same, so the
    # library default (one thread per core, per worker) oversubscribes the machine:
    # 16 closed-orbit workers took 3.4x longer per Hessian and pushed the load
    # average to ~290 on a 64-core host. 1 (default) keeps one thread per worker;
    # None leaves the library default.
    worker_blas_threads: int | None = field(default=1)

    # Probe the workers at the initial knobs before optimising and mask BPMs and
    # workers whose loss z-score exceeds the thresholds below, so grossly bad
    # data cannot dominate the fit. Disable to obtain the raw per-BPM behaviour.
    enable_preloop_outlier_screening: bool = field(default=True)
    bpm_loss_outlier_sigma: float = field(default=3.0)
    worker_loss_outlier_sigma: float = field(default=3.0)

    def __post_init__(self):
        if self.n_run_turns < 1:
            raise ValueError("SimulationConfig.n_run_turns must be >= 1")
        if not 0.0 < self.data_fraction <= 1.0:
            raise ValueError(
                f"SimulationConfig.data_fraction must be in (0, 1], got {self.data_fraction}"
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                "SimulationConfig.validation_fraction must be in [0, 1), "
                f"got {self.validation_fraction}"
            )

    def log_state(self) -> None:
        """Log the current simulation config settings."""
        logger.info("SimulationConfig: %s", self)


# Global schema constant for data files (this is appropriate as a global)
FILE_COLUMNS: tuple[str, ...] = (
    "name",
    "turn",
    "bunch_number",
    "x",
    "px",
    "y",
    "py",
    "var_x",
    "var_y",
    "var_px",
    "var_py",
)

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
TRAINING_RUNS_ROOT = ARTIFACTS_ROOT / "training" / "runs"
logger.info(f"Current project root: {PROJECT_ROOT}")

# Data files
NO_NOISE_FILE = PROJECT_ROOT / "data/track_data.parquet"  # Measurement Parquet file
CLEANED_FILE = PROJECT_ROOT / "data/filtered_data.parquet"  # Filtered TFS file

# Other files
# Ground-truth knob strengths
TRUE_STRENGTHS_FILE = PROJECT_ROOT / "data/true_strengths.txt"
# Where to write final strengths
OUTPUT_KNOBS = PROJECT_ROOT / "data/final_knobs.txt"
