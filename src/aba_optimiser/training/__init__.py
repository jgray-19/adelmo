"""Training loops and utilities for magnet knob optimisation.

The public entry points are the three tracking fitters -- :class:`ArcByArcFitter`,
:class:`ACDMarkerFitter` and :class:`KickerFitter` -- which orchestrate gradient
evaluation, learning-rate scheduling, and checkpointing.
"""

from aba_optimiser.training.base_fitter import BaseFitter
from aba_optimiser.training.config import (
    ACDArcByArcTrackingPlan,
    ACDTrackingPlan,
    CheckpointConfig,
    ConfigurationManager,
    KickerConfig,
    KickerTrackingPlan,
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
    TrackingModeSetup,
    TrackingPlan,
    WorkerRangeSpec,
    acd_marker_setup,
    arc_by_arc_setup,
    create_arc_measurement_config,
    kicker_setup,
)
from aba_optimiser.training.data_manager import DataManager, FileTracks
from aba_optimiser.training.optimisation.loop import OptimisationLoop
from aba_optimiser.training.optimisation.scheduler import LRScheduler
from aba_optimiser.training.tracking_fitter import (
    ACDMarkerFitter,
    ArcByArcFitter,
    KickerFitter,
    TrackingFitter,
)
from aba_optimiser.training.utils import (
    extract_bpm_range_names,
    filter_bad_bpms,
    normalise_true_strengths,
)
from aba_optimiser.training.workers.manager import WorkerManager
from aba_optimiser.training.workers.pool import WorkerPool

__all__ = [
    "ACDArcByArcTrackingPlan",
    "ACDMarkerFitter",
    "ACDTrackingPlan",
    "ArcByArcFitter",
    "BaseFitter",
    "CheckpointConfig",
    "ConfigurationManager",
    "DataManager",
    "FileTracks",
    "KickerConfig",
    "KickerFitter",
    "KickerTrackingPlan",
    "LRScheduler",
    "MeasurementConfig",
    "MeasurementDetails",
    "OptimisationLoop",
    "OutputConfig",
    "SequenceConfig",
    "TrackingFitter",
    "TrackingModeSetup",
    "TrackingPlan",
    "WorkerManager",
    "WorkerPool",
    "WorkerRangeSpec",
    "acd_marker_setup",
    "arc_by_arc_setup",
    "create_arc_measurement_config",
    "extract_bpm_range_names",
    "filter_bad_bpms",
    "kicker_setup",
    "normalise_true_strengths",
]
