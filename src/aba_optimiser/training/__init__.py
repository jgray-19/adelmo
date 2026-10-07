"""Training loops and utilities for magnet knob optimisation.

The public entry points are the three tracking fitters -- :class:`ArcByArcFitter`,
:class:`ACDMarkerFitter` and :class:`KickerFitter`. The closed-orbit and
closed-twiss fitters live in :mod:`aba_optimiser.poco`; both families share
:class:`MachineSetup`, :class:`WorkerPool` and :class:`FitResult` from here.
"""

from aba_optimiser.training.config import (
    ACDArcByArcTrackingPlan,
    ACDTrackingPlan,
    CheckpointConfig,
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
from aba_optimiser.training.machine_setup import MachineSetup
from aba_optimiser.training.pool import WorkerPool
from aba_optimiser.training.results import FitDiagnostics, FitResult
from aba_optimiser.training.sgd.loop import EpochState, SGDLoop
from aba_optimiser.training.sgd.scheduler import LRScheduler
from aba_optimiser.training.tracking.data_manager import DataManager, FileTracks
from aba_optimiser.training.tracking.fitter import (
    ACDMarkerFitter,
    ArcByArcFitter,
    FitterOptions,
    KickerFitter,
    TrackingFitter,
)
from aba_optimiser.training.tracking.session import TrackingSession
from aba_optimiser.training.utils import (
    extract_bpm_range_names,
    filter_bad_bpms,
    normalise_true_strengths,
)

__all__ = [
    "ACDArcByArcTrackingPlan",
    "ACDMarkerFitter",
    "ACDTrackingPlan",
    "ArcByArcFitter",
    "CheckpointConfig",
    "DataManager",
    "EpochState",
    "FileTracks",
    "FitDiagnostics",
    "FitResult",
    "FitterOptions",
    "KickerConfig",
    "KickerFitter",
    "KickerTrackingPlan",
    "LRScheduler",
    "MachineSetup",
    "MeasurementConfig",
    "MeasurementDetails",
    "OutputConfig",
    "SGDLoop",
    "SequenceConfig",
    "TrackingFitter",
    "TrackingModeSetup",
    "TrackingPlan",
    "TrackingSession",
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
