"""Tracking fits: magnet knobs from turn-by-turn BPM data by mini-batch gradient descent.

The public entry points are the three tracking fitters -- :class:`ArcByArcFitter`,
:class:`ACDMarkerFitter` and :class:`KickerFitter`. They share the machine setup,
worker pool and results of :mod:`aba_optimiser.fitting` with the closed-orbit and
closed-twiss fitters of :mod:`aba_optimiser.poco`.
"""

from aba_optimiser.tracking.config import (
    ACDArcByArcTrackingPlan,
    ACDTrackingPlan,
    CheckpointConfig,
    KickerConfig,
    KickerTrackingPlan,
    MeasurementConfig,
    MeasurementDetails,
    TrackingModeSetup,
    TrackingPlan,
    WorkerRangeSpec,
    acd_marker_setup,
    arc_by_arc_setup,
    create_arc_measurement_config,
    kicker_setup,
)
from aba_optimiser.tracking.config.tracking import extract_bpm_range_names
from aba_optimiser.tracking.data_manager import DataManager, FileTracks
from aba_optimiser.tracking.fitter import (
    ACDMarkerFitter,
    ArcByArcFitter,
    FitterOptions,
    KickerFitter,
    TrackingFitter,
)
from aba_optimiser.tracking.session import TrackingSession
from aba_optimiser.tracking.sgd.loop import EpochState, SGDLoop
from aba_optimiser.tracking.sgd.scheduler import LRScheduler

__all__ = [
    "ACDArcByArcTrackingPlan",
    "ACDMarkerFitter",
    "ACDTrackingPlan",
    "ArcByArcFitter",
    "CheckpointConfig",
    "DataManager",
    "EpochState",
    "FileTracks",
    "FitterOptions",
    "KickerConfig",
    "KickerFitter",
    "KickerTrackingPlan",
    "LRScheduler",
    "MeasurementConfig",
    "MeasurementDetails",
    "SGDLoop",
    "TrackingFitter",
    "TrackingModeSetup",
    "TrackingPlan",
    "TrackingSession",
    "WorkerRangeSpec",
    "acd_marker_setup",
    "arc_by_arc_setup",
    "create_arc_measurement_config",
    "extract_bpm_range_names",
    "kicker_setup",
]
