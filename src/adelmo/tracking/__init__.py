"""Tracking fits: magnet knobs from turn-by-turn BPM data by mini-batch gradient descent.

The public entry points are the three tracking fitters -- :class:`ArcByArcFitter`,
:class:`ACDMarkerFitter` and :class:`KickerFitter`. They share the machine setup,
worker pool and results of :mod:`adelmo.fitting` with the closed-orbit and
closed-twiss fitters of :mod:`adelmo.poco`.
"""

from adelmo.tracking.config import (
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
from adelmo.tracking.config.tracking import extract_bpm_range_names
from adelmo.tracking.data_manager import DataManager, FileTracks
from adelmo.tracking.fitter import (
    ACDMarkerFitter,
    ArcByArcFitter,
    FitterOptions,
    KickerFitter,
    TrackingFitter,
)
from adelmo.tracking.session import TrackingSession
from adelmo.tracking.sgd.loop import EpochState, SGDLoop
from adelmo.tracking.sgd.scheduler import LRScheduler

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
