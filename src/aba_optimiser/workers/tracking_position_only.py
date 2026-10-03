"""Position-only tracking worker specialization."""

from __future__ import annotations

from aba_optimiser.workers.tracking import TrackingWorker


class PositionOnlyConfigMixin:
    """Observable configuration for workers that compare only x/y positions.

    Shared by the training and validation position-only workers so the
    (x, y) / no-momentum settings are defined in exactly one place.
    """

    observables = ("x", "y")
    include_momentum = False


class PositionOnlyTrackingWorker(PositionOnlyConfigMixin, TrackingWorker):
    """Tracking worker that compares only x/y position observables."""
