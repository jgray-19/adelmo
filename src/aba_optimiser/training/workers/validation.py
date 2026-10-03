"""Worker payload type and the training/validation payload pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from aba_optimiser.workers import TrackingData, WorkerConfig


WorkerPayload: TypeAlias = tuple["TrackingData", "WorkerConfig", int]


def payload_track_count(payload: WorkerPayload) -> int:
    """Return number of tracked turns represented by one payload."""
    data, _config, _file_idx = payload
    return int(data.init_coords.shape[0])


@dataclass(frozen=True)
class ValidationSplitResult:
    """Training payloads and held-out validation payloads.

    Both are built from disjoint turns (the validation turns were removed from
    training upstream in ``DataManager``), so validation loss is a genuine
    out-of-sample signal.
    """

    training_payloads: list[WorkerPayload]
    validation_payloads: list[WorkerPayload]
