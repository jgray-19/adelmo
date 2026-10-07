"""Checkpoint persistence and knob-remapping for the optimisation loop.

The :class:`OptimisationCheckpointer` owns everything related to saving and
restoring an :class:`~aba_optimiser.tracking.sgd.loop.SGDLoop`:
serialising loop/optimiser state to JSON, deciding when a checkpoint is due, and
remapping a saved optimiser state onto the loop's current knob layout when the
two differ. It reads and writes the loop's state directly through a back
reference so the loop itself stays focused on running epochs.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.tracking.config.models import CheckpointConfig
    from aba_optimiser.tracking.sgd.loop import SGDLoop

LOGGER = logging.getLogger(__name__)


class CheckpointState(TypedDict):
    saved_epoch: int
    next_epoch: int
    current_knobs: dict[str, float]
    prev_loss: float | None


class OptimisationCheckpointer:
    """Saves and restores optimisation-loop state to/from a JSON checkpoint."""

    def __init__(
        self,
        loop: SGDLoop,
        checkpoint_config: CheckpointConfig | None = None,
    ) -> None:
        self._loop = loop
        if checkpoint_config is None:
            self.path: Path | None = None
            self.every_n_epochs = 0
            self.restore = False
        else:
            self.path = checkpoint_config.checkpoint_path
            self.every_n_epochs = checkpoint_config.checkpoint_every_n_epochs
            self.restore = checkpoint_config.restore_from_checkpoint

    @property
    def _knob_names(self) -> list[str]:
        return self._loop.knob_names

    def should_save_periodic(self, epoch: int) -> bool:
        """Return True when this epoch should trigger periodic checkpointing."""
        return (
            self.path is not None
            and self.every_n_epochs > 0
            and (epoch + 1) % self.every_n_epochs == 0
        )

    def should_save_final(self, last_completed_epoch: int) -> bool:
        """Return True when a final checkpoint should be written on loop exit."""
        return self.path is not None and self.every_n_epochs > 0 and last_completed_epoch >= 0

    def save(self, epoch: int, current_knobs: dict[str, float], prev_loss: float | None) -> None:
        """Save optimisation state so the run can be resumed later."""
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_epoch": int(epoch),
            "next_epoch": int(epoch + 1),
            "knob_names": self._knob_names,
            "current_knobs": {k: float(v) for k, v in current_knobs.items()},
            "best_knobs": {k: float(v) for k, v in self._loop.best.value.items()},
            "best_loss": float(self._loop.best.loss),
            "prev_loss": None if prev_loss is None else float(prev_loss),
            "smoothed_grad_norm": float(self._loop.smoothed_grad_norm),
            "smoothed_loss_change": float(self._loop.smoothed_loss_change),
            "optimiser_class": self._loop.optimiser.__class__.__name__,
            "optimiser_state": self._loop.optimiser.state_dict(),
        }
        self.path.write_text(json.dumps(payload, indent=2))

    def load(self, base_current_knobs: dict[str, float] | None = None) -> CheckpointState:
        """Load optimisation state from the checkpoint and apply it to the loop."""
        assert self.path is not None
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.path}")

        payload = cast("dict[str, Any]", json.loads(self.path.read_text()))

        saved_knob_names = [str(k) for k in payload.get("knob_names", [])]
        saved_knob_set = self._validate_checkpoint_knob_compatibility(saved_knob_names)

        merged_current_knobs = self._initialise_merged_current_knobs(base_current_knobs)

        checkpoint_current = self._parse_checkpoint_knob_values(payload, "current_knobs")

        missing_checkpoint_current = saved_knob_set.difference(set(checkpoint_current.keys()))
        if missing_checkpoint_current:
            raise ValueError(
                "Checkpoint current knobs are missing saved checkpoint knob values: "
                + self._clip_name_list_for_error(missing_checkpoint_current)
            )

        merged_current_knobs.update(checkpoint_current)
        self._validate_finite_knob_values(merged_current_knobs, label="Restored current knobs")

        checkpoint_best = self._parse_checkpoint_knob_values(payload, "best_knobs")
        self._validate_finite_knob_values(checkpoint_best, label="Checkpoint best_knobs")

        best_knobs = merged_current_knobs.copy()
        best_knobs.update(checkpoint_best)
        self._loop.best.record(float(payload.get("best_loss", float("inf"))), best_knobs)
        self._loop.smoothed_grad_norm = float(payload.get("smoothed_grad_norm", 0.0))
        self._loop.smoothed_loss_change = float(payload.get("smoothed_loss_change", 0.0))

        optimiser_state = payload.get("optimiser_state", {})
        if optimiser_state:
            try:
                remap = None
                if saved_knob_names != self._knob_names:

                    def remap(vector: list[float]) -> list[float]:
                        return self._expand_vector_to_current_knobs(
                            vector, saved_knob_names, self._knob_names, fill_value=0.0
                        )

                self._loop.optimiser.load_state_dict(
                    cast("dict[str, Any]", optimiser_state), remap=remap
                )
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.info("Skipping optimiser state restore: %s", exc)

        return {
            "saved_epoch": int(payload.get("saved_epoch", 0)),
            "next_epoch": int(payload.get("next_epoch", 0)),
            "current_knobs": merged_current_knobs,
            "prev_loss": (
                float(payload["prev_loss"]) if payload.get("prev_loss") is not None else None
            ),
        }

    @staticmethod
    def _clip_name_list_for_error(names: set[str]) -> str:
        """Format long knob-name sets for concise error messages."""
        return ", ".join(sorted(names)[:10]) + ("..." if len(names) > 10 else "")

    @staticmethod
    def _nonfinite_knob_names(knobs: dict[str, float]) -> list[str]:
        """Return knob names whose values are not finite."""
        return [name for name, value in knobs.items() if not np.isfinite(value)]

    def _validate_finite_knob_values(self, knobs: dict[str, float], *, label: str) -> None:
        """Reject knob maps that contain NaN or infinite values."""
        nonfinite = self._nonfinite_knob_names(knobs)
        if nonfinite:
            raise ValueError(
                f"{label} contains non-finite knob values: "
                + self._clip_name_list_for_error(set(nonfinite))
            )

    def _initialise_merged_current_knobs(
        self,
        base_current_knobs: dict[str, float] | None,
    ) -> dict[str, float]:
        """Build the starting current-knob map used when restoring checkpoints."""
        if base_current_knobs is None:
            return {k: (1e-6 if k == "pt" else 0.0) for k in self._knob_names}

        if set(base_current_knobs.keys()) != set(self._knob_names):
            raise ValueError(
                "base_current_knobs must contain exactly the current optimisation knob set."
            )
        merged = {str(k): float(v) for k, v in base_current_knobs.items()}
        self._validate_finite_knob_values(merged, label="base_current_knobs")
        return merged

    def _validate_checkpoint_knob_compatibility(self, saved_knob_names: list[str]) -> set[str]:
        """Ensure checkpoint knobs are a subset of the current optimisation knobs."""
        saved_knob_set = set(saved_knob_names)
        missing_in_current = saved_knob_set.difference(set(self._knob_names))
        if missing_in_current:
            raise ValueError(
                "Checkpoint knob names are not compatible with current optimisation setup. "
                "Current setup is missing checkpoint knobs: "
                + self._clip_name_list_for_error(missing_in_current)
            )
        return saved_knob_set

    def _parse_checkpoint_knob_values(
        self,
        payload: dict[str, Any],
        field_name: str,
    ) -> dict[str, float]:
        """Parse and validate a knob-value mapping from checkpoint payload."""
        checkpoint_values = {
            str(k): float(v) for k, v in cast("dict[str, Any]", payload.get(field_name, {})).items()
        }
        unknown_names = set(checkpoint_values.keys()).difference(set(self._knob_names))
        if unknown_names:
            raise ValueError(
                f"Checkpoint {field_name} contain names that are not in the current optimisation setup: "
                + self._clip_name_list_for_error(unknown_names)
            )
        self._validate_finite_knob_values(checkpoint_values, label=f"Checkpoint {field_name}")
        return checkpoint_values

    @staticmethod
    def _expand_vector_to_current_knobs(
        vector: list[float],
        saved_knob_names: list[str],
        current_knob_names: list[str],
        fill_value: float,
    ) -> list[float]:
        """Remap a saved optimizer vector onto the current knob layout with padding."""
        if len(vector) != len(saved_knob_names):
            raise ValueError(
                "Optimiser state vector length does not match checkpoint knob_names length."
            )
        saved_index = {name: i for i, name in enumerate(saved_knob_names)}
        return [
            float(vector[saved_index[name]]) if name in saved_index else fill_value
            for name in current_knob_names
        ]
