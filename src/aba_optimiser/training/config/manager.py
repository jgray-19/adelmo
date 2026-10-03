"""Configuration and setup management for the optimisation fitter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.mad.optimising_mad_interface import is_magnet_strength_name
from aba_optimiser.training.config.tracking import RangeContext, TrackingPlan

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.config.models import SequenceConfig


LOGGER = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages configuration and setup for the optimisation process."""

    def __init__(
        self,
        accelerator: Accelerator,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        optimise_knobs: list[str] | None = None,
        tracking_plan: TrackingPlan | None = None,
    ):
        self.mad_iface: GradientDescentMadInterface = None
        self.knob_names: list[str] = []
        self.elem_spos: list[float] = []
        self.all_bpms: list[str] = []
        self.initial_strengths: np.ndarray = np.array([])
        self.initial_model_values: dict[str, float] = {}
        self.fixed_start: str = ""
        self.fixed_end: str = ""

        self.accelerator: Accelerator = accelerator
        self.start_bpms: list[str] = bpm_start_points
        self.end_bpms: list[str] = bpm_end_points
        self.sequence_config = sequence_config
        self.magnet_range = sequence_config.magnet_range
        self.simulation_config = simulation_config
        self.optimise_knobs = optimise_knobs
        self.tracking_plan = tracking_plan if tracking_plan is not None else TrackingPlan()

    def setup_mad_interface(
        self,
        debug: bool = False,
        mad_logfile: Path | None = None,
        corrector_knobs: Path | None = None,
        tune_knobs: Path | None = None,
        b2_errors: Path | None = None,
    ) -> None:
        """Initialise the MAD-NG interface and get basic model parameters."""

        self.mad_iface = GradientDescentMadInterface(
            accelerator=self.accelerator,
            magnet_range=self.magnet_range,
            corrector_knobs=corrector_knobs,
            tune_knobs=tune_knobs,
            b2_errors=b2_errors,
            bad_bpms=self.sequence_config.bad_bpms,
            debug=debug,
            mad_logfile=mad_logfile,
            tracking_anchor_mode=self.tracking_plan.tracking_anchor_mode,
            tracking_anchor_markers=list(self.tracking_plan.tracking_anchor_sources),
        )
        self.knob_names = self.mad_iface.knob_names

        if self.tracking_plan.cycle_marker is not None:
            self.mad_iface.cycle_to_start(self.tracking_plan.cycle_marker)
            self.mad_iface.bpms_in_range, self.mad_iface.nbpms, self.mad_iface.all_bpms = (
                self.mad_iface.count_bpms(self.mad_iface.bpm_range)
            )

        self.elem_spos: list[int | float] = self.mad_iface.elem_spos

        self.all_bpms = self.mad_iface.all_bpms
        self.bpms_in_range = self.mad_iface.bpms_in_range
        range_label = (
            "full cycled sequence ($start/$end)"
            if self.magnet_range == "$start/$end"
            else self.magnet_range
        )
        LOGGER.info(
            "Total BPMs in model: %d, BPMs in configured observation range %s: %d",
            len(self.all_bpms),
            range_label,
            len(self.bpms_in_range),
        )

        # Marker-anchored modes (kicker/ACD) may start from installed or measured
        # marker anchors rather than ordinary BPMs, so keep those through the
        # BPM-range filter.
        allowed_starts = set(self.tracking_plan.tracking_anchor_sources)
        if self.tracking_plan.init_marker is not None:
            allowed_starts.add(self.tracking_plan.init_marker)
        self.start_bpms = [
            bpm for bpm in self.start_bpms if bpm in self.bpms_in_range or bpm in allowed_starts
        ]
        self.end_bpms = [bpm for bpm in self.end_bpms if bpm in self.bpms_in_range]

        # With use_fixed_bpm the fixed BPM window comes from magnet_range; otherwise
        # fixed_start/fixed_end stay empty and ranges come from start_bpms/end_bpms.
        if self.simulation_config.use_fixed_bpm and self.tracking_plan.uses_fixed_bpm_window:
            self.fixed_start, self.fixed_end = self.magnet_range.split("/", 1)

            # Validate fixed points are in the model
            if (
                self.fixed_start not in self.bpms_in_range
                or self.fixed_end not in self.bpms_in_range
            ):
                LOGGER.warning(
                    "Fixed BPMs from range %s not found in model, using first available",
                    range_label,
                )
                self.fixed_start = self.start_bpms[0] if self.start_bpms else self.fixed_start
                self.fixed_end = self.end_bpms[0] if self.end_bpms else self.fixed_end
        elif self.simulation_config.use_fixed_bpm:
            LOGGER.info(
                "Skipping fixed BPM derivation for this tracking plan; %s: %s",
                self.tracking_plan.start_point_label,
                self.start_bpms,
            )

    @property
    def bpm_pairs(self) -> list[tuple[str, str]]:
        """Return the plan's BPM ranges as explicit (start, end) tuples."""
        return self.tracking_plan.bpm_pairs(self._range_context())

    def _range_context(self) -> RangeContext:
        """Return the range-planning inputs for the tracking plan."""
        return RangeContext(
            start_bpms=self.start_bpms,
            end_bpms=self.end_bpms,
            all_bpms=self.all_bpms,
            use_fixed_bpm=self.simulation_config.use_fixed_bpm,
            fixed_start=self.fixed_start,
            fixed_end=self.fixed_end,
        )

    def initialise_knob_strengths(
        self,
        true_strengths: dict[str, float] | None = None,
        provided_initial_knobs: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Initialise knob strengths from MAD and filter true strengths.

        All inputs must be in optimisation space.
        Missing keys in provided_initial_knobs keep the current model defaults.
        """
        if self.mad_iface is None:
            raise ValueError("MAD interface must be setup first")

        knob_name_set = set(self.knob_names)

        self.initial_model_values = {}
        if provided_initial_knobs is not None:
            # Apply every settable initial value to the model, but keep the optimiser
            # state restricted to this stage's knob set.
            known_initial = {k: v for k, v in provided_initial_knobs.items() if k in knob_name_set}
            invalid_initial: list[str] = []
            for name, value in provided_initial_knobs.items():
                if name in knob_name_set:
                    continue
                if name == "pt" or is_magnet_strength_name(name):
                    self.initial_model_values[name] = value
                else:
                    invalid_initial.append(name)
            if invalid_initial:
                invalid_initial.sort()
                raise ValueError(
                    "Unknown optimisation knob names supplied for initialisation: "
                    + ", ".join(invalid_initial[:10])
                    + ("..." if len(invalid_initial) > 10 else "")
                )

            # Warm-start from the current model values and override only the
            # knobs that were explicitly provided.
            LOGGER.info("Using provided initial knob strengths from previous optimisation")
            self.initial_model_values.update(known_initial)
            self.mad_iface.apply_initial_model_values(self.initial_model_values)
        initial_strengths = self.mad_iface.receive_knob_values()

        self.initial_strengths = initial_strengths
        current_knobs = dict(zip(self.knob_names, initial_strengths))

        # Restrict true strengths to knobs we actually have in model
        if true_strengths is None or len(true_strengths) == 0:
            LOGGER.warning("No true strengths provided, skipping filtering")
            filtered_true_strengths = {}
        else:
            unknown_true = sorted(set(true_strengths) - knob_name_set)
            if unknown_true:
                LOGGER.warning(
                    "Ignoring %d true strengths outside the optimisation range: %s%s",
                    len(unknown_true),
                    ", ".join(unknown_true[:10]),
                    "..." if len(unknown_true) > 10 else "",
                )
            filtered_true_strengths = {
                knob: true_strengths[knob] for knob in self.knob_names if knob in true_strengths
            }
        return current_knobs, filtered_true_strengths
