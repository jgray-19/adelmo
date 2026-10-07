"""The model a fit runs on: MAD-NG interface, BPM ranges and starting knobs."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

from tensorboardX import SummaryWriter

from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.mad.optimising_mad_interface import is_magnet_strength_name
from aba_optimiser.training.config.models import OutputConfig
from aba_optimiser.training.config.tracking import RangeContext, TrackingPlan
from aba_optimiser.training.utils import filter_bad_bpms, normalise_true_strengths

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.config.models import SequenceConfig

LOGGER = logging.getLogger(__name__)


class MachineSetup:
    """Everything a fitter knows about the machine before any worker starts.

    Builds the fitter-side MAD-NG model, filters the BPM start/end points to the
    observed range, and resolves the starting knobs. Every value is in
    optimisation space: a ``deltap`` in the user's knobs becomes ``pt``.

    Attributes set here and read by the fitters:

    ``knob_names``
        The optimised knobs, in model order.
    ``initial_knobs``
        The starting value of every knob in ``knob_names``.
    ``initial_model_values``
        Fixed values (strengths and ``pt``) from ``initial_knob_strengths`` that are
        not optimised here but still have to be set in every model.
    ``true_strengths``
        The known answer, restricted to ``knob_names``; the initial knobs when none
        is given.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        *,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        output_config: OutputConfig | None = None,
        tracking_plan: TrackingPlan | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        debug: bool = False,
        machine_state: Path | Mapping[str, float] | None = None,
        b2_errors: Path | None = None,
    ):
        if not accelerator.has_any_optimisation():
            raise ValueError("No optimisation types enabled in the accelerator configuration.")
        self.accelerator = accelerator
        self.simulation_config = simulation_config
        self.sequence_config = sequence_config
        self.output_config = output_config if output_config is not None else OutputConfig()
        self.tracking_plan = tracking_plan if tracking_plan is not None else TrackingPlan()
        self.debug = debug

        self.output_config.log_state()
        simulation_config.log_state()
        sequence_config.log_state()

        self.start_bpms, self.end_bpms = filter_bad_bpms(
            bpm_start_points, bpm_end_points, sequence_config.bad_bpms
        )
        LOGGER.info(
            "After filtering bad BPMs, %s: %s; end points: %s",
            self.tracking_plan.start_point_label,
            self.start_bpms,
            self.end_bpms,
        )
        self.mad_iface = GradientDescentMadInterface(
            accelerator=accelerator,
            magnet_range=sequence_config.magnet_range,
            machine_state=machine_state,
            b2_errors=b2_errors,
            bad_bpms=sequence_config.bad_bpms,
            debug=debug,
            mad_logfile=self.output_config.mad_logfile,
            tracking_anchor_mode=self.tracking_plan.tracking_anchor_mode,
            tracking_anchor_markers=list(self.tracking_plan.tracking_anchor_sources),
        )
        self.knob_names: list[str] = self.mad_iface.knob_names
        self._resolve_bpms()

        true = self._to_pt(normalise_true_strengths(true_strengths))
        initial = self._to_pt(initial_knob_strengths)
        if initial is not None:
            initial = accelerator.normalise_initial_knobs(initial)
        self.initial_model_values: dict[str, float] = {}
        self.initial_knobs = self._initialise_knobs(initial)
        self.true_strengths = (
            self._restrict_true_strengths(true) if true else self.initial_knobs.copy()
        )
        self.output_knob_names = accelerator.format_result_knob_names(self.knob_names)

    @property
    def worker_start_knobs(self) -> dict[str, float]:
        """Every value a worker model starts from: the fixed values, then the initial knobs."""
        return {**self.initial_model_values, **self.initial_knobs}

    @property
    def bpm_pairs(self) -> list[tuple[str, str]]:
        """The tracking plan's BPM ranges as explicit ``(start, end)`` tuples."""
        return self.tracking_plan.bpm_pairs(
            RangeContext(
                start_bpms=self.start_bpms,
                end_bpms=self.end_bpms,
                all_bpms=self.all_bpms,
                use_fixed_bpm=self.simulation_config.use_fixed_bpm,
                fixed_start=self.fixed_start,
                fixed_end=self.fixed_end,
            )
        )

    def make_writer(self, log_suffix: str) -> SummaryWriter | None:
        """A TensorBoard writer under ``output_config.tensorboard_root``, or ``None`` when disabled."""
        if not self.output_config.write_tensorboard_logs:
            LOGGER.info("TensorBoard logging disabled")
            return None
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.output_config.tensorboard_root / f"{timestamp}_{log_suffix}"
        log_dir.parent.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    def close(self) -> None:
        """Shut down the fitter-side MAD process."""
        self.mad_iface.close()

    def _to_pt(self, knobs: dict[str, float] | None) -> dict[str, float] | None:
        """A copy of ``knobs`` with any ``deltap`` replaced by the equivalent ``pt``."""
        if knobs is None:
            return None
        knobs = knobs.copy()
        if "deltap" in knobs:
            knobs["pt"] = self.mad_iface.dp2pt(knobs.pop("deltap"))
        return knobs

    def _resolve_bpms(self) -> None:
        """Cycle the model if the plan asks, then restrict the start/end points to the observed range."""
        plan = self.tracking_plan
        mad_iface = self.mad_iface
        if plan.cycle_marker is not None:
            mad_iface.cycle_to_start(plan.cycle_marker)
            mad_iface.bpms_in_range, mad_iface.nbpms, mad_iface.all_bpms = mad_iface.count_bpms(
                mad_iface.bpm_range
            )
        self.all_bpms: list[str] = mad_iface.all_bpms
        self.bpms_in_range: list[str] = mad_iface.bpms_in_range
        magnet_range = self.sequence_config.magnet_range
        range_label = (
            "full cycled sequence ($start/$end)" if magnet_range == "$start/$end" else magnet_range
        )
        LOGGER.info(
            "Total BPMs in model: %d, BPMs in configured observation range %s: %d",
            len(self.all_bpms),
            range_label,
            len(self.bpms_in_range),
        )

        # Marker-anchored modes (kicker/ACD) may start from installed or measured
        # marker anchors rather than ordinary BPMs, so keep those through the filter.
        allowed_starts = set(plan.tracking_anchor_sources)
        if plan.init_marker is not None:
            allowed_starts.add(plan.init_marker)
        in_range = set(self.bpms_in_range)
        self.start_bpms = [b for b in self.start_bpms if b in in_range or b in allowed_starts]
        self.end_bpms = [b for b in self.end_bpms if b in in_range]

        # With use_fixed_bpm the fixed BPM window comes from magnet_range; otherwise
        # it stays empty and ranges come from start_bpms/end_bpms.
        self.fixed_start = ""
        self.fixed_end = ""
        if self.simulation_config.use_fixed_bpm and plan.uses_fixed_bpm_window:
            self.fixed_start, self.fixed_end = magnet_range.split("/", 1)
            if self.fixed_start not in in_range or self.fixed_end not in in_range:
                LOGGER.warning(
                    "Fixed BPMs from range %s not found in model, using first available",
                    range_label,
                )
                self.fixed_start = self.start_bpms[0] if self.start_bpms else self.fixed_start
                self.fixed_end = self.end_bpms[0] if self.end_bpms else self.fixed_end
        elif self.simulation_config.use_fixed_bpm:
            LOGGER.info(
                "Skipping fixed BPM derivation for this tracking plan; %s: %s",
                plan.start_point_label,
                self.start_bpms,
            )

    def _initialise_knobs(self, provided: dict[str, float] | None) -> dict[str, float]:
        """Apply ``provided`` to the model and return the starting value of every knob.

        Knobs that ``provided`` does not name keep the model's value. Values for
        magnet strengths or ``pt`` outside this fit's knobs go to
        :attr:`initial_model_values`; any other unknown name is an error.
        """
        if not self.knob_names:
            raise ValueError(
                "No optimisation knobs were created for this fitter configuration. "
                f"Optimisation is enabled, but the MAD model returned zero knobs for "
                f"magnet range '{self.sequence_config.magnet_range}'. Check that the "
                "selected optimisation flags match elements present in the loaded "
                "sequence and range."
            )
        knob_name_set = set(self.knob_names)
        if provided is not None:
            invalid: list[str] = []
            for name, value in provided.items():
                if name in knob_name_set:
                    continue
                if name == "pt" or is_magnet_strength_name(name):
                    self.initial_model_values[name] = value
                else:
                    invalid.append(name)
            if invalid:
                invalid.sort()
                raise ValueError(
                    "Unknown optimisation knob names supplied for initialisation: "
                    + ", ".join(invalid[:10])
                    + ("..." if len(invalid) > 10 else "")
                )
            LOGGER.info("Using provided initial knob strengths from previous optimisation")
            self.initial_model_values.update(
                {k: v for k, v in provided.items() if k in knob_name_set}
            )
            self.mad_iface.apply_initial_model_values(self.initial_model_values)
        values = self.mad_iface.receive_knob_values()
        if len(values) != len(self.knob_names):
            raise ValueError(
                "Knob initialisation produced an inconsistent result: "
                f"{len(self.knob_names)} knob names but {len(values)} initial values."
            )
        return dict(zip(self.knob_names, values))

    def _restrict_true_strengths(self, true: dict[str, float]) -> dict[str, float]:
        """``true`` restricted to this fit's knobs, warning about the rest."""
        unknown = sorted(set(true) - set(self.knob_names))
        if unknown:
            LOGGER.warning(
                "Ignoring %d true strengths outside the optimisation range: %s%s",
                len(unknown),
                ", ".join(unknown[:10]),
                "..." if len(unknown) > 10 else "",
            )
        return {knob: true[knob] for knob in self.knob_names if knob in true}
