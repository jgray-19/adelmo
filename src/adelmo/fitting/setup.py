"""The model a fit runs on: MAD-NG interface, BPM ranges and starting knobs."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pymadng_utils.io.utils import read_knobs
from tensorboardX import SummaryWriter

from adelmo.fitting.config import OutputConfig
from adelmo.machine.mad import GradientDescentMadInterface
from adelmo.machine.mad.optimising_mad_interface import is_magnet_strength_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from adelmo.config import SimulationConfig
    from adelmo.fitting.config import SequenceConfig
    from adelmo.machine.accelerators import Accelerator

LOGGER = logging.getLogger(__name__)


def normalise_true_strengths(
    true_strengths: Path | dict[str, float] | None,
) -> dict[str, float]:
    """Normalise true strengths to a dictionary format.

    Args:
        true_strengths: Can be None, a Path to a file, or a dict

    Returns:
        Dictionary of true strengths (empty if None was provided)
    """
    if true_strengths is None:
        return {}
    if isinstance(true_strengths, Path):
        return read_knobs(true_strengths)
    if isinstance(true_strengths, dict):
        return true_strengths.copy()
    raise TypeError(f"Unexpected type for true_strengths: {type(true_strengths)}")


class MachineSetup:
    """Everything a fitter knows about the machine before any worker starts.

    Builds the fitter-side MAD-NG model, reads its BPMs (``all_bpms`` and the
    observed ``bpms_in_range``), and resolves the starting knobs. Every value is in
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
        output_config: OutputConfig | None = None,
        cycle_marker: str | None = None,
        anchor_mode: str | None = None,
        anchor_markers: Sequence[str] = (),
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
        self.debug = debug

        self.output_config.log_state()
        simulation_config.log_state()
        sequence_config.log_state()

        self.mad_iface = GradientDescentMadInterface(
            accelerator=accelerator,
            magnet_range=sequence_config.magnet_range,
            machine_state=machine_state,
            b2_errors=b2_errors,
            bad_bpms=sequence_config.bad_bpms,
            debug=debug,
            mad_logfile=self.output_config.mad_logfile,
            tracking_anchor_mode=anchor_mode,
            tracking_anchor_markers=list(anchor_markers),
        )
        self.knob_names: list[str] = self.mad_iface.knob_names
        self._resolve_bpms(cycle_marker)

        true = self._to_pt(normalise_true_strengths(true_strengths))
        initial = self._to_pt(initial_knob_strengths)
        if initial is not None:
            initial = accelerator.normalise_initial_knobs(initial)
        self.initial_model_values: dict[str, float] = {}
        self.initial_knobs = self._initialise_knobs(initial)
        self.true_strengths = (
            self._restrict_true_strengths(true) if true else self.initial_knobs.copy()
        )

    @property
    def worker_start_knobs(self) -> dict[str, float]:
        """Every value a worker model starts from: the fixed values, then the initial knobs."""
        return {**self.initial_model_values, **self.initial_knobs}

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

    def _resolve_bpms(self, cycle_marker: str | None) -> None:
        """Cycle the model to ``cycle_marker`` if given, then read its BPMs."""
        mad_iface = self.mad_iface
        if cycle_marker is not None:
            mad_iface.cycle_to_start(cycle_marker)
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
