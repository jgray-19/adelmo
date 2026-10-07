"""LHC-specific accelerator implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymadng_utils.accelerators.lhc import LHC as BaseLHC  # noqa: N811

from aba_optimiser.machine.accelerators.base import Accelerator, KnobSpec, MagnetFamily
from aba_optimiser.machine.accelerators.magnet_grouping import normalise_lhcbend_magnets

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from aba_optimiser.machine.mad.optimising_mad_interface import GradientDescentMadInterface


class LHC(BaseLHC, Accelerator):
    """Large Hadron Collider accelerator configuration.

    This class encapsulates LHC-specific parameters like beam numbers,
    default BPMs, and sequence file locations.
    """

    # LHC magnet patterns as class constants
    PATTERN_MAIN_BEND = "MB%."
    PATTERN_RBEND = "MB[RXWAL]%w*%."
    PATTERN_MAIN_QUAD = "MQ%."
    PATTERN_CORRECTOR = "MCB"
    PATTERN_QUAD_NON_TUNE = "MQ[^.TSY]"  # Explicitly not MQS, MQT or MQ., but still quadrupoles
    # With the MQ. main quads (``quad`` family) this covers every "MQ[^ST]" quadrupole.
    PATTERN_QUAD_DISPLACEMENT_Y_OTHER = "MQ[^.ST]"  # Non-main quads with potential vertical misalignments
    PATTERN_QUAD_DISPLACEMENT_X = (
        "MQ[^.TSYM]"  # Triplet quadrupoles, warm quads with potential horizontal misalignments
    )
    PATTERN_SEXTUPOLE = "MSS?%."
    QUAD_ERROR_TABLE = {
        "MQ.": 18e-4,
        "MQM": 12e-4,
        "MQY": 8e-4,
        "MQX": 10e-4,
        "MQW": 15e-4,
        # "MQT": 75e-4,
    }
    BPM_PATTERN = "^BPM.*$"

    FAMILIES = {
        "bend": MagnetFamily(
            {"sbend": PATTERN_MAIN_BEND, "rbend": PATTERN_RBEND},
            errors=frozenset({"k0"}),
            nonzero_attr="k0",
        ),
        "quad": MagnetFamily(
            {"quadrupole": PATTERN_MAIN_QUAD},
            errors=frozenset({"k1"}),
            misalignments=frozenset({"dy"}),
            nonzero_attr="k1",
        ),
        "other_quad": MagnetFamily(
            {"quadrupole": PATTERN_QUAD_NON_TUNE},
            errors=frozenset({"k1"}),
            misalignments=frozenset({"dx", "dy"}),
            nonzero_attr="k1",
            attr_patterns={"dx": PATTERN_QUAD_DISPLACEMENT_X, "dy": PATTERN_QUAD_DISPLACEMENT_Y_OTHER},
        ),
        "sextupole": MagnetFamily(
            {"sextupole": PATTERN_SEXTUPOLE}, errors=frozenset({"k2"}), nonzero_attr="k2"
        ),
        "corrector": MagnetFamily(
            {"hkicker": PATTERN_CORRECTOR, "vkicker": PATTERN_CORRECTOR},
            errors=frozenset({"kick"}),
        ),
    }

    def __init__(
        self,
        beam: int,
        sequence_file: Path | str,
        kinetic_energy: float = 6800.0,
        particle: str = "proton",
        bpm_pattern: str = BPM_PATTERN,
        errors: Mapping[str, Iterable[str]] | None = None,
        misalignments: Mapping[str, Iterable[str]] | None = None,
        optimise_energy: bool = False,
        normalise_bends: bool | None = None,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise LHC accelerator for a specific beam.

        Args:
            beam: Beam number (1 or 2)
            sequence_file: Path to sequence file
            kinetic_energy: Particle kinetic energy in GeV
            bpm_pattern: Pattern for identifying BPMs in the sequence
            errors: Field errors to fit, keyed by ``FAMILIES`` (``bend``, ``quad``
                (MQ. main quads), ``other_quad``, ``sextupole``, ``corrector``)
            misalignments: Alignment errors to fit (``quad``: dy, ``other_quad``: dx/dy)
            optimise_energy: Whether to optimise beam energy
            normalise_bends: Whether to normalise bend strengths (default: when fitting bend k0)

        Raises:
            ValueError: If an invalid beam number is provided
        """
        if beam not in (1, 2):
            raise ValueError(f"LHC beam must be 1 or 2, got {beam}")

        super().__init__(
            beam=beam,
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern,
            particle=particle,
            errors=errors,
            misalignments=misalignments,
            optimise_energy=optimise_energy,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )
        if normalise_bends is None:
            normalise_bends = self.optimises("bend", "k0")
        self.normalise_bends = normalise_bends
        self.bend_lengths: dict[str, float] | None = None

    def copy_with(self, **overrides) -> LHC:
        """Return a new LHC instance with selected parameters overridden."""
        kwargs = {
            "beam": self.beam,
            "sequence_file": self.sequence_file,
            "kinetic_energy": self.kinetic_energy,
            "particle": self.particle,
            "bpm_pattern": self.bpm_pattern,
            "normalise_bends": self.normalise_bends,
            **self.selection_kwargs(),
        }
        return LHC(**{**kwargs, **overrides})

    def get_bend_lengths(self) -> dict[str, float] | None:
        """Return LHC bend lengths when bend normalisation is enabled."""
        if not (self.optimises("bend", "k0") and self.normalise_bends):
            return None
        return self.bend_lengths

    def normalise_true_strengths(
        self,
        true_strengths: dict[str, float],
        bend_lengths: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Normalise LHC bend strengths when applicable."""
        if bend_lengths is None:
            bend_lengths = self.bend_lengths
        if self.optimises("bend", "k0") and bend_lengths:
            return normalise_lhcbend_magnets(true_strengths, bend_lengths)
        return true_strengths

    def prepare_mad_for_knob_creation(
        self,
        mad_iface: GradientDescentMadInterface,
        specs: list[KnobSpec],
    ) -> None:
        """Prepare LHC-specific MAD state for knob creation."""
        super().prepare_mad_for_knob_creation(mad_iface, specs)
        if self.optimises("bend", "k0") and self.normalise_bends:
            mad_iface.mad.send(f"""
            bend_dict = {{}}
            bend_lengths = {{}}
            for i, e in loaded_sequence:siter(magnet_range) do
                if (e.kind == "sbend" or e.kind == "rbend") and e.k0 ~= 0 then
                    bend_dict[e.name .. ".dk0l"] = e.k0
                    bend_lengths[e.name .. ".dk0l"] = e.l
                end
            end
            {mad_iface.py_name}:send(bend_dict, true)
            {mad_iface.py_name}:send(bend_lengths, true)
            bend_dict = {mad_iface.py_name}:recv()
            """)
            true_strengths_dict: dict[str, float] = mad_iface.mad.recv()
            self.bend_lengths = mad_iface.mad.recv()
            normalised_names = normalise_lhcbend_magnets(true_strengths_dict, self.bend_lengths)
            mad_iface.mad.send(normalised_names)

    def get_mad_attr_spec(self, kind: str, attribute: str) -> dict[str, str]:
        """Return LHC-specific attr naming/value expressions."""
        if not self.normalise_bends or kind != "sbend" or attribute != "k0":
            return {}
        return {
            "name_expr": 'string.gsub(e.name, "(MB%.)([ABCD])([0-9]+[LR][1-8]%.B[12])", "%1%3") .. ".dk0l"',
            "mad_value": "bend_dict[k_str_name]",
        }

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """LHC BPMs measure both planes simultaneously."""
        del bpm_name
        return "HV"

    def get_perturbation_families(self) -> dict[str, dict[str, float | str | dict]]:
        """Return perturbation-family metadata for LHC."""
        return {
            "d": {
                "default_rel_std": 1e-4,
                "pattern": "MB\\.",
            },
            "q": {
                "relative_error_table": self.QUAD_ERROR_TABLE,
            },
            "s": {
                "default_rel_std": 1e-4,
                "pattern": "MS\\.",
            },
        }
