"""PSB-specific accelerator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymadng_utils.accelerators.psb import PSB as BasePSB  # noqa: N811

from aba_optimiser.machine.accelerators.base import MISALIGNMENT_ATTRS, Accelerator, MagnetFamily
from aba_optimiser.machine.accelerators.magnet_grouping import (
    collapse_psb_grouped_quadrupole_knobs,
    expand_psb_grouped_quadrupole_knobs,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


PSB_FLAT_BOTTOM_KINETIC_ENERGY_GEV = 0.160


class PSB(BasePSB, Accelerator):
    """Proton Synchrotron Booster accelerator configuration."""

    PATTERN_SBENDS = r"^BR%.BHZ%d+$"
    PATTERN_RBENDS = r"^BR%.BSW%d+L%d+%.%d+$"
    PATTERN_QUADRUPOLE = "^BR%.Q[FD][OE]%d+$"
    PATTERN_SEXTUPOLE = r"^BR%d+%.XNO%d+L1$"
    PATTERN_SKEW_SEXTUPOLE = r"^BR%d+%.XSK[26]L4$"
    PATTERN_CORRECTOR_H = r"^B[RE]%d+%.DHZ%d+L%d+$"
    PATTERN_CORRECTOR_V = r"^B[RE]%d+%.DVT%d+L%d+$"
    BEND_PERTURBATION_PATTERN = r"(?i)^BR\.(?:BHZ\d+|BSW\d+L\d+\.\d+)$"
    QUAD_PERTURBATION_PATTERN = r"(?i)^BR\.Q(?:FO\d+|DE\d+)$"
    BPM_PATTERN_TEMPLATE = "^BR{ring}%.BPM%d+L{ring}$"

    FAMILIES = {
        "quad": MagnetFamily(
            {"quadrupole": PATTERN_QUADRUPOLE},
            errors=frozenset({"k1", "k0s", "k1s"}),
            misalignments=MISALIGNMENT_ATTRS,
            nonzero_attr="k1",
        ),
        "bend": MagnetFamily(
            {"sbend": PATTERN_SBENDS, "rbend": PATTERN_RBENDS},
            errors=frozenset({"k0"}),
            misalignments=MISALIGNMENT_ATTRS,
            nonzero_attr="k0",
        ),
        "sextupole": MagnetFamily(
            {"multipole": PATTERN_SEXTUPOLE}, errors=frozenset({"knl[3]"}), nonzero_attr="knl[3]"
        ),
        "skew_sextupole": MagnetFamily(
            {"multipole": PATTERN_SKEW_SEXTUPOLE}, errors=frozenset({"ksl[3]"}), nonzero_attr="ksl[3]"
        ),
        "corrector": MagnetFamily(
            {"hkicker": PATTERN_CORRECTOR_H, "vkicker": PATTERN_CORRECTOR_V},
            errors=frozenset({"kick"}),
        ),
    }

    def __init__(
        self,
        ring: int,
        sequence_file: Path | str,
        kinetic_energy: float = PSB_FLAT_BOTTOM_KINETIC_ENERGY_GEV,
        particle: str = "proton",
        bpm_pattern: str | None = None,
        errors: Mapping[str, Iterable[str]] | None = None,
        misalignments: Mapping[str, Iterable[str]] | None = None,
        optimise_energy: bool = False,
        group_quadrupoles_by_cell: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise PSB accelerator for a specific ring.

        ``errors`` / ``misalignments`` are keyed by ``FAMILIES``: ``quad``,
        ``bend``, ``sextupole``, ``skew_sextupole``, ``corrector``.
        """
        if ring not in (1, 2, 3, 4):
            raise ValueError(f"PSB ring must be 1, 2, 3, or 4, got {ring}")

        super().__init__(
            ring=ring,
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern or self.BPM_PATTERN_TEMPLATE.format(ring=ring),
            particle=particle,
            errors=errors,
            misalignments=misalignments,
            optimise_energy=optimise_energy,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )
        self.group_quadrupoles_by_cell = bool(group_quadrupoles_by_cell)

    def copy_with(self, **overrides) -> PSB:
        """Return a new PSB instance with selected parameters overridden."""
        kwargs = {
            "ring": self.ring,
            "sequence_file": self.sequence_file,
            "kinetic_energy": self.kinetic_energy,
            "particle": self.particle,
            "bpm_pattern": self.bpm_pattern,
            "group_quadrupoles_by_cell": self.group_quadrupoles_by_cell,
            **self.selection_kwargs(),
        }
        return PSB(**{**kwargs, **overrides})

    def get_mad_attr_spec(self, kind: str, attribute: str) -> dict[str, str]:
        """Share the two QFO knobs in each cell when native grouping is enabled."""
        suffixes = {"k1": "dk1l", "k0s": "dk0sl", "k1s": "dk1sl", "dx": "dx", "dy": "dy", "ds": "ds", "tilt": "tilt"}
        if not self.group_quadrupoles_by_cell or kind != "quadrupole":
            return {}
        suffix = suffixes.get(attribute)
        if suffix is None:
            return {}
        return {
            "name_expr": (
                'string.gsub(e.name, "^(BR%.QFO)(%d+)(%d)$", '
                f'"%1CELL%2") .. ".{suffix}"'
            )
        }

    def format_result_knobs(self, knobs: dict[str, float]) -> dict[str, float]:
        """Return physical PSB knob values, expanding cell-grouped QFO knobs."""
        formatted = super().format_result_knobs(knobs)
        if not self.group_quadrupoles_by_cell:
            return formatted
        return expand_psb_grouped_quadrupole_knobs(formatted)

    def normalise_initial_knobs(self, knobs: dict[str, float]) -> dict[str, float]:
        """Collapse physical PSB QFO pairs when native grouping is enabled."""
        if not self.group_quadrupoles_by_cell:
            return super().normalise_initial_knobs(knobs)
        return collapse_psb_grouped_quadrupole_knobs(knobs)

    @property
    def seq_name(self) -> str:
        """Return the sequence name for the selected PSB ring."""
        return f"psb{self.ring}"

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return perturbation metadata for PSB ring bends and QFO/QDE quadrupoles."""
        return {
            "d": {
                "default_rel_std": 8e-4,
                "pattern": self.BEND_PERTURBATION_PATTERN,
            },
            "q": {
                "default_rel_std": 2e-3,
                "pattern": self.QUAD_PERTURBATION_PATTERN,
            },
        }

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from PSB monitor names, including ACD markers."""
        name = bpm_name.upper()
        if any(token in name for token in (".BPM", ".BWS", ".BPP", ".BPT")):
            return "HV"
        if name.endswith("_AFTER") or name.endswith("_BEFORE"):
            return "HV"
        raise ValueError(f"Unsupported PSB monitor name for plane inference: {bpm_name}")
