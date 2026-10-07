"""FCC-ee-specific accelerator implementation (LCC optics)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymadng_utils.accelerators.fcc import FCC as BaseFCC  # noqa: N811

from aba_optimiser.machine.accelerators.base import Accelerator, MagnetFamily

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class FCC(BaseFCC, Accelerator):
    """FCC-ee accelerator configuration: one k1 knob per arch/IR quadrupole."""

    # Lua patterns, matched against the upper-case MAD-NG element names.
    # Arc quadrupoles only (QD1A, QF2A, QF3A, QD1AM, QF... .N); the IR/straight-section quads carry an r/l suffix.
    PATTERN_QUAD = "^Q[DF][123]AM?%.%d+$"
    PATTERN_CORRECTOR = "^[HV]COR_"

    FAMILIES = {
        "quad": MagnetFamily(
            {"quadrupole": PATTERN_QUAD},
            errors=frozenset({"k1"}),
            nonzero_attr="k1",
        ),
        "corrector": MagnetFamily(
            {"kicker": PATTERN_CORRECTOR},
            errors=frozenset({"hkick", "vkick"}),
        ),
    }

    def __init__(
        self,
        sequence_file: Path | str,
        kinetic_energy: float | None = None,
        particle: str = "electron",
        bpm_pattern: str = BaseFCC.BPM_PATTERN,
        mode: str = "z",
        errors: Mapping[str, Iterable[str]] | None = None,
        misalignments: Mapping[str, Iterable[str]] | None = None,
        optimise_energy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise the FCC-ee accelerator.

        ``errors`` / ``misalignments`` are keyed by ``FAMILIES``: ``quad``, ``corrector``.
        """
        super().__init__(
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern,
            particle=particle,
            mode=mode,
            errors=errors,
            misalignments=misalignments,
            optimise_energy=optimise_energy,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )

    def copy_with(self, **overrides) -> FCC:
        """Return a new FCC instance with selected parameters overridden."""
        kwargs = {
            "sequence_file": self.sequence_file,
            "kinetic_energy": self.kinetic_energy,
            "particle": self.particle,
            "bpm_pattern": self.bpm_pattern,
            "mode": self.mode,
            **self.selection_kwargs(),
        }
        return FCC(**{**kwargs, **overrides})

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """FCC-ee BPMs measure both planes simultaneously."""
        del bpm_name
        return "HV"
