"""SPS-specific accelerator implementation with generic optimisation targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.accelerators.base import Accelerator, MagnetFamily

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class SPS(Accelerator):
    """Super Proton Synchrotron accelerator configuration.

    This class intentionally exposes only generic optimisation categories,
    without LHC-specific options (bends, correctors, quadrupole displacements).
    """

    # Restrict to main SPS families:
    # quadrupoles QF/QD/QFA/QDA, sextupoles LSF/LSD.
    PATTERN_QUADRUPOLE = "^Q[FD]A?%."
    PATTERN_SEXTUPOLE = "^LS[FD]A?%."
    BPM_PATTERN = "^BP[HV]%."

    FAMILIES = {
        "quad": MagnetFamily(
            {"quadrupole": PATTERN_QUADRUPOLE}, errors=frozenset({"k1"}), nonzero_attr="k1"
        ),
        "sextupole": MagnetFamily(
            {"sextupole": PATTERN_SEXTUPOLE}, errors=frozenset({"k2"}), nonzero_attr="k2"
        ),
    }

    def __init__(
        self,
        sequence_file: Path | str,
        kinetic_energy: float = 450.0,
        bpm_pattern: str = BPM_PATTERN,
        errors: Mapping[str, Iterable[str]] | None = None,
        misalignments: Mapping[str, Iterable[str]] | None = None,
        optimise_energy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        super().__init__(
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern,
            errors=errors,
            misalignments=misalignments,
            optimise_energy=optimise_energy,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )

    def copy_with(self, **overrides) -> SPS:
        """Return a new SPS instance with selected parameters overridden."""
        kwargs = {
            "sequence_file": self.sequence_file,
            "kinetic_energy": self.kinetic_energy,
            "bpm_pattern": self.bpm_pattern,
            **self.selection_kwargs(),
        }
        return SPS(**{**kwargs, **overrides})

    @property
    def seq_name(self) -> str:
        """Return the sequence name for SPS."""
        return "sps"

    @property
    def ac_dipole_name(self) -> str:
        """Return the simulated SPS AC-dipole location.

        omc3 installs the horizontal exciter at ZKHA.21991 and the vertical one at
        ZKV.21993, 1.5 m downstream with no BPM between. The marker machinery
        supports one exciter location, so both planes are driven at ZKHA.21991.
        """
        return "ZKHA.21991"

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return perturbation-family metadata for SPS main families."""
        #https://cds.cern.ch/record/66887/files/LABII-MA-Int-75-2.pdf?version=1
        return {
            "d": {
                "default_rel_std": 2e-5,
                "pattern": self.PATTERN_SEXTUPOLE.replace("%", "\\"),  # Change from lua to regex pattern
            },
            "q": {
                "default_rel_std": 2e-4,
                "pattern": self.PATTERN_QUADRUPOLE.replace("%", "\\"),  # Change from lua to regex pattern
            },
            "s": {
                "default_rel_std": 10e-4,
                "pattern": self.PATTERN_SEXTUPOLE.replace("%", "\\"),  # Change from lua to regex pattern
            },
        }

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from SPS BPM family name."""
        name = bpm_name.upper()
        if name.startswith("BPH"):
            return "H"
        if name.startswith("BPV"):
            return "V"
        raise ValueError(f"Unsupported SPS BPM name for plane inference: {bpm_name}")

    @property
    def tune_variables(self) -> tuple[str, str]:
        """Return SPS tune variable names."""
        return "kqf", "kqd"

    @property
    def tune_integers(self) -> tuple[int, int]:
        """Return SPS integer tunes."""
        return 20, 20
