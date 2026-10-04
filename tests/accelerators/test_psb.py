"""Tests for PSB accelerator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aba_optimiser.accelerators import PSB

if TYPE_CHECKING:
    from pathlib import Path


class TestPSBAccelerator:
    """Tests for the PSB accelerator class."""

    @pytest.fixture
    def test_sequence_file(self, data_dir: Path) -> Path:
        """Use the psb sequence in the data directory for testing."""
        return data_dir / "sequences" / "psb3_saved.seq"

    def test_init_basic(self, test_sequence_file: Path) -> None:
        """Test basic PSB initialisation."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)

        assert psb.ring == 3
        assert psb.sequence_file == test_sequence_file
        assert psb.kinetic_energy == pytest.approx(0.160)
        assert psb.energy == pytest.approx(0.160 + 0.9382720813)
        assert psb.bpm_pattern == "^BR3%.BPM%d+L3$"
        assert psb.errors == {}
        assert psb.misalignments == {}
        assert psb.optimise_energy is False
        assert psb.group_quadrupoles_by_cell is False

    def test_group_quadrupoles_by_cell_is_copied(self, test_sequence_file: Path) -> None:
        psb = PSB(
            ring=3,
            sequence_file=test_sequence_file,
            group_quadrupoles_by_cell=True,
        )
        assert psb.copy_with().group_quadrupoles_by_cell is True
        assert psb.copy_with(group_quadrupoles_by_cell=False).group_quadrupoles_by_cell is False

    def test_grouped_results_are_expanded_to_physical_magnets(
        self, test_sequence_file: Path
    ) -> None:
        psb = PSB(
            ring=3,
            sequence_file=test_sequence_file,
            group_quadrupoles_by_cell=True,
        )
        assert psb.format_result_knobs(
            {"BR.QFOCELL1.dk1l": 2e-4, "BR.QDE1.dk1l": -1e-4}
        ) == {
            "BR.QFO11.dk1l": 2e-4,
            "BR.QFO12.dk1l": 2e-4,
            "BR.QDE1.dk1l": -1e-4,
        }

    @pytest.mark.parametrize("ring", [1, 2, 3, 4])
    def test_seq_name_uses_ring_number(self, test_sequence_file: Path, ring: int) -> None:
        """Test sequence name follows the PSB ring convention."""
        psb = PSB(ring=ring, sequence_file=test_sequence_file)
        assert psb.seq_name == f"psb{ring}"

    @pytest.mark.parametrize("ring", [0, 5])
    def test_init_invalid_ring(self, test_sequence_file: Path, ring: int) -> None:
        """Test invalid ring numbers raise ValueError."""
        with pytest.raises(ValueError, match="PSB ring must be 1, 2, 3, or 4"):
            PSB(ring=ring, sequence_file=test_sequence_file)

    def test_init_custom_bpm_pattern(self, test_sequence_file: Path) -> None:
        """Test a custom BPM pattern overrides the ring default."""
        psb = PSB(
            ring=2,
            sequence_file=test_sequence_file,
            bpm_pattern="^CUSTOM%.BPM",
        )
        assert psb.bpm_pattern == "^CUSTOM%.BPM"

    def test_get_knob_specs(self, test_sequence_file: Path) -> None:
        """Test PSB exposes quadrupole knob specs."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            errors={"quad": {"k1"}},
        )

        assert psb.get_knob_specs() == [("quadrupole", "k1", "^BR%.Q[FD][OE]%d+$", "k1", "quad k1")]

    def test_init_with_corrector_errors(self, test_sequence_file: Path) -> None:
        """Test initialization with corrector optimization."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            errors={"corrector": {"kick"}},
        )
        assert psb.optimises("corrector", "kick")

    def test_get_knob_specs_with_correctors(self, test_sequence_file: Path) -> None:
        """Test PSB exposes sequence-name patterns for horizontal and vertical correctors."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            errors={"corrector": {"kick"}},
        )

        specs = psb.get_knob_specs()
        assert ("hkicker", "kick", "^B[RE]%d+%.DHZ%d+L%d+$", None, "corrector kick") in specs
        assert ("vkicker", "kick", "^B[RE]%d+%.DVT%d+L%d+$", None, "corrector kick") in specs

    def test_get_perturbation_families(self, test_sequence_file: Path) -> None:
        """Test PSB perturbation metadata is available for bends and quadrupoles."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert psb.get_perturbation_families() == {
            "d": {
                "default_rel_std": 8e-4,
                "pattern": r"(?i)^BR\.(?:BHZ\d+|BSW\d+L\d+\.\d+)$",
            },
            "q": {
                "default_rel_std": 2e-3,
                "pattern": r"(?i)^BR\.Q(?:FO\d+|DE\d+)$",
            },
        }

    @pytest.mark.parametrize(
        "monitor_name",
        [
            "BR3.BPM1L3",
            "BR3.BPMT3L1",
            "BR3.BWSH4L1",
            "BR3.BPP1L5",
        ],
    )
    def test_infer_monitor_plane(self, monitor_name: str) -> None:
        """Test PSB monitors are treated as dual-plane."""
        assert PSB.infer_monitor_plane(monitor_name) == "HV"

    def test_infer_monitor_plane_invalid(self) -> None:
        """Test unsupported PSB monitor names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported PSB monitor name"):
            PSB.infer_monitor_plane("BR3.QFO11")

    def test_tune_configuration(self, test_sequence_file: Path) -> None:
        """Test PSB tune variable names and integer tunes."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert psb.tune_variables == ("kBRQF", "kBRQD")
        assert psb.tune_integers == (4, 4)

    def test_has_any_optimisation(self, test_sequence_file: Path) -> None:
        """Test generic optimisation flags work for PSB."""
        psb = PSB(
            ring=3,
            sequence_file=test_sequence_file,
            errors={"quad": {"k1"}},
            optimise_energy=True,
            custom_knobs_to_optimise=["BR.QFO11.dk1l"],
        )
        assert psb.has_any_optimisation() is True

    def test_has_any_optimisation_correctors(self, test_sequence_file: Path) -> None:
        """Test corrector optimisation contributes to PSB optimisation state."""
        psb = PSB(
            ring=3,
            sequence_file=test_sequence_file,
            errors={"corrector": {"kick"}},
        )
        assert psb.has_any_optimisation() is True

    def test_format_result_knob_names_maps_indexed_sextupoles(self, test_sequence_file: Path) -> None:
        """Test PSB rewrites indexed sextupole knob names to public dk forms."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert psb.format_result_knob_names(["br3.xnoh0.4l1.knl[3]"]) == ["br3.xnoh0.4l1.dk2l"]
        assert psb.format_result_knob_names(["br3.osk4l1.ksl[3]"]) == ["br3.osk4l1.dk2sl"]

    def test_init_quad_k0s_k1s_default_off(self, test_sequence_file: Path) -> None:
        """Test the skew-multipole quadrupole errors are not fitted by default."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert not psb.optimises("quad", "k0s")
        assert not psb.optimises("quad", "k1s")

    def test_get_knob_specs_quad_k0s(self, test_sequence_file: Path) -> None:
        """Test PSB exposes a quadrupole skew dipole error knob spec when enabled."""
        psb = PSB(ring=1, sequence_file=test_sequence_file, errors={"quad": {"k0s"}})
        assert psb.get_knob_specs() == [
            ("quadrupole", "k0s", "^BR%.Q[FD][OE]%d+$", "k1", "quad k0s")
        ]

    def test_get_knob_specs_quad_k1s(self, test_sequence_file: Path) -> None:
        """Test PSB exposes a quadrupole skew gradient error knob spec when enabled."""
        psb = PSB(ring=1, sequence_file=test_sequence_file, errors={"quad": {"k1s"}})
        assert psb.get_knob_specs() == [
            ("quadrupole", "k1s", "^BR%.Q[FD][OE]%d+$", "k1", "quad k1s")
        ]

    def test_get_knob_specs_empty_by_default(self, test_sequence_file: Path) -> None:
        """Test no knob specs are produced when nothing is selected."""
        psb = PSB(ring=1, sequence_file=test_sequence_file)
        assert psb.get_knob_specs() == []

    def test_copy_with_overrides_errors(self, test_sequence_file: Path) -> None:
        """Test copy_with replaces the error selection wholesale."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, errors={"quad": {"k0s"}})
        copy = psb.copy_with(errors={"quad": {"k1s"}})
        assert copy.errors == {"quad": frozenset({"k1s"})}
        assert psb.errors == {"quad": frozenset({"k0s"})}

    def test_copy_with_preserves_selection(self, test_sequence_file: Path) -> None:
        """Test copy_with preserves errors/misalignments when not overridden."""
        psb = PSB(
            ring=3,
            sequence_file=test_sequence_file,
            errors={"quad": {"k0s", "k1s"}},
            misalignments={"quad": {"ds"}},
        )
        copy = psb.copy_with(optimise_energy=True)
        assert copy.errors == {"quad": frozenset({"k0s", "k1s"})}
        assert copy.misalignments == {"quad": frozenset({"ds"})}
        assert copy.optimise_energy is True
