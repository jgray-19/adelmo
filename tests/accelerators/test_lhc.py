"""Tests for LHC accelerator implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from pymadng_utils.physics import PROTON_MASS_GEV

from aba_optimiser.accelerators import LHC

if TYPE_CHECKING:
    from pathlib import Path

class TestLHCAccelerator:
    """Tests for LHC accelerator class."""

    @pytest.fixture
    def test_sequence_file(self, tmp_path: Path) -> Path:
        """Create a dummy sequence file for testing."""
        seq_file = tmp_path / "test.seq"
        seq_file.write_text("! Dummy sequence file\n")
        return seq_file

    def test_init_basic(self, test_sequence_file: Path) -> None:
        """Test basic LHC initialization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.beam == 1
        assert lhc.kinetic_energy == pytest.approx(6800.0)
        assert lhc.energy == pytest.approx(6800.0 + PROTON_MASS_GEV)
        assert lhc.errors == {}
        assert lhc.misalignments == {}
        assert lhc.normalise_bends is False
        assert lhc.optimise_energy is False

    def test_init_beam_2(self, test_sequence_file: Path) -> None:
        """Test LHC initialization with beam 2."""
        lhc = LHC(
            beam=2,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.beam == 2
        assert lhc.seq_name == "lhcb2"

    def test_init_invalid_beam(self, test_sequence_file: Path) -> None:
        """Test that invalid beam numbers raise ValueError."""
        with pytest.raises(ValueError, match="LHC beam must be 1 or 2"):
            LHC(
                beam=3,
                kinetic_energy=6800.0,
                sequence_file=str(test_sequence_file),
            )

    def test_init_beam_zero(self, test_sequence_file: Path) -> None:
        """Test that beam 0 raises ValueError."""
        with pytest.raises(ValueError, match="LHC beam must be 1 or 2"):
            LHC(
                beam=0,
                kinetic_energy=6800.0,
                sequence_file=str(test_sequence_file),
            )

    def test_init_with_optimise_energy(self, test_sequence_file: Path) -> None:
        """Test initialization with energy optimization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
        )
        assert lhc.optimise_energy is True

    def test_init_with_bend_errors(self, test_sequence_file: Path) -> None:
        """Test initialization with bend optimization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
        )
        assert lhc.optimises("bend", "k0")
        assert lhc.normalise_bends is True  # Should default to True when fitting bend k0

    def test_init_with_bend_errors_no_normalise(self, test_sequence_file: Path) -> None:
        """Test initialization with bend optimization but no normalization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
            normalise_bends=False,
        )
        assert lhc.optimises("bend", "k0")
        assert lhc.normalise_bends is False

    def test_init_normalise_bends_none_without_bends(self, test_sequence_file: Path) -> None:
        """Test normalise_bends defaults to False when bend k0 is not fitted."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            normalise_bends=None,
        )
        assert lhc.normalise_bends is False

    def test_init_with_corrector_errors(self, test_sequence_file: Path) -> None:
        """Test initialization with corrector optimization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"corrector": {"kick"}},
        )
        assert lhc.optimises("corrector", "kick")

    def test_init_with_all_optimisations(self, test_sequence_file: Path) -> None:
        """Test initialization with all optimizations enabled."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
            errors={"bend": {"k0"}, "quad": {"k1"}, "sextupole": {"k2"}, "corrector": {"kick"}},
        )
        assert lhc.optimise_energy is True
        assert lhc.optimises("bend", "k0")
        assert lhc.optimises("quad", "k1")
        assert lhc.optimises("sextupole", "k2")
        assert lhc.optimises("corrector", "kick")

    def test_get_seq_name_beam_1(self, test_sequence_file: Path) -> None:
        """Test seq_name returns correct sequence name for beam 1."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.seq_name == "lhcb1"

    def test_seq_name_beam_2(self, test_sequence_file: Path) -> None:
        """Test seq_name returns correct sequence name for beam 2."""
        lhc = LHC(
            beam=2,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.seq_name == "lhcb2"

    def test_has_any_optimisation_false(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns False with no optimizations."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.has_any_optimisation() is False

    def test_has_any_optimisation_energy(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True with energy optimization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
        )
        assert lhc.has_any_optimisation() is True

    def test_has_any_optimisation_bends(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True with bend optimization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
        )
        assert lhc.has_any_optimisation() is True

    def test_has_any_optimisation_correctors(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True with corrector optimization."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"corrector": {"kick"}},
        )
        assert lhc.has_any_optimisation() is True

    def test_log_optimisation_targets_none(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets with no optimizations."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        with caplog.at_level(logging.INFO):
            lhc.log_optimisation_targets()
        assert "No optimisation targets set" in caplog.text

    def test_log_optimisation_targets_all(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets logs all optimizations."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
            errors={"bend": {"k0"}, "quad": {"k1"}, "sextupole": {"k2"}, "corrector": {"kick"}},
        )
        with caplog.at_level(logging.INFO):
            lhc.log_optimisation_targets()
        assert "bend k0" in caplog.text
        assert "quad k1" in caplog.text
        assert "sextupole k2" in caplog.text
        assert "corrector kick" in caplog.text
        assert "beam energy" in caplog.text

    def test_get_bend_lengths_returns_none_no_bends(self, test_sequence_file: Path) -> None:
        """Test get_bend_lengths returns None when bends not optimized."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        result = lhc.get_bend_lengths()
        assert result is None

    def test_get_bend_lengths_returns_none_no_normalise(self, test_sequence_file: Path) -> None:
        """Test get_bend_lengths returns None when normalisation disabled."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
            normalise_bends=False,
        )
        result = lhc.get_bend_lengths()
        assert result is None

    def test_get_bend_lengths_returns_from_accelerator(self, test_sequence_file: Path) -> None:
        """Test get_bend_lengths returns bend_lengths from accelerator state."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
            normalise_bends=True,
        )
        lhc.bend_lengths = {"MB.A1.dk0l": 2.0, "MB.A2.dk0l": 2.0}
        result = lhc.get_bend_lengths()
        assert result == {"MB.A1.dk0l": 2.0, "MB.A2.dk0l": 2.0}

    def test_get_bend_lengths_handles_missing_state(self, test_sequence_file: Path) -> None:
        """Test get_bend_lengths handles missing accelerator bend_lengths."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
            normalise_bends=True,
        )
        result = lhc.get_bend_lengths()
        assert result is None

    def test_normalise_true_strengths_no_bends(self, test_sequence_file: Path) -> None:
        """Test normalise_true_strengths returns unchanged when bends not optimized."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        test_strengths = {"K1": 0.5}
        result = lhc.normalise_true_strengths(test_strengths, None)
        assert result == test_strengths

    def test_normalise_true_strengths_no_bend_lengths(self, test_sequence_file: Path) -> None:
        """Test normalise_true_strengths returns unchanged when no bend_lengths provided."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
        )
        test_strengths = {"MB.dk0l": 1.0}
        result = lhc.normalise_true_strengths(test_strengths, None)
        assert result == test_strengths

    def test_normalise_true_strengths_with_bend_lengths(
        self, test_sequence_file: Path
    ) -> None:
        """Test normalise_true_strengths calls normalise_lhcbend_magnets."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            errors={"bend": {"k0"}},
            normalise_bends=True,
        )
        # These would be the actual bend strengths from the sequence
        test_strengths = {
            "MB.A1[ABCD]1L1.B1.dk0l": 1.0,
            "MB.A1[ABCD]1L2.B1.dk0l": 1.0,
        }
        bend_lengths = {
            "MB.A1[ABCD]1L1.B1.dk0l": 14.2,
            "MB.A1[ABCD]1L2.B1.dk0l": 14.2,
        }
        result = lhc.normalise_true_strengths(test_strengths, bend_lengths)
        # The result should be a dict (we're not testing the actual normalisation logic here)
        assert isinstance(result, dict)

    def test_format_result_knob_names_with_energy(self, test_sequence_file: Path) -> None:
        """Test format_result_knob_names preserves controller-space pt."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
        )
        knob_names = ["K1.b1", "pt"]
        result = lhc.format_result_knob_names(knob_names)
        assert "pt" in result
        assert "deltap" not in result

    def test_format_result_knob_names_without_energy(self, test_sequence_file: Path) -> None:
        """Test format_result_knob_names leaves knobs unchanged without energy."""
        lhc = LHC(
            beam=1,
            kinetic_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=False,
        )
        knob_names = ["K1.b1", "pt"]
        result = lhc.format_result_knob_names(knob_names)
        assert result == knob_names
