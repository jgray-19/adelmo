"""
Tests for AbaMadInterface.

This module contains pytest tests for the AbaMadInterface class.
"""

from __future__ import annotations

import numpy as np
import pytest
import tfs

from adelmo.machine.accelerators import LHC
from adelmo.machine.mad.aba_mad_interface import AbaMadInterface
from tests.mad.helpers import (
    check_interface_basic_init,
    cleanup_interface,
)

ORBIT_CORRECTION_BPMS = [f"BPM.{i}R2.B1" for i in range(9, 13)]
ORBIT_CORRECTION_CORRECTORS = ["MCBCH.5R2.B1", "MCBCH.6R2.B1"]


@pytest.mark.parametrize(
    "py_name, expected_py_name, var_name, var_value",
    [
        (None, "py", "a", 2),
        ("test_py", "test_py", "b", 3),
    ],
    ids=["default_py_name", "custom_py_name"],
)
def test_init(py_name, expected_py_name, var_name, var_value, seq_b1) -> None:
    """Test initialization of AbaMadInterface."""
    accel = LHC(beam=1, sequence_file=seq_b1)
    interface = (
        AbaMadInterface(accelerator=accel)
        if py_name is None
        else AbaMadInterface(accelerator=accel, py_name=py_name)
    )
    check_interface_basic_init(interface, expected_py_name)
    interface.mad.send(f"{var_name} = {var_value}")
    assert getattr(interface.mad, var_name) == var_value
    cleanup_interface(interface)


def test_cycle_sequence(loaded_interface: AbaMadInterface) -> None:
    """Test cycling sequence to a marker."""
    loaded_interface.mad.send("""py:send(loaded_sequence:raw_get("__cycle"))""")
    assert loaded_interface.mad.recv() is None

    loaded_interface.cycle_sequence("IP5")
    loaded_interface.mad.send("""py:send(loaded_sequence:raw_get("__cycle"))""")
    cycle_marker = loaded_interface.mad.recv()
    assert cycle_marker == "IP5"

    loaded_interface.cycle_sequence()
    loaded_interface.mad.send("""py:send(loaded_sequence:raw_get("__cycle"))""")
    cycle_marker = loaded_interface.mad.recv()
    assert cycle_marker is None

def test_getset_variables(loaded_interface: AbaMadInterface) -> None:
    """Test setting MAD variables."""
    loaded_interface.set_variables(**{"KQTL_1L1_B1": 1.2, "KQTL_1L2_B1": 2.3})
    assert loaded_interface.mad.KQTL_1L1_B1 == 1.2
    assert loaded_interface.mad.KQTL_1L2_B1 == 2.3

    v1, v2 = loaded_interface.get_variables("KQTL_1L1_B1", "KQTL_1L2_B1")
    assert v1 == 1.2
    assert v2 == 2.3


def test_set_madx_variables(loaded_interface: AbaMadInterface) -> None:
    """Test setting MAD-X variables."""
    loaded_interface.set_madx_variables(**{"kqtl_1l1_b1": 1.5, "KQTL_1L2_B1": 2.5})
    assert loaded_interface.mad.MADX.KQTL_1L1_B1 == 1.5
    assert loaded_interface.mad.MADX.kqtl_1l2_b1 == 2.5


def test_twiss(loaded_interface: AbaMadInterface):
    """Test twiss function."""
    interface = loaded_interface
    twiss_df = interface.run_twiss()

    # Assert the columns we expect are present
    expected_columns = [
        "s",
        "beta11",
        "beta22",
        "alfa11",
        "alfa22",
        "mu1",
        "mu2",
        "dx",
        "dy",
    ]
    for col in expected_columns:
        assert col in twiss_df.columns, f"Expected column {col} not found in twiss output"
    assert "name" not in twiss_df.columns, "Column 'name' should be the index"
    assert twiss_df.index.name == "name", (
        f"Expected index name to be 'name', got {twiss_df.index.name}"
    )

    # There should only be one entry, since by default observe = 1 and nothing has been set to be observed
    assert len(twiss_df) == 1, f"Expected 1 twiss entry, got {len(twiss_df)}"
    # Check the marker is named $end
    assert twiss_df.index[0] == "$end", f"Expected marker name '$end', got {twiss_df.index[0]}"

    # Check the tunes
    assert abs(twiss_df.headers["q1"] - 62.28) < 3e-7, f"Unexpected Qx: {twiss_df.q1}"
    assert abs(twiss_df.headers["q2"] - 60.31) < 3e-7, f"Unexpected Qy: {twiss_df.q2}"

    # Now set observe to 0
    twiss_df = interface.run_twiss(observe=0)
    # There should be loads of entries now, including drifts
    assert len(twiss_df) > 1000, f"Expected >1000 twiss entries, got {len(twiss_df)}"
    assert "drift__3" in twiss_df.index, "Expected to find drift elements in twiss output"
    assert "MB.A33R2.B1" in twiss_df.index, "Expected to find magnet elements in twiss output"
    assert abs(twiss_df.headers["q1"] - 62.28) < 3e-7, f"Unexpected Qx: {twiss_df.headers['q1']}"
    assert abs(twiss_df.headers["q2"] - 60.31) < 3e-7, f"Unexpected Qy: {twiss_df.headers['q2']}"

    # Now observe BPMs
    interface.observe("BPM")
    twiss_df = interface.run_twiss()
    # There should only be BPMs observed
    assert len(twiss_df) == 563, f"Expected 563 twiss entries, got {len(twiss_df)}"
    assert all(twiss_df.index.str.match(r"^BPM.*")), "Expected only BPM elements in twiss output"
    assert abs(twiss_df.headers["q1"] - 62.28) < 3e-7, f"Unexpected Qx: {twiss_df.headers['q1']}"
    assert abs(twiss_df.headers["q2"] - 60.31) < 3e-7, f"Unexpected Qy: {twiss_df.headers['q2']}"
    interface.unobserve_elements(["BPM"])


@pytest.mark.parametrize(
    "target_qx,target_qy",
    [
        (0.2801, 0.3101),
        (0.29, 0.32),
        (0.27, 0.30),
    ],
)
def test_match_tunes(
    loaded_interface: AbaMadInterface,
    target_qx: float,
    target_qy: float,
):
    """Test matching tunes."""
    interface = loaded_interface
    tune_qx, tune_qy = interface.accelerator.tune_variables
    initial_knobs = {
        tune_qx: interface.mad.MADX[tune_qx],
        tune_qy: interface.mad.MADX[tune_qy],
    }
    print("Initial knobs:", initial_knobs)

    new_knobs = interface.match_tunes(
        target_qx=target_qx,
        target_qy=target_qy,
    )

    twiss_df = interface.run_twiss()
    assert abs((twiss_df.headers["q1"] % 1) - target_qx) < 1e-5, (
        f"Qx not matched: {twiss_df.headers['q1'] % 1} != {target_qx}"
    )
    assert abs((twiss_df.headers["q2"] % 1) - target_qy) < 1e-5, (
        f"Qy not matched: {twiss_df.headers['q2'] % 1} != {target_qy}"
    )

    # Check that knobs have been changed
    for knob, old_value in initial_knobs.items():
        new_value = interface.mad.MADX[knob]
        assert new_value != old_value, f"Knob {knob} value did not change"
        assert new_value == new_knobs[knob], (
            f"Returned knob value for {knob} does not match MAD value"
        )


class TestDp2pt:
    def test_zero_dp(self, loaded_interface: AbaMadInterface):
        """Test dp2pt with dp=0 returns 0"""
        interface = loaded_interface
        pt = interface.dp2pt(0.0)
        assert np.isclose(pt, 0.0, rtol=1e-12, atol=1e-15)

    def test_positive_dp(self, loaded_interface: AbaMadInterface):
        """Test dp2pt with positive dp"""
        interface = loaded_interface
        dp = 0.01
        pt = interface.dp2pt(dp)
        # Compare with physics calculation
        expected_pt = interface.accelerator.dp2pt(dp)
        assert np.isclose(pt, expected_pt, rtol=1e-12, atol=1e-13)

    def test_negative_dp(self, loaded_interface: AbaMadInterface):
        """Test dp2pt with negative dp"""
        interface = loaded_interface
        dp = -0.005
        pt = interface.dp2pt(dp)
        # Compare with physics calculation
        expected_pt = interface.accelerator.dp2pt(dp)
        assert np.isclose(pt, expected_pt, rtol=1e-12, atol=1e-13)

    def test_high_energy_approximation(self, loaded_interface: AbaMadInterface):
        """Test at high energy where pt ≈ dp"""
        interface = loaded_interface
        dp = 0.01
        pt = interface.dp2pt(dp)
        # At high energy, pt should be close to dp
        assert np.isclose(pt, dp, rtol=1e-3, atol=1e-6)


class TestPt2dp:
    def test_zero_pt(self, loaded_interface: AbaMadInterface):
        """Test pt2dp with pt=0 returns 0"""
        interface = loaded_interface
        dp = interface.pt2dp(0.0)
        assert np.isclose(dp, 0.0, rtol=1e-12, atol=1e-15)

    def test_positive_pt(self, loaded_interface: AbaMadInterface):
        """Test pt2dp with positive pt"""
        interface = loaded_interface
        pt = 0.01
        dp = interface.pt2dp(pt)
        # Check inverse consistency
        pt_back = interface.dp2pt(dp)
        assert np.isclose(pt_back, pt, rtol=1e-12, atol=1e-15)

    def test_negative_pt(self, loaded_interface: AbaMadInterface):
        """Test pt2dp with negative pt"""
        interface = loaded_interface
        pt = -0.005
        dp = interface.pt2dp(pt)
        # Check inverse consistency
        pt_back = interface.dp2pt(dp)
        assert np.isclose(pt_back, pt, rtol=1e-12, atol=1e-15)


def test_pt2dp_dp2pt_consistency(loaded_interface: AbaMadInterface):
    """Test that pt2dp and dp2pt are inverses of each other."""
    interface = loaded_interface
    test_dps = [0.0, 0.001, -0.001, 0.01, -0.005]
    test_pts = [0.0, 0.001, -0.001, 0.01, -0.005]

    for dp in test_dps:
        pt = interface.dp2pt(dp)
        dp_back = interface.pt2dp(pt)
        assert np.isclose(dp_back, dp, rtol=1e-12, atol=1e-15), (
            f"Inconsistency for dp={dp}: pt={pt}, dp_back={dp_back}"
        )

    for pt in test_pts:
        dp = interface.pt2dp(pt)
        pt_back = interface.dp2pt(dp)
        assert np.isclose(pt_back, pt, rtol=1e-12, atol=1e-15), (
            f"Inconsistency for pt={pt}: dp={dp}, pt_back={pt_back}"
        )


class TestPerformOrbitCorrection:
    def test_invalid_plane_raises(self, loaded_interface: AbaMadInterface) -> None:
        with pytest.raises(ValueError, match="Invalid orbit correction plane"):
            loaded_interface.perform_orbit_correction(machine_deltap=0.0, plane="z")

    def test_correct_tunes_requires_targets(self, loaded_interface: AbaMadInterface) -> None:
        with pytest.raises(ValueError, match="target tunes"):
            loaded_interface.perform_orbit_correction(machine_deltap=0.0, correct_tunes=True)

    def test_filters_bpms_and_correctors(
        self, loaded_interface: AbaMadInterface, tmp_path
    ) -> None:
        interface = loaded_interface
        corrector_file = tmp_path / "correctors.tfs"

        result = interface.perform_orbit_correction(
            machine_deltap=0.0,
            corrector_file=corrector_file,
            twiss_name="nil",
            correct_tunes=False,
            bpms=ORBIT_CORRECTION_BPMS,
            correctors=ORBIT_CORRECTION_CORRECTORS,
            plane="x",
        )

        qx_knob, qy_knob = interface.accelerator.tune_variables
        assert set(result) == {qx_knob, qy_knob}

        corrector_table = tfs.read(corrector_file)
        corrector_names = corrector_table["name"].astype(str).str.upper().tolist()
        kickers = corrector_table.loc[
            corrector_table["kind"].astype(str).str.contains("kicker"), "name"
        ]
        assert set(kickers.astype(str).str.upper()) <= {
            n.upper() for n in ORBIT_CORRECTION_CORRECTORS
        }
        assert corrector_names  # the correction actually produced output

        # The BPM/corrector filter must not leak: elements outside the
        # supplied names should still be selectable/observable afterwards.
        interface.observe("BPM")
        twiss_df = interface.run_twiss()
        assert len(twiss_df) > len(ORBIT_CORRECTION_BPMS)
        interface.unobserve_elements(["BPM"])


@pytest.mark.parametrize("end_num", [10, 11, 12])
def test_get_bpm_list(loaded_interface: AbaMadInterface, end_num: int) -> None:
    """Test getting list of BPM names within a range."""
    interface = loaded_interface

    # Set up BPM observation
    interface.observe("BPM")

    # Test getting BPM list for a range
    start_num = 9
    start_bpm = f"BPM.{start_num}R2.B1"
    end_bpm = f"BPM.{end_num}R2.B1"
    bpm_range = f"{start_bpm}/{end_bpm}"
    expected_bpms = [f"BPM.{i}R2.B1" for i in range(start_num, end_num + 1)]
    all_bpms, bpm_names = interface.get_bpm_list(bpm_range)

    # Verify all_bpms is a list of strings
    assert isinstance(all_bpms, list)
    assert all(isinstance(name, str) for name in all_bpms)

    # Verify bpm_names (in range) is a list of strings
    assert isinstance(bpm_names, list)
    assert all(isinstance(name, str) for name in bpm_names)

    # Check that the list has expected BPMs
    assert bpm_names == expected_bpms, f"Expected BPMs {expected_bpms}, got {bpm_names}"
    interface.unobserve_elements(["BPM"])
