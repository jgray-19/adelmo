from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.measurements.acd_pipeline import (
    ACDOpticsAnalysisConfig,
    long_frame_to_tbt_data,
    run_driven_and_compensated_optics,
)
from aba_optimiser.tracking.data_manager import FileTracks


def test_long_frame_to_tbt_data_preserves_name_and_turn_order() -> None:
    frame = pd.DataFrame(
        {
            "name": ["BPM2", "BPM1", "BPM2", "BPM1"],
            "turn": [1, 1, 0, 0],
            "x": [21.0, 11.0, 20.0, 10.0],
            "y": [-21.0, -11.0, -20.0, -10.0],
        }
    )
    result = long_frame_to_tbt_data(frame, source_file=Path("input.sdds"))

    assert result.nturns == 2
    assert result.meta["file"] == "input.sdds"
    assert result.matrices[0].X.index.tolist() == ["BPM2", "BPM1"]
    np.testing.assert_array_equal(result.matrices[0].X, [[20.0, 21.0], [10.0, 11.0]])


def test_run_driven_optics_rejects_harpy_cleaning(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "name": ["BPM1", "BPM1", "BPM2", "BPM2"],
            "turn": [0, 1, 0, 1],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
        }
    )

    with pytest.raises(ValueError, match="Harpy/OMC3 cleaning is disabled"):
        run_driven_and_compensated_optics(
            [(Path("input.sdds"), frame)],
            output_dir=tmp_path,
            config=ACDOpticsAnalysisConfig(
                model_dir=tmp_path / "model",
                harpy_options={"clean": True},
                optics_options={},
            ),
        )


def _acd_tracks() -> FileTracks:
    """One recorded turn (global turn 4) holding a single AC-dipole marker row."""
    return FileTracks(
        turns=np.array([4]),
        markers=["acd_before"],
        values={
            "x": np.array([[1.0]]),
            "px": np.array([[2.0]]),
            "y": np.array([[3.0]]),
            "py": np.array([[4.0]]),
            "var_x": np.array([[0.5]]),
            "var_px": np.array([[5.0]]),
            "var_y": np.array([[0.5]]),
            "var_py": np.array([[6.0]]),
        },
        kick_plane="xy",
    )


def _reconstruction() -> pd.DataFrame:
    """A reconstruction in file-local turn numbering, naming markers in upper case."""
    return pd.DataFrame(
        {
            "turn": [0],
            "name": ["ACD_BEFORE"],
            "px": [20.0],
            "py": [40.0],
            "var_px": [50.0],
            "var_py": [60.0],
        }
    )


def test_updated_momenta_match_markers_case_insensitively_and_keep_positions() -> None:
    updated = _acd_tracks().with_updated_momenta(_reconstruction())

    assert updated.values["px"][0, 0] == 20.0
    assert updated.values["py"][0, 0] == 40.0
    assert updated.values["var_px"][0, 0] == 50.0
    # An ACD refresh moves momenta only; the marker positions are unchanged.
    assert updated.values["x"][0, 0] == 1.0
    assert updated.values["y"][0, 0] == 3.0

