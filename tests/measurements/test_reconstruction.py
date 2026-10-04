"""Shared measurement preparation: momentum reconstruction and variance assignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import tfs
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from tmom_recon import ModelDetails

from aba_optimiser.accelerators import PSB
from aba_optimiser.measurements.reconstruction import (
    _scale_position_variances_after_svd,
    append_acd_marker_rows,
    process_single_dataframe,
)
from aba_optimiser.measurements.reference import reconstruction_frame
from aba_optimiser.measurements.variances import (
    assign_known_noise_variances,
    assign_uniform_variances,
)

MODEL_DIR = Path(__file__).parents[1] / "data" / "model_creator"


def _phase_space(names: list[str], x: float = 0.0, **extra: list[float]) -> pd.DataFrame:
    n = len(names)
    return pd.DataFrame(
        {"name": names, "turn": [1] * n, "x": [x] * n, "px": [0.0] * n, "y": [0.0] * n, "py": [0.0] * n, **extra}
    )


def test_append_acd_marker_rows_keeps_bpms_and_adds_only_markers() -> None:
    frame = _phase_space(["BR3.BPM1L3"], x=0.001)
    acd_result = _phase_space(["acd_before", "acd_after", "BR3.BPM1L3"], x=0.1)

    result = append_acd_marker_rows(frame, acd_result)

    assert list(result["name"]) == ["BR3.BPM1L3", "acd_before", "acd_after"]
    assert result.loc[1, "x"] == pytest.approx(0.1)


def test_append_acd_marker_rows_matches_suffix_case_insensitively() -> None:
    acd_result = _phase_space(["MKQA.6L4.B1_BEFORE", "BE3.KFA14L1_after"])

    result = append_acd_marker_rows(_phase_space(["BPM1"]), acd_result)

    assert list(result["name"]) == ["BPM1", "MKQA.6L4.B1_BEFORE", "BE3.KFA14L1_after"]


def test_append_acd_marker_rows_without_markers_returns_frame() -> None:
    frame = _phase_space(["BPM1"])

    assert append_acd_marker_rows(frame, _phase_space(["BPM2"])) is frame


def test_append_acd_marker_rows_reindexes_to_frame_columns() -> None:
    frame = _phase_space(["BPM1"])
    acd_result = _phase_space(["acd_after"], extra_only_in_acd_result=[999.0])

    result = append_acd_marker_rows(frame, acd_result)

    assert list(result.columns) == list(frame.columns)


def _fake_analysis_dir(output_dir: Path, twiss_path: Path) -> Path:
    """Write a minimal optics-measurement folder generated from a model twiss."""
    twiss = tfs.read(twiss_path, index="NAME")
    twiss.columns = [column.upper() for column in twiss.columns]
    twiss = twiss.rename(columns={"MU1": "MUX", "MU2": "MUY"})
    twiss.headers = {str(key).upper(): value for key, value in twiss.headers.items()}
    fake_measurement(
        twiss=twiss, outputdir=output_dir, parameters=["BETX", "BETY", "PHASEX", "PHASEY", "X", "Y"]
    )
    return output_dir


def test_process_single_dataframe_reconstructs_psb_momenta(tmp_path: Path) -> None:
    analysis_dir = _fake_analysis_dir(tmp_path / "analysis", MODEL_DIR / "psb3_twiss.dat")
    twiss = tfs.read(MODEL_DIR / "psb3_twiss_ac.dat", index="NAME")
    twiss.columns = [column.lower() for column in twiss.columns]
    twiss = twiss.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    twiss.index = twiss.index.astype(str)
    twiss.index.name = "name"
    # The fixture twiss predates vertical dispersion; PSB dy is ~0 and the
    # MAD-NG twiss table used in production includes it.
    twiss["dy"] = 0.0

    bpms = ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3"]
    df = pd.DataFrame(
        {
            "name": bpms * 2,
            "turn": [1, 1, 1, 2, 2, 2],
            "bunch_number": 0,
            "x": [1e-6, 2e-6, 3e-6, 1.5e-6, 2.5e-6, 3.5e-6],
            "y": [2e-6, 3e-6, 4e-6, 2.5e-6, 3.5e-6, 4.5e-6],
        }
    )
    orbit_zero = pd.DataFrame(0.0, index=pd.Index(bpms, name="name"), columns=["x", "y"])

    idx, result = process_single_dataframe(
        df_with_index=(7, df),
        twiss=twiss,
        bad_bpms=[],
        analysis_dir=analysis_dir,
        use_uniform_vars=True,
        beam=1,
        model_details=ModelDetails(
            accelerator=PSB(ring=3, sequence_file=MODEL_DIR / "psb3_saved.seq")
        ),
        frame=reconstruction_frame(orbit_zero, dynamic_planes=("x", "y")),
    )

    assert idx == 7
    assert {"px", "py", "var_x", "var_y", "var_px", "var_py"} <= set(result.columns)
    assert not result[["px", "py"]].isna().any().any()
    assert set(result["bunch_number"]) == {0}
    assert set(result["name"]) == set(bpms)


def test_post_svd_variance_scaling_follows_rank_over_bpm_count() -> None:
    """The SVD gain is rank/n_bpms, so PSB (16 BPMs, rank 2) gets 1/8, not 1/100."""
    df = pd.DataFrame({"var_x": [4.0, 4.0], "var_y": [9.0, 9.0], "px": [0.0, 0.0]})

    psb = _scale_position_variances_after_svd(df, n_bpms=16, svd_ranks=(2, 2))
    lhc = _scale_position_variances_after_svd(df, n_bpms=500, svd_ranks=(5, 5))

    assert psb["var_x"].tolist() == pytest.approx([0.5, 0.5])
    assert psb["var_y"].tolist() == pytest.approx([9.0 / 8.0, 9.0 / 8.0])
    assert lhc["var_x"].tolist() == pytest.approx([0.04, 0.04])


def test_assign_uniform_variances_zero_weights_bad_bpms() -> None:
    result = assign_uniform_variances(pd.DataFrame({"name": ["BPM1", "BPM2"]}), ["BPM2"], var_value=2.5)

    assert result.loc[0, "var_x"] == pytest.approx(2.5)
    assert result.loc[0, "var_py"] == pytest.approx(2.5)
    assert (result.loc[1, ["var_x", "var_y", "var_px", "var_py"]] == float("inf")).all()


def test_psb_non_bpm_monitors_get_nan_variances_from_patterns() -> None:
    indexed = pd.DataFrame(
        {"x": [0.0, 1e-6, 2e-6], "y": [0.0, 3e-6, 4e-6]},
        index=pd.Index(["BI3.KSW1L4", "BR3.BPM2L3", "BR3.BPMT3L1"], name="name"),
    )

    result = assign_known_noise_variances(
        indexed,
        bad_bpms=[],
        nan_variance_patterns=[r"^BI3\.KSW1L4$", r"^BR3\.BPMT3L1$"],
        accelerator_type="psb",
    )

    assert result.loc[["BI3.KSW1L4", "BR3.BPMT3L1"], ["var_x", "var_y"]].isna().all().all()
    assert (result.loc["BR3.BPM2L3", ["var_x", "var_y"]] > 0.0).all()
