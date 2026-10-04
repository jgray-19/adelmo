"""SPS AC-dipole fits with single-plane BPMs, driving both planes together or apart.

The tracking and fitter set-up come from ``examples/study_sps_acd_diagonal_vs_separate.py``,
so the study and these checks cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("xtrack_tools")

from tests.training.controller_test_utils import evaluate_controller_worker_losses

pytestmark = pytest.mark.serial

STUDY_PATH = Path(__file__).parents[3] / "examples" / "study_sps_acd_diagonal_vs_separate.py"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_sps_acd_diagonal_vs_separate", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: the study's dataclass resolves its annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(study, tmp_path_factory):
    return study.generate_cases(tmp_path_factory.mktemp("sps_acd"), flattop_turns=10)


@pytest.mark.parametrize("case", ["diagonal", "separate"])
def test_true_quadrupole_errors_minimise_the_acd_loss(study, data, case, tmp_path) -> None:
    """Both excitation schemes must single out the true errors through single-plane BPMs."""
    fitter = study.build_fitter(data, case, tmp_path, max_epochs=1)
    initial = fitter.initial_knobs
    true = {name: data.magnet_strengths[name] for name in initial}

    initial_loss, true_loss = evaluate_controller_worker_losses(fitter, [initial, true])

    assert true_loss < initial_loss * 1e-2, (
        f"{case}: loss at the true errors {true_loss:.3e} vs initial {initial_loss:.3e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", ["diagonal", "separate"])
def test_fit_removes_most_of_the_beta_beating(study, data, case, tmp_path) -> None:
    """Judged on beta-beating: neighbouring quadrupoles are degenerate, so the summed
    knob error falls far less than the optics error."""
    fitter = study.build_fitter(data, case, tmp_path, max_epochs=60)
    true_betas = study.bpm_betas(data, data.magnet_strengths)
    initial = study.beta_beating_rms(study.bpm_betas(data, None), true_betas)

    estimate, _uncertainties = fitter.run()

    fitted = study.beta_beating_rms(study.bpm_betas(data, estimate), true_betas)
    for plane, before, after in zip("xy", initial, fitted, strict=True):
        assert after < 0.3 * before, f"{case} {plane}: beta-beating {before:.2e} -> {after:.2e}"
