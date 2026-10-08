"""
Common pytest fixtures for MAD interface tests.

This module contains shared fixtures used across MAD interface test modules.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tfs
from omc3.model_creator import create_instance_and_model
from pymadng_utils.madx.make_sequence import make_madx_sequence

from adelmo.machine.accelerators import LHC, PSB
from adelmo.machine.mad.aba_mad_interface import AbaMadInterface

if TYPE_CHECKING:
    from collections.abc import Generator

# Configure logging for tests
logging.getLogger("xdeps").setLevel(logging.WARNING)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Prevent serial tests from running inside pytest-xdist workers."""
    if (
        item.get_closest_marker("serial")
        and getattr(item.config, "workerinput", None) is not None
    ):
        pytest.fail(f"{item.nodeid} is marked serial and cannot run under pytest-xdist")


@pytest.fixture(autouse=True)
def _serial_harpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Harpy's per-bunch analysis to run in-process.

    Harpy forks worker processes after BLAS/OpenMP thread pools have already
    been initialised, which aborts in ``threadpoolctl._load_libraries`` inside
    pytest. Tests do not need the wall-clock benefit of that parallelism, so
    it is disabled for the whole suite instead of threading ``n_jobs=1``
    through every ``harpy_options``/``harpy_kwargs`` call site.
    """
    monkeypatch.setattr("omc3.harpy._parallel.decide_n_workers", lambda *a, **k: 1)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to the example corrector file used by several tests."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def seq_b1(data_dir: Path) -> Path:
    """Path to the example sequence file for beam 1 used by several tests."""
    return data_dir / "sequences" / "lhcb1.seq"


@pytest.fixture(scope="session")
def seq_b1_crossing(data_dir: Path) -> Path:
    """Path to the example sequence file for beam 1 with crossing used by several tests."""
    return data_dir / "sequences" / "b1_120cm_crossing.seq"


@pytest.fixture(scope="session")
def seq_b2(data_dir: Path) -> Path:
    """Path to the example sequence file for beam 2 used by a test."""
    return data_dir / "sequences" / "lhcb2.seq"


def _create_psb_nominal_model(output_dir: Path, acc_models_dir: Path) -> None:
    """Generate the PSB nominal model, including its ACD sequence, with omc3."""
    create_instance_and_model(
        outputdir=output_dir,
        accel="psbooster",
        type="nominal",
        nat_tunes=[0.17, 0.225],
        dpp=0.0,
        fetch="path",
        path=acc_models_dir,
        scenario="lhc_indiv",
        year="2026",
        cycle_point="1_flat_bottom",
        str_file="psb_fb_lhcindiv.str",
        ring=3,
        driven_excitation="acd",
        drv_tunes=[0.162, 0.232],
        list_choices=False,
        show_help=False,
        logfile=None,
    )


@pytest.fixture(scope="session")
def psb_model_dir(tmp_path_factory: pytest.TempPathFactory, data_dir: Path) -> Path:
    """Generate the PSB model and ACD sequence with omc3 from local fixtures."""
    model_dir = tmp_path_factory.mktemp("psb_model") / "ring3_model"
    _create_psb_nominal_model(model_dir, data_dir / "acc-models-psb")
    make_madx_sequence(model_dir)
    return model_dir


@pytest.fixture(scope="session")
def seq_psb(psb_model_dir: Path) -> Path:
    """Path to the omc3-generated PSB ACD sequence."""
    return psb_model_dir / "psb3_saved.seq"


@pytest.fixture(scope="session")
def tune_knobs(data_dir: Path) -> Path:
    """Path to the tune knobs file."""
    return data_dir / "strengths" / "tune_knobs.txt"


@pytest.fixture(scope="session")
def corrector_knobs(data_dir: Path) -> Path:
    """Path to the corrector knobs file."""
    return data_dir / "correctors" / "corrector_knobs.txt"


@pytest.fixture(scope="session")
def corrector_file(data_dir: Path) -> Path:
    """Path to the corrector table file."""
    return data_dir / "correctors" / "corrector_table.tfs"


@pytest.fixture(scope="session")
def tracking_path(data_dir: Path) -> Path:
    """Path to the tracking data directory."""
    return data_dir / "analysis" / "tracking"


@pytest.fixture(scope="session")
def model_dir_b1() -> Path:
    """Path to the beam 1 model directory."""
    return Path(__file__).parent.parent / "models" / "lhcb1_12cm"


@pytest.fixture(scope="session")
def model_dir_b2() -> Path:
    """Path to the beam 2 model directory."""
    return Path(__file__).parent.parent / "models" / "lhcb2_12cm"


@pytest.fixture(scope="session")
def corrector_table(corrector_file: Path) -> tfs.TfsDataFrame:
    """Load and filter corrector table, removing monitor elements."""
    corrector_table = tfs.read(corrector_file)
    # Filter out monitor elements from the corrector table
    return corrector_table[corrector_table["kind"] != "monitor"]  # ty:ignore[invalid-return-type]


@pytest.fixture(scope="function")
def loaded_interface(seq_b1: Path) -> Generator[AbaMadInterface, None, None]:
    """Create a fresh AbaMadInterface for each test."""
    iface = AbaMadInterface(accelerator=LHC(beam=1, sequence_file=seq_b1, kinetic_energy=6800.0))
    yield iface
    with contextlib.suppress(Exception):
        del iface


@pytest.fixture(scope="function")
def loaded_psb_interface(seq_psb: Path) -> Generator[AbaMadInterface, None, None]:
    """Fixture that returns an interface with PSB ring 3 loaded and beam set up."""
    iface = AbaMadInterface(accelerator=PSB(ring=3, sequence_file=seq_psb))
    yield iface
    with contextlib.suppress(Exception):
        del iface


@pytest.fixture(scope="function")
def beam2_interface(seq_b2: Path) -> Generator[AbaMadInterface, None, None]:
    """Create a fresh AbaMadInterface for beam 2 tests."""
    iface = AbaMadInterface(accelerator=LHC(beam=2, sequence_file=seq_b2, kinetic_energy=6800.0))
    yield iface
    with contextlib.suppress(Exception):
        del iface
