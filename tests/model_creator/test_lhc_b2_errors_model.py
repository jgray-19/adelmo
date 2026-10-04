"""Cross-check MAD-NG b2 dipole errors against an omc3 MAD-X best-knowledge model.

omc3's ``best_knowledge`` model creator builds a MAD-X model with the b2 dipole
field errors (``seterr``) and their per-arc ``KQT[FD]`` trim-quad correction
(``b2_settings.madx``). This test reproduces that machine in MAD-NG -- loading the
nominal sequence, setting the same trim quads as custom tune knobs, and applying the
same b2 errors through ``dknl[2]`` -- and asserts the two twiss agree on phase
(MAD-NG ``mu1``/``mu2`` vs MAD-X ``MUX``/``MUY``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tfs
from cpymad.madx import Madx
from omc3.model_creator import create_instance_and_model
from pymadng_utils.io.utils import save_knobs
from pymadng_utils.madx.make_sequence import make_madx_sequence

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad.optimising_mad_interface import GenericMadInterface

BEAM = 1
ENERGY_GEV = 6800.0
NAT_TUNES = [0.28, 0.31]
DRV_TUNES = [0.27, 0.322]
OPTICS_MODIFIER = "R2025aRP_A12cmC12cmA10mL200cm_Flat.madx"
B2_STEM = "MB2022_6800.0GeV_0133cm"

# Per-arc trim quads (MAD-X names) set by the b2 correction, 8 arcs for KQTF and KQTD.
KQT_MADX_NAMES = [f"kqt{fd}.a{arc}b{BEAM}" for fd in "fd" for arc in ("12", "23", "34", "45", "56", "67", "78", "81")]


@pytest.fixture(scope="module")
def b2_model(tmp_path_factory: pytest.TempPathFactory, data_dir: Path) -> tuple[Path, Path]:
    """Create matching nominal and best-knowledge omc3 models (same optics modifier).

    Returns the nominal MAD-NG sequence and the best-knowledge model directory.
    """
    nominal_dir = tmp_path_factory.mktemp("nominal")
    create_instance_and_model(
        accel="lhc",
        fetch="path",
        path=data_dir / "acc-models-lhc",
        type="nominal",
        beam=BEAM,
        year="2025",
        driven_excitation="acd",
        energy=ENERGY_GEV,
        nat_tunes=NAT_TUNES,
        drv_tunes=DRV_TUNES,
        modifiers=[OPTICS_MODIFIER],
        outputdir=nominal_dir,
    )
    make_madx_sequence(nominal_dir, beam4=(BEAM == 2))

    best_knowledge_dir = tmp_path_factory.mktemp("best_knowledge")
    create_instance_and_model(
        accel="lhc",
        fetch="path",
        path=data_dir / "acc-models-lhc",
        type="best_knowledge",
        beam=BEAM,
        year="2025",
        energy=ENERGY_GEV,
        nat_tunes=NAT_TUNES,
        modifiers=[OPTICS_MODIFIER],
        b2_errors=str(data_dir / "strengths" / B2_STEM),  # For some reason stem is mandatory
        outputdir=best_knowledge_dir,
    )
    return nominal_dir / f"lhcb{BEAM}_saved.seq", best_knowledge_dir


def _extract_kqt_knobs(b2_settings_madx: Path) -> dict[str, float]:
    """Resolve the per-arc trim-quad strengths defined by the b2 correction file.

    The ``b2_settings.madx`` knob expressions are self-contained (they only depend on
    constants and ``kqtf.b1``/``kqtd.b1``, both set to zero in the file), so a bare
    MAD-X instance resolves them without loading the sequence. The MAD-NG MADX
    environment uses underscores, so ``kqtf.a12b1`` is stored as ``kqtf_a12b1``.
    """
    with Path(os.devnull).open("w") as devnull:
        madx = Madx(stdout=devnull)
        try:
            madx.call(str(b2_settings_madx))
            return {name.replace(".", "_"): float(madx.globals[name]) for name in KQT_MADX_NAMES}
        finally:
            madx.quit()


def _phase_advances(table: tfs.TfsDataFrame, column: str, bpms: list[str]) -> np.ndarray:
    """Phase at each BPM relative to the first, removing any absolute offset."""
    phases = table.loc[bpms, column].to_numpy()
    return phases - phases[0]


@pytest.mark.slow
def test_madng_b2_errors_match_omc3_best_knowledge_phase(
    b2_model: tuple[Path, Path], tmp_path: Path, data_dir: Path
) -> None:
    sequence_file, best_knowledge_dir = b2_model

    # MAD-X reference: best-knowledge model (b2 field errors + trim-quad correction).
    reference = tfs.read(best_knowledge_dir / "twiss_elements_best_knowledge.dat", index="NAME")

    # Extract the trim-quad correction from MAD-X and store it as a custom tune-knobs file.
    kqt_knobs = _extract_kqt_knobs(best_knowledge_dir / "b2_settings.madx")
    assert any(abs(value) > 0 for value in kqt_knobs.values())
    tune_knobs = tmp_path / "tune_knobs.txt"
    save_knobs(kqt_knobs, tune_knobs)

    # MAD-NG: nominal sequence + the same trim quads + the same b2 errors (via dknl).
    accelerator = LHC(
        beam=BEAM,
        sequence_file=sequence_file,
        kinetic_energy=ENERGY_GEV,
    )
    interface = GenericMadInterface(
        accelerator=accelerator,
        tune_knobs=tune_knobs,
        b2_errors=data_dir / "strengths" / (B2_STEM + ".errors"),
    )
    madng = interface.run_twiss(observe=1)

    # Tunes must agree.
    assert madng.headers["q1"] == pytest.approx(reference.headers["Q1"], abs=1e-4)
    assert madng.headers["q2"] == pytest.approx(reference.headers["Q2"], abs=1e-4)

    # Phase advances at every BPM must agree between MAD-X and MAD-NG.
    madng.index = [str(name).upper() for name in madng.index]
    reference.index = [str(name).upper() for name in reference.index]
    bpms = [name for name in madng.index if name in reference.index]
    assert len(bpms) > 500, "Not enough BPMs in common between MAD-X and MAD-NG twiss tables"

    for madx_col, madng_col in (("MUX", "mu1"), ("MUY", "mu2")):
        delta = np.abs(
            _phase_advances(reference, madx_col, bpms) - _phase_advances(madng, madng_col, bpms)
        )
        assert delta.max() < 1e-5, f"{madng_col}: max phase diff {delta.max():.2e}"
