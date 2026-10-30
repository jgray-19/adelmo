"""Wiring of a real :class:`ArcByArcFitter` on PSB ring 3.

- uncertainties are keyed by the same optimisation knob names as the fitted values;
  PSB sextupoles are used, whose ``knl[3]`` errors become ``...dk2l`` knobs;
- the simulation config has one owner, so clamping ``num_batches`` to the data
  reaches the loop, the data manager and the worker session alike.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from adelmo.config import OptimiserConfig
from adelmo.fitting.config import OutputConfig, SequenceConfig
from adelmo.machine.accelerators import PSB
from adelmo.machine.mad import merge_machine_states
from adelmo.tracking.config.helpers import create_arc_measurement_config
from adelmo.tracking.fitter import ArcByArcFitter, FitterOptions
from tests.training.controller.test_controller_quadrupole_psb import (
    PSB_BPM_END_POINTS,
    PSB_BPM_START_POINTS,
    PSB_TARGET_QX,
    PSB_TARGET_QY,
    PSB_TRACK_BPM_PATTERN,
)
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_simulation_config_quad,
)

if TYPE_CHECKING:
    from pathlib import Path

    from adelmo.machine.mad.aba_mad_interface import AbaMadInterface

pytestmark = [pytest.mark.slow, pytest.mark.serial]

# The nominal model has its sextupoles off, so no ``knl[3]`` knob would be created.
SEXTUPOLES_ON = {"kbr3xno311l1": 0.05}


def _fitter(
    tmp_path: Path, seq_psb: Path, interface: AbaMadInterface, *, num_batches: int
) -> ArcByArcFitter:
    track = tmp_path / "track.parquet"
    corrector, _, tune_knobs = _generate_nonoise_track(
        interface,
        64,
        track,
        0.0,
        bpm_pattern=PSB_TRACK_BPM_PATTERN,
        apply_orbit_correction=False,
        target_qx=PSB_TARGET_QX,
        target_qy=PSB_TARGET_QY,
    )
    return ArcByArcFitter(
        PSB(
            ring=3,
            kinetic_energy=interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors={"sextupole": {"knl[3]"}},
        ),
        OptimiserConfig(
            max_epochs=1,
            warmup_epochs=0,
            warmup_lr_start=1e-6,
            max_lr=1e-6,
            min_lr=1e-6,
            gradient_converged_value=0.0,
        ),
        dataclasses.replace(_make_simulation_config_quad(), num_workers=2, num_batches=num_batches),
        SequenceConfig("$start/$end"),
        create_arc_measurement_config(
            track,
            machine_state={**merge_machine_states(corrector, tune_knobs), **SEXTUPOLES_ON},
        ),
        bpm_start_points=PSB_BPM_START_POINTS[:1],
        bpm_end_points=PSB_BPM_END_POINTS[:1],
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "mad.log",
                write_run_log=False,
                include_uncertainty=True,
            ),
        ),
    )


def test_sextupole_uncertainties_keyed_like_knobs(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: AbaMadInterface
) -> None:
    fitter = _fitter(tmp_path, seq_psb, loaded_psb_interface, num_batches=1)
    result = fitter.run()

    assert any(name.endswith(".dk2l") for name in result.knobs), list(result.knobs)
    assert list(result.uncertainties) == list(result.knobs)
    assert all(np.isfinite(sigma) and sigma > 0 for sigma in result.uncertainties.values())


def test_num_batches_clamped_to_the_data_reaches_every_consumer(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: AbaMadInterface
) -> None:
    fitter = _fitter(tmp_path, seq_psb, loaded_psb_interface, num_batches=10_000)
    simulation_config = fitter.machine.simulation_config

    assert simulation_config.num_batches < 10_000
    assert fitter.loop.num_batches == simulation_config.num_batches
    assert fitter.data_manager.simulation_config is simulation_config
    session = fitter.start_session()
    try:
        assert session.simulation_config is simulation_config
    finally:
        session.terminate()
