"""MachineSetup and the tracking BPM ranges against a real MAD-NG model of the LHC beam-1 arc 12."""

from __future__ import annotations

import pytest

from aba_optimiser.config import SimulationConfig
from aba_optimiser.fitting.config import OutputConfig, SequenceConfig
from aba_optimiser.fitting.setup import MachineSetup
from aba_optimiser.machine.accelerators import LHC
from aba_optimiser.tracking.config.models import MeasurementConfig, MeasurementDetails
from aba_optimiser.tracking.config.tracking import TrackingPlan
from aba_optimiser.tracking.ranges import resolve_bpm_ranges

START, END = "BPM.9R1.B1", "BPM.9L2.B1"
KNOB = "MQ.11R1.B1.dk1l"  # a quadrupole inside START/END
OUTSIDE = "MQ.11L1.B1.dk1l"  # one before IP1, outside it


@pytest.fixture
def lhc(seq_b1):
    return LHC(beam=1, kinetic_energy=6800, sequence_file=seq_b1, errors={"quad": {"k1"}})


def _setup(accelerator, magnet_range=f"{START}/{END}", **kwargs) -> MachineSetup:
    return MachineSetup(
        accelerator,
        SimulationConfig(num_workers=1, num_batches=1),
        SequenceConfig(magnet_range),
        output_config=OutputConfig(write_tensorboard_logs=False),
        **kwargs,
    )


def test_initial_knobs_override_only_the_knobs_given(lhc) -> None:
    setup = _setup(lhc, initial_knob_strengths={KNOB: 1e-4}, true_strengths={KNOB: 9e-5, "x": 2.0})
    try:
        assert setup.initial_knobs[KNOB] == 1e-4
        assert set(setup.initial_knobs) == set(setup.knob_names)
        # The others keep the model's own (error-free) value.
        assert all(value == 0.0 for name, value in setup.initial_knobs.items() if name != KNOB)
        # True strengths are restricted to the fitted knobs.
        assert setup.true_strengths == {KNOB: 9e-5}
    finally:
        setup.close()


def test_true_strengths_default_to_the_initial_knobs(lhc) -> None:
    setup = _setup(lhc, initial_knob_strengths={KNOB: 1e-4})
    try:
        assert setup.true_strengths == setup.initial_knobs
        assert setup.true_strengths is not setup.initial_knobs
    finally:
        setup.close()


def test_values_outside_the_fit_are_kept_as_model_values(lhc) -> None:
    setup = _setup(lhc, initial_knob_strengths={KNOB: 1e-4, OUTSIDE: 2e-4, "deltap": 1e-3})
    try:
        pt = setup.mad_iface.dp2pt(1e-3)
        # The optimiser only sees this fit's knobs; every value reaches the workers.
        assert OUTSIDE not in setup.initial_knobs
        assert setup.initial_model_values == {KNOB: 1e-4, OUTSIDE: 2e-4, "pt": pt}
        assert setup.worker_start_knobs[OUTSIDE] == 2e-4
        assert setup.worker_start_knobs["pt"] == pt
    finally:
        setup.close()


def test_unknown_initial_knob_names_are_rejected(lhc) -> None:
    with pytest.raises(ValueError, match="Unknown optimisation knob names"):
        _setup(lhc, initial_knob_strengths={"not_a_magnet": 1.0})


def test_a_range_without_knobs_is_rejected(lhc) -> None:
    with pytest.raises(ValueError, match="No optimisation knobs were created"):
        _setup(lhc, magnet_range="BPM.12R1.B1/BPM.12R1.B1")


def test_bpm_points_are_filtered_and_the_fixed_window_set(lhc) -> None:
    setup = MachineSetup(
        lhc,
        SimulationConfig(num_workers=1, num_batches=1, use_fixed_bpm=True),
        SequenceConfig(f"{START}/{END}", bad_bpms=["BPM.10R1.B1"]),
        output_config=OutputConfig(write_tensorboard_logs=False),
    )
    try:
        ranges = resolve_bpm_ranges(
            setup, TrackingPlan(), [START, "BPM.10R1.B1", "BPM.9L1.B1"], [END]
        )
        # The bad BPM and the one outside the range are dropped.
        assert ranges.start_bpms == [START]
        assert (ranges.fixed_start, ranges.fixed_end) == (START, END)
    finally:
        setup.close()


def test_measurement_config_preserves_per_file_interface_options(tmp_path) -> None:
    config = MeasurementConfig(
        {
            tmp_path / "m0.parquet": MeasurementDetails(
                interface_options={"machine_state": tmp_path / "state0.txt"},
                machine_deltap=1e-4,
            ),
            tmp_path / "m1.parquet": MeasurementDetails(
                interface_options={
                    "machine_state": tmp_path / "correctors1_state.txt",
                },
                machine_deltap=2e-4,
            ),
        }
    )

    assert config.files == [tmp_path / "m0.parquet", tmp_path / "m1.parquet"]
    assert config.details == [
        MeasurementDetails(
            interface_options={"machine_state": tmp_path / "state0.txt"},
            machine_deltap=1e-4,
        ),
        MeasurementDetails(
            interface_options={
                "machine_state": tmp_path / "correctors1_state.txt",
            },
            machine_deltap=2e-4,
        ),
    ]


def test_measurement_config_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="at least one measurement file"):
        MeasurementConfig({})
