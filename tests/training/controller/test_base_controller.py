from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.training.base_fitter import BaseFitter
from aba_optimiser.training.config.manager import ConfigurationManager
from aba_optimiser.training.config.models import (
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
)


class DummyFitter(BaseFitter):
    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        return {}, {}


class EmptyKnobConfigurationManager(ConfigurationManager):
    def setup_mad_interface(
        self,
        debug: bool = False,
        mad_logfile=None,
        machine_state=None,
    ) -> None:
        del debug, mad_logfile, machine_state
        self.mad_iface = SimpleNamespace(  # ty:ignore[invalid-assignment]
            dp2pt=lambda value: value,
            pt2dp=lambda value: value,
        )
        self.knob_names = []
        self.elem_spos = []
        self.all_bpms = ["BPM.9R1.B1", "BPM.9L2.B1"]
        self.bpms_in_range = ["BPM.9R1.B1", "BPM.9L2.B1"]

    def initialise_knob_strengths(
        self,
        true_strengths: dict[str, float] | None = None,
        provided_initial_knobs: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        del true_strengths, provided_initial_knobs
        self.initial_strengths = np.array([])
        return {}, {}


class EmptyKnobController(DummyFitter):
    _configuration_manager_cls = EmptyKnobConfigurationManager


def test_configuration_manager_preserves_model_defaults_for_missing_initial_knobs(seq_b1) -> None:
    accelerator = LHC(
        beam=1,
        kinetic_energy=6800,
        sequence_file=seq_b1,
        errors={"quad": {"k1"}},
    )
    simulation_config = SimulationConfig(
        num_workers=1,
        num_batches=1,
    )
    manager = ConfigurationManager(
        accelerator=accelerator,
        simulation_config=simulation_config,
        sequence_config=SequenceConfig("BPM.9R1.B1/BPM.9L2.B1"),
        bpm_start_points=["BPM.9R1.B1"],
        bpm_end_points=["BPM.9L2.B1"],
    )
    initial_state = {"k1": 1.25, "k2": -0.75, "k3": 0.5}

    class FakeMadInterface:
        def __init__(self, initial_values: dict[str, float]) -> None:
            self.values = initial_values.copy()

        def receive_knob_values(self) -> np.ndarray:
            return np.asarray([self.values[name] for name in manager.knob_names], dtype=float)

        def update_knob_values(self, new_values: dict[str, float]) -> None:
            self.values.update(new_values)

        def apply_initial_model_values(self, values: dict[str, float] | None) -> None:
            if values:
                self.update_knob_values(values)

    manager.knob_names = list(initial_state)
    manager.mad_iface = FakeMadInterface(initial_state)  # ty:ignore[invalid-assignment]

    initial_knobs, filtered_true = manager.initialise_knob_strengths(
        true_strengths={"k1": 9.0, "outside": 2.0},
        provided_initial_knobs={"k2": 4.0},
    )

    assert initial_knobs == {"k1": 1.25, "k2": 4.0, "k3": 0.5}
    assert filtered_true == {"k1": 9.0}
    np.testing.assert_allclose(manager.initial_strengths, np.array([1.25, 4.0, 0.5]))


def _initial_model_values_manager(seq_b1):
    accelerator = LHC(
        beam=1,
        kinetic_energy=6800,
        sequence_file=seq_b1,
        errors={"quad": {"k1"}},
    )
    manager = ConfigurationManager(
        accelerator=accelerator,
        simulation_config=SimulationConfig(num_workers=1, num_batches=1),
        sequence_config=SequenceConfig("BPM.9R1.B1/BPM.9L2.B1"),
        bpm_start_points=["BPM.9R1.B1"],
        bpm_end_points=["BPM.9L2.B1"],
    )
    initial_state = {"MQ.1.dk1l": 1.25}

    class FakeMadInterface:
        def __init__(self, initial_values: dict[str, float]) -> None:
            self.values = initial_values.copy()
            self.magnet_strengths: dict[str, float] = {}
            self.initial_model_values: dict[str, float] = {}

        def receive_knob_values(self) -> np.ndarray:
            return np.asarray([self.values[name] for name in manager.knob_names], dtype=float)

        def update_knob_values(self, new_values: dict[str, float]) -> None:
            self.values.update(new_values)

        def set_magnet_strengths(self, strengths: dict[str, float]) -> None:
            self.magnet_strengths.update(strengths)

        def apply_initial_model_values(self, values: dict[str, float] | None) -> None:
            if not values:
                return
            self.initial_model_values.update(values)
            self.update_knob_values(
                {name: value for name, value in values.items() if name in manager.knob_names}
            )
            self.set_magnet_strengths(
                {
                    name: value
                    for name, value in values.items()
                    if name not in manager.knob_names and name != "pt"
                }
            )

    manager.knob_names = list(initial_state)
    manager.mad_iface = FakeMadInterface(initial_state)  # ty:ignore[invalid-assignment]
    return manager


def test_initialise_knob_strengths_applies_initial_model_values(seq_b1) -> None:
    manager = _initial_model_values_manager(seq_b1)

    initial_knobs, _ = manager.initialise_knob_strengths(
        provided_initial_knobs={"MQ.1.dk1l": 2.0, "BR.BHZ101.dk0l": 5.0, "pt": 1e-4},
    )

    # The full initial state is applied to the model, but only this stage's
    # optimisation knobs are returned to the optimiser.
    assert initial_knobs == {"MQ.1.dk1l": 2.0}
    assert manager.initial_model_values == {
        "MQ.1.dk1l": 2.0,
        "BR.BHZ101.dk0l": 5.0,
        "pt": 1e-4,
    }
    np.testing.assert_allclose(manager.initial_strengths, np.array([2.0]))
    assert manager.mad_iface.initial_model_values == manager.initial_model_values
    assert manager.mad_iface.magnet_strengths == {"BR.BHZ101.dk0l": 5.0}


def test_initialise_knob_strengths_still_rejects_genuinely_unknown_names(seq_b1) -> None:
    manager = _initial_model_values_manager(seq_b1)

    with pytest.raises(ValueError, match="Unknown optimisation knob names"):
        manager.initialise_knob_strengths(provided_initial_knobs={"not_a_magnet": 1.0})


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


def test_base_fitter_raises_when_no_knobs_created(seq_b1) -> None:
    accelerator = LHC(
        beam=1,
        kinetic_energy=6800,
        sequence_file=seq_b1,
        errors={"quad": {"k1"}},
    )
    optimiser_config = OptimiserConfig(
        max_epochs=1,
        warmup_epochs=0,
        warmup_lr_start=1e-6,
        max_lr=1e-6,
        min_lr=1e-6,
        gradient_converged_value=1e-12,
    )
    simulation_config = SimulationConfig(
        num_workers=1,
        num_batches=1,
    )

    with pytest.raises(ValueError, match="No optimisation knobs were created for this fitter configuration"):
        EmptyKnobController(
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            sequence_config=SequenceConfig("BPM.9R1.B1/BPM.9L2.B1"),
            bpm_start_points=["BPM.9R1.B1"],
            bpm_end_points=["BPM.9L2.B1"],
            output_config=OutputConfig(write_tensorboard_logs=False),
        )
