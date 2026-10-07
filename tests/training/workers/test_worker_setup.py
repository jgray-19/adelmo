"""Worker planning: ranges, observation plans, and per-file worker configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.machine.accelerators import PSB, SPS
from aba_optimiser.tracking.config.tracking import ACDTrackingPlan, TrackingPlan
from aba_optimiser.tracking.dispatch.setup import WorkerRangeSpec, WorkerSetupHelper

if TYPE_CHECKING:
    from pathlib import Path

ARC_BPMS = ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3", "BR3.BPM4L3"]
ACD_BPMS = [
    "BR3.BPM1L3",
    "BR3.BPM2L3",
    "HACMAP_before",
    "HACMAP_after",
    "BR3.BPM3L3",
    "BR3.BPM4L3",
]
# SPS monitors see one plane each: BPH horizontal, BPV vertical.
SPS_BPMS = ["BPH.13008", "BPV.13108", "BPH.13208", "BPV.13308", "BPH.13408", "BPV.13508"]


def _helper(tmp_path: Path, *, all_bpms: list[str], tracking_plan, accelerator=None, **overrides):
    seq_file = tmp_path / "machine.seq"
    seq_file.write_text("! placeholder sequence\n")
    defaults = {
        "accelerator": accelerator or PSB(ring=3, sequence_file=seq_file),
        "all_bpms": all_bpms,
        "fixed_start": all_bpms[0],
        "fixed_end": all_bpms[-1],
        "use_fixed_bpm": True,
        "bad_bpms": None,
        "file_kick_planes": {0: "x", 1: "xy"},
        "magnet_range": "$start/$end",
        "interface_options_per_file": [
            {"machine_state": tmp_path / "corr0_state.txt"},
            {"machine_state": tmp_path / "corr1_state.txt"},
        ],
        "debug": False,
        "mad_logfile": None,
        "python_logfile": None,
        "tracking_plan": tracking_plan,
    }
    return WorkerSetupHelper(**{**defaults, **overrides})


def _sps_helper(tmp_path: Path, **overrides) -> WorkerSetupHelper:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! placeholder sequence\n")
    return _helper(
        tmp_path,
        all_bpms=SPS_BPMS,
        tracking_plan=TrackingPlan(),
        accelerator=SPS(sequence_file=seq_file),
        use_fixed_bpm=False,
        **overrides,
    )


def test_fixed_bpm_ranges_pair_each_start_with_the_fixed_end(tmp_path: Path) -> None:
    helper = _helper(tmp_path, all_bpms=ARC_BPMS, tracking_plan=TrackingPlan())

    specs = helper.build_range_specs(start_bpms=["BR3.BPM1L3", "BR3.BPM2L3"], end_bpms=["BR3.BPM3L3"])

    assert specs == [
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM4L3", 1),
        WorkerRangeSpec("BR3.BPM2L3", "BR3.BPM4L3", 1),
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", -1),
    ]


def test_observation_plan_takes_its_kick_plane_from_the_measurement_file(tmp_path: Path) -> None:
    helper = _helper(tmp_path, all_bpms=ARC_BPMS, tracking_plan=TrackingPlan())
    spec = WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", 1)

    [single_plane] = helper.build_observation_plans(spec, 0, set(ARC_BPMS))
    [dual_plane] = helper.build_observation_plans(spec, 1, set(ARC_BPMS))

    assert single_plane.kick_plane == "x"
    assert dual_plane.kick_plane == "xy"
    assert single_plane.bpm_names == ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3"]
    assert single_plane.bad_bpms is None


def test_bpms_missing_from_a_file_are_unobserved_rather_than_dropping_the_worker(
    tmp_path: Path,
) -> None:
    helper = _helper(tmp_path, all_bpms=ARC_BPMS, tracking_plan=TrackingPlan())

    [plan] = helper.build_observation_plans(
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", 1), 0, {"BR3.BPM1L3", "BR3.BPM3L3"}
    )

    assert plan.bpm_names == ["BR3.BPM1L3", "BR3.BPM3L3"]
    assert plan.bad_bpms == ["BR3.BPM2L3"]


def test_worker_config_carries_its_own_files_measurement_artifacts(tmp_path: Path) -> None:
    helper = _helper(tmp_path, all_bpms=ARC_BPMS, tracking_plan=TrackingPlan())

    [plan] = helper.build_observation_plans(
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", 1), 1, set(ARC_BPMS)
    )
    config = helper.make_worker_config(plan)

    assert config.interface_options == {
        "machine_state": tmp_path / "corr1_state.txt",
    }
    assert config.kick_plane == "xy"


def test_acd_markers_supply_initial_conditions_and_are_not_observed(tmp_path: Path) -> None:
    """Each ACD direction initialises from its own marker and compares real BPMs only."""
    helper = _helper(
        tmp_path,
        all_bpms=ACD_BPMS,
        tracking_plan=ACDTrackingPlan(acd_name="HACMAP"),
        use_fixed_bpm=False,
        fixed_start="$start",
        fixed_end="$end",
        file_kick_planes={0: "xy"},
        interface_options_per_file=[{}],
    )

    specs = helper.build_range_specs(start_bpms=["HACMAP_after", "HACMAP_before"], end_bpms=[])
    assert specs == [
        WorkerRangeSpec("HACMAP_after", "HACMAP_before", 1),
        WorkerRangeSpec("HACMAP_after", "HACMAP_before", -1),
    ]

    [forward], [backward] = (
        helper.build_observation_plans(spec, 0, set(ACD_BPMS)) for spec in specs
    )

    assert forward.bpm_names == ["BR3.BPM3L3", "BR3.BPM4L3", "BR3.BPM1L3", "BR3.BPM2L3"]
    assert backward.bpm_names == ["BR3.BPM2L3", "BR3.BPM1L3", "BR3.BPM4L3", "BR3.BPM3L3"]
    assert helper.make_worker_config(forward).initial_condition_marker == "HACMAP_after"
    assert helper.make_worker_config(backward).initial_condition_marker == "HACMAP_before"


def test_single_plane_bpm_ranges_start_and_end_on_bpms_of_one_plane(tmp_path: Path) -> None:
    helper = _sps_helper(tmp_path)

    specs = helper.build_range_specs(
        start_bpms=["BPH.13008", "BPV.13108"], end_bpms=["BPH.13408", "BPV.13508"]
    )

    assert {(s.start_bpm, s.end_bpm) for s in specs} == {
        ("BPH.13008", "BPH.13408"),
        ("BPV.13108", "BPV.13508"),
    }


def test_single_plane_bpms_split_dual_plane_data_into_x_and_y_workers(tmp_path: Path) -> None:
    helper = _sps_helper(tmp_path)
    horizontal = WorkerRangeSpec("BPH.13008", "BPH.13408", 1)

    vertical = WorkerRangeSpec("BPV.13108", "BPV.13508", 1)

    # Each range yields only the worker whose plane its start BPM can see.
    [x_worker] = helper.build_observation_plans(horizontal, 1, set(SPS_BPMS))
    [y_worker] = helper.build_observation_plans(vertical, 1, set(SPS_BPMS))

    assert (x_worker.kick_plane, y_worker.kick_plane) == ("x", "y")
    assert x_worker.bpm_names == ["BPH.13008", "BPH.13208", "BPH.13408"]
    # MAD unobserves every blind BPM in the ring, not only those inside the range.
    assert x_worker.bad_bpms == ["BPV.13108", "BPV.13308", "BPV.13508"]


def test_single_plane_bpms_plan_acd_marker_ranges_and_split_workers_by_plane(tmp_path: Path) -> None:
    plan = ACDTrackingPlan(acd_name="ZKHA.21991")
    all_bpms = [*SPS_BPMS[:3], plan.acd_before, plan.acd_after, *SPS_BPMS[3:]]
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! placeholder sequence\n")
    helper = _helper(
        tmp_path,
        all_bpms=all_bpms,
        tracking_plan=plan,
        accelerator=SPS(sequence_file=seq_file),
        use_fixed_bpm=False,
    )

    # The markers carry both planes, so the ranges are not split per plane.
    specs = helper.build_range_specs(start_bpms=[plan.acd_after, plan.acd_before], end_bpms=[])
    assert specs == [
        WorkerRangeSpec(plan.acd_after, plan.acd_before, 1),
        WorkerRangeSpec(plan.acd_after, plan.acd_before, -1),
    ]

    # Dual-plane data gives one worker per plane, each blind to the other plane's BPMs.
    x_worker, y_worker = helper.build_observation_plans(specs[0], 1, set(all_bpms))
    assert (x_worker.kick_plane, y_worker.kick_plane) == ("x", "y")
    assert x_worker.bad_bpms == ["BPV.13108", "BPV.13308", "BPV.13508"]
    assert y_worker.bad_bpms == ["BPH.13008", "BPH.13208", "BPH.13408"]
    # A file driven in x only gives only the x worker.
    [x_only] = helper.build_observation_plans(specs[0], 0, set(all_bpms))
    assert x_only.kick_plane == "x"


def test_single_plane_file_drops_workers_starting_on_the_blind_plane(tmp_path: Path) -> None:
    helper = _sps_helper(tmp_path)

    assert helper.build_observation_plans(WorkerRangeSpec("BPV.13108", "BPV.13508", 1), 0, set(SPS_BPMS)) == []
