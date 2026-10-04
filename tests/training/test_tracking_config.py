"""Tests for tracking-mode planning in aba_optimiser.training.config.tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.config.models import KickerConfig
from aba_optimiser.training.config.tracking import (
    ACDArcByArcTrackingPlan,
    RangeContext,
    TrackingPlan,
    acd_marker_setup,
    arc_by_arc_setup,
    kicker_setup,
)

if TYPE_CHECKING:
    from pathlib import Path

RING = [f"BPM.{i}" for i in range(8)]


def _base_simulation_config() -> SimulationConfig:
    return SimulationConfig(num_workers=2, num_batches=1)


@pytest.fixture
def psb_accelerator(seq_psb: Path) -> PSB:
    return PSB(ring=3, sequence_file=seq_psb)


class TestACDArcByArcReroute:
    """``ACDArcByArcTrackingPlan`` must reroute ranges that straddle the exciter."""

    def _plan(self) -> ACDArcByArcTrackingPlan:
        return ACDArcByArcTrackingPlan(acd_name="ACD")

    def _ctx(self, *, start_bpms: list[str], end_bpms: list[str]) -> RangeContext:
        # The exciter markers sit adjacent between BPM.3 and BPM.4.
        all_bpms = [*RING[:4], "ACD_before", "ACD_after", *RING[4:]]
        return RangeContext(
            start_bpms=start_bpms,
            end_bpms=end_bpms,
            all_bpms=all_bpms,
            use_fixed_bpm=False,
            fixed_start=all_bpms[0],
            fixed_end=all_bpms[-1],
        )

    def test_range_crossing_exciter_is_rerouted(self) -> None:
        plan = self._plan()
        ctx = self._ctx(start_bpms=["BPM.2"], end_bpms=["BPM.5"])

        specs = plan.build_range_specs(ctx)

        # BPM.2 -> BPM.5 forward crosses ACD_before/ACD_after, so it must be
        # swapped to the complementary long-way-round arc (BPM.5 -> BPM.2).
        forward_specs = [s for s in specs if s.sdir == 1]
        assert all((s.start_bpm, s.end_bpm) == ("BPM.5", "BPM.2") for s in forward_specs)

    def test_range_not_crossing_exciter_is_unchanged(self) -> None:
        plan = self._plan()
        ctx = self._ctx(start_bpms=["BPM.5"], end_bpms=["BPM.6"])

        specs = plan.build_range_specs(ctx)

        forward_specs = [s for s in specs if s.sdir == 1]
        assert all((s.start_bpm, s.end_bpm) == ("BPM.5", "BPM.6") for s in forward_specs)

    def test_bpm_pairs_deduplicates_after_reroute(self) -> None:
        plan = self._plan()
        # Both pairs reroute to the same complementary arc, so they should
        # collapse to a single logical pair.
        ctx = self._ctx(start_bpms=["BPM.2", "BPM.1"], end_bpms=["BPM.5"])

        pairs = plan.bpm_pairs(ctx)

        assert len(pairs) == len(set(pairs))

    def test_observed_bpms_excludes_exciter_markers(self) -> None:
        plan = self._plan()
        all_bpms = [*RING[:4], "ACD_before", "ACD_after", *RING[4:]]

        observed = plan.observed_bpms(all_bpms, all_bpms)

        assert "ACD_before" not in observed
        assert "ACD_after" not in observed


class TestPlainArcByArcPlan:
    def test_plain_arc_by_arc_ranges_are_untouched(self) -> None:
        plan = TrackingPlan()
        all_bpms = RING
        ctx = RangeContext(
            start_bpms=["BPM.2"],
            end_bpms=["BPM.5"],
            all_bpms=all_bpms,
            use_fixed_bpm=False,
            fixed_start=all_bpms[0],
            fixed_end=all_bpms[-1],
        )

        specs = plan.build_range_specs(ctx)

        forward_specs = [s for s in specs if s.sdir == 1]
        assert all((s.start_bpm, s.end_bpm) == ("BPM.2", "BPM.5") for s in forward_specs)

    @pytest.mark.parametrize(
        ("use_fixed_bpm", "start_bpms", "end_bpms"),
        [
            # Cartesian product pairs BPM.5 with itself.
            (False, ["BPM.2", "BPM.5"], ["BPM.5", "BPM.6"]),
            # Fixed pairs: a start equal to the fixed end (the first end point).
            (True, ["BPM.2", "BPM.5"], ["BPM.5", "BPM.6"]),
            # Fixed pairs: an end equal to the fixed start (the first start point).
            (True, ["BPM.2", "BPM.3"], ["BPM.6", "BPM.2"]),
        ],
    )
    def test_range_starting_and_ending_at_the_same_bpm_is_rejected(
        self, use_fixed_bpm: bool, start_bpms: list[str], end_bpms: list[str]
    ) -> None:
        ctx = RangeContext(
            start_bpms=start_bpms,
            end_bpms=end_bpms,
            all_bpms=RING,
            use_fixed_bpm=use_fixed_bpm,
            fixed_start=start_bpms[0],
            fixed_end=end_bpms[0],
        )

        with pytest.raises(ValueError, match="same BPM"):
            TrackingPlan().build_range_specs(ctx)


class TestArcByArcSetup:
    def test_requires_bpm_start_points(self) -> None:
        with pytest.raises(ValueError, match="bpm_start_points"):
            arc_by_arc_setup(
                accelerator=None,
                simulation_config=_base_simulation_config(),
                bpm_start_points=[],
                bpm_end_points=["BPM.1"],
                acd_excited=False,
            )

    def test_acd_excited_requires_bpm_end_points(self, psb_accelerator: PSB) -> None:
        with pytest.raises(ValueError, match="bpm_end_points"):
            arc_by_arc_setup(
                accelerator=psb_accelerator,
                simulation_config=_base_simulation_config(),
                bpm_start_points=["BPM.1"],
                bpm_end_points=[],
                acd_excited=True,
            )

    def test_non_acd_excited_produces_plain_arc_by_arc_plan(self) -> None:
        setup = arc_by_arc_setup(
            accelerator=None,
            simulation_config=_base_simulation_config(),
            bpm_start_points=["BPM.1"],
            bpm_end_points=["BPM.2"],
            acd_excited=False,
        )

        assert type(setup.plan) is TrackingPlan
        assert setup.simulation_config.n_run_turns == 1

    def test_acd_excited_produces_acd_arc_by_arc_plan(self, psb_accelerator: PSB) -> None:
        setup = arc_by_arc_setup(
            accelerator=psb_accelerator,
            simulation_config=_base_simulation_config(),
            bpm_start_points=["BPM.1"],
            bpm_end_points=["BPM.2"],
            acd_excited=True,
        )

        assert isinstance(setup.plan, ACDArcByArcTrackingPlan)
        assert setup.plan.acd_name == psb_accelerator.ac_dipole_name
        assert setup.simulation_config.n_run_turns == 1


class TestKickerSetup:
    def test_produces_forward_only_config_with_one_worker_per_file(self) -> None:
        kicker_config = KickerConfig(kicker_name="KICKER1", turns_after_kicker=5)

        setup = kicker_setup(kicker_config, _base_simulation_config())

        # num_workers passes through: one worker per measurement file (momentum).
        assert setup.simulation_config.num_workers == 2
        assert setup.simulation_config.num_batches == 1
        assert setup.simulation_config.n_run_turns == 5
        assert setup.bpm_start_points == ["KICKER1"]
        assert setup.first_bpm_fallback == "KICKER1"

    def test_kicker_config_rejects_non_positive_turns(self) -> None:
        with pytest.raises(ValueError, match="turns_after_kicker"):
            KickerConfig(kicker_name="KICKER1", turns_after_kicker=0)


class TestAcdMarkerSetup:
    def test_produces_bidirectional_acd_plan(self, psb_accelerator: PSB) -> None:
        setup = acd_marker_setup(psb_accelerator, _base_simulation_config())

        assert setup.simulation_config.n_run_turns == 1
        assert setup.bpm_start_points == [
            psb_accelerator.acd_marker_name("after"),
            psb_accelerator.acd_marker_name("before"),
        ]
        assert setup.bpm_end_points == []
