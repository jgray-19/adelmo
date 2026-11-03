"""The BPM ranges a tracking fit tracks between."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from adelmo.tracking.config.tracking import RangeContext

if TYPE_CHECKING:
    from adelmo.fitting.setup import MachineSetup
    from adelmo.tracking.config.tracking import TrackingPlan

LOGGER = logging.getLogger(__name__)


def filter_bad_bpms(
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    bad_bpms: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Remove bad BPMs from start and end point lists.

    Args:
        bpm_start_points: List of starting BPM names
        bpm_end_points: List of ending BPM names
        bad_bpms: Optional list of BPM names to remove

    Returns:
        Tuple of (filtered_start_points, filtered_end_points)
    """
    if bad_bpms is None:
        return bpm_start_points, bpm_end_points

    filtered_start = bpm_start_points.copy()
    filtered_end = bpm_end_points.copy()

    for bpm in bad_bpms:
        if bpm in filtered_start:
            filtered_start.remove(bpm)
            LOGGER.warning(f"Removed bad BPM {bpm} from start points")
        if bpm in filtered_end:
            filtered_end.remove(bpm)
            LOGGER.warning(f"Removed bad BPM {bpm} from end points")

    return filtered_start, filtered_end


@dataclass
class BpmRanges:
    """The start and end points of a tracking fit, restricted to the observed range.

    ``fixed_start``/``fixed_end`` are the fixed BPM window taken from the magnet
    range when ``use_fixed_bpm`` is set and the plan uses one; empty otherwise.
    """

    plan: TrackingPlan
    start_bpms: list[str]
    end_bpms: list[str]
    all_bpms: list[str]
    use_fixed_bpm: bool
    fixed_start: str = ""
    fixed_end: str = ""

    @property
    def bpm_pairs(self) -> list[tuple[str, str]]:
        """The tracking plan's BPM ranges as explicit ``(start, end)`` tuples."""
        return self.plan.bpm_pairs(
            RangeContext(
                start_bpms=self.start_bpms,
                end_bpms=self.end_bpms,
                all_bpms=self.all_bpms,
                use_fixed_bpm=self.use_fixed_bpm,
                fixed_start=self.fixed_start,
                fixed_end=self.fixed_end,
            )
        )


def resolve_bpm_ranges(
    machine: MachineSetup,
    plan: TrackingPlan,
    bpm_start_points: list[str],
    bpm_end_points: list[str],
) -> BpmRanges:
    """Drop bad BPMs and those outside the observed range, and set the fixed BPM window."""
    sequence_config = machine.sequence_config
    use_fixed_bpm = machine.simulation_config.use_fixed_bpm
    start_bpms, end_bpms = filter_bad_bpms(
        bpm_start_points, bpm_end_points, sequence_config.bad_bpms
    )
    LOGGER.info(
        "After filtering bad BPMs, %s: %s; end points: %s",
        plan.start_point_label,
        start_bpms,
        end_bpms,
    )

    # Marker-anchored modes (kicker/ACD) may start from installed or measured
    # marker anchors rather than ordinary BPMs, so keep those through the filter.
    allowed_starts = set(plan.tracking_anchor_sources)
    if plan.init_marker is not None:
        allowed_starts.add(plan.init_marker)
    in_range = set(machine.bpms_in_range)
    ranges = BpmRanges(
        plan=plan,
        start_bpms=[b for b in start_bpms if b in in_range or b in allowed_starts],
        end_bpms=[b for b in end_bpms if b in in_range],
        all_bpms=machine.all_bpms,
        use_fixed_bpm=use_fixed_bpm,
    )

    # With use_fixed_bpm the fixed BPM window comes from magnet_range; otherwise
    # it stays empty and ranges come from start_bpms/end_bpms.
    magnet_range = sequence_config.magnet_range
    if use_fixed_bpm and plan.uses_fixed_bpm_window:
        ranges.fixed_start, ranges.fixed_end = magnet_range.split("/", 1)
        if ranges.fixed_start not in in_range or ranges.fixed_end not in in_range:
            range_label = (
                "full cycled sequence ($start/$end)"
                if magnet_range == "$start/$end"
                else magnet_range
            )
            LOGGER.warning(
                "Fixed BPMs from range %s not found in model, using first available",
                range_label,
            )
            ranges.fixed_start = ranges.start_bpms[0] if ranges.start_bpms else ranges.fixed_start
            ranges.fixed_end = ranges.end_bpms[0] if ranges.end_bpms else ranges.fixed_end
    elif use_fixed_bpm:
        LOGGER.info(
            "Skipping fixed BPM derivation for this tracking plan; %s: %s",
            plan.start_point_label,
            ranges.start_bpms,
        )
    return ranges
