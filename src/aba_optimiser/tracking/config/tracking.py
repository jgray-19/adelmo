"""Tracking-mode planning for arc-by-arc, kicker- and AC-dipole-excited runs.

This module owns everything mode-specific about a training run: which
:class:`TrackingPlan` applies, how the excitation method rewrites the
simulation config and BPM points (the `*_setup` helpers), and how each
plan expands BPM points into worker ranges.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.machine.accelerators import Accelerator
    from aba_optimiser.tracking.config.models import KickerConfig


def extract_bpm_range_names(
    all_bpms: list[str],
    start_bpm: str,
    end_bpm: str,
    sdir: int,
    allow_missing_start: bool = False,
) -> list[str]:
    """Extract BPM names between start and end BPMs, handling circular wrapping.

    Args:
        all_bpms: List of all BPM names
        start_bpm: Starting BPM name
        end_bpm: Ending BPM name
        sdir: Direction (1 for forward, -1 for reverse)

    Returns:
        List of BPM names in the range
    """
    if start_bpm in all_bpms:
        start_pos = all_bpms.index(start_bpm)
    elif allow_missing_start:
        if sdir == -1:
            raise ValueError(
                f"Start marker '{start_bpm}' is not in BPM list for reverse tracking"
            )
        start_pos = 0
    else:
        raise ValueError(f"Start BPM '{start_bpm}' not found in BPM list")

    if end_bpm not in all_bpms:
        raise ValueError(f"End BPM '{end_bpm}' not found in BPM list")
    end_pos = all_bpms.index(end_bpm) + 1

    if end_pos <= start_pos:
        # Circular wrapping
        extracted = all_bpms[start_pos:] + all_bpms[:end_pos]
    else:
        extracted = all_bpms[start_pos:end_pos]
    # Reverse for negative direction
    if sdir == -1:
        extracted = extracted[::-1]

    return extracted


def create_bpm_range_specs(
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    use_fixed_bpm: bool,
    fixed_start: str | None = None,
    fixed_end: str | None = None,
) -> list[tuple[str, str, int]]:
    """Create BPM range specifications for optimisation workers.

    Args:
        bpm_start_points: List of starting BPM names
        bpm_end_points: List of ending BPM names
        use_fixed_bpm: If True, use fixed BPM pairs; if False, create cartesian product
        fixed_start: Fixed start BPM for backward tracking (only used if use_fixed_bpm=True)
        fixed_end: Fixed end BPM for forward tracking (only used if use_fixed_bpm=True)

    Returns:
        List of (start_bpm, end_bpm, sdir) tuples where sdir is 1 for forward, -1 for reverse
    """
    if use_fixed_bpm:
        logger.warning("Using fixed BPM pairs for optimisation. This will create fewer, more constrained measurement combinations.")
        if fixed_start is None or fixed_end is None:
            raise ValueError("fixed_start and fixed_end must be provided when use_fixed_bpm=True")
        # Forward: start -> fixed_end; Backward: fixed_start -> end
        range_specs = [(s, fixed_end, 1) for s in bpm_start_points] + [
            (fixed_start, e, -1) for e in bpm_end_points
        ]
    else:
        # Cartesian product: every start with every end in both directions
        range_specs = [
            (s, e, sdir) for s in bpm_start_points for e in bpm_end_points for sdir in (1, -1)
        ]

    # MAD tracks "A/A" as the single element A: the worker would observe only its own
    # start point and constrain nothing.
    degenerate = sorted({s for s, e, _ in range_specs if s == e})
    if degenerate:
        raise ValueError(
            f"BPM ranges must not start and end at the same BPM: {degenerate}. "
            "Choose start and end points that differ (with use_fixed_bpm, the fixed start "
            "and end BPMs must also not appear among the end and start points)."
        )
    return range_specs


def _boundary_turns_for_track(track_turns: list, margin: int) -> list:
    """Return the boundary turns (first/last ``margin`` entries) for one track."""
    if len(track_turns) <= 2 * margin:
        return track_turns
    return track_turns[:margin] + track_turns[-margin:]


def _bpm_behind(all_bpms: list[str], bpm: str) -> str:
    """Return the BPM immediately behind ``bpm`` in ring order."""
    if bpm not in all_bpms:
        raise ValueError(f"Start BPM '{bpm}' not found in model BPM list")
    return all_bpms[all_bpms.index(bpm) - 1]


@dataclass(frozen=True)
class WorkerRangeSpec:
    """Logical BPM range assigned to a worker."""

    start_bpm: str
    end_bpm: str
    sdir: int

    @property
    def init_bpm(self) -> str:
        """Return the BPM used to initialise tracking for this direction."""
        return self.start_bpm if self.sdir > 0 else self.end_bpm


@dataclass(frozen=True)
class RangeContext:
    """Inputs a plan needs to expand BPM points into worker ranges."""

    start_bpms: list[str]
    end_bpms: list[str]
    all_bpms: list[str]
    use_fixed_bpm: bool
    fixed_start: str
    fixed_end: str


class TrackingPlan:
    """Mode-specific tracking policy; the base class is arc-by-arc BPM tracking.

    Subclasses override the class-level flags and the range-planning hooks below.
    """

    # Element supplying tracking initial conditions instead of a range BPM.
    init_marker: str | None = None
    # Whether the start element may be absent from the BPM list.
    allow_missing_start: bool = False
    # Whether validation workers should be enabled.
    enable_validation: bool = True
    # Whether fixed BPM start/end derivation applies to this plan.
    uses_fixed_bpm_window: bool = True
    # MAD-interface preparation mode for marker-anchored tracking.
    tracking_anchor_mode: str | None = None
    # Optional unobserved element used only for sequence cycling.
    cycle_marker: str | None = None
    # Non-BPM markers that must be kept in measurement data.
    extra_markers: tuple[str, ...] = ()
    # How this plan's start points are described in logs.
    start_point_label: str = "BPM tracking start points"

    @property
    def tracking_anchor_markers(self) -> tuple[str, ...]:
        """Return non-BPM tracking anchors that may not appear in the BPM range."""
        return self.extra_markers

    @property
    def tracking_anchor_sources(self) -> tuple[str, ...]:
        """Return source elements needed to prepare marker-anchored tracking."""
        return self.tracking_anchor_markers

    @property
    def observed_tracking_anchor_markers(self) -> tuple[str, ...]:
        """Return tracking anchors that should also be fit observations."""
        return self.tracking_anchor_markers

    def initial_condition_marker(self, range_spec: WorkerRangeSpec) -> str | None:
        """Return the measured marker used for a worker's initial coordinates."""
        del range_spec
        return self.init_marker

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """Return the BPMs compared against measurements."""
        return bpms_in_range

    def range_specs_per_batch(
        self,
        *,
        use_fixed_bpm: bool,
        num_starts: int,
        num_ends: int,
    ) -> tuple[int, str]:
        """Return the range-spec count used for worker planning."""
        if use_fixed_bpm:
            return num_starts + num_ends, f"fixed pairs ({num_starts} starts + {num_ends} ends)"
        return num_starts * num_ends * 2, f"2 directions x {num_starts} starts x {num_ends} ends"

    def select_available_turns(
        self,
        *,
        bunch_turns_by_file: dict[int, dict[int, list[int]]],
        simulation_config: SimulationConfig,
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        """Return boundary turns and the filtered list of usable start turns.

        A start turn is unusable when the multi-turn track it seeds would cross a
        bunch boundary. Each bunch's first/last turns are therefore removed, using
        the per-file ``bunch_number`` grouping read from the measurement data.
        """
        margin = max(1, simulation_config.n_run_turns)
        turns_to_remove = set()
        boundary_turns_by_file: dict[int, set[int]] = {}

        for file_idx, bunches in bunch_turns_by_file.items():
            for bunch_turns in bunches.values():
                boundary_turns = _boundary_turns_for_track(sorted(bunch_turns), margin)
                boundary_turns_by_file.setdefault(file_idx, set()).update(boundary_turns)
                turns_to_remove.update(boundary_turns)

        return boundary_turns_by_file, [t for t in available_turns if t not in turns_to_remove]

    def bpm_pairs(self, ctx: RangeContext) -> list[tuple[str, str]]:
        """Return logical fitter-side BPM pairs."""
        if ctx.use_fixed_bpm:
            return [(s, ctx.fixed_end) for s in ctx.start_bpms] + [
                (ctx.fixed_start, e) for e in ctx.end_bpms
            ]
        return [(s, e) for s in ctx.start_bpms for e in ctx.end_bpms]

    def build_range_specs(self, ctx: RangeContext) -> list[WorkerRangeSpec]:
        """Return the worker range specs for this plan."""
        return [
            WorkerRangeSpec(start_bpm, end_bpm, sdir)
            for start_bpm, end_bpm, sdir in create_bpm_range_specs(
                ctx.start_bpms,
                ctx.end_bpms,
                ctx.use_fixed_bpm,
                ctx.fixed_start,
                ctx.fixed_end,
            )
        ]

    def get_range_bpm_names(
        self,
        *,
        all_bpms: list[str],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None,
    ) -> list[str]:
        """Return the BPMs in one logical observation range."""
        bpm_names = extract_bpm_range_names(
            all_bpms,
            start_bpm,
            end_bpm,
            sdir,
            self.allow_missing_start,
        )
        excluded = set(bad_bpms or [])
        return [bpm for bpm in bpm_names if bpm not in excluded]

@dataclass(frozen=True)
class KickerTrackingPlan(TrackingPlan):
    """Forward-only tracking starting from a kicker initial-condition marker.

    The measured kicker marker supplies the initial coordinates, while the MAD
    sequence is cycled to an unobserved centre marker so the observed BPM order
    begins just after the kicker; only real BPMs are compared.
    """

    kicker_name: str

    allow_missing_start = True
    enable_validation = False
    tracking_anchor_mode = "kicker"
    start_point_label = "kicker tracking start marker(s)"

    @property
    def init_marker(self) -> str:
        return self.kicker_name

    @property
    def extra_markers(self) -> tuple[str, ...]:
        return (self.kicker_name,)

    @property
    def tracking_anchor_markers(self) -> tuple[str, ...]:
        return ()

    @property
    def tracking_anchor_sources(self) -> tuple[str, ...]:
        return (self.kicker_name,)

    @property
    def cycle_marker(self) -> str:
        return f"{self.kicker_name}_centre"

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        return all_bpms

    def range_specs_per_batch(
        self,
        *,
        use_fixed_bpm: bool,
        num_starts: int,
        num_ends: int,
    ) -> tuple[int, str]:
        return num_starts, f"kicker forward-only x {num_starts} start marker(s)"

    def select_available_turns(
        self,
        *,
        bunch_turns_by_file: dict[int, dict[int, list[int]]],
        simulation_config: SimulationConfig,
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        """Each bunch contributes exactly one track, seeded from its first turn."""
        boundary_turns_by_file = {file_idx: set() for file_idx in bunch_turns_by_file}
        kicker_start_turns = [
            min(bunch_turns)
            for bunches in bunch_turns_by_file.values()
            for bunch_turns in bunches.values()
        ]
        return boundary_turns_by_file, sorted(kicker_start_turns)

    def bpm_pairs(self, ctx: RangeContext) -> list[tuple[str, str]]:
        start = ctx.all_bpms[0]
        return [(start, _bpm_behind(ctx.all_bpms, start))]

    def build_range_specs(self, ctx: RangeContext) -> list[WorkerRangeSpec]:
        start = ctx.all_bpms[0]
        return [WorkerRangeSpec(start, _bpm_behind(ctx.all_bpms, start), sdir=1)]

    def get_range_bpm_names(
        self,
        *,
        all_bpms: list[str],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None,
    ) -> list[str]:
        excluded = set(bad_bpms or [])
        if start_bpm not in all_bpms:
            raise ValueError(f"Kicker tracking start BPM '{start_bpm}' not found in model BPM list")
        start_idx = all_bpms.index(start_bpm)
        marker_order = all_bpms[start_idx:] + all_bpms[:start_idx]
        return [bpm for bpm in marker_order if bpm not in excluded]

@dataclass(frozen=True)
class _AcdPlan(TrackingPlan):
    """Shared AC-dipole marker naming and MAD preparation mode."""

    acd_name: str

    tracking_anchor_mode = "acd"

    @property
    def acd_after(self) -> str:
        return f"{self.acd_name}_after"

    @property
    def acd_before(self) -> str:
        return f"{self.acd_name}_before"


@dataclass(frozen=True)
class ACDTrackingPlan(_AcdPlan):
    """Bidirectional tracking initialised at AC-dipole markers."""

    uses_fixed_bpm_window = False
    start_point_label = "ACD tracking start markers"

    @property
    def extra_markers(self) -> tuple[str, ...]:
        return (self.acd_after, self.acd_before)

    @property
    def observed_tracking_anchor_markers(self) -> tuple[str, ...]:
        return ()

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        return all_bpms

    def range_specs_per_batch(
        self,
        *,
        use_fixed_bpm: bool,
        num_starts: int,
        num_ends: int,
    ) -> tuple[int, str]:
        return 2, "ACD bidirectional (forward + backward)"

    def bpm_pairs(self, ctx: RangeContext) -> list[tuple[str, str]]:
        return [(self.acd_after, self.acd_before)]

    def build_range_specs(self, ctx: RangeContext) -> list[WorkerRangeSpec]:
        return [
            WorkerRangeSpec(start_bpm=self.acd_after, end_bpm=self.acd_before, sdir=sdir)
            for sdir in (1, -1)
        ]

    def initial_condition_marker(self, range_spec: WorkerRangeSpec) -> str:
        """Initialise each direction from the marker at its tracking start."""
        return range_spec.init_bpm

    def get_range_bpm_names(
        self,
        *,
        all_bpms: list[str],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None,
    ) -> list[str]:
        """Observe physical BPMs only; marker rows supply initial conditions."""
        markers = {self.acd_after, self.acd_before}
        return [
            bpm
            for bpm in super().get_range_bpm_names(
                all_bpms=all_bpms,
                start_bpm=start_bpm,
                end_bpm=end_bpm,
                sdir=sdir,
                bad_bpms=bad_bpms,
            )
            if bpm not in markers
        ]

@dataclass(frozen=True)
class ACDArcByArcTrackingPlan(_AcdPlan):
    """Arc-by-arc tracking of AC-dipole data over ordinary BPM ranges.

    Unlike :class:`ACDTrackingPlan` (which tracks bidirectionally from the AC-dipole
    ``before``/``after`` markers), this plan tracks the caller's cartesian product of
    ``bpm_start_points`` x ``bpm_end_points``. The exciter is a driven, turn-varying
    element, so a range that crosses it would compare BPMs on opposite sides of a kick
    the free-oscillation model cannot reproduce.

    Rather than drop such a range, this plan **reroutes** it: two BPMs are joined by two
    arcs around the ring, and only one contains the AC dipole, so a crossing pair keeps
    ``start`` and ``end`` connected via the complementary long-way-round arc (in both
    tracking directions).

    The ``before``/``after`` monitors are installed (``tracking_anchor_mode`` == ``"acd"``)
    only so they appear in ``all_bpms`` and mark the exciter's ring position for the
    crossing test; they are not observed against the measurement.
    """

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        # Keep the exciter markers out of the measurement comparison; they exist only
        # to locate the AC dipole for the crossing test.
        markers = {self.acd_after, self.acd_before}
        return [bpm for bpm in bpms_in_range if bpm not in markers]

    def _range_crosses_acd(self, all_bpms: list[str], start_bpm: str, end_bpm: str) -> bool:
        """Return whether the forward span ``start_bpm`` -> ``end_bpm`` straddles the exciter.

        The ``before``/``after`` monitors sit adjacent in ring order, bracketing the AC
        dipole, so a contiguous forward span crosses the exciter exactly when it contains
        both of them.
        """
        if self.acd_before not in all_bpms or self.acd_after not in all_bpms:
            return False
        span = extract_bpm_range_names(all_bpms, start_bpm, end_bpm, 1)
        return self.acd_before in span and self.acd_after in span

    def _reroute_pair(self, all_bpms: list[str], start_bpm: str, end_bpm: str) -> tuple[str, str]:
        """Return the (start, end) whose forward span avoids the AC dipole.

        If the natural ``start`` -> ``end`` arc crosses the exciter, swap the endpoints so
        the range follows the complementary long-way-round arc instead. The AC dipole lies
        in exactly one of the two arcs joining a pair, so the swapped span never crosses.
        """
        if self._range_crosses_acd(all_bpms, start_bpm, end_bpm):
            return end_bpm, start_bpm
        return start_bpm, end_bpm

    def build_range_specs(self, ctx: RangeContext) -> list[WorkerRangeSpec]:
        specs = super().build_range_specs(ctx)
        rerouted: list[WorkerRangeSpec] = []
        seen: set[tuple[str, str, int]] = set()
        n_rerouted = 0
        for spec in specs:
            start, end = self._reroute_pair(ctx.all_bpms, spec.start_bpm, spec.end_bpm)
            if (start, end) != (spec.start_bpm, spec.end_bpm):
                n_rerouted += 1
            key = (start, end, spec.sdir)
            if key in seen:
                continue
            seen.add(key)
            rerouted.append(WorkerRangeSpec(start_bpm=start, end_bpm=end, sdir=spec.sdir))
        if n_rerouted:
            logger.warning(
                "Rerouted %d ACD-crossing range(s) of %d the long way round the ring; "
                "ranges that would straddle the AC dipole (%s) are tracked via the "
                "complementary arc instead.",
                n_rerouted,
                len(specs),
                self.acd_name,
            )
        return rerouted

    def bpm_pairs(self, ctx: RangeContext) -> list[tuple[str, str]]:
        rerouted: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for start, end in super().bpm_pairs(ctx):
            pair = self._reroute_pair(ctx.all_bpms, start, end)
            if pair in seen:
                continue
            seen.add(pair)
            rerouted.append(pair)
        return rerouted


@dataclass(frozen=True)
class TrackingModeSetup:
    """Resolved tracking mode: the plan plus the inputs it rewrote.

    Each :class:`~aba_optimiser.tracking.fitter.TrackingFitter` entry point
    builds one of these (via the ``*_setup`` helpers below) to fix the parts of the
    simulation config and BPM points that are not free choices for its mode.
    """

    plan: TrackingPlan
    simulation_config: SimulationConfig
    bpm_start_points: list[str]
    bpm_end_points: list[str]
    # Per-file fallback for the BPM the measurement turns are recorded from.
    # Kicker files are written from the kicker marker; ACD and free-oscillation
    # files keep the sequence-file BPM order, so they leave this unset.
    first_bpm_fallback: str | None = None


def arc_by_arc_setup(
    *,
    accelerator: Accelerator,
    simulation_config: SimulationConfig,
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    acd_excited: bool,
) -> TrackingModeSetup:
    """Return the arc-by-arc setup, optionally accounting for the AC dipole.

    Ranges are the caller's ``bpm_start_points`` x ``bpm_end_points`` product. When
    the data is ``acd_excited`` the AC-dipole ``before``/``after`` markers are installed
    and any range that would straddle the exciter is rerouted the long way round the
    ring (:class:`ACDArcByArcTrackingPlan`); otherwise ranges track free oscillations
    directly (:class:`TrackingPlan`).
    """
    if not bpm_start_points:
        raise ValueError("Arc-by-arc mode requires bpm_start_points.")
    if not bpm_end_points:
        raise ValueError("Arc-by-arc mode requires bpm_end_points.")

    # An arc range is a single-turn transport between two BPMs.
    simulation_config = dataclasses.replace(simulation_config, n_run_turns=1)

    if acd_excited:
        plan: TrackingPlan = ACDArcByArcTrackingPlan(acd_name=accelerator.ac_dipole_name)
        logger.info(
            "ACD arc-by-arc mode enabled: %d start x %d end BPMs "
            "(ranges crossing %s are rerouted the long way round)",
            len(bpm_start_points),
            len(bpm_end_points),
            accelerator.ac_dipole_name,
        )
    else:
        plan = TrackingPlan()
        logger.info(
            "Arc-by-arc mode enabled: %d start x %d end BPMs",
            len(bpm_start_points),
            len(bpm_end_points),
        )

    return TrackingModeSetup(
        plan=plan,
        simulation_config=simulation_config,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
    )


def kicker_setup(
    kicker_config: KickerConfig,
    simulation_config: SimulationConfig,
) -> TrackingModeSetup:
    """Return the forward-only setup for kicker measurements.

    Each file holds one kicked track, so ``num_workers`` should be the number of
    files (one worker per momentum); a single file always gets one worker.
    """
    kicker_config.log_state()
    simulation_config = dataclasses.replace(
        simulation_config,
        num_workers=max(1, simulation_config.num_workers),
        num_batches=1,
        n_run_turns=kicker_config.turns_after_kicker,
    )
    logger.info(
        "Kicker mode enabled: start=%s, turns=%d",
        kicker_config.kicker_name,
        kicker_config.turns_after_kicker,
    )
    return TrackingModeSetup(
        plan=KickerTrackingPlan(kicker_name=kicker_config.kicker_name),
        simulation_config=simulation_config,
        bpm_start_points=[kicker_config.kicker_name],
        bpm_end_points=[],
        first_bpm_fallback=kicker_config.kicker_name,
    )


def acd_marker_setup(
    accelerator: Accelerator,
    simulation_config: SimulationConfig,
) -> TrackingModeSetup:
    """Return the bidirectional AC-dipole marker setup.

    Tracking runs both directions from the AC-dipole ``after``/``before`` markers,
    which supply the initial conditions; the whole ring is observed against the
    measurement (:class:`ACDTrackingPlan`).
    """
    acd_after = accelerator.acd_marker_name("after")
    acd_before = accelerator.acd_marker_name("before")
    simulation_config = dataclasses.replace(simulation_config, n_run_turns=1)
    logger.info("ACD marker mode enabled: after=%s, before=%s", acd_after, acd_before)
    return TrackingModeSetup(
        plan=ACDTrackingPlan(acd_name=accelerator.ac_dipole_name),
        simulation_config=simulation_config,
        bpm_start_points=[acd_after, acd_before],
        bpm_end_points=[],
    )
