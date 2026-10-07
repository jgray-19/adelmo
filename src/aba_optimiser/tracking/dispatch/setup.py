"""Worker planning: which workers exist for a set of BPM ranges and files.

LHC and PSB BPMs measure both planes, so workers follow the data alone: a file
kicked in one plane gives single-plane workers. On a machine with single-plane
BPMs (SPS ``BPH``/``BPV``) each worker additionally observes only the BPMs that
see its plane, ranges are planned per plane, and dual-plane data is split into
separate x and y workers.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aba_optimiser.fitting.worker import KickPlane, WorkerConfig
from aba_optimiser.tracking.config.tracking import RangeContext, WorkerRangeSpec

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.machine.accelerators import Accelerator
    from aba_optimiser.tracking.config.tracking import TrackingPlan


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerObservationPlan:
    """What one worker observes, for one range and one measurement file."""

    range_spec: WorkerRangeSpec
    file_idx: int
    kick_plane: KickPlane
    bpm_names: list[str]
    bad_bpms: list[str] | None
    init_marker: str | None = None

    @property
    def init_bpm(self) -> str:
        """Return the marker used to initialise tracking for this plan."""
        return self.init_marker or self.range_spec.init_bpm


@dataclass(frozen=True)
class WorkerRuntimeMetadata:
    """Fitter-side metadata retained for screening and diagnostics."""

    worker_id: int
    file_idx: int
    start_bpm: str
    end_bpm: str
    sdir: int
    kick_plane: KickPlane
    n_run_turns: int
    bpm_names: list[str]


class WorkerSetupHelper:
    """Build worker ranges, observation plans, and worker configs."""

    def __init__(
        self,
        accelerator: Accelerator,
        all_bpms: list[str],
        fixed_start: str,
        fixed_end: str,
        use_fixed_bpm: bool,
        bad_bpms: list[str] | None,
        file_kick_planes: dict[int, str | KickPlane],
        magnet_range: str,
        interface_options_per_file: list[dict],
        debug: bool,
        mad_logfile: Path | None,
        python_logfile: Path | None,
        tracking_plan: TrackingPlan,
    ) -> None:
        self.accelerator = accelerator
        self.all_bpms = all_bpms
        self.fixed_start = fixed_start
        self.fixed_end = fixed_end
        self.use_fixed_bpm = use_fixed_bpm
        self.bad_bpms = bad_bpms
        self.file_kick_planes = file_kick_planes
        self.magnet_range = magnet_range
        self.interface_options_per_file = interface_options_per_file
        self.debug = debug
        self.mad_logfile = mad_logfile
        self.python_logfile = python_logfile
        self.tracking_plan = tracking_plan
        self.single_plane_bpms = not all(self.measures(bpm, KickPlane.XY) for bpm in all_bpms)

    @staticmethod
    def merge_bad_bpms(*bad_bpm_lists: list[str] | None) -> list[str] | None:
        """Merge bad-BPM lists while preserving the first occurrence order."""
        merged: list[str] = []
        for bpm_list in bad_bpm_lists:
            for bpm in bpm_list or []:
                if bpm not in merged:
                    merged.append(bpm)
        return merged or None

    def measures(self, bpm: str, plane: KickPlane) -> bool:
        """Return whether ``bpm`` measures ``plane`` (``XY`` means both planes)."""
        # Tracking markers carry reconstructed coordinates in both planes.
        if bpm in self.tracking_plan.extra_markers:
            return True
        monitor = self.accelerator.infer_monitor_plane(bpm)
        if plane == KickPlane.X:
            return "H" in monitor
        if plane == KickPlane.Y:
            return "V" in monitor
        return "H" in monitor and "V" in monitor

    def get_range_bpm_names(
        self,
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None = None,
    ) -> list[str]:
        """Return the BPMs a range observes, in tracking order."""
        return self.tracking_plan.get_range_bpm_names(
            all_bpms=self.all_bpms,
            start_bpm=start_bpm,
            end_bpm=end_bpm,
            sdir=sdir,
            bad_bpms=bad_bpms,
        )

    def build_range_specs(self, start_bpms: list[str], end_bpms: list[str]) -> list[WorkerRangeSpec]:
        """Return the logical worker ranges for this tracking plan."""
        ctx = RangeContext(
            start_bpms=start_bpms,
            end_bpms=end_bpms,
            all_bpms=self.all_bpms,
            use_fixed_bpm=self.use_fixed_bpm,
            fixed_start=self.fixed_start,
            fixed_end=self.fixed_end,
        )
        # Marker-anchored plans (ACD markers, kicker) start on markers that carry both
        # planes; build_observation_plans splits their workers by plane instead.
        if not self.single_plane_bpms or set(start_bpms) <= set(self.tracking_plan.extra_markers):
            return self.tracking_plan.build_range_specs(ctx)

        # A range must start and end on BPMs of the plane it tracks.
        specs: list[WorkerRangeSpec] = []
        for plane in (KickPlane.X, KickPlane.Y):
            plane_bpms = [bpm for bpm in self.all_bpms if self.measures(bpm, plane)]
            starts = [bpm for bpm in start_bpms if bpm in plane_bpms]
            ends = [bpm for bpm in end_bpms if bpm in plane_bpms]
            if not starts or not ends:
                LOGGER.warning("No %s-plane start and end BPMs; no %s-plane ranges", plane.value, plane.value)
                continue
            plane_ctx = dataclasses.replace(
                ctx,
                start_bpms=starts,
                end_bpms=ends,
                all_bpms=plane_bpms,
                fixed_start=starts[0] if self.use_fixed_bpm else self.fixed_start,
                fixed_end=ends[0] if self.use_fixed_bpm else self.fixed_end,
            )
            specs.extend(self.tracking_plan.build_range_specs(plane_ctx))
        return specs

    def build_observation_plans(
        self,
        range_spec: WorkerRangeSpec,
        file_idx: int,
        available_bpms: set[str],
    ) -> list[WorkerObservationPlan]:
        """Return the worker plan(s) for one range and measurement file."""
        data_plane = KickPlane(self.file_kick_planes.get(file_idx, KickPlane.XY))
        if data_plane == KickPlane.XY and self.single_plane_bpms:
            planes = (KickPlane.X, KickPlane.Y)
        else:
            planes = (data_plane,)
        plans = (self._build_plan(range_spec, file_idx, plane, available_bpms) for plane in planes)
        return [plan for plan in plans if plan is not None]

    def _build_plan(
        self,
        range_spec: WorkerRangeSpec,
        file_idx: int,
        plane: KickPlane,
        available_bpms: set[str],
    ) -> WorkerObservationPlan | None:
        """Build one worker's plan, or ``None`` when the range is unusable.

        BPMs the worker cannot use are added to its bad-BPM list so MAD stops
        observing them; if the initial-condition marker is missing from the file
        there is nothing to track from and the worker is dropped.
        """
        bad_bpms = self.bad_bpms
        if self.single_plane_bpms and plane != KickPlane.XY:
            # Every blind BPM in the ring, not only those in the range: MAD's
            # observation flags are set on the whole sequence.
            blind = [bpm for bpm in self.all_bpms if not self.measures(bpm, plane)]
            bad_bpms = self.merge_bad_bpms(bad_bpms, blind)

        bpm_names = self.get_range_bpm_names(
            range_spec.start_bpm, range_spec.end_bpm, range_spec.sdir, bad_bpms
        )
        init_marker = self.tracking_plan.initial_condition_marker(range_spec)

        missing = [bpm for bpm in bpm_names if bpm not in available_bpms]
        if missing:
            LOGGER.warning(
                "File %d range %s/%s sdir=%d: %d BPMs missing from measurement data; "
                "adding to bad BPMs",
                file_idx,
                range_spec.start_bpm,
                range_spec.end_bpm,
                range_spec.sdir,
                len(missing),
            )
            bad_bpms = self.merge_bad_bpms(bad_bpms, missing)
            bpm_names = [bpm for bpm in bpm_names if bpm in available_bpms]
        if init_marker is not None and init_marker not in available_bpms:
            LOGGER.warning(
                "File %d range %s/%s sdir=%d: init marker %s missing from measurement data",
                file_idx,
                range_spec.start_bpm,
                range_spec.end_bpm,
                range_spec.sdir,
                init_marker,
            )
            return None

        if not bpm_names:
            return None
        if init_marker is None and range_spec.init_bpm not in bpm_names:
            return None

        return WorkerObservationPlan(
            range_spec=range_spec,
            file_idx=file_idx,
            kick_plane=plane,
            bpm_names=bpm_names,
            bad_bpms=bad_bpms,
            init_marker=init_marker,
        )

    def make_worker_config(self, plan: WorkerObservationPlan) -> WorkerConfig:
        """Build the worker configuration object for one plan."""
        return WorkerConfig(
            accelerator=self.accelerator,
            tracking_start_bpm=plan.range_spec.start_bpm,
            tracking_end_bpm=plan.range_spec.end_bpm,
            magnet_range=self.magnet_range,
            interface_options=self.interface_options_per_file[plan.file_idx],
            initial_condition_marker=plan.init_marker,
            sdir=plan.range_spec.sdir,
            kick_plane=plan.kick_plane,
            bad_bpms=plan.bad_bpms,
            debug=self.debug,
            mad_logfile=self.mad_logfile,
            python_logfile=self.python_logfile,
            tracking_anchor_mode=self.tracking_plan.tracking_anchor_mode,
            tracking_anchor_sources=list(self.tracking_plan.tracking_anchor_sources),
            observed_tracking_anchor_markers=list(
                self.tracking_plan.observed_tracking_anchor_markers
            ),
            cycle_marker=self.tracking_plan.cycle_marker,
        )

    @staticmethod
    def make_runtime_metadata(
        worker_id: int,
        file_idx: int,
        config: WorkerConfig,
        bpm_names: list[str],
        n_run_turns: int,
    ) -> WorkerRuntimeMetadata:
        """Return the metadata needed after a worker has started."""
        return WorkerRuntimeMetadata(
            worker_id=worker_id,
            file_idx=file_idx,
            start_bpm=config.tracking_start_bpm,
            end_bpm=config.tracking_end_bpm,
            sdir=config.sdir,
            kick_plane=config.kick_plane,
            n_run_turns=n_run_turns,
            bpm_names=bpm_names,
        )
