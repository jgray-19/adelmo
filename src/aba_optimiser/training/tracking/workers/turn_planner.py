"""Turn-batch planning for tracking workers.

Batching is intentionally simple: the caller (``DataManager``) has already carved
out the held-out validation turns and applied ``data_fraction`` sampling, so the
turns handed to the planner are exactly the training turns to be used. The planner
only has to spread those turns evenly across the batches implied by ``num_workers``
(one batch per turn-group, later fanned out over range specs). Every batch comes
from a single measurement file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from aba_optimiser.training.config.tracking import TrackingPlan

    ShuffleTurns = Callable[[list[int]], None]

LOGGER = logging.getLogger(__name__)


def group_turns_by_file(
    available_turns: list[int], file_map: dict[int, int]
) -> dict[int, list[int]]:
    """Group available turns by measurement file index."""
    turns_by_file: dict[int, list[int]] = {}
    for turn in available_turns:
        turns_by_file.setdefault(file_map[turn], []).append(turn)
    return turns_by_file


def allocate_batches_per_file(
    turns_by_file: dict[int, list[int]], num_turn_batches: int
) -> dict[int, int]:
    """Split ``num_turn_batches`` batches across files, keeping >=1 turn per batch.

    Batches are handed out one at a time to whichever file currently has the most
    turns per already-assigned batch, so files end up with batch counts roughly
    proportional to how much data they contribute while never exceeding one batch
    per turn (a batch must contain at least one turn).
    """
    file_ids = sorted(turns_by_file)
    counts = dict.fromkeys(file_ids, 0)
    capacity = sum(len(turns_by_file[f]) for f in file_ids)
    remaining = min(max(0, num_turn_batches), capacity)

    while remaining > 0:
        # Prefer the file with the largest turns-per-(batch+1); ties broken by id.
        candidates = [f for f in file_ids if counts[f] < len(turns_by_file[f])]
        if not candidates:
            break
        chosen = max(candidates, key=lambda f: (len(turns_by_file[f]) / (counts[f] + 1), -f))
        counts[chosen] += 1
        remaining -= 1

    return {f: c for f, c in counts.items() if c > 0}


def _split_even(turns: list[int], num_chunks: int) -> list[list[int]]:
    """Split ``turns`` into ``num_chunks`` contiguous, near-even chunks."""
    base, remainder = divmod(len(turns), num_chunks)
    chunks: list[list[int]] = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(turns[start : start + size])
        start += size
    return chunks


@dataclass(frozen=True)
class WorkerTurnBatchPlan:
    """Concrete turn-batch plan plus logging metadata."""

    turn_batches: list[list[int]]
    range_specs_per_batch: int
    range_specs_desc: str


class WorkerTurnPlanner:
    """Compute worker turn batches from the training turns and tracking mode."""

    def __init__(
        self,
        tracking_plan: TrackingPlan,
        simulation_config,
        *,
        shuffle_turns: ShuffleTurns,
    ) -> None:
        """Create a planner.

        Args:
            tracking_plan: Tracking-mode policy for BPM/range expansion.
            simulation_config: Worker and batching configuration.
            shuffle_turns: In-place turn ordering.
        """
        self.tracking_plan = tracking_plan
        self.simulation_config = simulation_config
        self.shuffle_turns = shuffle_turns

    def build_turn_batches(
        self,
        *,
        available_turns: list[int],
        file_map: dict[int, int],
        num_files: int,
        num_starts: int,
        num_ends: int,
    ) -> WorkerTurnBatchPlan:
        """Plan batches from the (already sampled) training turns and log the result."""
        turns_by_file = group_turns_by_file(available_turns, file_map)
        num_workers = self.simulation_config.num_workers
        range_specs_per_batch, range_specs_desc = self.tracking_plan.range_specs_per_batch(
            use_fixed_bpm=self.simulation_config.use_fixed_bpm,
            num_starts=num_starts,
            num_ends=num_ends,
        )

        # One turn batch is fanned out over ``range_specs_per_batch`` workers, so the
        # number of turn batches that realises ``num_workers`` workers is the ratio.
        # A batch must hold at least one turn, so the number of batches cannot
        # exceed the number of available training turns.
        worker_turn_batches = max(1, num_workers // max(1, range_specs_per_batch))
        total_turns = sum(len(turns) for turns in turns_by_file.values())
        num_turn_batches = min(worker_turn_batches, total_turns)
        # Every batch comes from one file, so fewer batches than files would leave
        # whole files untrained (lowest file ids win). Never drop data to honour the
        # worker cap: give every file with training turns at least one batch.
        if num_turn_batches < len(turns_by_file):
            LOGGER.warning(
                "num_workers=%d gives %d turn batches for %d files with training turns; "
                "raising to %d batches (%d workers) so that every file is trained",
                num_workers,
                num_turn_batches,
                len(turns_by_file),
                len(turns_by_file),
                len(turns_by_file) * range_specs_per_batch,
            )
            num_turn_batches = len(turns_by_file)

        LOGGER.info(
            "Worker planning: requested=%d workers, range_specs_per_batch=%d (%s), starts=%d, ends=%d",
            num_workers,
            range_specs_per_batch,
            range_specs_desc,
            num_starts,
            num_ends,
        )
        limiting_factor = (
            "training-turn count" if total_turns < worker_turn_batches else "num_workers cap"
        )
        LOGGER.info(
            "Turn-batch planning: num_workers_cap->%d batches, training turns=%d across %d files, "
            "selected=%d batches [limited by %s]",
            worker_turn_batches,
            total_turns,
            num_files,
            num_turn_batches,
            limiting_factor,
        )
        LOGGER.info(
            "Planned workers: %d turn batches x %d range specs = %d workers "
            "(num_batches=%d is MAD-internal sub-batching, does not affect worker count)",
            num_turn_batches,
            range_specs_per_batch,
            num_turn_batches * range_specs_per_batch,
            self.simulation_config.num_batches,
        )

        turn_batches = self._materialise_turn_batches(turns_by_file, num_turn_batches)

        return WorkerTurnBatchPlan(
            turn_batches=turn_batches,
            range_specs_per_batch=range_specs_per_batch,
            range_specs_desc=range_specs_desc,
        )

    def _materialise_turn_batches(
        self,
        turns_by_file: dict[int, list[int]],
        num_turn_batches: int,
    ) -> list[list[int]]:
        """Split each file's turns into its allotted number of near-even batches."""
        if num_turn_batches <= 0:
            return []

        for turns in turns_by_file.values():
            self.shuffle_turns(turns)

        batches_per_file = allocate_batches_per_file(turns_by_file, num_turn_batches)
        mad_num_batches = self.simulation_config.num_batches

        turn_batches: list[list[int]] = []
        for file_idx in sorted(batches_per_file):
            for chunk in _split_even(turns_by_file[file_idx], batches_per_file[file_idx]):
                # MAD sub-batches each worker's turns into ``num_batches`` groups; trim
                # to a multiple so the split is even (a no-op when the chunk is small).
                if len(chunk) >= mad_num_batches:
                    chunk = chunk[: (len(chunk) // mad_num_batches) * mad_num_batches]
                if chunk:
                    turn_batches.append(chunk)

        if len(turn_batches) < num_turn_batches:
            LOGGER.warning(
                "Created %d/%d worker turn batches (limited by available training turns)",
                len(turn_batches),
                num_turn_batches,
            )
        return turn_batches
