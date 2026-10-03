"""Reference orbits computed once and shared with every closed-orbit worker through shared memory.

Every reference-subtracted closed-orbit series needs the orbit (and its knob Jacobian) of the *reference* machine state:
all correctors at nominal, the current magnet knobs. That state is the same for every series, so a dedicated reference
worker solves it once per iteration, in parallel with the other workers' signal solves, and publishes it here. The
signal workers map the block instead of each solving the same reference (and sending 90 MB through a pipe).

Block layout: float64 ``(len(reference_pts), len(coords), n_bpms, 1 + n_knobs)``; ``[..., 0]`` is the orbit and
``[..., 1:]`` its Jacobian with respect to the knobs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory

import numpy as np


@dataclass(frozen=True)
class SharedReference:
    """Picklable description of a published block (what the fitter relays to the signal workers)."""

    name: str
    reference_pts: tuple[float, ...]
    coords: tuple[str, ...]
    shape: tuple[int, int, int, int]

    def rows(self, wanted: list[str]) -> slice | list[int]:
        """Rows of the coordinate axis holding ``wanted`` coordinates (a slice when they are all of them, in order)."""
        if tuple(wanted) == self.coords:
            return slice(None)
        return [self.coords.index(coord) for coord in wanted]


class ReferencePublisher:
    """Writer side: owns the shared-memory block and rewrites it every iteration."""

    def __init__(self) -> None:
        self._block: shared_memory.SharedMemory | None = None
        self._shape: tuple[int, ...] | None = None

    def publish(
        self,
        states: list[tuple[np.ndarray, np.ndarray]],
        reference_pts: tuple[float, ...],
        coords: tuple[str, ...],
    ) -> SharedReference:
        """Write one ``(orbit (n_coords, n_bpms), jacobian (n_coords, n_bpms, n_knobs))`` per reference momentum."""
        n_coords, n_bpms, n_knobs = states[0][1].shape
        shape = (len(states), n_coords, n_bpms, 1 + n_knobs)
        if self._block is None:
            self._block = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 8)
            self._shape = shape
        elif shape != self._shape:
            raise ValueError(f"Reference block shape changed from {self._shape} to {shape}")
        block = np.ndarray(shape, dtype=np.float64, buffer=self._block.buf)
        for index, (orbit, jacobian) in enumerate(states):
            block[index, ..., 0] = orbit
            block[index, ..., 1:] = jacobian
        return SharedReference(self._block.name, tuple(reference_pts), tuple(coords), shape)

    def close(self) -> None:
        """Release and delete the block (readers must be done with it)."""
        if self._block is not None:
            self._block.close()
            self._block.unlink()
            self._block = None


def _attach(name: str) -> shared_memory.SharedMemory:
    """Map an existing block without making this process responsible for deleting it.

    Before Python 3.13 attaching registers the block with the resource tracker, which then warns about a
    "leaked" block when the reader exits although the publisher has already unlinked it.
    """
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    block = shared_memory.SharedMemory(name=name)
    resource_tracker.unregister(block._name, "shared_memory")  # noqa: SLF001
    return block


class ReferenceReader:
    """Reader side: maps the block named by a :class:`SharedReference` (once) and returns views of it."""

    def __init__(self) -> None:
        self._block: shared_memory.SharedMemory | None = None

    def view(self, reference: SharedReference) -> np.ndarray:
        if self._block is None or self._block.name != reference.name:
            self.close()
            self._block = _attach(reference.name)
        return np.ndarray(reference.shape, dtype=np.float64, buffer=self._block.buf)

    def close(self) -> None:
        if self._block is not None:
            self._block.close()
            self._block = None
