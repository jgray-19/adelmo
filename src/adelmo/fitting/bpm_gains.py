"""BPM gains, shared by the closed-orbit (poco) and tracking fits.

A gain ``g`` scales what a BPM reports: ``reading = (1 + g) * position``. Only
positions are read by a BPM; momenta are never scaled. Gains travel in the knob
dict next to the magnet knobs as ``bpmgain.<x|y>.<BPM>``.
"""

from __future__ import annotations

import numpy as np

PREFIX = "bpmgain"
PLANES = ("x", "y")


def bpm_gain_name(plane: str, bpm: str) -> str:
    return f"{PREFIX}.{plane}.{bpm}"


def bpm_gain_names(bpms: list[str] | tuple[str, ...]) -> list[str]:
    """Every x gain, then every y gain, in ``bpms`` order."""
    return [bpm_gain_name(plane, bpm) for plane in PLANES for bpm in bpms]


def is_bpm_gain(name: str) -> bool:
    return name.startswith(f"{PREFIX}.")


def without_bpm_gains(knobs: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in knobs.items() if not is_bpm_gain(k)}


def only_bpm_gains(knobs: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in knobs.items() if is_bpm_gain(k)}


def bpm_gains(knobs: dict[str, float], plane: str, bpms: list[str] | tuple[str, ...]) -> np.ndarray:
    """The ``plane`` gain of each of ``bpms`` from ``knobs``; 0 where a gain is absent."""
    return np.array([knobs.get(bpm_gain_name(plane, bpm), 0.0) for bpm in bpms])


def bpm_reading(position: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """What a BPM with ``gains`` reports for ``position`` (gains broadcast on the last axis)."""
    return (1.0 + gains) * position


def point_gain_index(point_names: list[str], gain_bpms: tuple[str, ...]) -> np.ndarray:
    """Index of each observation point's BPM in ``gain_bpms``; -1 for points with no gain."""
    index = {bpm: i for i, bpm in enumerate(gain_bpms)}
    return np.array([index.get(name, -1) for name in point_names], dtype=np.int64)
