"""Accelerator definitions for the supported machines (LHC, PSB, SPS, FCC).

Each class carries the machine-specific knob specifications, BPM pattern and
tune configuration the MAD-NG interface needs.
"""

from adelmo.machine.accelerators.base import (
    MISALIGNMENT_ATTRS,
    Accelerator,
    KnobSpec,
    MagnetFamily,
)
from adelmo.machine.accelerators.fcc import FCC
from adelmo.machine.accelerators.lhc import LHC
from adelmo.machine.accelerators.psb import PSB
from adelmo.machine.accelerators.sps import SPS

__all__ = ["MISALIGNMENT_ATTRS", "Accelerator", "KnobSpec", "MagnetFamily", "FCC", "LHC", "PSB", "SPS"]
