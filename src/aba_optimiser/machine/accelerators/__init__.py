"""Accelerator definitions for the supported machines (LHC, PSB, SPS, FCC).

Each class carries the machine-specific knob specifications, BPM pattern and
tune configuration the MAD-NG interface needs.
"""

from aba_optimiser.machine.accelerators.base import (
    MISALIGNMENT_ATTRS,
    Accelerator,
    KnobSpec,
    MagnetFamily,
)
from aba_optimiser.machine.accelerators.fcc import FCC
from aba_optimiser.machine.accelerators.lhc import LHC
from aba_optimiser.machine.accelerators.psb import PSB
from aba_optimiser.machine.accelerators.sps import SPS

__all__ = ["MISALIGNMENT_ATTRS", "Accelerator", "KnobSpec", "MagnetFamily", "FCC", "LHC", "PSB", "SPS"]
