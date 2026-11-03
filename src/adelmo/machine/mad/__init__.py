"""
MAD-NG interface modules for accelerator simulation.

This package provides different interfaces for working with MAD-NG:
- AbaMadInterface: Repository-specific helper extensions
- GenericMadInterface: Generic setup functionality
- GradientDescentMadInterface: Generic gradient-descent interface
"""

from .aba_mad_interface import AbaMadInterface
from .machine_state import merge_machine_states
from .optimising_mad_interface import GenericMadInterface, GradientDescentMadInterface

__all__ = [
    "AbaMadInterface",
    "GenericMadInterface",
    "GradientDescentMadInterface",
    "merge_machine_states",
]
