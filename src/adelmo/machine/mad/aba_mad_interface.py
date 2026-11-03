"""Repository-specific MAD-NG interface extensions.

This module builds on shared classes from ``pymadng-utils``:
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymadng_utils.mad.knob_mad_interface import KnobMadInterface

if TYPE_CHECKING:
    from pathlib import Path

    from pymadng_utils.accelerators import Accelerator

logger = logging.getLogger(__name__)

__all__ = [
    "AbaMadInterface",
]


class AbaMadInterface(KnobMadInterface):
    """Repository-local extension of ``KnobMadInterface`` with helper utilities.

    The magnet-strength setters/getters (``set_magnet_strengths``,
    ``get_magnet_strengths``, ``get_base_magnet_strengths`` and their dknl/dksl
    helpers) are inherited from :class:`AcceleratorMadInterface`; only the
    accelerator-specific BPM/observation and momentum-conversion helpers remain here.
    """

    def __init__(self, accelerator: Accelerator, **kwargs):
        super().__init__(accelerator=accelerator, **kwargs)
        self.accelerator = accelerator

    def observe_bpms(
        self,
        bpm_pattern: str | None = None,
        bad_bpms: list[str] | None = None,
        unobserve_first: bool = True,
    ) -> None:
        """Set up the MAD-NG session to observe BPMs."""
        if bpm_pattern is None:
            bpm_pattern = self.accelerator.bpm_pattern
        super().observe_bpms(bpm_pattern, bad_bpms, unobserve_first)

    def get_bpm_list(self, bpm_range: str) -> tuple[list[str], list[str]]:
        """Get list of observed BPM names in the sequence and in a range."""
        logger.debug(f"Getting BPM list for range: {bpm_range}")

        get_bpms_mad = f"""
        local all_bpms = {{}}
        local bpm_in_range = {{}}
        for _, elm in loaded_sequence:iter() do
            if elm:is_observed() then
                table.insert(all_bpms, elm.name)
            end
        end
        for _, elm in loaded_sequence:iter("{bpm_range}") do
            if elm:is_observed() then
                table.insert(bpm_in_range, elm.name)
            end
        end
        {self.py_name}:send(all_bpms, true)
        {self.py_name}:send(bpm_in_range, true)
        """
        self.mad.send(get_bpms_mad)
        all_bpms = self.mad.receive()
        bpms_in_range = self.mad.receive()

        # Remove duplicate if the first and last BPM are the same physical element.
        if all_bpms[0] == all_bpms[-1] and len(all_bpms) > 1:
            all_bpms = all_bpms[:-1]
        bpms_in_range = [bpm for bpm in all_bpms if bpm in bpms_in_range]
        logger.debug(f"Found {len(bpms_in_range)} BPMs in range {bpm_range}")
        return all_bpms, bpms_in_range

    def _clear_disabled_elements(self) -> None:
        """Clear the MAD disabled flag on every element in the loaded sequence."""
        self.mad.send("loaded_sequence:deselect(MAD.element.flags.disabled)")

    def _enable_only_named_elements(self, names: list[str], element_class: str) -> None:
        """Disable an element class, then re-enable the provided element-name table."""
        self.mad.send(
            f"""
local disabled_flag = MAD.element.flags.disabled
loaded_sequence:select(disabled_flag, nil, {{class={element_class}}}) ! Disable all elements of the given class
loaded_sequence:deselect(disabled_flag, nil, {self.py_name}:recv())   ! Re-enable only the provided element names
{self.py_name}:send("Complete")
"""
        )
        self.mad.send(names)
        self._check_mad_response("Complete", "Error applying orbit-correction element filter")

    def _apply_orbit_correction_filters(
        self,
        bpms: list[str] | None,
        correctors: list[str] | None,
    ) -> bool:
        """Apply optional orbit-correction BPM/corrector filters."""
        if bpms is None and correctors is None:
            return False

        self._clear_disabled_elements()
        if bpms is not None:
            self._enable_only_named_elements(bpms, "MAD.element.monitor")
        if correctors is not None:
            self._enable_only_named_elements(correctors, "MAD.element.kicker")
        return True

    def _run_orbit_correction_mad(self, twiss_name: str) -> None:
        """Run MAD-NG orbit correction using already prepared sequence flags."""
        self.mad.send(
            rf"""
local correct, option in MAD

io.write("*** orbit correction using off momentum twiss\n")
local tws_offmom = twiss {{ sequence=loaded_sequence, deltap=machine_deltap }}

local fmt = option.numfmt ; option.numfmt = "% -.16e"
local tbl = correct {{ sequence=loaded_sequence, model=tws_offmom, target={twiss_name}, method="svd", info=1, plane=correction_plane }}
if correct_file then
    tbl:write(correct_file)
end
option.numfmt = fmt

{self.py_name}:send("Complete")
"""
        )
        self._check_mad_response("Complete", "Error during MAD-NG orbit correction")

    def perform_orbit_correction(
        self,
        machine_deltap: float,
        target_qx: float | None = None,
        target_qy: float | None = None,
        corrector_file: Path | None = None,
        twiss_name: str = "zero_twiss",
        correct_tunes: bool = True,
        bpms: list[str] | None = None,
        correctors: list[str] | None = None,
        plane: str = "x",
    ) -> dict[str, float]:
        """Perform orbit correction with optional BPM/corrector name filtering."""
        qx_knob, qy_knob = self.accelerator.tune_variables
        if plane not in {"x", "y", "xy"}:
            raise ValueError(f"Invalid orbit correction plane {plane!r}; expected 'x', 'y' or 'xy'")
        self.mad["machine_deltap"] = machine_deltap
        self.mad["correct_file"] = str(corrector_file.absolute()) if corrector_file else None
        self.mad["correction_plane"] = plane

        filters_applied = self._apply_orbit_correction_filters(bpms, correctors)
        try:
            self._run_orbit_correction_mad(twiss_name)
        finally:
            if filters_applied:
                self._clear_disabled_elements()

        if correct_tunes:
            if target_qx is None or target_qy is None:
                raise ValueError(
                    "Orbit has been corrected, but target tunes are not provided for matching, so MAD-NG cannot match tunes."
                )
            return self.match_tunes(target_qx, target_qy, deltap=machine_deltap)
        return {
            qx_knob: self.mad[f"MADX['{qx_knob}']"],
            qy_knob: self.mad[f"MADX['{qy_knob}']"],
        }
