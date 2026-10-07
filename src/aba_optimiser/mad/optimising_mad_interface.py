"""High-level MAD-NG interfaces used by optimisation and worker code.

The classes in this module build on :mod:`aba_optimiser.mad.aba_mad_interface`
to provide a fully configured MAD-NG session for optimisation workflows. They
handle sequence loading, BPM observation setup, optional machine-state
application, and knob discovery for gradient-based tuning.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import tfs
from pymadng_utils.mad.accelerator_mad_interface import (
    MAGNET_STRENGTH_SUFFIXES,
    MAX_MULTIPOLE,
    MULTIPOLE_ATTRS,
    MultipoleInfo,
)

from aba_optimiser.accelerators import LHC

from .aba_mad_interface import AbaMadInterface
from .machine_state import MachineState, assign_state, resolve_machine_state

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pymadng_utils.accelerators import Accelerator as PyMadAccelerator

    from aba_optimiser.accelerators import Accelerator, KnobSpec

BPM_PATTERN = "^BPM"
LOGGER = logging.getLogger(__name__)

_INDEXED_MULTIPOLE_RE = re.compile(r"^(knl|ksl)\[(\d+)\]$")


def is_magnet_strength_name(name: str) -> bool:
    """True if ``name`` is a settable magnet-strength name (e.g. ``MQ.1.dk1l``).

    These are exactly the names accepted by
    :meth:`AcceleratorMadInterface.set_magnet_strengths`, so callers can use this
    to tell a genuine magnet strength from a stray/typo'd knob name before routing
    values to the model.
    """
    return any(name.endswith(suffix) for suffix in MAGNET_STRENGTH_SUFFIXES)


def indexed_multipole_attr_info(attr: str) -> MultipoleInfo | None:
    """Return multipole metadata for indexed MAD attrs such as ``knl[3]`` or ``ksl[3]``."""
    match = _INDEXED_MULTIPOLE_RE.fullmatch(attr)
    if match is None:
        return None

    base_table, index_str = match.groups()
    index = int(index_str)
    if index < 1 or index > MAX_MULTIPOLE:
        return None

    order = index - 1
    base_attr = f"k{order}" if base_table == "knl" else f"k{order}s"
    return MULTIPOLE_ATTRS[base_attr]


def _ensure_cycleable_start_element(
    iface: GenericMadInterface,
    start_bpm: str,
    observed_bpms: list[str],
) -> None:
    """Replace a non-BPM start element with a same-name marker for MAD cycling."""
    if start_bpm in observed_bpms:
        return
    iface.mad.send(f"""
correct_elm = MADX['{start_bpm}']
local new_elm = MAD.element.marker ('{start_bpm}') {{ at=loaded_sequence:upos(correct_elm) }}
local replaced = loaded_sequence:replace({{new_elm}}, '{start_bpm}')
MADX['{start_bpm}'] = new_elm
{iface.py_name}:send(replaced and #replaced or 0)
correct_elm = nil
    """)
    if iface.mad.recv() != 1:
        raise ValueError(f"Failed to replace start element {start_bpm} with a marker")
    LOGGER.info("Replaced non-BPM start element with cycle marker: %s", start_bpm)


def _unique_names(names: Iterable[str]) -> list[str]:
    """Return names deduplicated in first-seen order."""
    return list(dict.fromkeys(names))


def _absolute_name_from_dk_knob(knob_name: str) -> str | None:
    """Map a dknl/dksl perturbation knob name back to its absolute strength attribute."""
    for abs_attr, mp in MULTIPOLE_ATTRS.items():
        if not mp.is_delta and knob_name.endswith(f".{mp.dk_suffix}"):
            return knob_name[: -(len(mp.dk_suffix) + 1)] + f".{abs_attr}"
    return None


def _deferred_table_helpers() -> str:
    """Lua helpers that defer dknl/dksl tables without resetting current values."""
    values = ",\n        ".join(f"old[{idx}] or 0.0" for idx in range(1, MAX_MULTIPOLE + 1))
    return f"""
local function make_dknl_deferred_knob(e)
    if not MAD.typeid.is_deferred(loaded_sequence[e.name].dknl) then
        local old = loaded_sequence[e.name].dknl or {{}}
        loaded_sequence[e.name].dknl = MAD.typeid.deferred {{
        {values},
        }}
    end
end
local function make_dksl_deferred_knob(e)
    if not MAD.typeid.is_deferred(loaded_sequence[e.name].dksl) then
        local old = loaded_sequence[e.name].dksl or {{}}
        loaded_sequence[e.name].dksl = MAD.typeid.deferred {{
        {values},
        }}
    end
end
"""


def read_b2_error_table(path: Path | str) -> dict[str, float]:
    """Read an OMC3-style ``.errors`` table as an element-name to K1L mapping."""
    table = tfs.read(path, index="NAME")
    return {str(name): float(value) for name, value in table["K1L"].items()}


def apply_b2_errors_to_sequence(
    mad,
    py_name: str,
    b2_errors: Path | None,
    machine_state: MachineState | None,
) -> None:
    """Route a b2 dipole error table into the loaded sequence's dknl[2] slots.

    Works on any MAD interface whose sequence is bound to the ``loaded_sequence``
    global (the reconstruction ``ACDipoleMadDriver`` and the optimisation
    ``GenericMadInterface`` both qualify). The b2 K1L is added to the quadrupole
    perturbation slot (dknl[2]), leaving the dipole slot dknl[1] untouched. b2
    errors shift the machine tunes, so a machine state holding the tune knobs is required
    to restore them.
    """
    if b2_errors is None:
        return
    if machine_state is None:
        raise ValueError(
            "The tune knobs in the machine state are designed to compensate for the known b2 errors."
            "Therefore it makes no sense to apply b2 errors without also applying a machine state."
        )

    b2_table = read_b2_error_table(b2_errors)
    if not b2_table:
        LOGGER.warning("No entries found in b2 error table %s", b2_errors)
        return

    zeros = ", ".join(["0.0"] * MAX_MULTIPOLE)
    mad.send(
        f"""
local b2_errors = {py_name}:recv()
local applied = {{}}
local missing = {{}}

for name, k1l in pairs(b2_errors) do
    local element = loaded_sequence[name]
    if element == nil then
        table.insert(missing, name)
    elseif k1l ~= 0 then
        -- Route the b2 K1L into the dknl perturbation table, leaving the
        -- dipole slot dknl[1] at 0 and adding the quadrupole error to dknl[2].
        if not MAD.typeid.is_deferred(element.dknl) then
            element.dknl = MAD.typeid.deferred {{{zeros}}}
        end
        element.dknl[2] = (element.dknl[2] or 0.0) + k1l
        applied[name] = element.dknl[2]
    end
end

{py_name}:send(applied, true)
{py_name}:send(missing, true)
"""
    )
    mad.send(b2_table)
    applied = mad.recv()
    missing = mad.recv()

    if missing:
        preview = ", ".join(sorted(str(name) for name in missing[:8]))
        raise ValueError(
            f"B2 error table {b2_errors} contains elements not present in the loaded "
            f"sequence: {preview}"
        )

    LOGGER.info("Applied %d b2 dipole error entries from %s", len(applied), b2_errors)


class GenericMadInterface(AbaMadInterface):
    """
    Generic MAD interface for all setup tasks EXCEPT knob creation.

    This interface handles:
    - Loading sequences and setting up beams
    - Observing BPMs and configuring ranges
    - Applying corrector strengths and tune knobs
    - General MAD-NG operations

    Knob creation (for gradient descent) is handled by ``GradientDescentMadInterface``.

    This separation allows non-gradient-descent use cases to use this interface
    without unnecessary knob creation overhead.
    """

    def __init__(
        self,
        accelerator: PyMadAccelerator,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        bad_bpms: list[str] | None = None,
        machine_state: MachineState | None = None,
        b2_errors: Path | None = None,
        py_name: str = "py",
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
        tracking_anchor_mode: str | None = None,
        tracking_anchor_markers: list[str] | None = None,
        observed_tracking_anchor_markers: list[str] | None = None,
    ):
        stdout, redirect_stderr = self._resolve_mad_stdout(mad_logfile, discard_mad_output)

        super().__init__(
            accelerator=accelerator,
            stdout=stdout,
            redirect_stderr=redirect_stderr,
            py_name=py_name,
            debug=debug,
        )

        self.accelerator = accelerator
        self.magnet_range = magnet_range
        self.bpm_range = bpm_range if bpm_range is not None else magnet_range
        self.knob_names: list[str] = []
        self.knob_name_set: set[str] = set()
        self.elem_spos: list[float] = []

        # Set MAD variables for ranges and patterns
        self.mad["magnet_range"] = self.magnet_range
        self.mad["bpm_range"] = self.bpm_range

        anchor_markers = self.prepare_tracking_anchors(
            mode=tracking_anchor_mode,
            marker_names=tracking_anchor_markers,
        )

        self.observe_bpms(bad_bpms=bad_bpms, unobserve_first=True)
        observed_anchor_markers = (
            anchor_markers
            if observed_tracking_anchor_markers is None
            else _unique_names(observed_tracking_anchor_markers)
        )
        for marker in observed_anchor_markers:
            self.observe_element(marker)
        self.bpms_in_range, self.nbpms, self.all_bpms = self.count_bpms(self.bpm_range)
        self.make_all_monitors_thin(list(set(self.all_bpms) - set(anchor_markers)))

        self._apply_b2_errors(b2_errors, machine_state)

        if machine_state is not None:
            self._set_machine_state(machine_state)
        else:
            LOGGER.info("Skipping machine state (not provided)")

    def prepare_tracking_anchors(
        self,
        *,
        mode: str | None,
        marker_names: list[str] | None,
    ) -> list[str]:
        """Prepare marker-anchored tracking modes through one monitor-anchor path.

        Returns the observed anchor markers the caller must keep through the BPM
        observation filter (the ACD before/after monitors; none for kicker mode).
        """
        anchor_markers = _unique_names(marker_names or [])
        if mode == "acd":
            acd_before, acd_after = self.insert_acd_markers()
            LOGGER.info("Installed ACD markers: before=%s, after=%s", acd_before, acd_after)
            anchor_markers = _unique_names([*anchor_markers, acd_before, acd_after])
        elif mode == "kicker":
            if not anchor_markers:
                raise ValueError("Kicker tracking anchor mode requires a marker name")
            for source_name in anchor_markers:
                self.make_element_thin(
                    source_name,
                    marker_name=f"{source_name}_centre",
                    observe_after=False,
                )
            anchor_markers = []
        elif mode is not None:
            raise ValueError(f"Unsupported tracking anchor mode: {mode!r}")
        return anchor_markers

    def cycle_to_start(self, start_marker: str) -> None:
        """Cycle the loaded sequence so it begins at ``start_marker``.

        Cycling is never done implicitly during construction; callers (e.g. a
        worker that must track a range as one contiguous segment) request it
        explicitly. A non-BPM start element is first replaced with a same-name
        marker so MAD can cycle to it.
        """
        _ensure_cycleable_start_element(self, start_marker, self.all_bpms)
        self.cycle_sequence(marker_name=start_marker)
        LOGGER.info("Cycled sequence to start at: %s", start_marker)

    @staticmethod
    def _resolve_mad_stdout(
        mad_logfile: Path | None, discard_mad_output: bool
    ) -> tuple[Path | str | None, bool]:
        """Return (stdout, redirect_stderr) for the MAD process."""
        if mad_logfile is not None:
            if discard_mad_output:
                LOGGER.warning(
                    "MAD logfile specified, but discard_mad_output is True. MAD output will be logged."
                )
            LOGGER.info(f"MAD logfile set to: {mad_logfile.absolute()}")
            return mad_logfile, True
        if discard_mad_output:
            return "/dev/null", True
        return None, False

    def _apply_b2_errors(self, b2_errors: Path | None, machine_state: MachineState | None) -> None:
        """Route a b2 dipole error table into this interface's loaded sequence."""
        if b2_errors is not None and not isinstance(self.accelerator, LHC):
            raise ValueError("b2_errors are only supported for LHC MAD interfaces.")
        apply_b2_errors_to_sequence(self.mad, self.py_name, b2_errors, machine_state)

    def count_bpms(self, bpm_range: str) -> tuple[list[str], int, list[str]]:
        """Count the number of BPM elements in the specified range."""
        all_bpms, bpms_in_range = self.get_bpm_list(bpm_range)
        LOGGER.info(
            "Counted %d BPMs in observation range: %s",
            len(bpms_in_range),
            self._format_range_for_log(bpm_range),
        )
        return bpms_in_range, len(bpms_in_range), all_bpms

    def make_all_monitors_thin(self, monitors: list[str], observe_after: bool = True) -> None:
        """Replace monitor elements with markers in the specified BPM range."""
        self.make_elements_thin(
            monitors, observe_after=observe_after, require_kind="monitor"
        )
        LOGGER.info(
            "Replaced %d monitor BPMs with thin observation markers in range: %s",
            len(monitors),
            self._format_range_for_log(self.bpm_range),
        )

    @staticmethod
    def _format_range_for_log(bpm_range: str) -> str:
        if bpm_range == "$start/$end":
            return "full cycled sequence ($start/$end)"
        return bpm_range

    def _set_machine_state(self, machine_state: MachineState) -> None:
        """Put the model in *machine_state* (see :mod:`aba_optimiser.mad.machine_state`)."""
        state = resolve_machine_state(machine_state)
        assign_state(self.mad, state)
        LOGGER.info("Set %d machine-state entries", len(state))


class GradientDescentMadInterface(GenericMadInterface):
    """
    MAD interface for gradient descent optimisation.

    Extends GenericMadInterface with knob creation capabilities. Accelerator-specific
    behaviour (which knobs to create, naming, MAD preparation) is provided via
    accelerator hooks, keeping this interface generic.
    """

    accelerator: Accelerator  # narrows the parent's PyMadAccelerator type

    def __init__(
        self,
        accelerator: Accelerator,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        bad_bpms: list[str] | None = None,
        machine_state: MachineState | None = None,
        b2_errors: Path | None = None,
        initial_model_values: dict[str, float] | None = None,
        py_name: str = "py",
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
        tracking_anchor_mode: str | None = None,
        tracking_anchor_markers: list[str] | None = None,
        observed_tracking_anchor_markers: list[str] | None = None,
    ):
        super().__init__(
            accelerator,
            magnet_range,
            bpm_range,
            bad_bpms,
            machine_state,
            b2_errors,
            py_name,
            debug,
            mad_logfile,
            discard_mad_output,
            tracking_anchor_mode,
            tracking_anchor_markers,
            observed_tracking_anchor_markers,
        )

        if accelerator.has_any_optimisation():
            self._make_adj_knobs()
        else:
            LOGGER.warning(
                "Gradient descent optimisation interface initialised without any optimisation enabled."
                "\nUse GenericMadInterface if no optimisation is required."
            )

        # Apply after knob creation
        self.apply_initial_model_values(initial_model_values)

    def apply_initial_model_values(self, values: dict[str, float] | None) -> None:
        """Apply a full initial machine-state map without changing the trainable set."""
        if not values:
            return

        unknown = [
            name
            for name in values
            if name != "pt" and not is_magnet_strength_name(name) and name not in self.knob_name_set
        ]
        if unknown:
            raise ValueError(
                "Unknown initial model value names: "
                + ", ".join(sorted(unknown)[:10])
                + ("..." if len(unknown) > 10 else "")
            )

        if "pt" in values:
            self.mad["loaded_sequence['pt']"] = float(values["pt"])

        knob_values = {
            name: float(value)
            for name, value in values.items()
            if name in self.knob_name_set
        }
        if knob_values:
            self.update_knob_values(knob_values)

        magnet_values = {
            name: float(value)
            for name, value in values.items()
            if name != "pt" and name not in self.knob_name_set and is_magnet_strength_name(name)
        }
        if magnet_values:
            self.set_magnet_strengths(magnet_values)

    def get_knob_specs(self) -> list[KnobSpec]:
        """Return the knob specs selected by the accelerator's errors / misalignments."""
        return self.accelerator.get_knob_specs()

    def _build_attr_block(self, attr_conditions: list[tuple[str, str, str]]) -> str:
        """
        Build the MAD-NG Lua block that assigns a deferred knob variable to each matched element.

        Each entry in attr_conditions is (kind, attr, lua_condition). For multipole
        attrs (k1, k1s, dk1l, …) the knob routes through dknl/dksl so the base
        strength is never mutated. For direct attrs (kick, dx, …) the element
        attribute itself becomes a deferred variable.
        """
        lines: list[str] = []

        for kind, attr, condition in attr_conditions:
            spec = self.accelerator.get_mad_attr_spec(kind, attr)
            mp = MULTIPOLE_ATTRS.get(attr) or indexed_multipole_attr_info(attr)

            # The knob variable name in MAD (e.g. "MQXA.1R1.dk1l")
            default_name_expr = (
                f'e.name .. ".{mp.dk_suffix}"' if mp is not None else f'e.name .. ".{attr}"'
            )
            name_expr = spec.get("name_expr", default_name_expr)
            mad_value = spec.get("mad_value", f"e.{attr}")

            tmpl = [
                f"if {condition} then",
                f"    local k_str_name = {name_expr}",
            ]
            if mp is not None:
                tmpl += [
                    f"    make_{mp.dk_table}_deferred_knob(e)",
                    (
                        f"    loaded_sequence[k_str_name] = loaded_sequence[k_str_name] "
                        f"or loaded_sequence[e.name].{mp.dk_table}[{mp.index}] or 0.0"
                    ),
                    f"    loaded_sequence[e.name].{mp.dk_table}[{mp.index}] = \\->loaded_sequence[k_str_name]",
                ]
            else:
                tmpl += [
                    f"    loaded_sequence[k_str_name] = loaded_sequence[k_str_name] or {mad_value}",
                    f"    loaded_sequence[e.name].{attr} = \\->loaded_sequence[k_str_name]",
                ]
            tmpl.append(f"    store_knobs(k_str_name, {mad_value}, '{attr}', s)")
            tmpl.append("end")
            lines.extend(f"    {line}" for line in tmpl)

        return "\n".join(lines) if lines else "        -- no attributes selected"

    def get_absolute_knob_values(self, knob_names: list[str]) -> dict[str, float]:
        """Return underlying absolute strengths for optimisation knob names.

        For ``*.dk0l/*.dk1l/*.dk2l`` (and skew equivalents) this returns the base
        element strength before any perturbation. For direct knobs it returns the
        current value from the sequence.
        """
        absolute_names = [_absolute_name_from_dk_knob(knob) or knob for knob in knob_names]
        return self.get_base_magnet_strengths(absolute_names)

    def _make_adj_knobs(self) -> None:
        """Create deferred-strength knobs in MAD for all elements matching the knob specs."""
        specs = self.get_knob_specs()
        self.accelerator.prepare_mad_for_knob_creation(self, specs)

        attr_block = ""
        if specs:
            attr_conditions = [
                (
                    kind, attr,
                    f'(e.kind == "{kind}" {"and e." + nonzero_attr + " ~=0 " if nonzero_attr else ""}and e.name:match("{pattern}"))',
                )
                for kind, attr, pattern, nonzero_attr, _label in specs
            ]
            attr_block = f"""
local function store_knobs(k_str_name, mad_value, attr, spos)
    if not used[k_str_name] then
        used[k_str_name] = #knob_names + 1  ! deduplicate (e.g. shared PSB QFO knobs)
        counts[k_str_name] = 1
        table.insert(knob_names, k_str_name)
        table.insert(spos_list, spos)
    else
        local count = counts[k_str_name] + 1
        local index = used[k_str_name]
        spos_list[index] = (spos_list[index] * (count - 1) + spos) / count
        counts[k_str_name] = count
    end
end
{_deferred_table_helpers()}
for i, e, s, ds in loaded_sequence:siter(magnet_range) do
{self._build_attr_block(attr_conditions)}
end
"""

        energy_block = ""
        if self.accelerator.optimise_energy:
            energy_block = """
loaded_sequence['pt'] = loaded_sequence['pt'] or 1e-6
table.insert(knob_names, "pt")
"""

        self.mad.send(f"""
local knob_names = {{}}
local spos_list = {{}}
local used = {{}}
local counts = {{}}
{attr_block}
{energy_block}
{self.py_name}:send(knob_names, true)
{self.py_name}:send(spos_list, true)
        """)
        knob_names_all: list[str] = self.mad.recv()
        elem_spos_all: list[float] = self.mad.recv()

        # Optionally restrict to a user-specified subset of knobs
        if self.accelerator.custom_knobs_to_optimise is not None:
            keep = set(self.accelerator.custom_knobs_to_optimise)
            pairs = [(k, s) for k, s in zip(knob_names_all, elem_spos_all) if k in keep]
            knob_names_all, elem_spos_all = (list(x) for x in zip(*pairs)) if pairs else ([], [])
            LOGGER.info(f"Filtered to {len(knob_names_all)} knobs based on custom_knobs_to_optimise")

        self.knob_names = knob_names_all
        self.elem_spos = elem_spos_all
        self.knob_name_set = set(self.knob_names)
        self.mad["knob_names"] = self.knob_names

        if self.elem_spos:
            LOGGER.info(
                f"Created {len(self.knob_names)} knobs from {self.elem_spos[0]} to {self.elem_spos[-1]}"
            )
            LOGGER.debug(f"Knob names: {self.knob_names}")
        else:
            LOGGER.info("No knobs created")

    def _set_multipole_perturbation(
        self,
        element_name: str,
        attr: str,
        info: MultipoleInfo,
        integrated_delta: float,
        target_strength: float,
    ) -> None:
        """Apply a perturbation through an existing optimisation knob when possible."""
        knob_name = f"{element_name}.{info.dk_suffix}"
        if self.accelerator.has_any_optimisation() and knob_name in self.knob_name_set:
            self.mad[f"loaded_sequence['{knob_name}']"] = integrated_delta
            return

        super()._set_multipole_perturbation(
            element_name,
            attr,
            info,
            integrated_delta,
            target_strength,
        )

    def receive_knob_values(self) -> np.ndarray:
        """Retrieve the current values of all knobs from the MAD-NG session."""
        var_names = [f"loaded_sequence['{k}']" for k in self.knob_names]
        values = self.mad.recv_vars(*var_names)
        # recv_vars returns a scalar when only one variable is requested
        if len(self.knob_names) == 1:
            values = [values]
        return np.array(values, dtype=float)

    def update_knob_values(self, knob_values: dict[str, float]) -> None:
        """Update knob values in the MAD-NG session."""
        unknown = [n for n in knob_values if n not in self.knob_name_set]
        if unknown:
            raise ValueError(
                "Unknown knob names supplied to update_knob_values: "
                + ", ".join(unknown[:10])
                + ("..." if len(unknown) > 10 else "")
            )
        commands = [f"loaded_sequence['{n}'] = {v}" for n, v in knob_values.items()]
        if commands:
            self.mad.send("\n".join(commands))
        LOGGER.info(f"Updated {len(commands)} knobs from {len(knob_values)} provided")
