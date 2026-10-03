"""Helpers for loading or generating MAD-NG tracking scripts."""

from __future__ import annotations

import logging
from pathlib import Path

mad_dir = Path(__file__).absolute().parent
MAD_SCRIPTS_DIR = mad_dir / "mad_scripts"

CLOSED_TWISS_INIT = MAD_SCRIPTS_DIR / "run_closed_twiss_init.mad"

TRACKING_OBSERVABLES = ("x", "y", "px", "py")
LOGGER = logging.getLogger(__name__)
TAB = "\t"
PYTHON_IN_MAD = "python"


def dump_debug_script(
    script_name: str,
    text: str,
    *,
    debug: bool,
    mad_logfile: Path | None = None,
    worker_id: int | None = None,
) -> Path | None:
    """Write a generated MAD script to disk when debug mode is enabled."""
    if not debug:
        return None

    base_dir = (
        mad_logfile.parent / "generated_mad_scripts"
        if mad_logfile is not None
        else Path.cwd() / "generated_mad_scripts"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_worker_{worker_id}" if worker_id is not None else ""
    output_path = base_dir / f"{script_name}{suffix}.mad"
    output_path.write_text(text)
    LOGGER.debug("Wrote generated MAD script to %s", output_path)
    return output_path


def _validate_observables(observables: tuple[str, ...]) -> tuple[str, ...]:
    invalid = set(observables) - set(TRACKING_OBSERVABLES)
    if invalid:
        raise ValueError(f"Unsupported tracking observables: {sorted(invalid)}")
    if not observables:
        raise ValueError("At least one observable is required")
    return observables


def _join_lines(lines: list[str]) -> str:
    """Join script lines while keeping the assembly logic easy to read."""
    return "\n".join(lines)


def _indent(level: int, text: str) -> str:
    """Indent one generated Lua line with tabs."""
    return f"{TAB * level}{text}"


def _per_observable(observables: tuple[str, ...], template: str, *, level: int = 0) -> list[str]:
    """Render ``template`` once per observable as indented Lua lines.

    ``template`` is a normal ``str.format`` template using ``{o}`` for the
    observable name, e.g. ``"{o}[i]:zeros()"``.
    """
    return [_indent(level, template.format(o=observable)) for observable in observables]


def _table_definitions(observables: tuple[str, ...], *, include_derivatives: bool = True) -> str:
    lines = _per_observable(observables, "{o} = table.new(batch_size, 0)")
    if include_derivatives:
        lines += _per_observable(observables, "d{o}_dk = table.new(batch_size, 0)")
        lines += _per_observable(observables, "d{o}_dc0 = table.new(batch_size, 0)")
    return _join_lines(lines)


def _allocation_block(observables: tuple[str, ...], *, include_derivatives: bool = True) -> str:
    lines = _per_observable(observables, "{o}[i] = vector(nbpms * n_run_turns)", level=1)
    if include_derivatives:
        lines += _per_observable(
            observables, "d{o}_dk[i] = matrix(matrix_size, nbpms * n_run_turns)", level=1
        )
        lines += _per_observable(
            observables, "d{o}_dc0[i] = matrix(2, nbpms * n_run_turns)", level=1
        )
    return _join_lines(lines)


def _save_scalar_block(observables: tuple[str, ...]) -> str:
    return _join_lines(
        _per_observable(observables, "{o}[i]:seti(observe_count, mflw[i].{o}:get0())", level=3)
    )


def _save_knob_derivative_block(observables: tuple[str, ...]) -> str:
    return _join_lines(
        _per_observable(
            observables,
            "d{o}_dk[i]:setsub(set_range, observe_count, get_knob_vec(mflw[i].{o}))",
            level=4,
        )
    )


def _save_energy_derivative_block(observables: tuple[str, ...]) -> str:
    return _join_lines(
        _per_observable(
            observables,
            "d{o}_dk[i]:setsub(dpt_idx, observe_count, mflw[i].{o}:get(7))",
            level=4,
        )
    )


def _save_start_derivative_block(observables: tuple[str, ...]) -> str:
    """Store d(observable)/d(x0) and d(observable)/d(y0), TPSA monomials 2 and 4."""
    lines = _per_observable(
        observables, "d{o}_dc0[i]:set(1, observe_count, mflw[i].{o}:get(2))", level=4
    )
    lines += _per_observable(
        observables, "d{o}_dc0[i]:set(2, observe_count, mflw[i].{o}:get(4))", level=4
    )
    return _join_lines(lines)


def _reset_block(observables: tuple[str, ...], *, include_derivatives: bool = True) -> str:
    lines = _per_observable(observables, "{o}[i]:zeros()", level=2)
    if include_derivatives:
        lines += _per_observable(observables, "d{o}_dk[i]:zeros()", level=2)
        lines += _per_observable(observables, "d{o}_dc0[i]:zeros()", level=2)
    return _join_lines(lines)


def _send_block(
    observables: tuple[str, ...],
    *,
    include_derivatives: bool = True,
    include_start_derivatives: bool = False,
) -> str:
    lines = _per_observable(observables, f"{PYTHON_IN_MAD}:send({{o}}, true)")
    if include_derivatives:
        lines += _per_observable(observables, f"{PYTHON_IN_MAD}:send(d{{o}}_dk, true)")
    if include_start_derivatives:
        lines += _per_observable(observables, f"{PYTHON_IN_MAD}:send(d{{o}}_dc0, true)")
    return _join_lines(lines)


def _track_call(*, x0: str, atexit: str, level: int = 0) -> str:
    """Render a MAD ``track{...}`` call with the standard tracking arguments.

    Only ``x0`` and the ``atexit`` callback differ between the tracking,
    preflight, validation and Hessian scripts; ``level`` indents the whole call
    for use inside a ``for`` loop.
    """
    fields = [
        "sequence=loaded_sequence",
        f"X0={x0}",
        "nturn=n_run_turns",
        "save=false",
        f"atexit={atexit}",
        "range=tracking_range",
        "dir=sdir",
        "aperture={kind='circle', 100}",
        "method=6",
    ]
    return _join_lines(
        [
            _indent(level, "local _, mflw = track {"),
            *(_indent(level + 1, f"{field},") for field in fields),
            _indent(level, "}"),
        ]
    )


def _observation_gate_definition() -> str:
    """Return the MAD predicate shared by every observation callback.

    An element is an observation point on its exit slice (``slc == -4``) when it is
    flagged observed. Defining it once keeps ``save_data``, the validation saver and
    the preflight counter on exactly the same gate, so the preflight count can never
    drift from what the real tracking callbacks store.
    """
    return "function is_observation_point(elm, slc)\nreturn slc == -4 and elm:is_observed()\nend"


def build_tracking_init_script(
    observables: tuple[str, ...], *, start_on_first_turn: bool = False
) -> str:
    """Build the tracking initialisation script for the requested observables."""
    observables = _validate_observables(observables)
    initial_observe_count = "1" if start_on_first_turn else "nbpms + 1"

    return f"""! Generated tracking init script
num_knobs = #knob_names
local matrix_size = optimise_energy and (num_knobs + 1) or num_knobs
local set_range, get_range, get_knob_vec

if num_knobs > 0 then
    set_range = vector(num_knobs):seq()
    get_range = vector(num_knobs):seq(7)

    function get_knob_vec(coord_tpsa)
        local all_first_deriv = coord_tpsa:getvec(1, num_knobs + 7)
        return all_first_deriv:getvec(get_range)
    end
end

{_table_definitions(observables)}

for i=1,batch_size do
{_allocation_block(observables)}
end

{_observation_gate_definition()}

observe_count = {initial_observe_count}
-- Only the uncertainty propagation needs the start-coordinate derivatives.
save_start_derivatives = false
function save_data(elm, mflw, _, slc)
    if is_observation_point(elm, slc) then
        for i=1,batch_size do
{_save_scalar_block(observables)}

            if set_range then
{_save_knob_derivative_block(observables)}
            end

            if optimise_energy then
                local dpt_idx = num_knobs + 1
{_save_energy_derivative_block(observables)}
            end

            if save_start_derivatives then
{_save_start_derivative_block(observables)}
            end
        end
        observe_count = observe_count + 1
    end
end

function reset_before_tracking()
    observe_count = 1
    for i=1,batch_size do
{_reset_block(observables)}
    end
end
"""


def build_tracking_script(
    observables: tuple[str, ...], *, include_start_derivatives: bool = False
) -> str:
    """Build the tracking script for the requested observables.

    ``include_start_derivatives`` also sends ``d{o}_dc0``; the caller must set
    ``save_start_derivatives = true`` in MAD first.
    """
    observables = _validate_observables(observables)
    return f"""! Generated tracking script
reset_before_tracking()
{_track_call(x0="da_x0_c[batch]", atexit="save_data")}

local n_lost = 0
for i=1,batch_size do
    if mflw[i] and mflw[i].status == 'lost' then n_lost = n_lost + 1 end
end
{PYTHON_IN_MAD}:send({{n_lost=n_lost, n_total=batch_size}}, true)
{_send_block(observables, include_start_derivatives=include_start_derivatives)}
"""


def build_tracking_preflight_script() -> str:
    """Build the single-particle preflight script.

    Tracks one particle through the configured range/turns with a counting
    ``atexit`` and reports how many BPM points were observed. The worker compares
    this against the allocated result-vector size (``nbpms * n_run_turns``) once,
    before the optimisation loop, so an observe/range misconfiguration fails with a
    clear message here instead of as an opaque ``seti`` index-out-of-bounds deep in
    a tracking run. Observable-independent: only the observation geometry matters.
    """
    return f"""! Generated tracking preflight script
local observed = 0
local function preflight_counter(elm, mflw, _, slc)
    if is_observation_point(elm, slc) then observed = observed + 1 end
end
{_track_call(x0="da_x0_c[1][1]", atexit="preflight_counter")}
{PYTHON_IN_MAD}:send({{
    observed=observed,
    expected=nbpms * n_run_turns,
    lost=(mflw[1] and mflw[1].status == 'lost') or false,
}}, true)
"""


def build_validation_init_script(observables: tuple[str, ...]) -> str:
    """Build the validation initialisation script without derivative storage."""
    observables = _validate_observables(observables)

    return f"""! Generated validation init script
{_table_definitions(observables, include_derivatives=False)}

for i=1,batch_size do
{_allocation_block(observables, include_derivatives=False)}
end

{_observation_gate_definition()}

observe_count = nbpms + 1
function save_val_data(elm, mflw, _, slc)
    if is_observation_point(elm, slc) then
        for i=1,batch_size do
{_save_scalar_block(observables)}
        end
        observe_count = observe_count + 1
    end
end

function reset_before_validation()
    observe_count = 1
    for i=1,batch_size do
{_reset_block(observables, include_derivatives=False)}
    end
end
"""


def build_validation_script(observables: tuple[str, ...]) -> str:
    """Build the validation tracking script without derivative returns."""
    observables = _validate_observables(observables)
    return f"""! Generated validation script
reset_before_validation()
{_track_call(x0="da_x0_c[batch]", atexit="save_val_data")}

{_send_block(observables, include_derivatives=False)}
"""


if __name__ == "__main__":
    scripts = {
        "tracking init": build_tracking_init_script(TRACKING_OBSERVABLES),
        "tracking": build_tracking_script(TRACKING_OBSERVABLES),
        "tracking preflight": build_tracking_preflight_script(),
        "validation init": build_validation_init_script(TRACKING_OBSERVABLES),
        "validation": build_validation_script(TRACKING_OBSERVABLES),
        "tracking uncertainty": build_tracking_script(
            TRACKING_OBSERVABLES, include_start_derivatives=True
        ),
    }
    for name, script in scripts.items():
        print(f"{'=' * 80}\n{name}\n{'=' * 80}\n{script}")
