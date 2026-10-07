"""The machine state a model is put in: name/value pairs applied to the MAD-NG ``MADX`` environment.

A name is either a MAD-X variable (``kbrqf``, ``kqtf.a12b1``, a corrector's ``kbr3dhz2l4``) or an
element attribute written ``<element>.<attribute>`` (``br3.dhz2l4.kick``). Element names are
upper case in the loaded sequence and matched case-insensitively. Assigning an element attribute
replaces its deferred expression (a kicker's ``kick := kbr3dhz2l4``) with the number.

A state is given as a mapping, as a ``name<TAB>value`` knobs file, or as a TFS corrector table
(columns ``ename``, ``kind``, ``hkick``/``vkick``), which becomes one ``<ename>.<attribute>`` entry per kicker.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import tfs
from pymadng_utils.io.utils import read_knobs

if TYPE_CHECKING:
    from pymadng import MAD

MachineState: TypeAlias = Mapping[str, float] | str | Path

#: Kicker kinds of a corrector table: the attribute set and the table column holding its value.
_KICKER_ATTRIBUTES: dict[str, tuple[tuple[str, str], ...]] = {
    "hkicker": (("kick", "hkick"),),
    "vkicker": (("kick", "vkick"),),
    "tkicker": (("hkick", "hkick"), ("vkick", "vkick")),
}
_TABLE_COLUMNS = {"ename", "kind", "hkick", "hkick_old", "vkick", "vkick_old"}


def resolve_machine_state(state: MachineState) -> dict[str, float]:
    """Name/value pairs of *state*, whether given directly, as a knobs file or as a TFS corrector table."""
    if not isinstance(state, str | Path):
        return {str(name): float(value) for name, value in state.items()}
    path = Path(state)
    if path.suffix.lower() == ".tfs":
        return _corrector_table_state(tfs.read(path))
    return {name: float(value) for name, value in read_knobs(path).items()}


def merge_machine_states(*states: MachineState | None) -> dict[str, float] | None:
    """One state from several, later ones overriding earlier ones; ``None`` when none is given."""
    given = [state for state in states if state is not None]
    if not given:
        return None
    merged: dict[str, float] = {}
    for state in given:
        merged.update(resolve_machine_state(state))
    return merged


def _corrector_table_state(table: tfs.TfsDataFrame) -> dict[str, float]:
    """One ``<ename>.<attribute>`` entry per kicker whose strength differs from ``*_old``."""
    missing = _TABLE_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(
            f"Corrector table is missing required columns: {', '.join(sorted(missing))}"
        )
    changed = (table["hkick"] != table["hkick_old"]) | (table["vkick"] != table["vkick_old"])
    state = {}
    for row in table[changed].itertuples():
        for attribute, column in _KICKER_ATTRIBUTES.get(row.kind, ()):
            state[f"{row.ename}.{attribute}"] = float(getattr(row, column))
    return state


def _key(name: str) -> str:
    """Lua expression for the ``MADX.__var`` key of *name*: MAD-NG stores a dotted variable (``k_hcor_qf2a.56``) with ``.`` as ``_``,
    although ``MADX['k_hcor_qf2a.56']`` maps it on access (a raw ``__var`` lookup does not)."""
    return repr(name.replace(".", "_"))


def check_state_names(mad: MAD, py_name: str, names: list[str]) -> None:
    """Raise ``ValueError`` for a name that is neither a MAD-X variable nor an element attribute.

    Indexing ``MADX`` with an unknown name silently creates it as zero, so existence is tested first.
    """
    for name in names:
        element, _, attribute = name.rpartition(".")
        mad.send(
            f"{py_name}:send(MADX.__var[{_key(name)}] ~= nil or "
            f"('{element}' ~= '' and loaded_sequence[('{element}'):upper()] ~= nil and loaded_sequence[('{element}'):upper()]['{attribute}'] ~= nil))"
        )
        if not mad.recv():
            raise ValueError(
                f"machine_state names {name!r}, which is neither a MAD-X variable nor an element attribute"
            )


def read_state(mad: MAD, py_name: str, names: list[str]) -> dict[str, float]:
    """The current value of each name (checked first, see :func:`check_state_names`)."""
    check_state_names(mad, py_name, names)
    values = {}
    for name in names:
        element, _, attribute = name.rpartition(".")
        mad.send(
            f"{py_name}:send(MADX.__var[{_key(name)}] ~= nil and MADX['{name}'] or loaded_sequence[('{element}'):upper()]['{attribute}'])"
        )
        values[name] = float(mad.recv())
    return values


def assign_state(mad: MAD, state: Mapping[str, float]) -> None:
    """Set every name of *state*: an existing variable, else an element attribute, else a new variable."""
    if state:
        mad.send("\n".join(_assignment(name, value) for name, value in state.items()))


def _assignment(name: str, value: float) -> str:
    element, _, attribute = name.rpartition(".")
    target = f"loaded_sequence[('{element}'):upper()]"
    return (
        f"if MADX.__var[{_key(name)}] ~= nil then MADX['{name}'] = {value!r} "
        f"elseif '{element}' ~= '' and {target} ~= nil and {target}['{attribute}'] ~= nil then {target}['{attribute}'] = {value!r} "
        f"else MADX['{name}'] = {value!r} end"
    )
