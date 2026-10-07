"""CLI spelling and helpers for accelerator ``errors`` / ``misalignments`` selections.

Every :class:`~aba_optimiser.machine.accelerators.base.Accelerator` takes ``errors`` and
``misalignments`` as mappings from magnet family (keys of ``FAMILIES``) to the
attributes to free. This module holds the one CLI spelling of those mappings
(``--errors quad:k1 bend:k0``, ``--misalign quad:tilt,dy``) and the helpers that
combine and describe them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from aba_optimiser.machine.accelerators.base import Accelerator

Selection = dict[str, set[str]]

#: Token that selects nothing, so an empty selection can be spelled on the CLI.
NONE_TOKEN = "none"


def parse_selection(tokens: Iterable[str] | None) -> Selection:
    """Parse ``["quad:k1,tilt", "bend:k0"]`` into ``{"quad": {"k1", "tilt"}, ...}``.

    Repeated families are merged. ``none`` (alone) yields an empty selection.
    """
    selection: Selection = {}
    for token in tokens or ():
        token = token.strip()
        if not token or token.lower() == NONE_TOKEN:
            continue
        family, sep, attrs = token.partition(":")
        names = {attr.strip() for attr in attrs.split(",") if attr.strip()}
        if not sep or not family.strip() or not names:
            raise ValueError(
                f"Invalid selection {token!r}: expected FAMILY:ATTR[,ATTR...], "
                "e.g. quad:k1,tilt"
            )
        selection.setdefault(family.strip(), set()).update(names)
    return selection


def merge_selections(*selections: Mapping[str, Iterable[str]] | None) -> Selection:
    """Union several selections, dropping families left empty."""
    merged: Selection = {}
    for selection in selections:
        for family, attrs in (selection or {}).items():
            merged.setdefault(family, set()).update(attrs)
    return {family: attrs for family, attrs in merged.items() if attrs}


def selects(selection: Mapping[str, Iterable[str]] | None, family: str, attr: str) -> bool:
    """Whether *selection* frees *attr* on *family*."""
    return attr in set((selection or {}).get(family, ()))


def describe_selections(
    errors: Mapping[str, Iterable[str]] | None,
    misalignments: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """Human-readable labels such as ``["quad k1", "bend k0", "quad dy"]``."""
    return [
        f"{family} {attr}"
        for selection in (errors, misalignments)
        for family, attrs in (selection or {}).items()
        for attr in sorted(attrs)
    ]


def _families_help(accelerator: type[Accelerator] | None, allowed: str) -> str:
    if accelerator is None:
        return ""
    parts = [
        f"{name} ({','.join(sorted(getattr(family, allowed)))})"
        for name, family in accelerator.FAMILIES.items()
        if getattr(family, allowed)
    ]
    return f" {accelerator.__name__} families: {', '.join(parts)}."


def add_selection_args(
    parser: Any,
    *,
    accelerator: type[Accelerator] | None = None,
    errors_default: Sequence[str] = (),
    misalign_default: Sequence[str] = (),
) -> None:
    """Add ``--errors`` / ``--misalign`` to *parser* (raw tokens; see :func:`parse_selection`).

    Pass *accelerator* to list its families and valid attributes in the help text.
    """
    parser.add_argument(
        "--errors",
        nargs="*",
        default=list(errors_default),
        metavar="FAMILY:ATTR[,ATTR]",
        help="Field errors to fit, per magnet family, e.g. 'quad:k1 bend:k0'. "
        "'none' selects nothing." + _families_help(accelerator, "errors"),
    )
    parser.add_argument(
        "--misalign",
        nargs="*",
        default=list(misalign_default),
        metavar="FAMILY:ATTR[,ATTR]",
        help="Misalignments to fit, per magnet family, e.g. 'quad:tilt,dy'. "
        "'none' selects nothing." + _families_help(accelerator, "misalignments"),
    )
