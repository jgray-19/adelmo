"""Imports between the packages only point down the layers.

``tracking`` and ``poco`` are the two fitting engines and never import each other;
both build on ``fitting``, which builds on ``machine``. Imports under
``TYPE_CHECKING`` count too: a layer should not need to know about the ones above.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import adelmo

PACKAGE_ROOT = Path(adelmo.__file__).parent

#: Packages each package must never import.
FORBIDDEN = {
    "machine": {"fitting", "tracking", "poco", "measurements", "analysis", "momentum_reference"},
    "optimisers": {"machine", "fitting", "tracking", "poco"},
    "fitting": {"tracking", "poco", "momentum_reference"},
    "tracking": {"poco", "momentum_reference"},
    "poco": {"tracking", "momentum_reference"},
}


def _imported_packages(path: Path) -> set[str]:
    packages = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "adelmo" and len(parts) > 1:
                packages.add(parts[1])
    return packages


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_imports_only_lower_layers(package: str) -> None:
    violations = {
        str(path.relative_to(PACKAGE_ROOT)): sorted(_imported_packages(path) & FORBIDDEN[package])
        for path in (PACKAGE_ROOT / package).rglob("*.py")
    }
    assert {path: found for path, found in violations.items() if found} == {}
