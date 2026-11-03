"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import importlib.metadata
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -- Project information -----------------------------------------------------

project = "adelmo"
author = "Joshua Gray"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"  # noqa: A001

try:
    release = importlib.metadata.version("adelmo")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    # Document each object only in the module that defines it, not again in the
    # package that re-exports it through ``__all__``.
    "ignore-module-all": True,
}
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
# Render "Attributes:" sections as field lists, so dataclass fields are not
# documented twice.
napoleon_use_ivar = True
autodoc_mock_imports = [
    "bokeh",
    "matplotlib",
    "nxcals",
    "numpy",
    "omc3",
    "pandas",
    "panel",
    "psutil",
    "pymadng",
    "pymadng_utils",
    "scipy",
    "tfs",
    "tmom_recon",
    "turn_by_turn",
    "uncertainties",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "reports",
    "Thumbs.db",
    ".DS_Store",
]

# Type aliases defined under ``TYPE_CHECKING`` cannot be resolved at build time.
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    "sphinx_autodoc_typehints.guarded_import",
]

todo_include_todos = True
nitpicky = False


def setup(app):  # type: ignore[override]
    """Inject custom configuration at Sphinx start-up."""
    app.add_css_file("custom.css")
    app.add_js_file("glossary_tooltips.js")


# -- Options for HTML output -------------------------------------------------

html_theme = "shibuya"
html_static_path = ["_static"]
html_title = "adelmo documentation"

# -- Intersphinx configuration ----------------------------------------------

# Keep the docs build self-contained in offline environments.
intersphinx_mapping: dict[str, tuple[str, str | None]] = {}

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
