"""Sphinx configuration for sheaf_mpnn."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# -- Path setup --------------------------------------------------------------
# Make the source packages importable without installation (CI installs them,
# but local builds may not).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# Resolve the GitHub repo slug for source/edit links and Pages URLs. In CI,
# `GITHUB_REPOSITORY` is set automatically (e.g. "alessioborgi/pytorch-..."),
# so the same config works for any fork without further edits.
_REPO_SLUG = os.environ.get(
    "GITHUB_REPOSITORY", "alessioborgi/pytorch-SheafNeuralNetworks"
)
_REPO_OWNER, _REPO_NAME = _REPO_SLUG.split("/", 1)
_REPO_URL = f"https://github.com/{_REPO_SLUG}"
_PAGES_URL = f"https://{_REPO_OWNER}.github.io/{_REPO_NAME}/"

# -- Project information -----------------------------------------------------
project = "sheaf_mpnn"
author = "Sheaf MPNN contributors"
copyright = f"{datetime.now().year}, {author}"

try:
    release = _pkg_version("sheaf_mpnn")
except PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "undoc-members": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
autodoc_preserve_defaults = True
always_document_param_types = True
typehints_use_rtype = True
typehints_defaults = "comma"

# Napoleon (Google-style docstrings — matches ruff pydocstyle convention).
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "tasklist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "logo": {
        "text": project,
    },
    "use_edit_page_button": True,
    "github_url": _REPO_URL,
    "icon_links": [],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_depth": 4,
    "show_nav_level": 1,
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": ["sphinx-version"],
    "primary_sidebar_end": ["indices.html"],
    "secondary_sidebar_items": ["page-toc"],
}

html_context = {
    "github_user": _REPO_OWNER,
    "github_repo": _REPO_NAME,
    "github_version": "main",
    "doc_path": "docs",
}

# -- MathJax -----------------------------------------------------------------
mathjax3_config = {
    "tex": {
        "macros": {
            "R": r"\mathbb{R}",
            "stalks": r"\mathcal{F}",
            "Lap": r"\mathbf{L}_{\mathcal{F}}",
        },
        "tags": "ams",
    },
}

# -- Copy button -------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: "
copybutton_prompt_is_regexp = True

# -- OpenGraph ---------------------------------------------------------------
ogp_site_url = _PAGES_URL

# -- Linkcheck configuration -------------------------------------------------
linkcheck_ignore = [
    # Ignore workflow files and repository blobs to avoid GitHub rate-limiting
    rf"{_REPO_URL}/blob/.*",
    r"https://github\.com/.*\.yml",
]

linkcheck_anchors_ignore_for_url = [
    # PyTorch pages use JS/dynamic layouts rendering static anchor checks useless
    r"https://docs\.pytorch\.org/.*",
    r"https://pytorch\.org/.*",
]

# -- Misc --------------------------------------------------------------------
nitpicky = False  # Toggle once docstrings are fully typed.
todo_include_todos = True
add_module_names = False
python_use_unqualified_type_names = True

suppress_warnings = [
    "autosummary",  # tolerate autosummary stubs during incremental rollout
    "ref.obj",      # inherited PyG attrs (self.raw_dir, self.processed_dir, etc.)
    "ref.meth",     # inherited Lightning methods (prepare_data, optimizer_step, etc.)
]

# -- Custom roles for inherited third-party docstrings -----------------------
from docutils import nodes  # noqa: E402
from docutils.parsers.rst import roles as _roles  # noqa: E402


def _passthrough_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    return [nodes.literal(rawtext, text)], []


def setup(app):
    app.add_role("pyg", _passthrough_role)
    app.add_role("paramref", _passthrough_role)
