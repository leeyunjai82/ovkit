"""Sphinx configuration for the ovkit documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the package importable for autodoc without requiring an install.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "ovkit"
author = "ovkit contributors"
copyright = f"{datetime.now():%Y}, {author}"

try:
    from ovkit import __version__ as release
except Exception:  # pragma: no cover - docs build without install
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Autodoc imports the package; mock heavy optional deps so the build never needs
# them. (Core deps like numpy/openvino are imported lazily inside functions.)
autodoc_mock_imports = [
    "cv2",
    "openvino",
    "openvino_genai",
    "nncf",
    "anomalib",
    "huggingface_hub",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"ovkit {release}"
html_static_path = ["_static"]
