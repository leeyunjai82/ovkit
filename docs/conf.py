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
    "sphinx_design",
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

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "attrs_inline"]
myst_heading_anchors = 3
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
# The Korean docs live in docs/ko/ (built separately); keep them out of the
# English build so autodoc objects aren't documented twice.
exclude_patterns = ["_build", "ko", "ko/**", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"ovkit {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

_BRAND = "#0a7d8c"
_BRAND_DARK = "#22b8cf"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/leeyunjai82/ovkit/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/leeyunjai82/ovkit",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
                "5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49"
                "-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08"
                ".58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64"
                "-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21"
                " 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2"
                "-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65"
                "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01"
                ' 8.01 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
    "light_css_variables": {
        "color-brand-primary": _BRAND,
        "color-brand-content": _BRAND,
        "font-stack": "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        "font-stack--monospace": "JetBrains Mono, ui-monospace, SFMono-Regular, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": _BRAND_DARK,
        "color-brand-content": _BRAND_DARK,
    },
}

# Language switcher (EN <-> KO) at the top of the sidebar.
html_sidebars = {
    "**": [
        "lang-switch.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
