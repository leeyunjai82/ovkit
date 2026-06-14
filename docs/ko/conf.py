"""Korean (ko) Sphinx config — extends the English config with overrides."""

from __future__ import annotations

import os

_here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_here, "..", "conf.py")).read())  # noqa: S102

language = "ko"
html_title = f"ovkit {release}"  # noqa: F821 (release from the English conf)
templates_path = ["_templates"]
html_static_path = ["../_static"]
html_css_files = ["custom.css"]
html_logo = "../_static/logo.svg"
html_favicon = "../_static/logo.svg"
