"""Composed solutions — now implemented in :mod:`ovkit.pipelines`.

These names are kept as the documented import path; the implementations live
next to each other in ``ovkit.pipelines`` so they can share the source handling
and model caching. The friendly entry point is ``ovkit.vis(name)``.
"""

from __future__ import annotations

from ..pipelines import ReID, Tracker
from ..pipelines.text import TextReader as OCR
from .anomaly import AnomalyModel

__all__ = ["OCR", "ReID", "Tracker", "AnomalyModel"]
