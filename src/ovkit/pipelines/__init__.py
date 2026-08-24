"""Composed capabilities — several models chained into one intuitive call.

``Model("face_detection")`` gives you boxes. What you usually want is *who is in
the frame, how old they look and whether they are smiling* — which is a
detector plus three more models plus the code to crop and join them. That is
what a pipeline is:

    from ovkit import Model

    for r in Model("face_analyze")("group.jpg"):
        print(r.summary())     # 2 faces: age 31 · male 98% · happy 92%, ...
        r.save("faces.jpg")

Every pipeline takes the same sources as :class:`~ovkit.Model` (path, ndarray,
folder, video, camera index) and returns the same :class:`~ovkit.Results`, so
they are drop-in replacements for a model anywhere in your code.

``ovkit.list_pipelines()`` shows every capability and what it does.
"""

from __future__ import annotations

from typing import Any

from .analyze import FaceAnalyzer, PersonAnalyzer, VehicleAnalyzer
from .base import Pipeline
from .gaze import GazeEstimator
from .reid import ReID
from .text import TextReader
from .tracking import Tracker

#: Every composed capability, by the name :func:`vis` accepts.
PIPELINES: dict[str, type[Pipeline]] = {
    "face_analyze": FaceAnalyzer,
    "person_analyze": PersonAnalyzer,
    "vehicle_analyze": VehicleAnalyzer,
    "read_text": TextReader,
    "track": Tracker,
    "gaze": GazeEstimator,
    "face_match": ReID,
}

#: Friendlier spellings people reach for first.
ALIASES = {
    "ocr": "read_text",
    "text": "read_text",
    "face": "face_analyze",
    "faces": "face_analyze",
    "emotion_age": "face_analyze",
    "person": "person_analyze",
    "people": "person_analyze",
    "vehicle": "vehicle_analyze",
    "car": "vehicle_analyze",
    "tracking": "track",
    "reid": "face_match",
    "match": "face_match",
}


def resolve_name(name: str) -> str | None:
    """Return the canonical pipeline name for ``name``, or ``None``."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    return key if key in PIPELINES else None


def is_pipeline(name: str) -> bool:
    """True when ``name`` is a composed capability rather than one network."""
    return resolve_name(name) is not None


def build_pipeline(name: str, device: str = "AUTO", **kwargs: Any) -> Pipeline:
    """Build a composed pipeline by name — normally reached via ``Model(name)``.

    >>> from ovkit import Model
    >>> Model("face_analyze")("group.jpg")[0].summary()
    >>> Model("read_text")("sign.jpg")[0].text
    >>> Model("track")(0)                   # webcam, ids kept across frames
    """
    key = resolve_name(name)
    if key is None:
        known = ", ".join(sorted(PIPELINES))
        raise ValueError(f"Unknown pipeline '{name}'. Available: {known}.")
    return PIPELINES[key](device=device, **kwargs)


def list_pipelines() -> dict[str, str]:
    """Return ``{name: description}`` for every composed pipeline."""
    return {name: cls.description for name, cls in sorted(PIPELINES.items())}


__all__ = [
    "build_pipeline",
    "is_pipeline",
    "resolve_name",
    "list_pipelines",
    "PIPELINES",
    "Pipeline",
    "FaceAnalyzer",
    "PersonAnalyzer",
    "VehicleAnalyzer",
    "TextReader",
    "Tracker",
    "GazeEstimator",
    "ReID",
]
