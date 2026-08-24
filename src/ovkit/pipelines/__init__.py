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
from .attention import AttentionAnalyzer
from .base import Pipeline
from .gaze import GazeEstimator
from .plates import PlateReader
from .privacy import Anonymizer
from .reid import ReID
from .scene import SceneReport
from .teach import Teach
from .temporal import DrowsinessMonitor, GestureRecognizer
from .text import TextReader
from .tracking import Tracker

#: Every composed capability, by the name :func:`vis` accepts.
PIPELINES: dict[str, type[Pipeline]] = {
    # describe what is in the frame
    "face_analyze": FaceAnalyzer,
    "person_analyze": PersonAnalyzer,
    "vehicle_analyze": VehicleAnalyzer,
    "scene": SceneReport,
    # read something out of it
    "read_text": TextReader,
    "read_plate": PlateReader,
    # follow it over time
    "track": Tracker,
    "gesture": GestureRecognizer,
    "drowsiness": DrowsinessMonitor,
    # where people are looking
    "gaze": GazeEstimator,
    "attention": AttentionAnalyzer,
    # identity: match it, or remove it
    "face_match": ReID,
    "anonymize": Anonymizer,
    # make your own
    "teach": Teach,
}

#: Friendlier spellings people reach for first.
ALIASES = {
    "ocr": "read_text",
    "anpr": "read_plate",
    "plate": "read_plate",
    "blur": "anonymize",
    "privacy": "anonymize",
    "sign_language": "gesture",
    "driver": "drowsiness",
    "sleepy": "drowsiness",
    "describe": "scene",
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
    from ..core.i18n import canonical

    key = canonical(str(name)).strip().lower()
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


def capability_using(network: str) -> str | None:
    """Which capability drives this network, if one does.

    A multi-input model like ``gaze_estimation_adas_0002`` cannot be run from a
    picture — but a pipeline builds its inputs. Looking that up here lets the
    error say so instead of leaving the caller stuck.
    """
    import inspect

    from ..core.registry import resolve

    for name, cls in PIPELINES.items():
        for param in inspect.signature(cls.__init__).parameters.values():
            default = param.default
            if not isinstance(default, str):
                continue
            if default == network:
                return name
            # Pipelines name their parts by friendly alias ("face_detection");
            # the error reports the canonical model, so compare both.
            entry = resolve(default)
            if entry is not None and entry.name == network:
                return name
    return None


def list_pipelines() -> dict[str, str]:
    """Return ``{name: description}`` for every composed pipeline."""
    return {name: cls.description for name, cls in sorted(PIPELINES.items())}


__all__ = [
    "build_pipeline",
    "capability_using",
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
    "AttentionAnalyzer",
    "Anonymizer",
    "DrowsinessMonitor",
    "GestureRecognizer",
    "PlateReader",
    "SceneReport",
    "Teach",
]
