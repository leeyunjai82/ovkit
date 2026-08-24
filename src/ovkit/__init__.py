"""ovkit — a simple Python inference API for OpenVINO.

One import, one :class:`Model` class, a callable object, and clean
:class:`Results` — plus OpenVINO's strengths (AUTO/NPU devices, async, INT8).

One model
---------
>>> from ovkit import Model
>>> model = Model("rtdetr_r50")          # name -> auto download/convert/cache
>>> results = model("img.jpg", conf=0.25)
>>> for r in results:
...     print(r.summary())               # '2x person, car'
...     r.save("out.jpg")

Several models, one answer
--------------------------
>>> from ovkit import Model
>>> for r in Model("face_analyze")("group.jpg"):
...     print(r.summary())               # '2 faces: age 31 · male 98% · happy 92%, ...'

A capability name composes the models the answer needs — detection plus
whatever describes what was found. :func:`list_pipelines` shows them all.
"""

from __future__ import annotations

from .core.errors import (
    ConversionError,
    DownloadError,
    GatedModelError,
    LicenseError,
    MirrorMissingError,
    ModelNotFoundError,
    OfflineError,
    OVKitError,
    TaskDetectionError,
)
from .core.model import Model
from .core.registry import list_models
from .core.results import Boxes, Keypoints, Masks, Probs, Results
from .pipelines import Pipeline, list_pipelines

__version__ = "0.1.2"

__all__ = [
    "Model",
    "Pipeline",
    "list_pipelines",
    "Results",
    "Boxes",
    "Masks",
    "Keypoints",
    "Probs",
    "list_models",
    "OVKitError",
    "ModelNotFoundError",
    "OfflineError",
    "DownloadError",
    "GatedModelError",
    "MirrorMissingError",
    "ConversionError",
    "TaskDetectionError",
    "LicenseError",
    "__version__",
]
