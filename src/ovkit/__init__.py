"""ovkit — a simple Python inference API for OpenVINO.

One import, one :class:`Model` class, a callable object, and clean
:class:`Results` — plus OpenVINO's strengths (AUTO/NPU devices, async, INT8).

Example
-------
>>> from ovkit import Model
>>> model = Model("rtdetr_r50")          # name -> auto download/convert/cache
>>> results = model("img.jpg", conf=0.25)
>>> for r in results:
...     print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
...     r.save("out.jpg")
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

__version__ = "0.1.0"

__all__ = [
    "Model",
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
