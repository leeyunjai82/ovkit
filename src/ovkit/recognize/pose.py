"""Pose-estimation adapter — INTERFACE STUB.

TODO(v0+): implement keypoint decoding (heatmaps or regression heads) and
register a permissive pose model in the manifest.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Results
from .base import BaseAdapter


class PoseAdapter(BaseAdapter):
    """Adapter for keypoint/pose estimation (not yet implemented)."""

    task = "pose"

    def run(self, backend: Backend, image: np.ndarray, **kwargs: Any) -> Results:
        raise NotImplementedError(
            "Pose estimation is not implemented in v0. "
            "Planned: keypoint heatmap/regression decoding. Contributions welcome."
        )
