"""Segmentation adapter — INTERFACE STUB.

TODO(v0+): implement mask decoding (e.g. prototype masks + coefficients, or a
per-pixel logits head) and register a permissive segmentation model in the
manifest.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Results
from .base import BaseAdapter


class SegmentAdapter(BaseAdapter):
    """Adapter for instance/semantic segmentation (not yet implemented)."""

    task = "segment"

    def run(self, backend: Backend, image: np.ndarray, **kwargs: Any) -> Results:
        raise NotImplementedError(
            "Segmentation is not implemented in v0. "
            "Planned: prototype masks + per-instance coefficients. Contributions welcome."
        )
