"""Semantic segmentation adapter (OMZ segmentation models).

Handles the two common OMZ output shapes:

* ``[1, C, H, W]`` — per-class scores; the class map is ``argmax`` over ``C``.
* ``[1, 1, H, W]`` / ``[1, H, W]`` — an already-decoded class-index map.

The class map is resized (nearest-neighbour) back to the original image and
returned as a single ``(1, H, W)`` entry in :attr:`Results.masks`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Masks, Results
from .base import BaseAdapter


class SegmentAdapter(BaseAdapter):
    """Adapter for semantic segmentation."""

    task = "segment"

    def run(self, backend: Backend, image: np.ndarray, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))  # OMZ seg: raw BGR
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)

        class_map = self._class_map(np.asarray(next(iter(outputs.values()))))

        import cv2

        h, w = image.shape[:2]
        class_map = cv2.resize(class_map.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        return Results(image, task=self.task, names=self.names, masks=Masks(class_map[None]))

    @staticmethod
    def _class_map(out: np.ndarray) -> np.ndarray:
        """Reduce a raw segmentation output to a 2-D ``(H, W)`` class map."""
        a = out[0] if out.ndim == 4 else out  # -> [C, H, W] / [1, H, W] / [H, W]
        if a.ndim == 3:
            return a[0] if a.shape[0] == 1 else a.argmax(axis=0)
        return a
