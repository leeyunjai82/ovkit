"""Pose-estimation adapter (OMZ human-pose models) — heatmap peak decoding.

OMZ pose models emit keypoint heatmaps ``[1, K, H, W]`` (bottom-up models also
emit part-affinity fields as a second, wider tensor — we take the heatmap, the
one with fewer channels). Each keypoint is the per-channel peak.

v0 decodes one set of keypoints (the dominant peak per channel); full
multi-person grouping (PAF / associative-embedding) is a future refinement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Keypoints, Results
from .base import BaseAdapter


class PoseAdapter(BaseAdapter):
    """Adapter for keypoint/pose estimation (single-instance heatmap peaks)."""

    task = "pose"

    def run(self, backend: Backend, image: np.ndarray, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)

        heat = self._heatmap(outputs)
        kpts = self._peaks(heat, image.shape[:2])
        return Results(image, task=self.task, names=self.names, keypoints=Keypoints(kpts))

    @staticmethod
    def _heatmap(outputs: dict[str, np.ndarray]) -> np.ndarray:
        """Pick the keypoint heatmap ``[K, H, W]`` (fewest channels) from outputs."""
        maps = []
        for v in outputs.values():
            a = np.asarray(v, dtype=np.float32)
            a = a[0] if a.ndim == 4 else a
            if a.ndim == 3:
                maps.append(a)
        if not maps:
            raise ValueError("No [K, H, W] heatmap found among pose outputs.")
        return min(maps, key=lambda a: a.shape[0])

    @staticmethod
    def _peaks(heat: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
        """Return ``(1, K, 3)`` ``[x, y, conf]`` from per-channel argmax peaks."""
        k, hh, ww = heat.shape
        h, w = orig_hw
        out = np.zeros((1, k, 3), dtype=np.float32)
        for i in range(k):
            flat = int(np.argmax(heat[i]))
            y, x = divmod(flat, ww)
            out[0, i] = [x / ww * w, y / hh * h, float(heat[i].flat[flat])]
        return out
