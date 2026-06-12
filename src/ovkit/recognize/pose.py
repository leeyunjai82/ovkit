"""Pose-estimation adapter (OMZ human-pose models) — multi-peak heatmap decoding.

OMZ pose models emit keypoint heatmaps ``[1, K, H, W]`` (bottom-up models also
emit part-affinity fields as a second, wider tensor — we take the heatmap, the
one with fewer channels). For each keypoint channel we extract **all local
maxima** above a threshold, then group peaks into person instances by their
rank (strongest peaks per channel = instance 0, next = instance 1, ...).

This handles multiple people without full PAF association; limb-accurate
grouping (true PAF/AE decoding) can replace the rank-based grouping later
without changing the public API.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Keypoints, Results
from .base import BaseAdapter


class PoseAdapter(BaseAdapter):
    """Adapter for keypoint/pose estimation (multi-instance heatmap peaks)."""

    task = "pose"

    def run(self, backend: Backend, image: np.ndarray, *, conf: float = 0.1, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)

        heat = self._heatmap(outputs)
        kpts = self._peaks(heat, image.shape[:2], thr=float(self.post.get("thr", conf)))
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
    def _local_maxima(ch: np.ndarray, thr: float, limit: int = 10) -> list[tuple[int, int, float]]:
        """Return up to ``limit`` strict local maxima (x, y, score) above ``thr``."""
        h, w = ch.shape
        pad = np.full((h + 2, w + 2), -np.inf, dtype=np.float32)
        pad[1:-1, 1:-1] = ch
        center = pad[1:-1, 1:-1]
        is_peak = (
            (center >= pad[:-2, 1:-1])
            & (center >= pad[2:, 1:-1])
            & (center >= pad[1:-1, :-2])
            & (center >= pad[1:-1, 2:])
            & (center > thr)
        )
        ys, xs = np.nonzero(is_peak)
        scores = center[ys, xs]
        order = np.argsort(scores)[::-1][:limit]
        return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]

    def _peaks(self, heat: np.ndarray, orig_hw: tuple[int, int], thr: float) -> np.ndarray:
        """Return ``(N, K, 3)`` ``[x, y, conf]`` grouping peaks by rank."""
        k, hh, ww = heat.shape
        h, w = orig_hw
        per_channel = [self._local_maxima(heat[i], thr) for i in range(k)]
        n = max((len(p) for p in per_channel), default=0)
        if n == 0:
            return np.zeros((0, k, 3), dtype=np.float32)

        out = np.zeros((n, k, 3), dtype=np.float32)
        for ki, peaks in enumerate(per_channel):
            for rank, (x, y, s) in enumerate(peaks):
                out[rank, ki] = [x / ww * w, y / hh * h, s]
        return out
