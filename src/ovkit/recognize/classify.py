"""Classification adapter — end-to-end for single-output ``[N, C]`` classifiers.

Runs preprocessing → inference → (optional) softmax → :class:`Results` with
``probs``. Most ImageNet-style models emit a single ``[N, C]`` (or
``[N, C, 1, 1]``) tensor; we squeeze it to ``C`` and expose ``top1``/``top5``.

Per-model preprocessing (input size, channel order, mean/std) comes from the
model's own input shape and the manifest ``preprocess`` field; OMZ classifiers
default to raw BGR ``[0, 255]`` (override via ``preprocess`` for timm-style RGB).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.constants import class_names
from ..core.results import Probs, Results
from .base import BaseAdapter


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class ClassifyAdapter(BaseAdapter):
    """Adapter for image classification."""

    task = "classify"

    def run(self, backend: Backend, image: np.ndarray, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))  # OMZ classifiers: raw BGR
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)

        arr = np.asarray(next(iter(outputs.values())))
        arr = arr[0] if arr.ndim >= 2 else arr  # drop batch dim
        scores = arr.ravel().astype(np.float32)
        if self.post.get("softmax", True):
            scores = _softmax(scores)

        names = self.names or class_names(self.post.get("classes"), len(scores))
        return Results(image, task=self.task, names=names, probs=Probs(scores))
