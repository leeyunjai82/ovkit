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


#: Known multi-head output names (documented OMZ interfaces), keyed by output name.
_KNOWN_HEADS = {
    "type": ["car", "bus", "truck", "van"],
    "color": ["white", "gray", "yellow", "red", "green", "blue", "black"],
}

#: Landmark regressors emit [1, 2K] normalized coords; K in {5, 35, 98}.
_LANDMARK_SIZES = frozenset({10, 70, 196})


def _looks_like_embedding(scores: np.ndarray) -> bool:
    """True for a descriptor vector rather than a class distribution.

    Embeddings are unnormalized (they carry negatives and do not sum to one),
    while classifier outputs are logits or probabilities over a class table.
    """
    if scores.size < 64:
        return False
    total = float(np.sum(scores))
    return bool(np.any(scores < -0.01)) and not (0.99 <= total <= 1.01)


def _looks_like_multilabel(scores: np.ndarray) -> bool:
    """True for a small vector of independent sigmoids (attribute heads)."""
    if not 2 <= scores.size <= 16:
        return False
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        return False
    return float(np.sum(scores)) > 1.05  # a distribution would sum to ~1


class ClassifyAdapter(BaseAdapter):
    """Adapter for image classification."""

    task = "classify"

    def run(self, backend: Backend, image: np.ndarray, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))  # OMZ classifiers: raw BGR
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)

        arrs = {n: np.asarray(a) for n, a in outputs.items()}
        if len(arrs) > 1:
            return self._multi_head(image, arrs, outputs)

        arr = next(iter(arrs.values()))
        arr = arr[0] if arr.ndim >= 2 else arr  # drop batch dim
        scores = arr.ravel().astype(np.float32)

        # Some OMZ "classify"-labelled models are really landmark regressors
        # ([1, 2K] coords in [0, 1] — not a probability distribution). Showing a
        # "top-1 class" for those is meaningless; return keypoints instead.
        if (
            self.post.get("classes") is None
            and scores.size in _LANDMARK_SIZES
            and np.all(np.isfinite(scores))
            and float(scores.min()) >= -0.5
            and float(scores.max()) <= 1.5
            and float(scores.sum()) > 1.5  # a softmax over K classes sums to 1
        ):
            h, w = image.shape[:2]
            pts = scores.reshape(-1, 2)
            kpts = np.stack([pts[:, 0] * w, pts[:, 1] * h, np.ones(len(pts), np.float32)], axis=1)
            from ..core.results import Keypoints

            return Results(image, task=self.task, names=self.names, keypoints=Keypoints(kpts[None]))

        # Re-identification / retrieval models are registered as "classify"
        # because they emit one vector, but that vector is a descriptor, not a
        # distribution — "top1 class_190" is meaningless for them.
        if self.post.get("kind") == "embedding" or (
            self.post.get("classes") is None and _looks_like_embedding(scores)
        ):
            r = Results(image, task=self.task, names=self.names, tensors=outputs)
            r.text = f"{scores.size}-d embedding (compare with cosine similarity)"
            return r

        # Attribute heads emit independent sigmoids, not a distribution: every
        # attribute above the threshold is "on", so a single top-1 hides most
        # of the answer.
        if self.post.get("classes") is None and _looks_like_multilabel(scores):
            labels = self.names or class_names(self.post.get("attributes"), len(scores))
            hits = np.where(scores >= 0.5)[0]
            order = hits[np.argsort(scores[hits])[::-1]][:5]
            on = [f"{labels.get(int(i), f'attr_{i}')} {scores[i]:.2f}" for i in order]
            r = Results(image, task=self.task, names=labels, probs=Probs(scores))
            more = f" (+{len(hits) - len(order)} more)" if len(hits) > len(order) else ""
            r.text = (" · ".join(on) + more) if on else "no attribute above 0.5"
            return r

        if self.post.get("softmax", True):
            scores = _softmax(scores)

        names = self.names or class_names(self.post.get("classes"), len(scores))
        r = Results(image, task=self.task, names=names, probs=Probs(scores))
        top = int(np.argmax(scores))
        r.text = f"{names.get(top, f'class_{top}')} {float(scores[top]):.2f}"
        return r

    def _multi_head(
        self, image: np.ndarray, arrs: dict[str, np.ndarray], outputs: dict[str, Any]
    ) -> Results:
        """Multi-output classifier (e.g. vehicle type + color): one line per head."""
        parts: list[str] = []
        first: np.ndarray | None = None
        for name, arr in arrs.items():
            flat = arr.reshape(-1).astype(np.float32)
            if flat.size == 0:
                continue
            if first is None:
                first = flat
            labels = _KNOWN_HEADS.get(name.lower())
            idx = int(np.argmax(flat))
            label = labels[idx] if labels and idx < len(labels) else f"#{idx}"
            parts.append(f"{name}: {label} ({float(flat[idx]):.2f})")
        r = Results(
            image,
            task=self.task,
            names=self.names,
            probs=Probs(first) if first is not None else None,
            tensors=outputs,
        )
        r.text = " · ".join(parts)
        return r
