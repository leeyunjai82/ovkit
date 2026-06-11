"""Base class shared by task adapters.

An *adapter* turns a generic :class:`~ovkit.core.backend.Backend` plus an input
image into a task-specific :class:`~ovkit.core.results.Results`. The adapter owns
the pre/post-processing; the backend only moves tensors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Results


class BaseAdapter:
    """Common configuration + interface for every task adapter."""

    task: str = "base"

    def __init__(
        self,
        *,
        imgsz: int = 640,
        preprocess: dict[str, Any] | None = None,
        postprocess: dict[str, Any] | None = None,
        names: dict[int, str] | None = None,
    ) -> None:
        self.imgsz = imgsz
        self.pre = preprocess or {}
        self.post = postprocess or {}
        self.names = names or {}

    def run(self, backend: Backend, image: np.ndarray, **kwargs: Any) -> Results:
        """Run the full pre/infer/post pipeline for a single image."""
        raise NotImplementedError

    # -- shared preprocessing helpers --------------------------------------

    def _scale_mean_std(self) -> tuple[float, np.ndarray, np.ndarray]:
        scale = float(self.pre.get("scale", 255.0))
        mean = np.asarray(self.pre.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        std = np.asarray(self.pre.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
        return scale, mean, std

    def preprocess_square(self, image: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Resize to ``imgsz`` square, normalize, and return an NCHW float tensor."""
        from ..image import ops

        img = ops.resize(image, self.imgsz)
        if rgb:
            img = ops.bgr_to_rgb(img)
        arr = img.astype(np.float32)
        scale, mean, std = self._scale_mean_std()
        if scale != 1.0:
            arr = arr / scale
        if np.any(mean != 0.0) or np.any(std != 1.0):
            arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None]  # HWC -> NCHW
        return np.ascontiguousarray(arr, dtype=np.float32)
