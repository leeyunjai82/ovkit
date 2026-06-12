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
        return self.preprocess(image, (self.imgsz, self.imgsz), rgb=rgb)

    def model_input_hw(self, backend: Backend) -> tuple[int, int]:
        """Return the model's static spatial ``(h, w)`` (last two dims), or ``imgsz``.

        Works for 4-D ``[N,C,H,W]`` and higher-rank inputs (e.g. video clips
        ``[N,C,T,H,W]``); only the trailing two dims are taken as ``H, W``.
        """
        shape = backend.input_shape  # full shape, -1 for dynamic dims
        if len(shape) >= 2 and shape[-1] and shape[-1] > 0 and shape[-2] and shape[-2] > 0:
            return int(shape[-2]), int(shape[-1])
        return self.imgsz, self.imgsz

    def preprocess(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        rgb: bool = True,
        scale: float | None = None,
    ) -> np.ndarray:
        """Resize to ``(h, w)``, normalize, and return an NCHW float tensor.

        ``scale`` overrides the manifest/default divisor (e.g. ``1.0`` to keep
        raw ``[0, 255]`` input, ``255.0`` to map to ``[0, 1]``).
        """
        from ..image import ops

        h, w = size
        img = ops.resize(image, (w, h))
        if rgb:
            img = ops.bgr_to_rgb(img)
        arr = img.astype(np.float32)
        default_scale, mean, std = self._scale_mean_std()
        scale = default_scale if scale is None else float(scale)
        if scale != 1.0:
            arr = arr / scale
        if np.any(mean != 0.0) or np.any(std != 1.0):
            arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None]  # HWC -> NCHW
        return np.ascontiguousarray(arr, dtype=np.float32)
