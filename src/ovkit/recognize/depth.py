"""Depth and background adapters — a map of floats made into a picture.

Both models answer with a single-channel map the size of the input, and both
are useless as raw numbers: what a caller wants is a colourised depth image,
or the subject cut out of its background. Turning the map into that, and into
a sentence, is this module's whole job.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.i18n import lang, position
from ..core.results import Results
from .base import BaseAdapter


def _msg(ko: str, en: str) -> str:
    return ko if lang() == "ko" else en


def _single_map(outputs: dict[str, np.ndarray]) -> np.ndarray:
    """The first output squeezed to ``(H, W)`` floats."""
    arr = np.asarray(next(iter(outputs.values())), np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _to_original(map_2d: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(map_2d, (width, height), interpolation=cv2.INTER_LINEAR)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    low, high = float(arr.min()), float(arr.max())
    return (arr - low) / (high - low) if high - low > 1e-9 else np.zeros_like(arr)


class DepthAdapter(BaseAdapter):
    """Monocular depth: a colourised map, plus where the nearest thing is."""

    task = "depth"

    def run(self, backend: Backend, image: np.ndarray, **_: Any) -> Results:
        import cv2

        size = self.model_input_hw(backend)
        feed = self.preprocess(image, size, rgb=bool(self.pre.get("rgb", True)))
        outputs = backend.infer(feed)

        h, w = image.shape[:2]
        depth = _normalize01(_to_original(_single_map(outputs), h, w))

        result = Results(image, task=self.task, tensors={"depth": depth})
        # Depth Anything emits *inverse* depth: bright is near.
        near_y, near_x = np.unravel_index(int(np.argmax(depth)), depth.shape)
        where = position(float(near_x), float(near_y), w, h)
        close = float((depth > 0.7).mean() * 100)
        result.text = _msg(
            f"가장 가까운 곳: {where} · 화면의 {close:.0f}%가 가까움",
            f"nearest: {where} · {close:.0f}% of the frame is close",
        )
        result.display = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        return result


class BackgroundAdapter(BaseAdapter):
    """Salient-object segmentation: keep the subject, drop everything else."""

    task = "background"

    def run(self, backend: Backend, image: np.ndarray, *, conf: float = 0.5, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        feed = self.preprocess(image, size, rgb=bool(self.pre.get("rgb", True)))
        outputs = backend.infer(feed)

        h, w = image.shape[:2]
        # U2-Net stacks its side outputs; the first is the fused prediction.
        alpha = _normalize01(_to_original(_single_map(outputs), h, w))
        mask = (alpha >= float(conf)).astype(np.uint8)

        result = Results(image, task=self.task, tensors={"alpha": alpha})
        covered = float(mask.mean() * 100)
        result.text = (
            _msg(f"피사체가 화면의 {covered:.0f}%", f"subject covers {covered:.0f}% of the frame")
            if covered >= 1
            else _msg("뚜렷한 피사체를 못 찾았어요", "no clear subject found")
        )
        # What you see is the subject on a neutral ground; save(".png") keeps
        # the transparency instead.
        result.display = (image * alpha[:, :, None]).astype(np.uint8)
        return result
