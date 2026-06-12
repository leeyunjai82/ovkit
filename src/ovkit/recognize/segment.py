"""Segmentation adapter (OMZ semantic + instance segmentation).

Two output families, auto-detected:

* **Semantic** — a single map: ``[1, C, H, W]`` (argmax over ``C``) or
  ``[1, 1, H, W]`` / ``[1, H, W]`` (already a class-index map). Returned as one
  ``(1, H, W)`` class map in :attr:`Results.masks`.
* **Instance** (Mask R-CNN-style, e.g. ``instance-segmentation-*``) — boxes
  ``[N, 5]`` (x1, y1, x2, y2, conf) + labels ``[N]`` + per-instance mask
  prototypes ``[N, h, w]``. Boxes land in :attr:`Results.boxes`; each prototype
  is resized into its box and pasted into a full-image per-instance mask in
  :attr:`Results.masks` (``(N, H, W)``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Boxes, Masks, Results
from .base import BaseAdapter


class SegmentAdapter(BaseAdapter):
    """Adapter for semantic and instance segmentation."""

    task = "segment"

    def run(self, backend: Backend, image: np.ndarray, *, conf: float = 0.25, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))  # OMZ seg: raw BGR
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)

        if self._is_instance(outputs):
            return self._run_instance(outputs, image, size, conf=conf)
        return self._run_semantic(outputs, image)

    # -- semantic ------------------------------------------------------------

    def _run_semantic(self, outputs: dict[str, np.ndarray], image: np.ndarray) -> Results:
        import cv2

        class_map = self._class_map(np.asarray(next(iter(outputs.values()))))
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

    # -- instance ------------------------------------------------------------

    @staticmethod
    def _is_instance(outputs: dict[str, np.ndarray]) -> bool:
        """Instance models emit a boxes [N,5] tensor plus per-instance masks."""
        shapes = [np.asarray(v).shape for v in outputs.values()]
        has_boxes = any(len(s) >= 2 and s[-1] == 5 for s in shapes)
        return len(shapes) >= 2 and has_boxes

    def _run_instance(
        self,
        outputs: dict[str, np.ndarray],
        image: np.ndarray,
        in_hw: tuple[int, int],
        *,
        conf: float,
    ) -> Results:
        import cv2

        boxes_arr = labels_arr = masks_arr = None
        for v in outputs.values():
            a = np.asarray(v)
            if a.ndim >= 2 and a.shape[-1] == 5:
                boxes_arr = a.reshape(-1, 5)
            elif a.ndim >= 3:
                masks_arr = a.reshape(-1, a.shape[-2], a.shape[-1])  # [N, h, w]
            elif a.ndim == 1 or (a.ndim == 2 and a.shape[-1] == 1):
                labels_arr = a.reshape(-1)
        if boxes_arr is None:
            return Results(
                image, task=self.task, names=self.names, masks=Masks(np.zeros((0, 1, 1)))
            )

        keep = boxes_arr[:, 4] >= conf
        boxes_arr = boxes_arr[keep]
        labels = (
            labels_arr[keep[: len(labels_arr)]]
            if labels_arr is not None and len(labels_arr) == len(keep)
            else np.zeros(len(boxes_arr))
        )
        protos = masks_arr[keep[: len(masks_arr)]] if masks_arr is not None else None

        h, w = image.shape[:2]
        ih, iw = in_hw
        xyxy = boxes_arr[:, :4].astype(np.float32)
        if xyxy.max(initial=0.0) > 2.0:  # input-pixel coords -> scale to image
            xyxy = xyxy / np.array([iw, ih, iw, ih], dtype=np.float32)
        xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)

        # Paste each mask prototype into its box on a full-size canvas.
        n = len(boxes_arr)
        full = np.zeros((n, h, w), dtype=np.uint8)
        if protos is not None:
            for i in range(n):
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
                bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
                m = cv2.resize(protos[i].astype(np.float32), (bw, bh))
                full[i, y1 : y1 + bh, x1 : x1 + bw] = (m[: h - y1, : w - x1] > 0.5).astype(np.uint8)

        data = np.concatenate(
            [xyxy, boxes_arr[:, 4:5], labels[:n].reshape(-1, 1).astype(np.float32)], axis=1
        )
        return Results(
            image, task=self.task, names=self.names, boxes=Boxes(data), masks=Masks(full)
        )
